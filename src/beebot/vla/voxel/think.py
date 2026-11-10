"""The think-then-act voxel policy: stream.py's plain causal stream, with a reply that first thinks
in words and then either acts (the act marker and one action slot per scene patch) or answers in
text. One example is one segment:

    CONTEXT_BEFORE [scene x n^3] CONTEXT_AFTER REPLY [slot x n^3, acting examples only]

    <|im_start|>user\\n INSTRUCTION [scene] <|im_end|>\\n<|im_start|>assistant\\n <think>\\n THOUGHT \\n</think>\\n\\n <tool_call> [slot x n^3]
    <|im_start|>user\\n QUESTION    [scene] <|im_end|>\\n<|im_start|>assistant\\n <think>\\n\\n</think>\\n\\n ANSWER <|im_end|>

An example's context_text holds the placeholder SCENE exactly once; the scene's patch embeddings
(kind 0) go there, and the text before and after it is tokenized as it is (the data source writes
the chat markup; a context without "<|im_start|>" is wrapped in stream.py's user turn). A slot is
its scene patch embedded again (kind 1); the readout turns each slot's final state into its
patch's actions, so every slot sees the context, the scene, the thought and the earlier slots.

Losses of a batch:
    text   = cross-entropy of every reply token (the thought with its markers and the act marker,
             or the text answer with its END), predicted at reply positions only, mean over all
             reply tokens of the batch; the vocabulary-wide logits are made ce_chunk positions at
             a time and recomputed in the backward pass, never for the whole sequence;
    action = cross-entropy of each robot voxel's action code (codes) from its slot, mean over the
             robot voxels (observation channel 0) of the batch's acting examples;
    loss   = text_weight * text + action_weight * action.

Batching: examples are packed into rows of at most pack_length tokens (first fit, longest first);
each example keeps its own positions from 0 and attends only within itself (a block-diagonal causal
mask, handed to SDPA as a 4D boolean mask), so packing changes nothing but speed. pack_length 0 puts
one example per row (right padding).
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from beebot.vla.voxel.lm import PATCH, patchify3
from beebot.vla.voxel.stream import OPEN, TURN, StreamVoxelPolicy, positions
from beebot.vla.voxel.world import N_ACTIONS, N_CHANNELS

SCENE = "SCENE"  # the placeholder of the scene tokens in an example's context_text
THINK = "<think>"
THINK_END = "</think>"


def chat_context(instruction):
    """The context text of an instruction or question in stream.py's user turn (scene right after the words)."""
    return f"{OPEN}{instruction}{SCENE}{TURN}"


def chat(context):
    """context as the LM sees it: as given when it carries the chat markup, else wrapped in the user turn."""
    return context if "<|im_start|>" in context else f"{OPEN}{context}{TURN}"


@dataclass
class Layout:
    """One example as a token stream: before ids, n scene patches, after ids, reply ids, n slots if it acts."""

    before: list
    after: list
    reply: list
    n: int
    act: bool

    @property
    def reply_start(self):
        return len(self.before) + self.n + len(self.after)

    @property
    def length(self):
        return self.reply_start + len(self.reply) + (self.n if self.act else 0)


@dataclass
class ThinkStep:
    """A greedy think-then-act step of a batch of B rows.

    reply_ids: each row's generated token ids (ending in the act marker or END when finished);
    thought, answer: the text inside <think> ... </think>, and the text after it (empty for acting rows);
    acts: (B,) rows that reached the act marker; finished: (B,) rows that reached the act marker or END;
    logits: (B, n^3, PATCH^3, 27) action logits (meaningful for acting rows only), None when no row acts."""

    reply_ids: list
    thought: list
    answer: list
    acts: torch.Tensor
    finished: torch.Tensor
    logits: torch.Tensor | None


def pack(lengths, pack_length):
    """Rows of example indices whose lengths sum to at most pack_length each: first fit, longest
    first; an example longer than pack_length gets a row of its own. pack_length 0: one example
    per row, in order."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i]) if pack_length else range(len(lengths))
    rows, room = [], []
    for i in order:
        fit = next((r for r in range(len(rows)) if pack_length and room[r] >= lengths[i]), None)
        if fit is None:
            rows.append([i])
            room.append(pack_length - lengths[i])
        else:
            rows[fit].append(i)
            room[fit] -= lengths[i]
    return rows


def chunked_cross_entropy(head, states, targets, chunk=512):
    """Sum of the cross-entropy of head(states) (N, V) against targets (N,), chunk rows at a time;
    each chunk's logits are recomputed in the backward pass, so at most chunk x V logits exist."""

    def part(s, t):
        return F.cross_entropy(head(s).float(), t, reduction="sum")

    total = states.new_zeros((), dtype=torch.float32)
    for i in range(0, states.shape[0], chunk):
        total = total + checkpoint(part, states[i : i + chunk], targets[i : i + chunk], use_reentrant=False)
    return total


class ThinkVoxelPolicy(StreamVoxelPolicy):
    def __init__(self, lm, tokenizer, lora_rank=0, freeze_embeddings=True, channels=N_CHANNELS, text_weight=1.0, action_weight=1.0, pack_length=0, ce_chunk=512):
        super().__init__(lm, tokenizer, lora_rank=lora_rank, freeze_embeddings=freeze_embeddings, channels=channels)
        self.tokenizer = tokenizer
        self.think_id = tokenizer.convert_tokens_to_ids(THINK)
        self.think_end_id = tokenizer.convert_tokens_to_ids(THINK_END)
        self.text_weight, self.action_weight = text_weight, action_weight
        self.pack_length, self.ce_chunk = pack_length, ce_chunk

    def encode(self, text):
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def split_context(self, context):
        """(ids before SCENE, ids after SCENE) of a context text."""
        before, found, after = chat(context).partition(SCENE)
        assert found and SCENE not in after, f"context_text needs exactly one {SCENE}: {context!r}"
        return self.encode(before), self.encode(after)

    def layout(self, example):
        before, after = self.split_context(example.context_text)
        reply = self.encode(example.reply_text)
        act = bool(example.act)
        assert reply and (not act or reply[-1] == self.act_id), f"an acting reply ends in the act marker: {example.reply_text!r}"
        d, h, w = example.obs.shape[-3:]
        return Layout(before, after, reply, (d // PATCH) * (h // PATCH) * (w // PATCH), act)

    def scene_and_slots(self, examples, device):
        """Per example: (scene patch embeddings (n^3, H), slot embeddings (n^3, H) or None)."""
        groups = {}
        for i, e in enumerate(examples):
            groups.setdefault(tuple(e.obs.shape), []).append(i)
        out = [None] * len(examples)
        for idx in groups.values():
            obs = torch.stack([examples[i].obs for i in idx]).to(device, torch.float32)
            scene = self.patches(obs, 0)
            acting = [k for k, i in enumerate(idx) if examples[i].act]
            slots = self.patches(obs[acting], 1) if acting else None
            for k, i in enumerate(idx):
                out[i] = (scene[k], slots[acting.index(k)] if k in acting else None)
        return out

    def states(self, examples):
        """Run the packed batch: (final states (R * T, H) flattened over rows, layouts, each
        example's flat index of its first token, real tokens)."""
        device = self.open_ids.device
        layouts = [self.layout(e) for e in examples]
        rows = pack([l.length for l in layouts], self.pack_length)
        width = max(sum(layouts[i].length for i in row) for row in rows)
        ids = torch.full((len(rows), width), self.end_id, dtype=torch.long)
        segment = torch.full((len(rows), width), -1, dtype=torch.long)
        pos = torch.zeros(len(rows), width, dtype=torch.long)
        starts = [0] * len(examples)
        for r, row in enumerate(rows):
            offset = 0
            for i in row:
                l = layouts[i]
                filler = [self.end_id] * l.n
                ids[r, offset : offset + l.length] = torch.tensor(l.before + filler + l.after + l.reply + (filler if l.act else []))
                segment[r, offset : offset + l.length] = i
                pos[r, offset : offset + l.length] = torch.arange(l.length)
                starts[i] = r * width + offset
                offset += l.length
        x = self.embed(ids.to(device))
        hidden = x.shape[-1]
        places, values = [], []
        for i, (scene, slots) in enumerate(self.scene_and_slots(examples, device)):
            l = layouts[i]
            first = starts[i] + len(l.before)
            places.append(torch.arange(first, first + l.n))
            values.append(scene)
            if l.act:
                first = starts[i] + l.reply_start + len(l.reply)
                places.append(torch.arange(first, first + l.n))
                values.append(slots)
        x = x.reshape(-1, hidden).index_put((torch.cat(places).to(device),), torch.cat(values).to(x.dtype)).view(len(rows), width, hidden)
        segment = segment.to(device)
        causal = torch.ones(width, width, dtype=torch.bool, device=device).tril()
        mask = (segment.unsqueeze(2) == segment.unsqueeze(1)) & (segment.unsqueeze(1) >= 0) & causal
        mask = mask | torch.eye(width, dtype=torch.bool, device=device)  # padding attends to itself: no empty rows
        out = self.lm.model(inputs_embeds=x, attention_mask=mask.unsqueeze(1), position_ids=pos.to(device)).last_hidden_state
        return out.reshape(-1, hidden), layouts, starts, sum(l.length for l in layouts)

    def slot_logits(self, examples):
        """(sum over acting examples of n^3, PATCH^3, 27) action logits of the acting examples, in
        order (at least one example must act)."""
        h, layouts, starts, _ = self.states(examples)
        return self.readout_of(h, layouts, starts)

    def readout_of(self, h, layouts, starts):
        idx = torch.cat([torch.arange(s + l.reply_start + len(l.reply), s + l.length) for l, s in zip(layouts, starts) if l.act])
        return self.readout(h[idx.to(h.device)].float()).view(-1, PATCH**3, N_ACTIONS)

    def forward(self, examples):
        """Losses of a batch of examples: {"loss", "text", "action"} tensors, and counts
        ("text_tokens", "robots", "tokens", "examples", "acting")."""
        h, layouts, starts, n_tokens = self.states(examples)
        device = h.device
        reply_idx = torch.cat([torch.arange(s + l.reply_start - 1, s + l.reply_start - 1 + len(l.reply)) for l, s in zip(layouts, starts)]).to(device)
        targets = torch.tensor([t for l in layouts for t in l.reply], device=device)
        text = chunked_cross_entropy(self.lm.get_output_embeddings(), h[reply_idx], targets, self.ce_chunk) / len(targets)
        acting = [e for e in examples if e.act]
        action = h.new_zeros((), dtype=torch.float32)
        robots = 0.0
        if acting:
            out = self.readout_of(h, layouts, starts)
            codes = torch.stack([e.codes for e in acting]).to(device).unsqueeze(1).float()
            obs = torch.stack([e.obs[:1] for e in acting]).to(device).float()
            target = patchify3(codes).round().long().reshape(-1)
            weight = (patchify3(obs) > 0.5).float().reshape(-1)
            ce = F.cross_entropy(out.reshape(-1, N_ACTIONS), target, reduction="none")
            robots = weight.sum()
            action = (ce * weight).sum() / robots.clamp(min=1.0)
            robots = robots.item()
        loss = self.text_weight * text + self.action_weight * action
        return {"loss": loss, "text": text, "action": action, "text_tokens": len(targets), "robots": robots, "tokens": n_tokens, "examples": len(examples), "acting": len(acting)}

    @torch.no_grad()
    def think_then_act(self, obs, contexts, max_new_tokens=256):
        """Greedy reply of each row (obs (B, C, D, H, W), contexts: B texts with SCENE) with the KV
        cache: tokens until the act marker or END (or max_new_tokens), so the thought ends at
        </think> and the token that follows decides; rows that reached the act marker then run their
        slots in one more pass over the cache. Rows are left-padded; masked tokens change nothing."""
        b = len(contexts)
        device = obs.device
        scene = self.patches(obs.float(), 0)
        prefixes = []
        for i, context in enumerate(contexts):
            before, after = self.split_context(context)
            prefixes.append(torch.cat([self.embed(torch.tensor(before, dtype=torch.long, device=device)), scene[i], self.embed(torch.tensor(after, dtype=torch.long, device=device))]))
        width = max(len(p) for p in prefixes)
        x = scene.new_zeros(b, width, scene.shape[-1])
        mask = torch.zeros(b, width, dtype=torch.long, device=device)
        for i, p in enumerate(prefixes):
            x[i, width - len(p) :] = p
            mask[i, width - len(p) :] = 1
        out = self.lm.model(inputs_embeds=x, attention_mask=mask, position_ids=positions(mask), use_cache=True)
        cache, state = out.past_key_values, out.last_hidden_state[:, -1]
        head = self.lm.get_output_embeddings()
        replies = [[] for _ in range(b)]
        going = torch.ones(b, dtype=torch.bool, device=device)  # rows still generating
        acts = torch.zeros(b, dtype=torch.bool, device=device)
        for _ in range(max_new_tokens):
            if not going.any():
                break
            token = head(state).float().argmax(-1)
            for i, t in enumerate(token.tolist()):
                if going[i]:
                    replies[i].append(t)
            fed = going.clone()  # the act marker itself goes into the cache, before the slots
            acts |= going & (token == self.act_id)
            going &= (token != self.act_id) & (token != self.end_id)
            mask = torch.cat([mask, fed.long().view(-1, 1)], 1)
            out = self.lm.model(inputs_embeds=self.embed(token.view(-1, 1)), attention_mask=mask, position_ids=positions(mask)[:, -1:], past_key_values=cache, use_cache=True)
            cache, state = out.past_key_values, out.last_hidden_state[:, -1]
        logits = None
        if acts.any():
            slots = self.patches(obs.float(), 1)
            n = slots.shape[1]
            mask = torch.cat([mask, acts.long().view(-1, 1).expand(-1, n)], 1)
            h = self.lm.model(inputs_embeds=slots, attention_mask=mask, position_ids=positions(mask)[:, -n:], past_key_values=cache, use_cache=True).last_hidden_state
            logits = self.readout(h.float()).view(b, n, PATCH**3, N_ACTIONS)
        thought, answer = [], []
        for ids in replies:
            cut = ids.index(self.think_end_id) if self.think_end_id in ids else len(ids)
            inner = [t for t in ids[:cut] if t != self.think_id]
            rest = [t for t in ids[cut + 1 :] if t not in (self.act_id, self.end_id)]
            thought.append(self.tokenizer.decode(inner).strip())
            answer.append("" if self.act_id in ids else self.tokenizer.decode(rest).strip())
        return ThinkStep(replies, thought, answer, acts.cpu(), (acts | ~going).cpu(), logits)


def checkpoint_meta(policy, model):
    """What it takes to rebuild the policy of a checkpoint."""
    lora = [m for m in policy.lm.modules() if type(m).__name__ == "LoRALinear"]
    return {"model": model, "lora_rank": lora[0].a.shape[0] if lora else 0, "channels": policy.channels, "trained_embeddings": policy.lm.get_input_embeddings().weight.requires_grad}


def trainable_state(policy, dtype=None):
    trainable = {n for n, p in policy.named_parameters() if p.requires_grad}
    return {k: (v.to(dtype) if dtype is not None else v.clone()) for k, v in policy.state_dict().items() if k in trainable}


def load_policy(path, device, dtype=torch.float32, pack_length=0):
    """(policy, tokenizer) of a think checkpoint ({"weights", "meta"}) on device; the LM from the
    local Hugging Face cache."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    saved = torch.load(path, map_location="cpu")
    meta = saved["meta"]
    tokenizer = AutoTokenizer.from_pretrained(meta["model"], local_files_only=True)
    lm = AutoModelForCausalLM.from_pretrained(meta["model"], dtype=torch.float32, local_files_only=True)
    policy = ThinkVoxelPolicy(lm, tokenizer, lora_rank=meta["lora_rank"], freeze_embeddings=not meta["trained_embeddings"], channels=meta["channels"], pack_length=pack_length)
    missing = [n for n, p in policy.named_parameters() if p.requires_grad and n not in saved["weights"]]
    assert not missing, f"checkpoint lacks trainable weights: {missing[:5]}"
    policy.load_state_dict({k: v.float() for k, v in saved["weights"].items()}, strict=False)
    return policy.to(device, dtype).eval(), tokenizer
