"""The think-then-act voxel policy: stream.py's plain causal stream, with a user turn that may carry
images and a reply that first thinks in words and then either acts (the act marker and one action
slot per voxel patch) or answers in text. One example is one row:

    <|im_start|>user\\n [images] INSTRUCTION [scene] <|im_end|>\\n<|im_start|>assistant\\n REPLY [slot x n^3, acting only]

    REPLY = <think>\\n THOUGHT \\n</think>\\n\\n <tool_call>          (acts: the slots follow)
          | <think>\\n\\n</think>\\n\\n ANSWER <|im_end|>              (speaks)

Placeholders in an example's context_text, each at most once:
    <IMAGES>  the example's images (0, 1, 2, ... PIL images of any size), each as the LM's own
              <|vision_start|> <|image_pad|> x k <|vision_end|> block, k = its merged patches; the
              vision encoder's outputs replace the pad tokens (a vision-language LM such as
              Qwen3.5; positions are its multimodal rotary positions, get_rope_index).
    SCENE     the voxel scene as n^3 patch embeddings (kind 0), for text-only LMs (Qwen3).
A context without "<|im_start|>" is wrapped in stream.py's user turn; otherwise its text is used as
it is (the data source writes the chat markup). A slot is its voxel patch embedded again (kind 1);
the readout turns each slot's final state into its patch's actions, so every slot sees the images,
the words, the thought and the earlier slots.

Losses of a batch:
    text   = cross-entropy of every reply token (the thought with its markers and the act marker,
             or the text answer with its END), predicted at reply positions only, mean over all
             reply tokens of the batch; the vocabulary-wide logits are made ce_chunk positions at
             a time and recomputed in the backward pass, never for the whole sequence;
    action = cross-entropy of each robot voxel's action code (codes) from its slot, mean over the
             robot voxels (observation channel 0) of the batch's acting examples;
    loss   = text_weight * text + action_weight * action.

Batching: one example per row, right-padded (the trainer buckets examples of similar length into
the same micro-batch). No packing: the linear-attention layers of hybrid LMs (Qwen3.5) carry a
recurrent state along the row, which a block-diagonal mask cannot cut.
"""

import importlib
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from beebot.vla.voxel.lm import PATCH, patchify3
from beebot.vla.voxel.stream import OPEN, TURN, StreamVoxelPolicy, positions
from beebot.vla.voxel.world import N_ACTIONS, N_CHANNELS

IMAGES = "<IMAGES>"  # the place of the example's images in context_text
SCENE = "SCENE"  # the place of the voxel scene tokens in context_text (text-only LMs)
THINK = "<think>"
THINK_END = "</think>"
IMAGE_PAD = "<|image_pad|>"
VISION = "<|vision_start|>{}<|vision_end|>"  # one image's block, as the Qwen3.5 / Qwen3-VL chat template writes it


def chat_context(instruction, images=False, scene=True):
    """The context text of an instruction or question in stream.py's user turn: images first,
    then the words, then the voxel scene."""
    return f"{OPEN}{IMAGES if images else ''}{instruction}{SCENE if scene else ''}{TURN}"


def chat(context):
    """context as the LM sees it: as given when it carries the chat markup, else wrapped in the user turn."""
    return context if "<|im_start|>" in context else f"{OPEN}{context}{TURN}"


@dataclass
class Prepared:
    """One example as a token stream, ready for the batch: ids holds the whole row (context with its
    image pad tokens and scene placeholders, reply, slot placeholders); scene_at, slots_at: the
    first scene and slot positions (-1: none); pixels, grid: the images' patches and (t, h, w)
    grids for the vision encoder (None: no images)."""

    ids: list
    scene_at: int
    reply_start: int
    reply_len: int
    slots_at: int
    n: int
    pixels: torch.Tensor | None
    grid: torch.Tensor | None
    obs: torch.Tensor
    codes: torch.Tensor | None
    act: bool

    @property
    def length(self):
        return len(self.ids)


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


def chunked_cross_entropy(head, states, targets, chunk=512):
    """Sum of the cross-entropy of head(states) (N, V) against targets (N,), chunk rows at a time;
    each chunk's logits are recomputed in the backward pass, so at most chunk x V logits exist."""

    def part(s, t):
        return F.cross_entropy(head(s).float(), t, reduction="sum")

    total = states.new_zeros((), dtype=torch.float32)
    for i in range(0, states.shape[0], chunk):
        total = total + checkpoint(part, states[i : i + chunk], targets[i : i + chunk], use_reentrant=False)
    return total


def linear_attention_kernels(lm):
    """For a hybrid LM, which implementation its linear-attention layers get: the fla and
    causal_conv1d packages when importable (transformers then uses them), else transformers' torch
    reference fallback. {} for an LM without linear attention."""
    types = getattr(lm.config.get_text_config(), "layer_types", None) or []
    out = {}
    if "linear_attention" in types:
        for part, package in (("gated_delta_rule", "fla"), ("causal_conv1d", "causal_conv1d")):
            try:
                importlib.import_module(package)
                out[part] = package
            except Exception:
                out[part] = "torch fallback"
    return out


class ThinkVoxelPolicy(StreamVoxelPolicy):
    """lm: a causal LM (Qwen3ForCausalLM) or a vision-language LM (Qwen3_5ForConditionalGeneration,
    with image_processor, e.g. Qwen2VLImageProcessorPil). Full weights by default (lora_rank 0);
    token embeddings and the (tied) output head frozen unless freeze_embeddings is False. Vision
    LMs: the vision encoder trains with train_vision, its projector (merger) with train_projector."""

    def __init__(self, lm, tokenizer, image_processor=None, lora_rank=0, freeze_embeddings=True, train_vision=False, train_projector=True, channels=N_CHANNELS, text_weight=1.0, action_weight=1.0, ce_chunk=512):
        if getattr(lm.config, "hidden_size", None) is None:  # a composite (vision-language) config
            lm.config.hidden_size = lm.config.get_text_config().hidden_size
        super().__init__(lm, tokenizer, lora_rank=lora_rank, freeze_embeddings=freeze_embeddings, channels=channels)
        self.tokenizer, self.image_processor = tokenizer, image_processor
        self.think_id = tokenizer.convert_tokens_to_ids(THINK)
        self.think_end_id = tokenizer.convert_tokens_to_ids(THINK_END)
        self.image_id = tokenizer.convert_tokens_to_ids(IMAGE_PAD)
        self.text_weight, self.action_weight, self.ce_chunk = text_weight, action_weight, ce_chunk
        self.mrope = hasattr(lm.model, "get_rope_index")
        if self.visual is not None:
            for p in self.visual.parameters():
                p.requires_grad_(train_vision)
            for p in self.visual.merger.parameters():
                p.requires_grad_(train_vision or train_projector)

    @property
    def visual(self):
        """The vision encoder of a vision-language LM, else None."""
        return getattr(self.lm.model, "visual", None)

    @property
    def text_model(self):
        """The LM's decoder stack (without head)."""
        return getattr(self.lm.model, "language_model", self.lm.model)

    def encode(self, text):
        return self.tokenizer(text, add_special_tokens=False).input_ids

    # ---- examples as token streams

    def prepare(self, context, obs, images=()):
        """The context part of a row (reply empty, no slots)."""
        text = chat(context)
        assert text.count(IMAGES) <= 1 and text.count(SCENE) <= 1, f"{IMAGES} and {SCENE} at most once each: {context!r}"
        assert IMAGES in text or not images, f"images need an {IMAGES} place in the context: {context!r}"
        pixels = grid = None
        blocks = ""
        if images:
            assert self.image_processor is not None, "images need a vision-language LM"
            out = self.image_processor(images=list(images), return_tensors="pt")
            pixels, grid = out["pixel_values"], out["image_grid_thw"]
            blocks = "".join(VISION.format(IMAGE_PAD * int(g.prod() // self.image_processor.merge_size**2)) for g in grid)
        before, found, after = text.replace(IMAGES, blocks).partition(SCENE)
        d, h, w = obs.shape[-3:]
        n = (d // PATCH) * (h // PATCH) * (w // PATCH)
        ids = self.encode(before)
        scene_at = len(ids) if found else -1
        ids = ids + ([self.end_id] * n if found else []) + self.encode(after)
        return Prepared(ids, scene_at, len(ids), 0, -1, n, pixels, grid, obs, None, False)

    def with_reply(self, prepared, reply, act, codes=None):
        """prepared (a context) followed by the reply ids and, when it acts, the slots."""
        assert reply and (not act or reply[-1] == self.act_id), "an acting reply ends in the act marker"
        ids = prepared.ids[: prepared.reply_start] + list(reply)
        slots_at = len(ids) if act else -1
        ids = ids + ([self.end_id] * prepared.n if act else [])
        return replace(prepared, ids=ids, reply_len=len(reply), slots_at=slots_at, codes=codes, act=bool(act))

    def prepare_example(self, example):
        """A data source's Example (obs, context_text, reply_text, act, codes, optional images) as a row."""
        context = self.prepare(example.context_text, example.obs, getattr(example, "images", None) or ())
        return self.with_reply(context, self.encode(example.reply_text), example.act, example.codes)

    # ---- the batch

    def embed_rows(self, prepared, left=False):
        """(inputs (B, T, H), ids (B, T), key mask (B, T), position ids, each row's offset): rows
        padded on the right (left: on the left), images, scene and slots put in their places."""
        device = self.readout.weight.device
        width = max(p.length for p in prepared)
        offsets = [width - p.length if left else 0 for p in prepared]
        ids = torch.full((len(prepared), width), self.end_id, dtype=torch.long)
        mask = torch.zeros(len(prepared), width, dtype=torch.long)
        for i, (p, o) in enumerate(zip(prepared, offsets)):
            ids[i, o : o + p.length] = torch.tensor(p.ids)
            mask[i, o : o + p.length] = 1
        ids, mask = ids.to(device), mask.to(device)
        x = self.embed(ids)
        hidden = x.shape[-1]
        shown = [p for p in prepared if p.pixels is not None]
        grid = torch.cat([p.grid for p in shown]).to(device) if shown else None
        if shown:
            pixels = torch.cat([p.pixels for p in shown]).to(device, self.visual.dtype)
            features = self.visual(pixels, grid_thw=grid).pooler_output
            x = x.masked_scatter((ids == self.image_id).unsqueeze(-1), features.to(x.dtype))
        places, values = [], []
        for p, o, i in zip(prepared, offsets, range(len(prepared))):
            if p.scene_at >= 0 or p.act:
                obs = p.obs.unsqueeze(0).to(device, torch.float32)
                for at, kind in ((p.scene_at, 0), (p.slots_at, 1)):
                    if at >= 0:
                        places.append(torch.arange(at, at + p.n, device=device) + i * width + o)
                        values.append(self.patches(obs, kind)[0])
        if places:
            x = x.reshape(-1, hidden).index_put((torch.cat(places),), torch.cat(values).to(x.dtype)).view(len(prepared), width, hidden)
        if self.mrope:
            pos = self.lm.model.get_rope_index(ids, mm_token_type_ids=(ids == self.image_id).int(), image_grid_thw=grid, attention_mask=mask)[0]
        else:
            pos = positions(mask)
        return x, ids, mask, pos, offsets

    def states(self, prepared):
        """Final states (B * T, H) of right-padded rows, flattened, and the row width."""
        x, _, mask, pos, _ = self.embed_rows(prepared)
        out = self.text_model(inputs_embeds=x, attention_mask=mask, position_ids=pos).last_hidden_state
        return out.reshape(-1, out.shape[-1]), x.shape[1]

    def rows(self, examples):
        return [e if isinstance(e, Prepared) else self.prepare_example(e) for e in examples]

    def reply_index(self, prepared, width):
        return torch.cat([torch.arange(p.reply_start - 1, p.reply_start - 1 + p.reply_len) + i * width for i, p in enumerate(prepared)])

    def slot_index(self, prepared, width):
        return torch.cat([torch.arange(p.slots_at, p.slots_at + p.n) + i * width for i, p in enumerate(prepared) if p.act])

    def slot_logits(self, examples):
        """(sum over acting examples of n^3, PATCH^3, 27) action logits of the acting examples, in
        order (at least one example must act). examples: Examples or Prepared rows."""
        prepared = self.rows(examples)
        h, width = self.states(prepared)
        return self.readout(h[self.slot_index(prepared, width).to(h.device)].float()).view(-1, PATCH**3, N_ACTIONS)

    def reply_logits(self, examples):
        """Per example, the (reply length, vocabulary) logits that predict its reply tokens."""
        prepared = self.rows(examples)
        h, width = self.states(prepared)
        out = self.lm.get_output_embeddings()(h[self.reply_index(prepared, width).to(h.device)]).float()
        return list(out.split([p.reply_len for p in prepared]))

    def forward(self, examples):
        """Losses of a batch (Examples or Prepared rows): {"loss", "text", "action"} tensors, and
        counts ("text_tokens", "robots", "tokens", "examples", "acting")."""
        prepared = self.rows(examples)
        h, width = self.states(prepared)
        device = h.device
        targets = torch.tensor([t for p in prepared for t in p.ids[p.reply_start : p.reply_start + p.reply_len]], device=device)
        text = chunked_cross_entropy(self.lm.get_output_embeddings(), h[self.reply_index(prepared, width).to(device)], targets, self.ce_chunk) / len(targets)
        acting = [p for p in prepared if p.act]
        action = h.new_zeros((), dtype=torch.float32)
        robots = 0.0
        if acting:
            out = self.readout(h[self.slot_index(prepared, width).to(device)].float()).view(-1, PATCH**3, N_ACTIONS)
            codes = torch.cat([patchify3(p.codes.view(1, 1, *p.codes.shape).float()) for p in acting]).to(device)
            weight = torch.cat([patchify3(p.obs[:1].unsqueeze(0).float()) for p in acting]).to(device) > 0.5
            ce = F.cross_entropy(out.reshape(-1, N_ACTIONS), codes.round().long().reshape(-1), reduction="none")
            weight = weight.float().reshape(-1)
            robots = weight.sum()
            action = (ce * weight).sum() / robots.clamp(min=1.0)
            robots = robots.item()
        loss = self.text_weight * text + self.action_weight * action
        return {"loss": loss, "text": text, "action": action, "text_tokens": len(targets), "robots": robots, "tokens": sum(p.length for p in prepared), "examples": len(prepared), "acting": len(acting)}

    # ---- rollouts

    @torch.no_grad()
    def think_then_act(self, obs, contexts, images=None, max_new_tokens=256):
        """Greedy reply of each row with the KV cache (obs (B, C, D, H, W); contexts: B texts;
        images: B lists of PIL images or None): tokens until the act marker or END (or
        max_new_tokens), so the thought ends at </think> and the token after it decides. Rows that
        reached the act marker then run their slots in one more pass over the whole row, exactly
        the training stream (a cached pass would not do for hybrid LMs: the linear-attention state
        of a row that stopped early has moved on). Rows are left-padded for the generation."""
        b = len(contexts)
        prepared = [self.prepare(c, obs[i].cpu(), images[i] if images else ()) for i, c in enumerate(contexts)]
        x, _, mask, pos, _ = self.embed_rows(prepared, left=True)
        device = x.device
        following = (pos.amax(dim=(0, 2)) if self.mrope else pos.amax(-1)) + 1  # each row's next position
        out = self.text_model(inputs_embeds=x, attention_mask=mask, position_ids=pos, use_cache=True)
        cache, state = out.past_key_values, out.last_hidden_state[:, -1]
        head = self.lm.get_output_embeddings()
        replies = [[] for _ in range(b)]
        going = torch.ones(b, dtype=torch.bool, device=device)  # rows still generating
        acts = torch.zeros(b, dtype=torch.bool, device=device)
        for _ in range(max_new_tokens):
            token = head(state).float().argmax(-1)
            for i, t in enumerate(token.tolist()):
                if going[i]:
                    replies[i].append(t)
            acts |= going & (token == self.act_id)
            going &= (token != self.act_id) & (token != self.end_id)
            if not going.any():
                break
            mask = torch.cat([mask, torch.ones(b, 1, dtype=torch.long, device=device)], 1)
            step_pos = following.view(1, b, 1).expand(3, b, 1) if self.mrope else following.view(b, 1)
            out = self.text_model(inputs_embeds=self.embed(token.view(-1, 1)), attention_mask=mask, position_ids=step_pos, past_key_values=cache, use_cache=True)
            cache, state = out.past_key_values, out.last_hidden_state[:, -1]
            following = following + 1
        logits = None
        acting = acts.nonzero().flatten().tolist()
        if acting:
            rows = [self.with_reply(prepared[i], replies[i], True) for i in acting]
            out = self.slot_logits(rows).view(len(acting), -1, PATCH**3, N_ACTIONS)
            logits = out.new_zeros(b, *out.shape[1:])
            logits[acting] = out
        thought, answer = [], []
        for ids in replies:
            cut = ids.index(self.think_end_id) if self.think_end_id in ids else len(ids)
            inner = [t for t in ids[:cut] if t != self.think_id]
            rest = [t for t in ids[cut + 1 :] if t not in (self.act_id, self.end_id)]
            thought.append(self.tokenizer.decode(inner).strip())
            answer.append("" if self.act_id in ids else self.tokenizer.decode(rest).strip())
        return ThinkStep(replies, thought, answer, acts.cpu(), (acts | ~going).cpu(), logits)


# ---- loading and saving


def load_lm(name, dtype=torch.float32):
    """(LM, tokenizer, image processor or None) from the local Hugging Face cache: a
    vision-language LM (its config has a vision_config) through AutoModelForImageTextToText with
    the PIL image processor (no torchvision needed), else AutoModelForCausalLM."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    vision = hasattr(AutoConfig.from_pretrained(name, local_files_only=True), "vision_config")
    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
    image_processor = None
    if vision:
        from transformers.models.qwen2_vl import Qwen2VLImageProcessorPil

        image_processor = Qwen2VLImageProcessorPil.from_pretrained(name, local_files_only=True)
    lm = (AutoModelForImageTextToText if vision else AutoModelForCausalLM).from_pretrained(name, dtype=dtype, local_files_only=True)
    return lm, tokenizer, image_processor


def checkpoint_meta(policy, model):
    """What it takes to rebuild the policy of a checkpoint."""
    lora = [m for m in policy.lm.modules() if type(m).__name__ == "LoRALinear"]
    visual = policy.visual
    return {
        "model": model,
        "lora_rank": lora[0].a.shape[0] if lora else 0,
        "channels": policy.channels,
        "trained_embeddings": policy.lm.get_input_embeddings().weight.requires_grad,
        "train_vision": bool(visual is not None and next(p for n, p in visual.named_parameters() if not n.startswith("merger")).requires_grad),
        "train_projector": bool(visual is not None and next(visual.merger.parameters()).requires_grad),
    }


def trainable_state(policy, dtype=None):
    trainable = {n for n, p in policy.named_parameters() if p.requires_grad}
    return {k: (v.to(dtype) if dtype is not None else v.clone()) for k, v in policy.state_dict().items() if k in trainable}


def load_policy(path, device, dtype=torch.float32):
    """(policy, tokenizer) of a think checkpoint ({"weights", "meta"}) on device, the LM in dtype
    (the new weights stay fp32)."""
    saved = torch.load(path, map_location="cpu")
    meta = saved["meta"]
    lm, tokenizer, image_processor = load_lm(meta["model"])
    policy = ThinkVoxelPolicy(lm, tokenizer, image_processor, lora_rank=meta["lora_rank"], freeze_embeddings=not meta["trained_embeddings"], train_vision=meta.get("train_vision", False), train_projector=meta.get("train_projector", True), channels=meta["channels"])
    missing = [n for n, p in policy.named_parameters() if p.requires_grad and n not in saved["weights"]]
    assert not missing, f"checkpoint lacks trainable weights: {missing[:5]}"
    policy.load_state_dict({k: v.float() for k, v in saved["weights"].items()}, strict=False)
    policy = policy.to(device)
    policy.lm.to(dtype)
    return policy.eval(), tokenizer
