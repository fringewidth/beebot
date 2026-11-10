"""The voxel LM policy as one plain causal stream: the LM's own causal mask and positions, nothing
custom. A turn is a chat exchange whose user message is the instruction (or a question) followed
by the scene tokens, and whose reply is either text or the act marker followed by one action slot
per scene patch:

    <|im_start|>user\\n INSTRUCTION [scene x n^3] <|im_end|>\\n<|im_start|>assistant\\n <tool_call> [slot x n^3]
    <|im_start|>user\\n QUESTION    [scene x n^3] <|im_end|>\\n<|im_start|>assistant\\n ANSWER <|im_end|>

A slot is its scene patch embedded again (kind 1 instead of kind 0); the readout turns each slot's
final state into its patch's actions, so every slot sees the instruction and the whole scene.
Whether to act or to speak is the LM head's choice of the first reply token (ACT or a word), so
one model in one context does both. Instructions are right-aligned in the batch (padding on the
left), which keeps the scene and the slots at the same distance from the instruction's end in
every row.
"""

import torch
import torch.nn.functional as F

from beebot.vla.voxel.lm import PATCH, VoxelLMPolicy, patchify3, unpatchify3
from beebot.vla.voxel.world import N_ACTIONS

ACT = "<tool_call>"  # an existing token of the LM's vocabulary: acting is the reply's first token
OPEN = "<|im_start|>user\n"
TURN = "<|im_end|>\n<|im_start|>assistant\n"
END = "<|im_end|>"


def positions(mask):
    """Position ids counted from each row's first real token, so that left padding changes
    nothing (rotary attention is relative, but this keeps the arithmetic identical too)."""
    return (mask.cumsum(1) - 1).clamp(min=0)


def right_align(ids, mask):
    """Rows of (ids, mask) with their padding moved to the left, token order kept."""
    order = mask.long().cpu().argsort(dim=1, stable=True).to(ids.device)
    return ids.gather(1, order), mask.gather(1, order)


class StreamVoxelPolicy(VoxelLMPolicy):
    def __init__(self, lm, tokenizer, lora_rank=0, freeze_embeddings=True, marker_weight=0.1):
        super().__init__(lm, lora_rank=lora_rank, freeze_embeddings=freeze_embeddings)
        encode = lambda s: torch.tensor(tokenizer(s, add_special_tokens=False).input_ids)  # noqa: E731
        self.register_buffer("open_ids", encode(OPEN), persistent=False)
        self.register_buffer("turn_ids", encode(TURN), persistent=False)
        self.act_id = tokenizer.convert_tokens_to_ids(ACT)
        self.end_id = tokenizer.convert_tokens_to_ids(END)
        self.marker_weight = marker_weight

    def embed(self, ids):
        return self.lm.get_input_embeddings()(ids)

    def patches(self, obs, kind):
        pos = self.positions3(obs.shape[-1], obs.device)
        return (self.scene_embed(patchify3(obs)) + pos + self.kind[kind]).to(self.lm.get_input_embeddings().weight.dtype)

    def user_turn(self, obs, text_ids, text_mask, null=None):
        """Embeddings and key mask of the stream up to the start of the reply: (B, T, H), (B, T)."""
        b = obs.shape[0]
        ones = lambda n: torch.ones(b, n, dtype=torch.long, device=obs.device)  # noqa: E731
        ids, mask = self.open_ids.expand(b, -1), ones(len(self.open_ids))
        if text_ids is not None:
            text_mask = text_mask.long()
            if null is not None:  # the null instruction: an empty one (its tokens become padding)
                text_mask = text_mask * (~null).long().view(-1, 1)
            ids, mask = right_align(torch.cat([ids, text_ids], 1), torch.cat([mask, text_mask], 1))
        scene = self.patches(obs, 0)
        parts = [self.embed(ids), scene, self.embed(self.turn_ids).expand(b, -1, -1)]
        masks = [mask, ones(scene.shape[1]), ones(len(self.turn_ids))]
        return torch.cat(parts, 1), torch.cat(masks, 1)

    def act_states(self, obs, text_ids=None, text_mask=None, null=None):
        """Final states of the reply that acts: (the state that predicts the reply's first token
        (B, H), the slots' states (B, n^3, H))."""
        x, mask = self.user_turn(obs, text_ids, text_mask, null)
        slots = self.patches(obs, 1)
        act = self.embed(torch.full((obs.shape[0], 1), self.act_id, device=obs.device))
        x = torch.cat([x, act, slots], 1)
        mask = torch.cat([mask, torch.ones(obs.shape[0], 1 + slots.shape[1], dtype=torch.long, device=obs.device)], 1)
        out = self.lm.model(inputs_embeds=x, attention_mask=mask, position_ids=positions(mask)).last_hidden_state
        n = slots.shape[1]
        return out[:, -n - 2], out[:, -n:]

    def hidden(self, obs, goal=None, text_ids=None, text_mask=None, null=None):
        return self.act_states(obs, text_ids, text_mask, null)[1]

    def loss(self, obs, codes, text_ids, text_mask, null=None):
        """Cross-entropy on each robot's action code, plus marker_weight times the cross-entropy
        of ACT as the reply's first token."""
        first, h = self.act_states(obs, text_ids, text_mask, null)
        out = self.readout(h.float()).view(h.shape[0], h.shape[1], PATCH**3, N_ACTIONS)
        target = patchify3(codes.unsqueeze(1).float()).round().long()
        robots = patchify3(obs[:, :1])
        ce = F.cross_entropy(out.reshape(-1, N_ACTIONS), target.reshape(-1), reduction="none").view_as(robots)
        action = (ce * robots).sum() / robots.sum().clamp(min=1.0)
        marker = F.cross_entropy(self.lm.get_output_embeddings()(first).float(), torch.full((len(obs),), self.act_id, device=obs.device))
        return action + self.marker_weight * marker

    def text_loss(self, obs, text_ids, text_mask, answer_ids, answer_mask):
        """Mean cross-entropy of the answer tokens (right-padded, each row ending in END) to a
        question about the scene."""
        x, mask = self.user_turn(obs, text_ids, text_mask)
        x = torch.cat([x, self.embed(answer_ids[:, :-1])], 1)
        mask = torch.cat([mask, answer_mask[:, :-1].long()], 1)
        out = self.lm.model(inputs_embeds=x, attention_mask=mask, position_ids=positions(mask)).last_hidden_state
        logits = self.lm.get_output_embeddings()(out[:, -answer_ids.shape[1] :]).float()
        target = answer_ids.masked_fill(~answer_mask.bool(), -100)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=-100)

    @torch.no_grad()
    def reply(self, obs, text_ids, text_mask, max_new_tokens=24):
        """Greedy reply token ids (B, <= max_new_tokens); a row that starts with ACT chose to act."""
        x, mask = self.user_turn(obs, text_ids, text_mask)
        return self.lm.generate(inputs_embeds=x, attention_mask=mask, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=self.end_id, pad_token_id=self.end_id)

    @torch.no_grad()
    def act(self, obs, text_ids, text_mask, guidance=1.0):
        """Sampled action codes (B, D, H, W)."""
        out = self.logits(obs, text_ids, text_mask)
        if guidance != 1.0:
            null = self.logits(obs, text_ids, text_mask, null=torch.ones(len(obs), dtype=torch.bool, device=obs.device))
            out = null + guidance * (out - null)
        moves = torch.distributions.Categorical(logits=out.float()).sample()
        return unpatchify3(moves, obs.shape[-1])
