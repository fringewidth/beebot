"""The readout policy on a pretrained language model (Qwen3): the LM's own transformer reads
the instruction as text tokens, the goal as end-frame patch tokens and the scene as patch tokens,
and a linear readout turns each scene token into its squares' next moves (9 choices each).

Attention is prefix-LM: text tokens stay causal among themselves (exactly as the LM was trained,
so its language behaviour is unchanged on text), while goal and scene tokens attend to every
valid token both ways. Text gets the LM's own positions 0..L-1; goal and scene tokens all sit at
position L (no rotary offset between them) and carry their place on the grid as learned row and
column embeddings.

The null goal (classifier-free guidance) is the scene alone: goal and text keys are masked out.
Optional LoRA on the LM's attention and MLP projections (rank 0 means full fine-tuning).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from beebot.vla.data import CHUNK, decode_field, encode_field
from beebot.vla.model import N_ACTIONS, patchify

PATCH = 4
MAX_SIDE = 32  # patches per grid side the position embeddings cover (grids up to 128 squares)
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank update B A, B starting at zero."""

    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / rank
        self.enabled = True

    def forward(self, x):
        if not self.enabled:
            return self.base(x)
        return self.base(x) + (x @ self.a.t().to(x.dtype)) @ self.b.t().to(x.dtype) * self.scale


class lora_disabled:
    """Context: every LoRA update switched off, so the model is the original LM."""

    def __init__(self, lm):
        self.layers = [m for m in lm.modules() if isinstance(m, LoRALinear)]

    def __enter__(self):
        for m in self.layers:
            m.enabled = False

    def __exit__(self, *exc):
        for m in self.layers:
            m.enabled = True


def add_lora(lm, rank, alpha=None):
    """Freeze every LM weight and wrap the projections in LoRA."""
    for p in lm.parameters():
        p.requires_grad_(False)
    for name, module in list(lm.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in LORA_TARGETS and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child, rank, alpha or 2 * rank))


class LMPolicy(nn.Module):
    def __init__(self, lm, obs_channels=10, lora_rank=0, freeze_embeddings=False):
        super().__init__()
        self.lm = lm
        if lora_rank:
            add_lora(lm, lora_rank)
        elif freeze_embeddings:  # tied in Qwen3-0.6B: freezes the output head too
            lm.get_input_embeddings().weight.requires_grad_(False)
            lm.get_output_embeddings().weight.requires_grad_(False)
        hidden = lm.config.hidden_size
        self.obs_channels = obs_channels
        self.decode = "readout"
        self.scene_embed = nn.Linear(obs_channels * PATCH * PATCH, hidden)
        self.goal_embed = nn.Linear((obs_channels + 1) * PATCH * PATCH, hidden)
        self.row = nn.Embedding(MAX_SIDE, hidden)
        self.col = nn.Embedding(MAX_SIDE, hidden)
        self.kind = nn.Parameter(torch.zeros(2, hidden))
        self.readout = nn.Linear(hidden, PATCH * PATCH * N_ACTIONS)
        scale = lm.get_input_embeddings().weight.std().item()  # match the LM's token embedding scale
        for layer in (self.scene_embed, self.goal_embed):
            nn.init.normal_(layer.weight, std=scale / math.sqrt(layer.in_features) * 4)
            nn.init.zeros_(layer.bias)
        for table in (self.row, self.col):
            nn.init.normal_(table.weight, std=scale)

    def new_parameters(self):
        lm_ids = {id(p) for p in self.lm.parameters()}
        return [p for p in self.parameters() if id(p) not in lm_ids]

    def lm_parameters(self):
        return [p for p in self.lm.parameters() if p.requires_grad]

    # Goals: the end frame (B, C, G, G) of a window's start and end frames, plus a channel of ones.
    def goal_encoder(self, frames):
        end = frames[:, self.obs_channels :]
        return torch.cat([end, torch.ones_like(end[:, :1])], dim=1)

    def grid_positions(self, g, device):
        n = g // PATCH
        r = torch.arange(n, device=device)
        return (self.row(r).unsqueeze(1) + self.col(r).unsqueeze(0)).view(n * n, -1)

    def scene_tokens(self, obs):
        """(B, (G / PATCH)^2, H) scene tokens of obs (B, C, G, G), in the LM's dtype."""
        pos = self.grid_positions(obs.shape[-1], obs.device)
        return (self.scene_embed(patchify(obs, PATCH)) + pos + self.kind[0]).to(self.lm.get_input_embeddings().weight.dtype)

    def hidden(self, obs, goal=None, text_ids=None, text_mask=None, null=None):
        """Final hidden states of the scene tokens (B, T_scene, H). goal: (B, C + 1, G, G) or None;
        text_ids, text_mask: (B, L) or None; null: (B,) bool rows with no goal."""
        b = obs.shape[0]
        device = obs.device
        dtype = self.lm.get_input_embeddings().weight.dtype
        scene = self.scene_tokens(obs)
        parts, valid, is_text = [], [], []
        length = 0
        if text_ids is not None:
            parts.append(self.lm.get_input_embeddings()(text_ids))
            valid.append(text_mask.bool())
            is_text.append(torch.ones(text_ids.shape[1], dtype=torch.bool, device=device))
            length = text_ids.shape[1]
        if goal is not None:
            pos = self.grid_positions(goal.shape[-1], device)
            parts.append((self.goal_embed(patchify(goal, PATCH)) + pos + self.kind[1]).to(dtype))
            valid.append(torch.ones(b, pos.shape[0], dtype=torch.bool, device=device))
            is_text.append(torch.zeros(pos.shape[0], dtype=torch.bool, device=device))
        if null is not None and parts:
            valid = [v & ~null.view(-1, 1) for v in valid]  # the null goal: scene alone
        parts.append(scene)
        valid.append(torch.ones(b, scene.shape[1], dtype=torch.bool, device=device))
        is_text.append(torch.zeros(scene.shape[1], dtype=torch.bool, device=device))
        x = torch.cat(parts, dim=1)
        keys = torch.cat(valid, dim=1)  # (B, T)
        text = torch.cat(is_text)  # (T,)
        t = x.shape[1]
        idx = torch.arange(t, device=device)
        causal = idx.view(-1, 1) >= idx.view(1, -1)
        # Text rows: causal over text keys only. Other rows: every valid key.
        allowed = torch.where(text.view(-1, 1), causal & text.view(1, -1), torch.ones_like(causal))
        mask = allowed.unsqueeze(0) & keys.unsqueeze(1)
        mask = mask | torch.eye(t, dtype=torch.bool, device=device).unsqueeze(0)  # no empty rows
        positions = torch.where(text, idx, torch.full_like(idx, length)).unsqueeze(0).expand(b, -1)
        out = self.lm.model(inputs_embeds=x, attention_mask=mask.unsqueeze(1), position_ids=positions).last_hidden_state
        return out[:, -scene.shape[1] :]

    def logits(self, obs, goal=None, text_ids=None, text_mask=None, null=None):
        """(B, (G / PATCH)^2, PATCH * PATCH, 9)."""
        h = self.hidden(obs, goal, text_ids, text_mask, null)
        return self.readout(h.float()).view(h.shape[0], h.shape[1], PATCH * PATCH, N_ACTIONS)

    def policy_loss(self, obs, actions, weights, goal=None, text_ids=None, text_mask=None, null=None):
        """Cross-entropy on each robot's next move (the chunk's first step)."""
        codes = encode_field(actions[:, 0]).long()
        robots = (obs[:, 0] > 0.5) & (weights[:, 0] > 0)
        out = self.logits(obs, goal, text_ids, text_mask, null)
        target = patchify(codes.unsqueeze(1), PATCH)
        counted = patchify(robots.unsqueeze(1).float(), PATCH)
        ce = F.cross_entropy(out.reshape(-1, N_ACTIONS), target.reshape(-1), reduction="none").view_as(counted)
        return (ce * counted).sum() / counted.sum().clamp(min=1.0)

    @torch.no_grad()
    def sample(self, obs, goal=None, text_ids=None, text_mask=None, guidance=1.0):
        """A chunk (B, CHUNK, 2, G, G) whose first step holds every square's sampled move."""
        b, _, g, _ = obs.shape
        out = self.logits(obs, goal, text_ids, text_mask)
        if guidance != 1.0:
            null = self.logits(obs, goal, text_ids, text_mask, null=torch.ones(b, dtype=torch.bool, device=obs.device))
            out = null + guidance * (out - null)
        moves = torch.distributions.Categorical(logits=out).sample()
        n = g // PATCH
        codes = moves.view(b, n, n, PATCH, PATCH).permute(0, 1, 3, 2, 4).reshape(b, g, g)
        chunk = torch.zeros(b, CHUNK, 2, g, g, device=obs.device)
        chunk[:, 0] = decode_field(codes, obs.device)
        return chunk


class EvalAdapter:
    """What goal_eval.run expects of a policy. z is either a goal frame (B, C + 1, G, G) or, for
    captions, token ids (B, L) as floats with -1 for padding (goal_eval passes text through
    text_head as floats)."""

    def __init__(self, policy, guidance=1.0):
        self.policy, self.guidance = policy, guidance
        self.obs_channels = policy.obs_channels

    def goal_encoder(self, frames):
        return self.policy.goal_encoder(frames)

    def text_head(self, ids):
        return ids

    def goal_z(self, z, null_mask=None):
        return z

    def sample(self, obs, z, steps=1, guidance=1.0):
        if z.dim() == 4:
            return self.policy.sample(obs, goal=z, guidance=guidance)
        ids = z.long()
        mask = ids >= 0
        return self.policy.sample(obs, text_ids=ids.clamp(min=0), text_mask=mask, guidance=guidance)
