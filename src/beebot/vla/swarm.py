"""Relative-position policies: the backbone-readout policy with every token placed by rotary
position embeddings (attention between two tokens depends only on their offset; no token ever
sees an absolute coordinate), in two views of the world:

- "global": one token per PATCH x PATCH patch of the whole grid (the world's-eye view). The patch
  size is fixed, so a bigger grid means more tokens, and the model runs unchanged on any size.
- "swarm": one token per robot, holding only the CROP x CROP cells around it (the swarm's view:
  each robot's local sensing, relative positions, and every robot attending to every other).

The goal is the window's end frame as positioned patch tokens (a goal the user hands over, laid
out in the same relative frame), never the start frame, which would give the swarm view a global
snapshot of the present. Each robot (or each patch's squares) reads out its next move as a
categorical over the 9 actions; the policy re-plans every step (execute 1).

Same interface as model.Policy where train.py and goal_eval.py use it; z is the goal frame
itself (B, C + 1, G, G), the extra channel 1 for a real goal and 0 for the null goal. With text
(a caption run), z can instead be a frozen caption embedding (B, 1, D), with no goal tokens and no
position, so nothing global enters but what the robots see and tell each other: text_mode "add"
projects it and adds it to every scene token at the input; "cross" projects it to a few text
tokens that every block reads through a position-free cross-attention (TextCross). The all-zero embedding is the
text null goal. No contrastive loss ties text to goal frames; each is trained only through the
moves it leads to.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from beebot.vla.data import CHUNK, decode_field, encode_field
from beebot.vla.model import HEAD_DIM, HEADS, N_ACTIONS, patchify

PATCH = 4
CROP = 9
ROPE_BASE = 100.0


def rope_freqs(head_dim=HEAD_DIM, base=ROPE_BASE, device=None):
    """(head_dim // 4,) rotation per cell of each frequency, fastest first."""
    quarter = head_dim // 4
    return base ** (-torch.arange(quarter, device=device, dtype=torch.float32) / quarter)


def rope_angles(pos, head_dim=HEAD_DIM, base=ROPE_BASE, freq_scale=None):
    """(B, N, 2) positions in cells -> (B, N, head_dim // 2) rotation angles: the first half of
    the pairs turn with the row, the second half with the column. freq_scale (head_dim // 4,)
    multiplies each frequency's positions (frequency-aware interpolation)."""
    freqs = rope_freqs(head_dim, base, pos.device)
    if freq_scale is not None:
        freqs = freqs * freq_scale.to(pos.device)
    return torch.cat([pos[..., :1] * freqs, pos[..., 1:] * freqs], dim=-1)


def frequency_aware_scale(trained_extent, test_extent, low=1.0, high=4.0, head_dim=HEAD_DIM, base=ROPE_BASE):
    """Per-frequency position multipliers for a world bigger than in training (the YaRN ramp from
    long-context LLMs): a frequency that turns more than high times across the trained extent
    encodes short offsets and stays exact; one that turns less than low times encodes long
    distances and is squeezed by trained / test extent; in between, linearly blended."""
    turns = trained_extent * rope_freqs(head_dim, base) / (2 * math.pi)
    ramp = ((turns - low) / (high - low)).clamp(0.0, 1.0)
    squeeze = trained_extent / test_extent
    return squeeze + (1.0 - squeeze) * ramp


def rotate(x, angles):
    """Rotate consecutive pairs of x (B, H, N, D) by angles (B, N, D // 2)."""
    cos, sin = angles.cos().unsqueeze(1), angles.sin().unsqueeze(1)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class RotaryBlock(nn.Module):
    """Pre-norm transformer block with 2D rotary positions and a key padding mask; with cross,
    also a position-free cross-attention to text tokens (TextCross) after the self-attention."""

    def __init__(self, width, heads=HEADS, head_dim=HEAD_DIM, cross=False):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * heads * head_dim)
        self.out = nn.Linear(heads * head_dim, width)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))
        self.q_scale = 1.0  # > 1 sharpens attention (an inference-time probe for more tokens than in training)
        self.cross = TextCross(width, heads, head_dim) if cross else None

    def forward(self, x, angles, valid, text=None):
        b, n, _ = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = rotate(q, angles) * self.q_scale, rotate(k, angles)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=valid.view(b, 1, 1, n))
        x = x + self.out(attn.transpose(1, 2).reshape(b, n, -1))
        if text is not None:
            x = x + self.cross(x, text)
        return x + self.mlp(self.norm2(x))


class TextCross(nn.Module):
    """Cross-attention from every token to the instruction's tokens, with no positions on either
    side: what a robot reads from the instruction cannot depend on where it stands. The output
    projection starts at zero, so the block starts as if there were no text."""

    def __init__(self, width, heads=HEADS, head_dim=HEAD_DIM):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        self.norm = nn.LayerNorm(width)
        self.text_norm = nn.LayerNorm(width)
        self.q = nn.Linear(width, heads * head_dim)
        self.kv = nn.Linear(width, 2 * heads * head_dim)
        self.out = nn.Linear(heads * head_dim, width)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, text):
        b, n, _ = x.shape
        q = self.q(self.norm(x)).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(self.text_norm(text)).view(b, text.shape[1], 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        attn = F.scaled_dot_product_attention(q, k, v)
        return self.out(attn.transpose(1, 2).reshape(b, n, -1))


def robot_crops(obs, crop):
    """Each robot's local view. obs (B, C, G, G) -> crops (B, N, C * crop * crop), positions
    (B, N, 2) in cells, flat square index (B, N) and valid (B, N), N the most robots in the
    batch. Outside the grid looks like wall (channel 1)."""
    b, c, g, _ = obs.shape
    r = crop // 2
    padded = F.pad(obs, (r, r, r, r))
    padded[:, 1, :r] = 1.0
    padded[:, 1, -r:] = 1.0
    padded[:, 1, :, :r] = 1.0
    padded[:, 1, :, -r:] = 1.0
    robots = obs[:, 0] > 0.5
    counts = robots.flatten(1).sum(1)
    n = max(int(counts.max()), 1)
    # Robots first (stable order), then padding.
    order = torch.argsort((~robots).flatten(1).float(), dim=1, stable=True)[:, :n]
    valid = torch.arange(n, device=obs.device).view(1, -1) < counts.view(-1, 1)
    rows, cols = order // g, order % g
    offsets = torch.arange(crop, device=obs.device)
    rr = (rows.unsqueeze(-1) + offsets).unsqueeze(-1)  # padded coordinates: row + offset - r + r
    cc = (cols.unsqueeze(-1) + offsets).unsqueeze(-2)
    flat = (rr * (g + 2 * r) + cc).view(b, 1, n * crop * crop).expand(-1, c, -1)
    patches = padded.flatten(2).gather(2, flat).view(b, c, n, crop * crop)
    crops = patches.permute(0, 2, 1, 3).reshape(b, n, c * crop * crop)
    positions = torch.stack([rows, cols], dim=-1).float()
    return crops, positions, order, valid


def patch_positions(g, b, device):
    """(B, (g / PATCH)^2, 2): centre of each patch, in cells."""
    n = g // PATCH
    centre = torch.arange(n, device=device, dtype=torch.float32) * PATCH + (PATCH - 1) / 2
    grid = torch.stack(torch.meshgrid(centre, centre, indexing="ij"), dim=-1).view(1, n * n, 2)
    return grid.expand(b, -1, -1)


class RelativePolicy(nn.Module):
    def __init__(self, view="global", width=192, layers=4, obs_channels=10, crop=CROP, z_dropout=0.0, text_dim=0, text_mode="add", text_tokens=8):
        super().__init__()
        assert view in ("global", "swarm")
        self.view, self.crop, self.obs_channels = view, crop, obs_channels
        self.decode = "readout"
        self.pos_scale = 1.0  # < 1 squeezes rotary positions (an inference-time probe for bigger worlds)
        self.freq_scale = None  # per-frequency multipliers (frequency_aware_scale), the same probe by frequency
        scene_in = obs_channels * (crop * crop if view == "swarm" else PATCH * PATCH)
        self.scene_embed = nn.Linear(scene_in, width)
        self.goal_embed = nn.Linear((obs_channels + 1) * PATCH * PATCH, width)
        self.kind = nn.Parameter(torch.randn(2, width) * 0.02)  # scene token, goal token
        assert text_mode in ("add", "cross")
        self.text_mode, self.n_text_tokens = text_mode, text_tokens
        cross = bool(text_dim) and text_mode == "cross"
        self.blocks = nn.ModuleList([RotaryBlock(width, cross=cross) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.readout = nn.Linear(width, N_ACTIONS * (1 if view == "swarm" else PATCH * PATCH))
        if text_dim:  # only caption runs have these, so goal-only checkpoints load unchanged
            # "add": one vector added to every scene token at the input. "cross": text_tokens
            # tokens that every block's TextCross reads.
            out = width * (text_tokens if text_mode == "cross" else 1)
            self.text_embed = nn.Sequential(nn.Linear(text_dim, width), nn.GELU(), nn.Linear(width, out))

    # The goal: the window's end frame, plus a channel that is 1 for a real goal.
    def goal_encoder(self, frames):
        end = frames[:, self.obs_channels :]
        return torch.cat([end, torch.ones_like(end[:, :1])], dim=1)

    def text_head(self, embedding):
        """A caption embedding (B, D) as z (B, 1, D)."""
        return embedding.float().unsqueeze(1)

    def goal_z(self, raw, null_mask=None):
        if null_mask is None:
            return raw
        return torch.where(null_mask.view(-1, *[1] * (raw.dim() - 1)), torch.zeros_like(raw), raw)

    def contrastive_loss(self, raw_goal, raw_text, negative_mask):
        """None here (text and goal frames are not tied); zero, so train.py's sum is unchanged."""
        return torch.zeros((), device=raw_text.device)

    def null_z(self, z):
        return torch.zeros_like(z)

    def loss_noise(self, b, device):
        return torch.zeros(b, device=device), torch.zeros(b, device=device)

    def scene_tokens(self, obs):
        """Scene tokens (B, N, width), their positions, and what each owns: for "swarm" the flat
        square of its robot (B, N) and valid (B, N); for "global" None and all valid."""
        b, _, g, _ = obs.shape
        if self.view == "swarm":
            crops, positions, owner, valid = robot_crops(obs, self.crop)
            return self.scene_embed(crops), positions, owner, valid
        tokens = self.scene_embed(patchify(obs, PATCH))
        return tokens, patch_positions(g, b, obs.device), None, torch.ones(tokens.shape[:2], dtype=torch.bool, device=obs.device)

    def logits(self, obs, z):
        """Per scene token: (B, N, 9) for "swarm", (B, N, PATCH * PATCH, 9) for "global"."""
        b, _, g, _ = obs.shape
        scene, scene_pos, owner, valid = self.scene_tokens(obs)
        text = None
        if z.dim() == 4:  # a goal frame, as positioned patch tokens after the scene tokens
            goal = self.goal_embed(patchify(z, PATCH)) + self.kind[1]
            goal_pos = patch_positions(g, b, obs.device)
        else:  # a caption embedding (B, 1, D): no goal tokens, and nothing positioned
            if self.text_mode == "cross":
                text = self.text_embed(z[:, 0]).view(b, self.n_text_tokens, -1)
            else:
                scene = scene + self.text_embed(z[:, 0]).unsqueeze(1)
            goal = scene[:, :0]
            goal_pos = scene_pos[:, :0]
        x = torch.cat([scene + self.kind[0], goal], dim=1)
        angles = rope_angles(torch.cat([scene_pos, goal_pos], dim=1) * self.pos_scale, freq_scale=self.freq_scale)
        mask = torch.cat([valid, torch.ones(goal.shape[:2], dtype=torch.bool, device=obs.device)], dim=1)
        for block in self.blocks:
            x = block(x, angles, mask, text)
        out = self.readout(self.norm(x[:, : scene.shape[1]]))
        if self.view == "global":
            out = out.view(b, -1, PATCH * PATCH, N_ACTIONS)
        return out, owner, valid

    def flow_loss(self, obs, z, actions, weights, t=None, e=None, future=None):
        """Cross-entropy on each robot's next move (the chunk's first step), robots only in both views."""
        codes = encode_field(actions[:, 0]).long()  # (B, G, G)
        robots = (obs[:, 0] > 0.5) & (weights[:, 0] > 0)
        out, owner, valid = self.logits(obs, z)
        if self.view == "swarm":
            target = codes.flatten(1).gather(1, owner)
            counted = (valid & robots.flatten(1).gather(1, owner)).float()
            ce = F.cross_entropy(out.reshape(-1, N_ACTIONS), target.reshape(-1), reduction="none").view_as(counted)
        else:
            target = patchify(codes.unsqueeze(1), PATCH)
            counted = patchify(robots.unsqueeze(1).float(), PATCH)
            ce = F.cross_entropy(out.reshape(-1, N_ACTIONS), target.reshape(-1), reduction="none").view_as(counted)
        return (ce * counted).sum() / counted.sum().clamp(min=1.0)

    def distill_loss(self, obs, z, teacher_z, weights, keep):
        """Cross-entropy of the moves under z against this same policy's move distribution under
        teacher_z (no gradient through the teacher), per robot, over the rows in keep (B,) bool:
        a text-conditioned student copying the goal-conditioned policy."""
        with torch.no_grad():
            teacher, _, _ = self.logits(obs, teacher_z)
        out, owner, valid = self.logits(obs, z)
        robots = (obs[:, 0] > 0.5) & (weights[:, 0] > 0)
        if self.view == "swarm":
            counted = (valid & robots.flatten(1).gather(1, owner)).float()
        else:
            counted = patchify(robots.unsqueeze(1).float(), PATCH)
        counted = counted * keep.float().view(-1, *[1] * (counted.dim() - 1))
        ce = -(F.softmax(teacher, dim=-1) * F.log_softmax(out, dim=-1)).sum(-1)
        return (ce * counted).sum() / counted.sum().clamp(min=1.0)

    @torch.no_grad()
    def sample(self, obs, z, steps=1, guidance=1.0, noise=None):
        """A chunk (B, CHUNK, 2, G, G) whose first step holds each robot's sampled move (the rest
        zero: run it with execute 1)."""
        b, _, g, _ = obs.shape
        out, owner, valid = self.logits(obs, z)
        if guidance != 1.0:
            null, _, _ = self.logits(obs, self.null_z(z))
            out = null + guidance * (out - null)
        moves = torch.distributions.Categorical(logits=out).sample()
        codes = torch.full((b, g * g), N_ACTIONS - 1, dtype=torch.long, device=obs.device)  # stay
        if self.view == "swarm":
            codes.scatter_(1, owner, torch.where(valid, moves, N_ACTIONS - 1))
        else:
            n = g // PATCH
            codes = moves.view(b, n, n, PATCH, PATCH).permute(0, 1, 3, 2, 4).reshape(b, g * g)
        chunk = torch.zeros(b, CHUNK, 2, g, g, device=obs.device)
        chunk[:, 0] = decode_field(codes.view(b, g, g), obs.device)
        return chunk
