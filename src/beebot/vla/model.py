"""The policy: a goal encoder and a text head into goal tokens z (one by default), a backbone over
the current scene plus z, and an action expert that generates a chunk of velocity fields by flow matching.

The action expert has its own weights. At every layer its queries attend to that backbone layer's
keys and values and to its own; the backbone never attends to the expert, so the backbone runs
once per decision and its keys and values are reused for every denoising step.

On a grid of size G the patch is G/8 squares, so there are always 64 scene tokens and 128 action
tokens (64 patches x 2 halves of the chunk), and compute barely depends on G.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from beebot.vla.data import CHUNK, FUTURE_ROBOT, N_FUTURE, decode_field, encode_field
from beebot.vla.world import N_CHANNELS

Z_DIM = 64
HEADS = 6
HEAD_DIM = 32
N_PATCHES = 64
N_ACTIONS = 9  # per square and step: 8 directions and stay, as data.encode_field codes them
MASK = N_ACTIONS  # the masked state of an action slot (objective "masked")
HALVES = 2
STEPS_PER_TOKEN = CHUNK // HALVES


class Block(nn.Module):
    """Pre-norm transformer block whose attention can also read keys and values from elsewhere."""

    def __init__(self, width, mlp, heads=HEADS, head_dim=HEAD_DIM):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * heads * head_dim)
        self.out = nn.Linear(heads * head_dim, width)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, mlp), nn.GELU(), nn.Linear(mlp, width))

    def forward(self, x, extra_kv=None, key_mask=None):
        """Returns (x, (k, v)) so callers can hand this layer's keys and values to another stream."""
        b, n, _ = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        own = (k, v)
        if extra_kv is not None:
            k = torch.cat([extra_kv[0], k], dim=2)
            v = torch.cat([extra_kv[1], v], dim=2)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask)
        x = x + self.out(attn.transpose(1, 2).reshape(b, n, -1))
        x = x + self.mlp(self.norm2(x))
        return x, own


def patchify(x, patch):
    """(B, C, G, G) -> (B, 64, C * patch * patch)."""
    b, c, g, _ = x.shape
    n = g // patch
    return x.view(b, c, n, patch, n, patch).permute(0, 2, 4, 1, 3, 5).reshape(b, n * n, c * patch * patch)


def chunk_to_tokens(actions, patch):
    """(B, CHUNK, 2, G, G) -> (B, 128, STEPS_PER_TOKEN * 2 * patch * patch); token = half * 64 + patch index."""
    b, t, c, g, _ = actions.shape
    n = g // patch
    x = actions.view(b, HALVES, STEPS_PER_TOKEN, c, n, patch, n, patch)
    x = x.permute(0, 1, 4, 6, 2, 3, 5, 7)  # B, half, row-patch, col-patch, step, comp, py, px
    return x.reshape(b, HALVES * n * n, STEPS_PER_TOKEN * c * patch * patch)


def tokens_to_chunk(tokens, patch, grid, channels=2):
    b = tokens.shape[0]
    n = grid // patch
    x = tokens.view(b, HALVES, n, n, STEPS_PER_TOKEN, channels, patch, patch)
    x = x.permute(0, 1, 4, 5, 2, 6, 3, 7)  # B, half, step, comp, row-patch, py, col-patch, px
    return x.reshape(b, CHUNK, channels, grid, grid)


def codes_to_tokens(codes, patch):
    """(B, CHUNK, G, G) per-square values -> (B, 128, STEPS_PER_TOKEN * patch * patch) slots."""
    return chunk_to_tokens(codes.unsqueeze(2).contiguous(), patch)


def tokens_to_codes(tokens, patch, grid):
    return tokens_to_chunk(tokens, patch, grid, channels=1).squeeze(2)


def time_embedding(t, dim=64):
    half = dim // 2
    freqs = torch.exp(-math.log(1000.0) * torch.arange(half, device=t.device) / half)
    angles = t.view(-1, 1) * freqs.view(1, -1) * 1000.0
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class GoalEncoder(nn.Module):
    """Window start and end frames (20 channels) -> raw goal tokens (B, n_tokens, Z_DIM).

    pool="mean" averages the patch tokens and projects to n_tokens vectors; pool="attention" has
    n_tokens learned queries attend over the patch tokens, which keeps where things are;
    pool="none" keeps all 64 patch tokens (n_tokens must be 64): no bottleneck, a diagnostic.
    """

    def __init__(self, patch, n_tokens=1, pool="mean", width=128, layers=3):
        super().__init__()
        assert pool in ("mean", "attention", "none")
        assert pool != "none" or n_tokens == N_PATCHES
        self.patch, self.n_tokens, self.pool = patch, n_tokens, pool
        self.embed = nn.Linear(2 * N_CHANNELS * patch * patch, width)
        self.pos = nn.Parameter(torch.randn(1, N_PATCHES, width) * 0.02)
        self.blocks = nn.ModuleList([Block(width, 4 * width, heads=4) for _ in range(layers)])
        self.queries = nn.Parameter(torch.randn(1, n_tokens, width) * 0.02)
        self.attend = nn.MultiheadAttention(width, 4, batch_first=True)
        self.out = nn.Linear(width, n_tokens * Z_DIM if pool == "mean" else Z_DIM)

    def forward(self, frames):
        x = self.embed(patchify(frames, self.patch)) + self.pos
        for block in self.blocks:
            x, _ = block(x)
        if self.pool == "none":
            return self.out(x)
        if self.pool == "attention":
            pooled, _ = self.attend(self.queries.expand(len(x), -1, -1), x, x, need_weights=False)
            return self.out(pooled)
        return self.out(x.mean(dim=1)).view(len(x), self.n_tokens, Z_DIM)


class TextHead(nn.Module):
    """A frozen caption embedding -> raw goal tokens (B, n_tokens, Z_DIM), the goal encoder's shape."""

    def __init__(self, text_dim=1024, hidden=256, n_tokens=1):
        super().__init__()
        self.n_tokens = n_tokens
        self.net = nn.Sequential(nn.Linear(text_dim, hidden), nn.GELU(), nn.Linear(hidden, n_tokens * Z_DIM))

    def forward(self, embedding):
        return self.net(embedding).view(len(embedding), self.n_tokens, Z_DIM)


class Policy(nn.Module):
    def __init__(self, grid, width=192, expert_width=128, layers=4, z_dropout=0.1, text_dim=1024, goal_tokens=1, goal_pool="mean", obs_channels=N_CHANNELS, objective="flow", world_model=False, insulate=False):
        super().__init__()
        assert grid % 8 == 0, "the grid size must be a multiple of 8"
        self.grid = grid
        self.patch = grid // 8
        self.token_dim = STEPS_PER_TOKEN * 2 * self.patch * self.patch
        self.goal_tokens = goal_tokens
        self.goal_encoder = GoalEncoder(self.patch, goal_tokens, goal_pool)
        self.text_head = TextHead(text_dim, n_tokens=goal_tokens)
        self.z_null = nn.Parameter(torch.randn(goal_tokens, Z_DIM) * 0.02)
        self.z_norm = nn.LayerNorm(Z_DIM)  # shared by both goal sources and the null goal
        self.z_dropout = nn.Dropout(z_dropout)

        assert objective in ("flow", "mse", "masked", "readout")
        # "mse": an ablation where the action expert regresses the chunk directly (input tokens
        # all zero, noise level 1) instead of generating it by flow matching.
        # "masked": each square's action is one of N_ACTIONS choices (robots are discrete), and the
        # expert generates them by masked discrete diffusion: trained to fill in randomly masked
        # slots, sampled by unmasking the most confident slots over a few steps (MaskGIT).
        # "readout": no action expert. The backbone's scene tokens are read out directly into each
        # square's 8 actions as categoricals (cross-entropy), sampled once, independently per square.
        self.objective = objective
        # world_model (masked only): each token also holds the future state of its squares after
        # each of its steps (empty, robot, cheese), masked and filled in with the actions, so one
        # objective trains a policy, a world model and inverse dynamics together.
        # With "readout", world_model adds a second readout: each square's future states, from the
        # same backbone tokens, so the world-model loss shapes the backbone's representation.
        assert not world_model or objective in ("masked", "readout")
        self.world_model = world_model
        self.slots = STEPS_PER_TOKEN * self.patch * self.patch  # action slots per token (and future slots, with world_model)
        if objective == "masked":
            expert_in = self.slots * (N_ACTIONS + 1) + (self.slots * (N_FUTURE + 1) if world_model else 0)
            expert_out = self.slots * N_ACTIONS + (self.slots * N_FUTURE if world_model else 0)
        else:
            expert_in = expert_out = self.token_dim
        self.obs_channels = obs_channels  # more than N_CHANNELS only for diagnostics with extra inputs
        self.obs_embed = nn.Linear(obs_channels * self.patch * self.patch, width)
        self.z_embed = nn.Linear(Z_DIM, width)
        self.obs_pos = nn.Parameter(torch.randn(1, N_PATCHES + goal_tokens, width) * 0.02)
        self.backbone = nn.ModuleList([Block(width, 4 * width) for _ in range(layers)])
        # insulate (knowledge insulation, as in pi0.5 KI): the backbone gets its own per-square
        # action cross-entropy readout from its scene tokens, and the expert reads the backbone's
        # keys and values through a stop-gradient, so only the readout (and nothing from the
        # expert) trains the backbone and the goal encoders.
        self.insulate = insulate
        self.decode = "readout" if objective == "readout" else "expert"  # "readout": act from the backbone's own one-shot readout
        has_readout = insulate or objective == "readout"
        self.backbone_readout = nn.Linear(width, CHUNK * self.patch * self.patch * N_ACTIONS) if has_readout else None
        self.backbone_future = nn.Linear(width, CHUNK * self.patch * self.patch * N_FUTURE) if objective == "readout" and world_model else None
        expert_layers = 0 if objective == "readout" else layers
        self.action_in = nn.Sequential(nn.Linear(expert_in + 64, expert_width), nn.GELU(), nn.Linear(expert_width, expert_width)) if expert_layers else None
        self.action_pos = nn.Parameter(torch.randn(1, HALVES * N_PATCHES, expert_width) * 0.02) if expert_layers else None
        self.expert = nn.ModuleList([Block(expert_width, 4 * expert_width) for _ in range(expert_layers)])
        self.action_out = nn.Sequential(nn.LayerNorm(expert_width), nn.Linear(expert_width, expert_out)) if expert_layers else None

        # Sigmoid contrastive loss between goal z and text z (the caption half): learned scale and bias.
        self.contrast_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.contrast_bias = nn.Parameter(torch.tensor(-10.0))

    def goal_z(self, raw, null_mask=None):
        """Shared normalization for any goal source; rows in null_mask become the null goal."""
        if null_mask is not None:
            raw = torch.where(null_mask.view(-1, 1, 1), self.z_null.expand_as(raw), raw)
        return self.z_dropout(self.z_norm(raw))

    def contrastive_loss(self, raw_goal, raw_text, negative_mask):
        """Sigmoid contrastive loss between matching goal and text z (row i with row i). Off-diagonal
        pairs count as negatives only where negative_mask (B, B) is True. Uses the normalized z
        before dropout."""
        g = F.normalize(self.z_norm(raw_goal).flatten(1), dim=-1)
        t = F.normalize(self.z_norm(raw_text).flatten(1), dim=-1)
        logits = g @ t.T * self.contrast_scale.exp() + self.contrast_bias
        eye = torch.eye(len(g), dtype=torch.bool, device=g.device)
        labels = torch.where(eye, 1.0, -1.0)
        counted = eye | negative_mask
        return -(F.logsigmoid(labels * logits) * counted).sum() / len(g)

    def null_z(self, batch_size):
        return self.goal_z(self.z_null.expand(batch_size, -1, -1))

    def prefix_tokens(self, obs, z):
        """Run the backbone once on the scene tokens plus the goal tokens z (B, goal_tokens, Z_DIM);
        returns its per-layer keys and values, and its final tokens."""
        x = torch.cat([self.obs_embed(patchify(obs, self.patch)), self.z_embed(z)], dim=1) + self.obs_pos
        kvs = []
        for block in self.backbone:
            x, kv = block(x)
            kvs.append(kv)
        return kvs, x

    def prefix(self, obs, z):
        return self.prefix_tokens(obs, z)[0]

    def encode(self, obs, z, actions, weights):
        """The backbone's keys and values for the action loss, and the insulation readout loss
        (0 without insulate); with insulate the keys and values are cut from the graph."""
        kvs, x = self.prefix_tokens(obs, z)
        if not self.insulate:
            return kvs, torch.zeros((), device=obs.device)
        return [(k.detach(), v.detach()) for k, v in kvs], self.readout_ce(x, actions, weights)

    def readout_ce(self, x, actions, weights):
        """The backbone readout's weighted cross-entropy on each square's 8 actions, from the
        backbone's final tokens x."""
        b = x.shape[0]
        logits = self.backbone_readout(x[:, :N_PATCHES]).view(b, N_PATCHES, -1, N_ACTIONS)
        codes = patchify(encode_field(actions).long(), self.patch)  # (B, 64, CHUNK * patch * patch)
        w = patchify(weights, self.patch)
        ce = F.cross_entropy(logits.reshape(-1, N_ACTIONS), codes.reshape(-1), reduction="none").view_as(w)
        return (ce * w).sum() / w.sum().clamp(min=1.0)

    def readout_loss(self, obs, z, actions, weights, future):
        """objective "readout": the action readout's cross-entropy, plus (world_model) the future
        readout's cross-entropy on each square's state after each step inside the window."""
        _, x = self.prefix_tokens(obs, z)
        loss = self.readout_ce(x, actions, weights)
        if self.world_model:
            b = x.shape[0]
            logits = self.backbone_future(x[:, :N_PATCHES]).view(b, N_PATCHES, -1, N_FUTURE)
            inside = (weights.flatten(2).amax(-1) > 0)[..., None, None].expand_as(future).float()
            w = patchify(inside, self.patch)
            ce = F.cross_entropy(logits.reshape(-1, N_FUTURE), patchify(future, self.patch).reshape(-1), reduction="none").view_as(w)
            loss = loss + (ce * w).sum() / w.sum().clamp(min=1.0)
        return loss

    def velocity(self, kvs, x_t, t):
        """The action expert: predicted flow velocity in token space for noisy tokens x_t at noise
        level t (B,); for objective "masked", per-slot logits for one-hot slots x_t at mask ratio t."""
        emb = time_embedding(t).unsqueeze(1).expand(-1, x_t.shape[1], -1)
        a = self.action_in(torch.cat([x_t, emb], dim=-1)) + self.action_pos
        for block, kv in zip(self.expert, kvs):
            a, _ = block(a, extra_kv=kv)
        return self.action_out(a)

    def loss_noise(self, b, device):
        """(t, e) for the action loss: noise level and noise, or mask ratio and mask draws."""
        if self.objective == "masked":
            u = torch.rand(b, device=device)
            n_slots = self.slots * (2 if self.world_model else 1)
            return torch.cos(0.5 * math.pi * u), torch.rand(b, HALVES * N_PATCHES, n_slots, device=device)
        beta = torch.distributions.Beta(torch.tensor(1.5, device=device), torch.tensor(1.0, device=device))
        return 0.999 * (1 - beta.sample((b,))), torch.randn(b, HALVES * N_PATCHES, self.token_dim, device=device)

    def expert_input(self, action_slots, future_slots):
        """One-hot expert input from (B, 128, slots) long slots, MASK-able (future_slots None without world_model)."""
        parts = [F.one_hot(action_slots, N_ACTIONS + 1).flatten(2)]
        if self.world_model:
            parts.append(F.one_hot(future_slots, N_FUTURE + 1).flatten(2))
        return torch.cat(parts, dim=-1).float()

    def split_logits(self, out):
        """Expert output -> action logits (B, 128, slots, N_ACTIONS) and future logits (or None)."""
        b, n = out.shape[:2]
        actions = out[..., : self.slots * N_ACTIONS].reshape(b, n, self.slots, N_ACTIONS)
        future = out[..., self.slots * N_ACTIONS :].reshape(b, n, self.slots, N_FUTURE) if self.world_model else None
        return actions, future

    def known_robots(self, obs, future_slots):
        """(B, 128, slots) bool: action slots where a robot is known to stand before that step: at
        step 0 from the scene, at step k > 0 from a revealed (not MASK) future state of step k - 1
        that holds a robot. future_slots (B, 128, slots) long or None (nothing revealed)."""
        b = obs.shape[0]
        later = torch.zeros(b, CHUNK - 1, self.grid, self.grid, dtype=torch.bool, device=obs.device)
        if future_slots is not None:
            later = tokens_to_codes(future_slots, self.patch, self.grid)[:, :-1] == FUTURE_ROBOT
        return codes_to_tokens(torch.cat([obs[:, :1] > 0.5, later], dim=1), self.patch)

    def masked_loss(self, obs, z, actions, weights, t=None, e=None, future=None):
        """Cross-entropy on masked action slots (and, with world_model, masked future-state slots,
        weighted 1 on every square inside the window). Each slot is masked where e < t (t: mask
        ratio per sample, cosine-distributed as in sampling)."""
        # Only actions of robots are ever context: an action slot is revealed only where a robot is
        # known to stand before that step (step 0 from the scene; later steps from a revealed future
        # state, so only with world_model), and only robots' slots are trained. A program's field on
        # empty squares (smooth, so it leaks the program's intent through revealed neighbours) is
        # never seen. Sampling follows the same rule (sample_masked), with its own decoded future.
        occupied = torch.cat([obs[:, :1] > 0.5, future[:, :-1] == FUTURE_ROBOT], dim=1)  # robot there before step k
        inside = (weights.flatten(2).amax(-1) > 0)[..., None, None].expand_as(occupied)
        codes = codes_to_tokens(encode_field(actions).long(), self.patch)
        b = codes.shape[0]
        if t is None or e is None:
            t, e = self.loss_noise(b, codes.device)
        masked = e < t.view(b, 1, 1)
        masked_actions = masked[..., : self.slots]
        future_slots = future_x = None
        if self.world_model:
            future_slots = codes_to_tokens(future, self.patch)
            masked_future = masked[..., self.slots :]
            future_x = torch.where(masked_future, torch.full_like(future_slots, N_FUTURE), future_slots)
        reveal = self.known_robots(obs, future_x) & ~masked_actions
        x = torch.where(reveal, codes, torch.full_like(codes, MASK))
        kvs, readout = self.encode(obs, z, actions, weights)
        out = self.velocity(kvs, self.expert_input(x, future_x), t)
        action_logits, future_logits = self.split_logits(out)
        ce = F.cross_entropy(action_logits.reshape(-1, N_ACTIONS), codes.reshape(-1), reduction="none").view_as(codes)
        counted = codes_to_tokens((occupied & inside).float(), self.patch) * ~reveal
        loss = (ce * counted).sum() / counted.sum().clamp(min=1.0)
        if self.world_model:
            w = codes_to_tokens(inside.float(), self.patch)
            future_ce = F.cross_entropy(future_logits.reshape(-1, N_FUTURE), future_slots.reshape(-1), reduction="none").view_as(w)
            future_counted = w * masked_future
            loss = loss + (future_ce * future_counted).sum() / future_counted.sum().clamp(min=1.0)
        return loss + readout

    def flow_loss(self, obs, z, actions, weights, t=None, e=None, future=None):
        """Weighted flow-matching loss. x_t = t a + (1 - t) e, target a - e; t favours noise.
        t and e can be passed in, so two conditionings can be compared on identical noise.
        (The action loss for every objective: "masked" goes to masked_loss.)"""
        if self.objective == "masked":
            return self.masked_loss(obs, z, actions, weights, t, e, future)
        if self.objective == "readout":
            return self.readout_loss(obs, z, actions, weights, future)
        a = chunk_to_tokens(actions, self.patch)
        w = chunk_to_tokens(weights.unsqueeze(2).expand(-1, -1, 2, -1, -1).contiguous(), self.patch)
        b = a.shape[0]
        if self.objective == "mse":
            pred = self.velocity(self.prefix(obs, z), torch.zeros_like(a), torch.ones(b, device=a.device))
            return ((pred - a) ** 2 * w).sum() / w.sum().clamp(min=1.0)
        if t is None or e is None:
            t, e = self.loss_noise(b, a.device)
        x_t = t.view(b, 1, 1) * a + (1 - t.view(b, 1, 1)) * e
        kvs, readout = self.encode(obs, z, actions, weights)
        v = self.velocity(kvs, x_t, t)
        return ((v - (a - e)) ** 2 * w).sum() / w.sum().clamp(min=1.0) + readout

    @torch.no_grad()
    def sample(self, obs, z, steps=10, guidance=1.0, noise=None):
        """A chunk of fields (B, CHUNK, 2, G, G) by Euler integration from noise, with classifier-free guidance."""
        b = obs.shape[0]
        if self.decode == "readout":
            return self.sample_readout(obs, z, guidance)
        kvs = self.prefix(obs, z)
        if self.objective == "mse":
            zeros = torch.zeros(b, HALVES * N_PATCHES, self.token_dim, device=obs.device)
            return tokens_to_chunk(self.velocity(kvs, zeros, torch.ones(b, device=obs.device)), self.patch, self.grid)
        kvs_null = self.prefix(obs, self.null_z(b)) if guidance != 1.0 else None
        if self.objective == "masked":
            return self.sample_masked(obs, kvs, kvs_null, steps, guidance)
        x = torch.randn(b, HALVES * N_PATCHES, self.token_dim, device=obs.device) if noise is None else noise
        for i in range(steps):
            t = torch.full((b,), i / steps, device=obs.device)
            v = self.velocity(kvs, x, t)
            if kvs_null is not None:
                v_null = self.velocity(kvs_null, x, t)
                v = v_null + guidance * (v - v_null)
            x = x + v / steps
        return tokens_to_chunk(x, self.patch, self.grid)

    def sample_readout(self, obs, z, guidance=1.0):
        """One-shot actions from the backbone's per-square readout (no expert): every square's 8
        steps sampled independently from its categorical, with classifier-free guidance on the
        logits. (B, CHUNK, 2, G, G)."""
        b, n = obs.shape[0], self.grid // self.patch
        logits = self.backbone_readout(self.prefix_tokens(obs, z)[1][:, :N_PATCHES]).view(b, N_PATCHES, -1, N_ACTIONS)
        if guidance != 1.0:
            null = self.backbone_readout(self.prefix_tokens(obs, self.null_z(b))[1][:, :N_PATCHES]).view_as(logits)
            logits = null + guidance * (logits - null)
        codes = torch.distributions.Categorical(logits=logits).sample()
        codes = codes.view(b, n, n, CHUNK, self.patch, self.patch).permute(0, 3, 1, 4, 2, 5).reshape(b, CHUNK, self.grid, self.grid)
        return decode_field(codes, obs.device)

    def sample_masked(self, obs, kvs, kvs_null, steps, guidance):
        """MaskGIT decoding: start fully masked; each step samples every masked slot, keeps the most
        confident ones and re-masks the rest on a cosine schedule. With world_model, action and
        future-state slots are decoded together (one confidence ranking). As in training, an action
        slot is kept only where a robot is known to stand (known_robots, from the scene and the
        future states revealed so far); the others are drawn once more in the last step. Returns
        the action fields (B, CHUNK, 2, G, G)."""
        b, device = obs.shape[0], obs.device
        n_tokens = HALVES * N_PATCHES
        sizes = [N_ACTIONS] + ([N_FUTURE] if self.world_model else [])
        x = [torch.full((b, n_tokens, self.slots), size, dtype=torch.long, device=device) for size in sizes]  # MASK = size
        total = n_tokens * self.slots * len(sizes)
        for i in range(steps):
            ratio = torch.full((b,), math.cos(0.5 * math.pi * i / steps), device=device)
            known = self.known_robots(obs, x[1] if self.world_model else None)
            inputs = self.expert_input(x[0], x[1] if self.world_model else None)
            logits = [l for l in self.split_logits(self.velocity(kvs, inputs, ratio)) if l is not None]
            if kvs_null is not None:
                null = [l for l in self.split_logits(self.velocity(kvs_null, inputs, ratio)) if l is not None]
                logits = [n + guidance * (l - n) for l, n in zip(logits, null)]
            drawn, confidence = [], []
            for slots, part, size in zip(x, logits, sizes):
                probs = part.softmax(-1)
                d = torch.distributions.Categorical(probs=probs).sample()
                c = probs.gather(-1, d.unsqueeze(-1)).squeeze(-1)
                open_slot = slots == size
                drawn.append(torch.where(open_slot, d, slots))
                confidence.append(torch.where(open_slot, c, torch.full_like(c, float("inf"))))
            if i == steps - 1:
                x = drawn
                break
            flat = torch.cat([c.flatten(1) for c in confidence], dim=1)
            n_masked = int(total * math.cos(0.5 * math.pi * (i + 1) / steps))
            lowest = flat.argsort(dim=1)[:, :n_masked]
            remask = torch.zeros(b, total, dtype=torch.bool, device=device).scatter_(1, lowest, True)
            per_part = n_tokens * self.slots
            remask[:, :per_part] |= ~known.flatten(1)
            x = [torch.where(remask[:, j * per_part : (j + 1) * per_part].view_as(d), torch.full_like(d, size), d) for j, (d, size) in enumerate(zip(drawn, sizes))]
        return decode_field(tokens_to_codes(x[0], self.patch, self.grid), device)
