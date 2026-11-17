"""The low-level controller every player shares: attractor points in, a velocity field out.

A player places up to MAX_ATTRACTORS points. Each square sends its mass one step along the
shortest walking path to the attractor nearest to it (by walking distance), so a swarm with
several attractors splits by proximity. In routed mode, paths go around walls and cheeses, and
an attractor placed on a cheese targets that cheese's ring. In straight mode, paths go around
walls only, so mass heads through a cheese and presses on it: that is how a player pushes.

Attractors fill outward, from the inside: by walking distance from the attractor's targets, the
settled robots are those no farther out than the nearest empty square, and everyone else routes
around them to the empty squares at that distance. So an attractor grows a compact blob one
layer at a time, and a ring target gets surrounded instead of piled onto from one side. A square
with no path to any attractor stays put.

An attractor can ask for a share of the swarm. When any attractor in a scene does, squares are
assigned to attractors by a small auction (walking distance plus a per-attractor price, prices
raised on attractors that draw more than their share), so "split into four equal groups" is
something a player can actually do. Attractors without a share split what the others leave.

The field is dense: every open square gets the vector it would follow, mass or not, so the
stored action carries the player's intent everywhere.
"""

import math

import torch
import torch.nn.functional as F

from beebot.vla.world import DIRS, cheese_masks, look, ring_masks

MAX_ATTRACTORS = 4
SETTLED = 1.0
AUCTION_ROUNDS = 80
AUCTION_DECAY = 0.94
INF = float("inf")
FAR = 1e9  # finite stand-in for infinity inside walking_distance


def _dilate(mask):
    padded = F.pad(mask.float().unsqueeze(1), (1, 1, 1, 1), value=0.0)
    return F.max_pool2d(padded, 3, stride=1).squeeze(1) > 0


def walking_distance(sources, passable, through=None):
    """(N, H, W) bool masks -> (N, H, W) float moves to the nearest source.

    Moves go to any of the 8 neighbors. Every passable square gets a distance, but paths only
    continue through squares in `through` (default: all passable squares). A source counts as
    reachable even if it is not passable itself (an attractor on a wall still pulls mass next to
    it). Unreachable squares are inf.

    Computed with a large finite stand-in for infinity and explicit padding: max_pool2d on MPS
    mishandles inf values and implicit padding (distances came out as 0 or 1 everywhere).
    """
    open_ = passable | sources
    relay = (passable if through is None else through) | sources
    d = torch.where(sources, 0.0, FAR)
    h, w = sources.shape[-2:]
    for i in range(h * w):
        relayed = torch.where(relay, d, FAR)
        padded = F.pad(-relayed.unsqueeze(1), (1, 1, 1, 1), value=-FAR)
        nearest = -F.max_pool2d(padded, 3, stride=1).squeeze(1)
        new = torch.where(open_, torch.minimum(d, nearest + 1), FAR)
        new = torch.where(sources, 0.0, new)
        if i % 4 == 3 and torch.equal(new, d):
            break
        d = new
    return torch.where(d >= FAR / 2, INF, d)


def assign(dist, mass, valid, share, price=None, warm=None):
    """Which attractor each square follows: ((B, H, W) index, (B, A) auction prices).

    Without shares, the nearest. With shares (B, A), NaN where unset, an auction balances the mass
    each attractor draws toward its share. Scenes marked warm start from the given prices with a
    small step, so assignments evolve smoothly from one step to the next instead of dithering.
    """
    b, a = valid.shape
    wants = ~torch.isnan(share) & valid
    use_auction = wants.any(dim=1)
    if price is None:
        price = torch.zeros(b, a, device=dist.device)
        warm = torch.zeros(b, dtype=torch.bool, device=dist.device)
    if not use_auction.any():
        return dist.min(dim=1).indices, price
    given = torch.where(wants, share.clamp(min=0), torch.zeros_like(share))
    given = given / given.sum(dim=1, keepdim=True).clamp(min=1.0)
    leftover = (1 - given.sum(dim=1, keepdim=True)).clamp(min=0)
    unset = valid & ~wants
    target = torch.where(unset, leftover / unset.sum(dim=1, keepdim=True).clamp(min=1), given)
    target = target / target.sum(dim=1, keepdim=True).clamp(min=1e-9)

    total = mass.sum(dim=(-2, -1)).clamp(min=1e-9)  # (B,)
    price = torch.where(warm.view(b, 1), price, torch.zeros_like(price))
    finite = torch.where(torch.isfinite(dist), dist, torch.full_like(dist, 1e4))
    size = dist.shape[-1]
    step = torch.where(warm, size * 0.05, size * 0.5).view(b, 1)
    best = dist.min(dim=1).indices
    best_price = price.clone()
    best_gap = torch.full((b,), INF, device=dist.device)
    for _ in range(AUCTION_ROUNDS):
        cost = torch.where(valid.view(b, a, 1, 1), finite + price.view(b, a, 1, 1), INF)
        which = cost.min(dim=1).indices
        drawn = torch.stack([(mass * (which == j)).sum(dim=(-2, -1)) for j in range(a)], dim=1) / total.view(b, 1)
        gap = ((drawn - target).abs() * valid).amax(dim=1)
        better = gap < best_gap
        best = torch.where(better.view(b, 1, 1), which, best)
        best_gap = torch.where(better, gap, best_gap)
        best_price = torch.where(better.view(b, 1), price, best_price)
        price = price + step * (drawn - target) * valid
        step = step * AUCTION_DECAY
    nearest = dist.min(dim=1).indices
    return torch.where(use_auction.view(b, 1, 1), best, nearest), best_price


def attractor_field(world, points, valid, straight, share=None, price=None, warm=None):
    """The velocity field for every scene.

    points:   (B, A, 2) long (row, col), clamped onto the grid
    valid:    (B, A) bool, which attractors are in use
    straight: (B, A) bool, straight mode per attractor
    share:    (B, A) float share of the swarm each attractor asks for, NaN where unset (optional)
    price, warm: auction state carried over from the previous step (see Controller)
    Returns (field (B, 2, H, W), auction prices (B, A)).
    The field's channel 0 is the column component, channel 1 the row component.
    """
    b, a = valid.shape
    h, w = world.mass.shape[-2:]
    device = world.mass.device
    points = torch.stack([points[..., 0].clamp(0, h - 1), points[..., 1].clamp(0, w - 1)], dim=-1)

    rows = torch.arange(h, device=device).view(1, 1, h, 1)
    cols = torch.arange(w, device=device).view(1, 1, 1, w)
    at_point = (rows == points[..., 0].view(b, a, 1, 1)) & (cols == points[..., 1].view(b, a, 1, 1))
    masks = cheese_masks(world)  # (B, K, H, W)
    on_cheese = masks.any(dim=1)
    routed_ok = ~world.walls & ~on_cheese
    straight_ok = ~world.walls
    straight_ = straight.view(b, a, 1, 1)
    passable = torch.where(straight_, straight_ok.unsqueeze(1), routed_ok.unsqueeze(1))

    # Targets: the point itself, or in routed mode on a cheese, that cheese's ring.
    rings = ring_masks(world, masks)  # (B, K, H, W)
    hit = (masks.unsqueeze(1) & at_point.unsqueeze(2)).any(dim=(-2, -1))  # (B, A, K)
    ring_target = (hit.unsqueeze(-1).unsqueeze(-1) & rings.unsqueeze(1)).any(dim=2)
    use_ring = ~straight_ & hit.any(dim=-1).view(b, a, 1, 1)
    targets = torch.where(use_ring, ring_target, at_point) & valid.view(b, a, 1, 1)

    # Settled region: a ball filled from the inside out. By walking distance from the targets, r
    # is the distance of the nearest empty square; robots no farther than r stay put, and every
    # other robot heads for the empty squares at distance r. So a blob grows compactly, one layer
    # at a time, whatever order robots arrive in (a line of arrivals doesn't freeze as a line).
    full = (world.mass >= SETTLED).unsqueeze(1) & passable
    base = walking_distance(targets.view(b * a, h, w), passable.reshape(b * a, h, w)).view(b, a, h, w)
    empty = passable & ~full & torch.isfinite(base)
    radius = torch.where(empty, base, INF).amin(dim=(-2, -1), keepdim=True)
    settled = full & (base <= radius)
    frontier = empty & (base <= radius)
    sources = torch.where(frontier.any(dim=(-2, -1), keepdim=True), frontier, targets)

    through = passable & ~settled
    dist = walking_distance(
        sources.view(b * a, h, w), passable.reshape(b * a, h, w), through.reshape(b * a, h, w)
    ).view(b, a, h, w)
    dist = torch.where(settled, 0.0, dist)
    dist = torch.where(valid.view(b, a, 1, 1), dist, INF)
    relay = torch.where(settled, INF, dist)  # a settled square is never a next step

    # Each square follows its nearest attractor, or the one the auction gives it.
    if share is None:
        share = torch.full((b, a), float("nan"), device=device)
    which, price = assign(dist, world.mass, valid, share, price, warm)
    own = dist.gather(1, which.unsqueeze(1)).squeeze(1)  # (B, H, W)

    # The best neighbor under that attractor's distance, ties broken toward the attractor.
    target_rows = points[..., 0].float().gather(1, which.view(b, -1)).view(b, h, w)
    target_cols = points[..., 1].float().gather(1, which.view(b, -1)).view(b, h, w)
    chosen = relay.gather(1, which.unsqueeze(1)).squeeze(1)
    best_score = torch.full((b, h, w), INF, device=device)
    best_dir = torch.zeros(b, h, w, dtype=torch.long, device=device)
    grid_rows = rows.view(1, h, 1).float()
    grid_cols = cols.view(1, 1, w).float()
    # Among equally short moves, prefer a square no robot is on (robots can't share squares, so
    # otherwise a crowd queues behind one square while an equally good one stays empty); then
    # the move that ends nearest the attractor.
    occupied = (world.mass >= SETTLED).float()
    for k, (dy, dx) in enumerate(DIRS):
        neighbor = look(chosen, dy, dx, INF)
        euclid = torch.sqrt((grid_rows + dy - target_rows) ** 2 + (grid_cols + dx - target_cols) ** 2)
        score = neighbor + 0.5 * look(occupied, dy, dx, 0.0) + 1e-3 * euclid
        better = score < best_score
        best_score = torch.where(better, score, best_score)
        best_dir = torch.where(better, torch.full_like(best_dir, k), best_dir)

    improves = (best_score < own) & torch.isfinite(own) & (own > 0)
    unit = torch.tensor([[dx / math.hypot(dx, dy), dy / math.hypot(dx, dy)] for dy, dx in DIRS], device=device)
    vectors = unit[best_dir]  # (B, H, W, 2)
    field = torch.where(improves.unsqueeze(-1), vectors, torch.zeros_like(vectors))
    field = torch.where(world.walls.unsqueeze(-1), torch.zeros_like(field), field)
    return field.permute(0, 3, 1, 2).contiguous(), price


class Controller:
    """attractor_field with memory: keeps each scene's auction prices across steps while its
    commands stay the same, and resets them when the commands change."""

    def __init__(self):
        self.price = None
        self.key = None

    def __call__(self, world, points, valid, straight, share):
        key = torch.cat(
            [points.flatten(1).float(), valid.float(), straight.float(), torch.nan_to_num(share, nan=-1.0)], dim=1
        )
        warm = None
        if self.key is not None and self.key.shape == key.shape:
            warm = (key == self.key).all(dim=1)
        field, self.price = attractor_field(world, points, valid, straight, share, self.price if warm is not None else None, warm)
        self.key = key
        return field


def commands_to_tensors(commands, batch_size, device="cpu"):
    """Per-scene lists of {'row', 'col', 'straight', 'share'} dicts -> (points, valid, straight, share)."""
    points = torch.zeros(batch_size, MAX_ATTRACTORS, 2, dtype=torch.long)
    valid = torch.zeros(batch_size, MAX_ATTRACTORS, dtype=torch.bool)
    straight = torch.zeros(batch_size, MAX_ATTRACTORS, dtype=torch.bool)
    share = torch.full((batch_size, MAX_ATTRACTORS), float("nan"))
    for i, scene_commands in enumerate(commands):
        for j, command in enumerate((scene_commands or [])[:MAX_ATTRACTORS]):
            points[i, j, 0] = int(command["row"])
            points[i, j, 1] = int(command["col"])
            straight[i, j] = bool(command.get("straight", False))
            if command.get("share") is not None:
                share[i, j] = float(command["share"])
            valid[i, j] = True
    return points.to(device), valid.to(device), straight.to(device), share.to(device)
