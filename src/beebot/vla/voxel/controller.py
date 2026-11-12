"""The attractor controller: turns a per-scene set of target voxels into every robot's action.

Each robot heads for the nearest target voxel no robot holds yet (or, when every target is held,
the nearest target), taking the one of its 27 actions whose destination is closest to that goal
among destinations that are inside the grid and not a wall or cheese. A robot already on a
target stays. Optionally, robots touching a chosen cheese on its back side press into it along a
push direction. A share of robots take a random action instead (jitter). An `allowed` mask limits
both the chosen and the jittered actions to what the hardware can do (world.CUBE_MOVES).

Every intent of the scripted player is expressed as such a target set: a point (go somewhere),
two points (split), a cheese's shell (eat), a column (tower), a span (bridge), the face behind a
cheese (push).

Risk (odds, from a measured outcome model: world.OutcomeModel.odds): each move is weighed by
what it is expected to leave, the destination with its odds of landing and the robot's own cell
otherwise (a move that stays is simply chosen again next step), and moves that risk harm are left
out when a safer one also gets closer (risky).
"""

import torch

from beebot.vla.voxel.world import DIR_TABLE, N_ACTIONS, STAY, cheese_id_map, flat_index, gather

JITTER_PROB = 0.1
FAR = 1e9
RISK_SLACK = 0.02  # harm odds within this of the safest move that gets closer count as just as safe (about what a table of a few hundred trials per primitive resolves)


def risky(odds, progress):
    """(N, 27) bool: moves a planner that weighs risk leaves out. odds (N, 27, 3): (reached, stayed,
    harmful) per move (world.OutcomeModel.odds); progress (N, 27) bool: the moves that would get
    the robot closer to where it is going. Left out: a move more likely to harm than to land (its
    expected harm outweighs its expected step), and a move whose harm odds exceed those of the
    safest move that gets closer by more than RISK_SLACK. STAY is never risky."""
    reach, harm = odds[..., 0], odds[..., 2]
    floor = torch.where(progress & (harm <= reach), harm, torch.full_like(harm, 2.0)).min(dim=1, keepdim=True).values
    floor = torch.where(floor > 1.0, torch.zeros_like(floor), floor)  # nothing gets closer: only moves as safe as standing still
    out = (harm > reach) | (harm > floor + RISK_SLACK)
    out[:, STAY] = False
    return out


def robot_list(world):
    """Every robot as (global flat index, scene, (z, y, x))."""
    b, d, h, w = world.mass.shape
    v = d * h * w
    robots = (world.mass > 0.5).reshape(-1).nonzero().squeeze(1)
    local = robots % v
    here = torch.stack([local // (h * w), (local // w) % h, local % w], dim=1)
    return robots, robots // v, here


def act(world, targets, valid, press_cheese=None, press_dir=None, jitter=JITTER_PROB, generator=None, allowed=None, odds=None, note=None):
    """Action codes (B, D, H, W) int8 (STAY where no robot is). note (a dict): filled with "before",
    the codes the controller chose before the jitter (-1 off the robots), for the intended-action record (suite.Expert.intended).

    targets (B, S, 3) long, valid (B, S) bool: each scene's target voxels (a scene with none
    stays put). press_cheese (B,) long (-1 for none) and press_dir (B, 3) long: robots whose
    move along press_dir lands on that cheese take that move. allowed (N, 27) bool, over the robots
    in robot_list order: actions the robots can carry out (hardware limits); None allows all.
    Jittered robots draw their random action among the allowed ones. odds (N, 27, 3), in the same
    order: (reached, stayed, harmful) per move (world.OutcomeModel.odds); each move then costs
    what it is expected to leave, and risky moves are left out, jittered ones too (module
    docstring, Risk). None: every move lands."""
    b, d, h, w = world.mass.shape
    device = world.mass.device
    robots, scene, here = robot_list(world)
    codes = torch.full((b * d * h * w,), STAY, dtype=torch.int8, device=device)
    if robots.numel() == 0:
        return codes.view(b, d, h, w)
    held = gather(world.mass, targets, 0.0) > 0.5
    free = valid & ~held
    tgt = targets[scene].float()  # (N, S, 3)
    dist = ((tgt - here.unsqueeze(1).float()) ** 2).sum(-1)
    near_valid = dist.masked_fill(~valid[scene], FAR)
    near_free = dist.masked_fill(~free[scene], FAR)
    any_free = free[scene].any(dim=1)
    choice = torch.where(any_free, near_free.argmin(dim=1), near_valid.argmin(dim=1))
    goal = tgt.gather(1, choice.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
    active = valid[scene].any(dim=1) & (near_valid.min(dim=1).values > 0)

    table = DIR_TABLE.to(device)
    dest = here.unsqueeze(1) + table.unsqueeze(0)  # (N, 27, 3)
    flat, inside = flat_index(dest, (d, h, w))
    ids = cheese_id_map(world).reshape(b, -1)
    blocked = world.walls.reshape(b, -1)[scene.unsqueeze(1), flat] | (ids[scene.unsqueeze(1), flat] >= 0) | ~inside
    if allowed is not None:
        blocked |= ~allowed
    blocked[:, STAY] = False
    cost = ((dest.float() - goal.unsqueeze(1)) ** 2).sum(-1)
    if odds is not None:
        odds = odds.to(device)
        here_cost = cost[:, STAY : STAY + 1]
        blocked = blocked | risky(odds, (cost < here_cost) & ~blocked)
        cost = here_cost - odds[..., 0] * (here_cost - cost)  # expected: the destination if it lands, else here
        if allowed is not None:
            allowed = allowed & ~blocked
    cost = cost + 1e-3 * torch.rand(cost.shape, generator=generator, device=device)
    move = cost.masked_fill(blocked, FAR).argmin(dim=1)
    move = torch.where(active, move, STAY)

    if press_cheese is not None:
        pc = press_cheese[scene]
        step_code = direction_code(press_dir)[scene]
        landing = here + press_dir[scene]
        lf, lin = flat_index(landing, (d, h, w))
        on = lin & (pc >= 0) & (ids[scene, lf] == pc)
        move = torch.where(on, step_code, move)

    jittered = torch.rand(move.shape, generator=generator, device=device) < jitter
    if allowed is None:
        random_move = torch.randint(N_ACTIONS, move.shape, generator=generator, device=device)
    else:
        random_move = torch.multinomial(allowed.float(), 1, generator=generator).squeeze(1)
    if note is not None:
        before = torch.full_like(codes, -1)
        before[robots] = move.to(torch.int8)
        note["before"] = before.view(b, d, h, w)
    move = torch.where(jittered, random_move, move)
    codes[robots] = move.to(torch.int8)
    return codes.view(b, d, h, w)


def direction_code(direction):
    """(..., 3) long unit steps -> action codes (STAY for zero)."""
    table = DIR_TABLE.to(direction.device)
    match = (direction.unsqueeze(-2) == table).all(dim=-1)
    return match.float().argmax(dim=-1)
