"""Domain randomization of the world's physics and of what the policy sees, beyond views.py's visual
randomization. Opt-in everywhere; nothing here runs unless a `Randomization` is passed (the live
reverse source: ThinkData(live={"physics": {...}}), reverse_live.Live(physics=...)).

What is randomized (strength s in [0, 1] per item, `Randomization`; default 0.3, so most draws are
nominal; 0 turns an item off). A physical item is present in an episode with probability s:

    failure    one multiplier 2^U(-1, 1) (0.5x-2x) on the MuJoCo table's failure rates, harmless (stay)
               and harmful (stuck, knocked cubes) alike; without it nothing fails
    weak       1-4 cubes fail U(2, 4) times as often as the table says, the weakness following the
               cube as it moves (follow); switches the table on (at x1 when failure is not drawn)
    broken     with probability 0.5 s, 1 cube (2 with probability 0.3), one that the plan moves at step 0
               if there is one, on a cell the goal holds, stops responding: world.stuck from the start, every
               command to it a no-op (it looks like any robot, see ops node stuck-visible)
    drop       a commanded move is not executed this step with probability 0.04 per moving cube
    late       a commanded move is executed one step late with probability 0.03 per moving cube
    placement  rendering only, always on, size = s: each robot drawn off its cell by up to 0.1 s cells
               and turned by up to 5 s degrees (views.Placement), a fixed draw per (episode, cell)
               with slow drift in t; the state stays on the grid

Reference ranges (HARDWARE.md, MuJoCo "Variation per trial"): friction 0.3-0.7, magnet strength +-25 %
per cube, placement errors up to 0.03 L. The failure and weak ranges are wider than those on purpose:
the table is one measurement of one design and the real cube's failure rates are unknown (a factor 2
either way is a prior, not a measurement). The placement maximum (0.1 cells) is above 0.03 L: it is
for images that look like real or MuJoCo cubes, not for state error.

Episodes. The live reverse episodes are planned under nominal rules (reverse.take_apart verifies every
step with world.step, no failures). `randomize_row` executes the plan from its start under one physics
draw (DrawnPhysics). After each step the actual state is compared with the plan's frame; when a failure
(a failed move, a dropped command, a broken cube, a harmful end) makes them differ, the expert plans
again from the actual state: a fresh guided walk back from the goal to the actual state
(reverse.take_apart with the actual state as the start A; stuck robots cannot move there), verified
step by step as usual, and carries on from it. A late command is waited for with one step in which
every robot stays. The recorded frames are the realized states and the recorded codes the commands. An
episode that does not reach the goal within the step cap (or in MAX_REPLANS plans, or whose plan
cannot be found, e.g. a robot stuck where the goal has none) is dropped: randomize_row returns None.
Push episodes (objects move) get placement only. A re-plan is bounded (REPLAN_STEPS walk steps, REPLAN_SECONDS)
and the episode is dropped when it runs out.

Valid means (replay_drawn): every frame t + 1 equals DrawnPhysics.step(frame t, the codes of step t, t),
a pure function of the draw, the state and t (drops, late commands and outcome noise come from a
generator seeded by (draw, t)); no recorded command is one the pivot rule forbids on the state it is
given in; the last frame is the goal. The policy sees the current frame only (plus its own earlier
thoughts as text), so a failed move shows as the state the next example starts from.

    draw = draw_physics(Randomization(), rng, world, plan_codes, goal_cells)   # one episode's physics
    physics = DrawnPhysics(draw, world)                                        # state: late commands, weak cubes
    world, realized = physics.step(world, codes, t)                            # for the transfer measurement
"""

import contextlib
import signal
import threading
import time
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import torch

from beebot.vla.voxel import world as W
from beebot.vla.voxel.world import DIR_TABLE, STAY, OutcomeModel

RULE = "switch_cube"  # the demo's rule, as reverse.RULE


@dataclass
class Randomization:
    """Strengths of the items (module docstring); an item left at None takes `strength`, 0 turns it off."""

    strength: float = 0.3  # the share of episodes that have each physical item (draw_physics)
    failure: float = None
    weak: float = None
    broken: float = None
    drop: float = None
    late: float = None
    harm: float = 0.5  # of the episodes with failures, the share where the harmful outcomes happen too (else only stays)
    placement: float = 0.5  # offset 0.05 cells, yaw 2.5 degrees at 0.5 (views.Placement)
    outcomes: str = "mujoco_hinge_v1"  # world.OUTCOME_MODELS: the table the failure items scale

    def level(self, item):
        value = getattr(self, item)
        return float(self.strength if value is None else value)

    def any(self):
        return any(self.level(item) > 0 for item in ("failure", "weak", "broken", "drop", "late", "placement"))


@dataclass
class PhysicsDraw:
    """One episode's physics (JSON-safe via asdict)."""

    seed: int = 0
    outcomes: str = None  # None: no outcome model (nothing fails but by drop and late)
    multiplier: float = 1.0
    harm: float = 1.0  # 1: the table's harmful outcomes happen; 0: only the harmless "stayed" ones
    weak: list = field(default_factory=list)  # [[z, y, x, factor]] at the start
    broken: list = field(default_factory=list)  # [[z, y, x]] at the start
    p_drop: float = 0.0
    p_late: float = 0.0
    offset: float = 0.0  # placement, cells
    yaw: float = 0.0  # placement, degrees

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


def draw_physics(rand, rng, world=None, plan=None, goal=None):
    """A PhysicsDraw for one episode. Each physical item at strength s is present in an episode with
    probability s (broken: 0.5 s), so a low strength leaves most draws nominal; placement is always
    drawn, its size scaling with its strength. world: the start (one scene), where the weak and broken
    cubes are chosen among the robots; plan: (T, D, H, W) codes of the nominal plan, whose step 0
    movers are preferred for the broken cubes, and goal: (D, H, W) bool, the goal's cells, where a
    broken cube must be (it can never leave its cell, so the goal has to hold it); none such: no
    broken cube."""
    draw = PhysicsDraw(seed=int(rng.integers(2**31)))
    if rng.random() < rand.level("failure"):
        draw.outcomes = rand.outcomes
        draw.multiplier = float(2.0 ** rng.uniform(-1, 1))
        draw.harm = float(rng.random() < rand.level("harm"))
    if rng.random() < rand.level("drop"):
        draw.p_drop = 0.04
    if rng.random() < rand.level("late"):
        draw.p_late = 0.03
    s = rand.level("placement")
    draw.offset, draw.yaw = 0.1 * s, 5.0 * s
    if world is None:
        return draw
    robots = (world.mass[0] > 0.5).nonzero().tolist()
    movers = []
    if plan is not None and len(plan):
        movers = [c for c in robots if int(plan[0][c[0], c[1], c[2]]) != STAY]
    holds = [c for c in robots if goal is None or bool(goal[c[0], c[1], c[2]])]
    holds = [c for c in holds if c in movers] or holds  # the plan's step 0 movers first
    if holds and rng.random() < 0.5 * rand.level("broken"):
        count = 2 if rng.random() < 0.3 and len(holds) > 1 else 1
        draw.broken = [list(holds[i]) for i in rng.choice(len(holds), size=count, replace=False)]
    if robots and rng.random() < rand.level("weak"):
        pool = [c for c in (movers or robots) if c not in draw.broken]
        count = min(int(rng.integers(1, 5)), len(pool))
        if count:
            picks = rng.choice(len(pool), size=count, replace=False)
            draw.weak = [[*pool[i], float(rng.uniform(2, 4))] for i in picks]
            draw.outcomes = rand.outcomes  # a factor on the table: the table is on, at its own rates if failure is not
    return draw


class ScaledOutcomes(OutcomeModel):
    """A measured OutcomeModel with every failure probability (stayed and harmful) multiplied by
    `scale` (a float) and, per cube, by `cube` (B, D, H, W) float at its cell, the sum capped at 1."""

    def __init__(self, base, scale=1.0, harm=1.0):
        self.__dict__.update(base.__dict__)
        self.scale = float(scale)
        self.harm = float(harm)
        self.cube = None

    def draw(self, world, codes, rule, ids, noise):
        base_reach, base_stay = self.reach, self.stay
        cubes = world.mass > 0.5
        movers = (cubes & (codes != STAY)).nonzero(as_tuple=True)
        grid = None if self.cube is None else self.cube
        try:
            self.reach, self.stay = _scaled(base_reach, base_stay, self.scale, self.harm)
            if grid is None:
                return OutcomeModel.draw(self, world, codes, rule, ids, noise)
            # per cube: draw once per distinct factor (few), each on the cubes that carry it
            out, events = None, None
            factors = torch.unique(grid[movers])
            kept = []
            for factor in factors.tolist():
                here = grid == factor
                only = torch.where(here, codes, torch.full_like(codes, STAY))
                self.reach, self.stay = _scaled(base_reach, base_stay, self.scale * factor, self.harm)
                one, ev = OutcomeModel.draw(self, world, only, rule, ids, noise)
                out = one if out is None else torch.where(here, one, out)
                kept.append(ev)
            if out is None:
                return codes, (torch.zeros(0, dtype=torch.long),) * 2 + (torch.zeros(0, 3, dtype=torch.long),) * 2 + (torch.zeros(0, dtype=torch.long),)
            # cells that were not movers keep their (STAY) code; movers took their factor's outcome
            out = torch.where(codes == STAY, codes, out)
            events = tuple(torch.cat([e[i] for e in kept]) for i in range(5))
            return out, events
        finally:
            self.reach, self.stay = base_reach, base_stay


def _scaled(reach, stay, factor, harm_share=1.0):
    """(reach, stay) with the stayed and harmful probabilities times factor (the sum capped at 1) and the
    harmful ones also times harm_share (0: harmful outcomes do not happen)."""
    harm = (1 - reach - stay).clamp(min=0.0) * harm_share
    miss = stay + harm
    f = torch.where(miss > 0, torch.clamp(torch.full_like(miss, factor), max=1.0 / miss.clamp(min=1e-9)), torch.ones_like(miss))
    return 1 - miss * f, stay * f


def follow(scale, realized, after):
    """The per-cube factor grid scale (1, D, H, W) after a step: each weak cube's factor goes to
    where the cube is now (its move happened, it stayed, or it fell), lost if it cannot be found."""
    out = torch.ones_like(scale)
    d, h, w = after.shape[-3:]
    for z, y, x in (scale[0] != 1).nonzero().tolist():
        factor = scale[0, z, y, x]
        code = int(realized[0, z, y, x])
        target = None
        if code != STAY:
            dz, dy, dx = DIR_TABLE[code].tolist()
            nz, ny, nx = z + dz, y + dy, x + dx
            if 0 <= nz < d and 0 <= ny < h and 0 <= nx < w and after[0, nz, ny, nx] > 0.5 and after[0, z, y, x] < 0.5:
                target = (nz, ny, nx)
        if target is None and after[0, z, y, x] > 0.5:
            target = (z, y, x)
        if target is None:
            for cz, cy, cx in ((z - 1, y, x),) + (((z + DIR_TABLE[code][0].item() - 1, y + DIR_TABLE[code][1].item(), x + DIR_TABLE[code][2].item()),) if code != STAY else ()):
                if 0 <= cz < d and 0 <= cy < h and 0 <= cx < w and after[0, cz, cy, cx] > 0.5:
                    target = (cz, cy, cx)
                    break
        if target is not None:
            out[0][target] = factor
    return out


class DrawnPhysics:
    """One episode's physics for a one-scene world: step(world, codes, t) is world.step under the
    draw (module docstring). It carries the late commands and the weak cubes between steps; call
    it for t = 0, 1, ... in order (a new DrawnPhysics replays the episode)."""

    def __init__(self, draw, world, base=None):
        self.draw = draw
        self.model = None
        if draw.outcomes is not None:
            self.model = ScaledOutcomes(base or OutcomeModel.load(draw.outcomes), draw.multiplier, draw.harm)
        self.scale = torch.ones(world.mass.shape)
        for z, y, x, factor in draw.weak:
            self.scale[0, z, y, x] = factor
        self.waiting = {}  # cell -> code, commands to execute at the next step
        self.dropped, self.delayed = set(), set()  # cells of the last step

    def start(self, world):
        """world with the draw's broken cubes stuck."""
        if not self.draw.broken:
            return world
        stuck = torch.zeros(world.mass.shape, dtype=torch.bool) if world.stuck is None else world.stuck.clone()
        for z, y, x in self.draw.broken:
            stuck[0, z, y, x] = True
        return replace(world, stuck=stuck)

    def step(self, world, codes, t):
        """(world after step t, the codes that were executed). codes (1, D, H, W) integer, the
        commands; a moving cube's command is dropped or delayed per the draw."""
        gen = torch.Generator().manual_seed((self.draw.seed * 1_000_003 + t) & 0x7FFFFFFFFFFFFFFF)
        shape = world.mass.shape
        u_drop, u_late, noise = (torch.rand(shape, generator=gen) for _ in range(3))
        codes = codes.long()
        moving = (codes != STAY) & (world.mass > 0.5)
        drop = moving & (u_drop < self.draw.p_drop)
        late = moving & ~drop & (u_late < self.draw.p_late)
        realized = torch.where(drop | late, torch.full_like(codes, STAY), codes)
        for (z, y, x), code in self.waiting.items():
            if world.mass[0, z, y, x] > 0.5 and codes[0, z, y, x] == STAY:
                realized[0, z, y, x] = code
        self.dropped = {tuple(c) for c in drop[0].nonzero().tolist()}
        self.delayed = {tuple(c) for c in late[0].nonzero().tolist()}
        self.waiting = {c: int(codes[0][c]) for c in self.delayed}
        if self.model is not None:
            self.model.cube = self.scale if (self.scale != 1).any() else None
        after, _ = W.step(world, realized, rule=RULE, outcomes=self.model, noise=noise if self.model is not None else None)
        if (self.scale != 1).any():
            self.scale = follow(self.scale, realized, after.mass)
        return after, realized


# ---------------------------------------------------------------------------------------------
# Episodes

MAX_REPLANS = 12


def _legal(world, codes):
    """codes with every move the pivot rule forbids on this state left out (a stuck cube's moves are
    not forbidden here: the cube does not know it is stuck)."""
    free = replace(world, stuck=None) if world.stuck is not None else world
    return W.refuse(free, codes, RULE)


REPLAN_STEPS = 200  # a re-plan's walk is at most this many steps
REPLAN_SECONDS = 10.0  # ... and this many CPU seconds (wall time of a main-thread alarm; none off the main thread)


class Budget(Exception):
    pass


@contextlib.contextmanager
def _seconds(limit):
    """Raise Budget after `limit` seconds in the block (SIGALRM; only in the main thread, else no limit)."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def alarm(*_):
        raise Budget()

    old = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, limit)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _plan_from(world, goal, rng, limit):
    """A verified plan from the actual state `world` (one scene; its stuck robots cannot move) to the
    occupancy of `goal` (a one-scene world): (frames (n + 1, D, H, W) bool, codes (n, D, H, W) long) or
    (None, why). A fresh guided walk back from the goal to the actual state (reverse.take_apart), so every
    step is verified by world.step as in the original episode."""
    from beebot.vla.voxel import reverse as R

    start = world.mass[0] > 0.5
    target = goal.mass[0] > 0.5
    if int(start.sum()) != int(target.sum()):
        return None, "robot count"
    if world.stuck is not None and bool((world.stuck[0] & ~target).any()):
        return None, "stuck cell not in the goal"
    end = replace(goal, cheese_pos=world.cheese_pos, cheese_alive=world.cheese_alive, stuck=world.stuck)
    try:
        with _seconds(REPLAN_SECONDS):
            ep = R.take_apart([R.Goal("replan", "", end, target, start=start)], rng, max_steps=min(limit, REPLAN_STEPS))[0][0]
    except Budget:
        return None, "cpu budget"
    if not ep.natural or ep.distance != 0:
        return None, "walk failed"
    return (ep.mass.bool(), ep.actions.long()), ""


REJOIN = (3, 6)  # a re-plan first tries to rejoin the original plan this many steps after the failure
REJOIN_STEPS = 12  # the cap of such a walk


def _replan(world, goal_world, frames, codes, k, rng, left, stats):
    """After a harmful failure at plan step k - 1 (the state is not frames[k]): a guided walk from the
    actual state to a later state of the original plan (k + 3, k + 6) that the stuck robots allow, then
    the plan goes on from there; failing that, a walk to the goal. Returns (frames, codes) to continue
    from step 0, or None (stats counts the cause)."""
    stuck = None if world.stuck is None else world.stuck[0].nonzero().tolist()
    for j in [k + r for r in REJOIN if k + r < len(codes)]:
        if stuck and not all(bool(frames[j:, z, y, x].all()) and bool((codes[j:, z, y, x] == STAY).all()) for z, y, x in stuck):
            continue  # a stuck robot the rest of the plan moves
        meet = replace(goal_world, mass=frames[j].float().unsqueeze(0))
        found, why = _plan_from(world, meet, rng, min(REJOIN_STEPS, left))
        if found is not None:
            stats["rejoined"] = stats.get("rejoined", 0) + 1
            return torch.cat([found[0], frames[j + 1 :]]), torch.cat([found[1], codes[j:]])
    found, why = _plan_from(world, goal_world, rng, left)
    if found is None:
        stats["replan failed: " + why] = stats.get("replan failed: " + why, 0) + 1
        return None
    stats["to goal"] = stats.get("to goal", 0) + 1
    return found


def randomize_row(row, rand, seed, stats=None):
    """The packed live episode `row` (reverse_live.pack) executed under one physics draw from `seed`,
    re-planned from the actual state whenever a failure changes it (module docstring). Returns a row of
    the same layout (realized frames, commanded codes, stuck frames, meta["physics"], meta["placement"],
    meta["replans"], meta["failed_steps"]) or None when the episode does not reach the goal within the
    step cap (a dropped episode: no episode is kept unfinished). stats: a dict counting why."""
    from beebot.vla.voxel import reverse as R
    from beebot.vla.voxel import reverse_live as L

    stats = {} if stats is None else stats
    ep = L.LiveEpisode(row)
    rng = np.random.default_rng([int(seed), 7])
    plan = torch.stack([ep.codes(t) for t in range(ep.steps)]) if ep.steps else None
    goal_world = ep.world(ep.steps)
    draw = draw_physics(rand, rng, ep.world(0), plan, goal_world.mass[0] > 0.5)
    meta = dict(row["meta"])
    meta["physics"] = draw.to_dict()
    meta["placement"] = {"offset": draw.offset, "yaw": draw.yaw} if draw.offset > 0 or draw.yaw > 0 else None
    meta["replans"] = 0
    meta["failed_steps"] = 0
    meta["repeats"] = 0
    physical = draw.outcomes is not None or draw.p_drop > 0 or draw.p_late > 0 or draw.broken
    if not physical or meta["family"] == "push":  # push episodes move objects: left as planned
        meta["physics"] = PhysicsDraw(seed=draw.seed, offset=draw.offset, yaw=draw.yaw).to_dict()
        return {**row, "meta": meta}
    physics = DrawnPhysics(draw, ep.world(0))
    world = physics.start(ep.world(0))
    limit = min(R.MAX_STEPS_CAP, R.STEPS_BASE + R.STEPS_PER_ROBOT * int(meta["robots"]))
    frames = torch.stack([torch.from_numpy(ep.occupancy(t)) for t in range(ep.steps + 1)])
    codes = plan if plan is not None else torch.zeros(0, L.SIZE, L.SIZE, L.SIZE, dtype=torch.long)
    k, replans = 0, 0
    masses, stucks, pos, alive, codes_out = [], [], [], [], []

    def record(world, cmd):
        masses.append(world.mass[0] > 0.5)
        stucks.append(torch.zeros(world.mass.shape[1:], dtype=torch.bool) if world.stuck is None else world.stuck[0].clone())
        pos.append(world.cheese_pos[0].clone())
        alive.append(world.cheese_alive[0].clone())
        codes_out.append(cmd[0].to(torch.int8))

    def stuck_of(world):
        return torch.zeros(world.mass.shape[1:], dtype=torch.bool) if world.stuck is None else world.stuck[0]

    goal_cells = goal_world.mass[0] > 0.5
    dirs = DIR_TABLE
    failed = repeats = 0
    pending, redo = {}, 0  # cells whose command did not happen (cell -> code) and how many times in a row the step was redone
    plan_new = False  # set when a failure changed the state in a way a redo does not explain: plan again from it
    while True:
        if torch.equal(world.mass[0] > 0.5, goal_cells) and not physics.waiting and not pending:
            break
        if len(codes_out) >= limit:
            stats["dropped"] = "step cap"
            return None
        if plan_new or (k >= len(codes) and not pending):
            if replans >= MAX_REPLANS:
                stats["dropped"] = "too many re-plans"
                return None
            replans += 1
            stats["replanned"] = stats.get("replanned", 0) + 1
            t0 = time.process_time()
            fresh = _replan(world, goal_world, frames, codes, k, rng, limit - len(codes_out), stats)
            stats.setdefault("replan_cpu", []).append(time.process_time() - t0)
            if fresh is None:
                stats["dropped"] = next(key for key in stats if key.startswith("replan failed"))[len("replan failed: ") :]
                return None
            frames, codes = fresh
            k, plan_new, pending, redo = 0, False, {}, 0
            continue
        waiting = bool(physics.waiting)
        cmd = torch.full((1,) + tuple(codes.shape[1:]), STAY, dtype=torch.long)
        if not waiting:
            if pending:  # redo the step: the cubes whose move did not happen make it again, the others have made theirs
                for cell, code in pending.items():
                    cmd[0][cell] = code
            else:
                cmd = codes[k].unsqueeze(0)
        record(world, cmd)
        before = stuck_of(world).clone()
        world, _ = physics.step(world, cmd, len(codes_out) - 1)
        if not waiting and not pending:
            k += 1
        if physics.delayed:
            continue  # the next step waits for the late move, then compares
        # the plan's frame after the step, with every cube whose move did not happen back on its cell
        expected = frames[k].clone()
        still = {}
        for cell in (cmd[0] != STAY).nonzero().tolist():
            cell = tuple(cell)
            if world.mass[0][cell] > 0.5:
                still[cell] = int(cmd[0][cell])
        for cell, code in still.items():
            there = tuple(c + int(d) for c, d in zip(cell, dirs[code]))
            expected[there] = False
        for cell in still:
            expected[cell] = True
        if torch.equal(world.mass[0] > 0.5, expected) and torch.equal(stuck_of(world), before):
            repeats += len(still)
            redo = redo + 1 if still else 0
            pending = still
            if redo > 3:
                plan_new, pending = True, {}
            continue
        failed += 1
        plan_new = True
        pending = {}
    record(world, torch.full(world.mass.shape, STAY, dtype=torch.long))
    n = len(masses) - 1
    meta["steps"] = n
    meta["replans"] = replans
    meta["failed_steps"] = failed
    meta["repeats"] = repeats
    meta["moving"] = float(np.mean([(c != STAY).sum().item() for c in codes_out[:-1]])) if n else 0.0
    flat = torch.stack(masses).reshape(n + 1, -1).numpy()
    packed, offsets = [], [0]
    for j in range(n):
        here = flat[j].nonzero()[0]
        packed.append(codes_out[j].reshape(-1).numpy()[here].astype(np.int8))
        offsets.append(offsets[-1] + len(here))
    return {
        **row,
        "meta": meta,
        "mass": np.packbits(flat, axis=1),
        "codes": np.concatenate(packed) if packed else np.zeros(0, np.int8),
        "offsets": np.asarray(offsets, np.int64),
        "cheese_pos": torch.stack(pos),
        "cheese_alive": torch.stack(alive),
        "goal_mass": row["mass"][ep.steps],
        "stuck": np.packbits(torch.stack(stucks).reshape(n + 1, -1).numpy(), axis=1),
    }


def replay_drawn(row):
    """Play a randomized row forward under its own draw and compare (module docstring): {"passed": bool,
    "first_bad": int step or -1, "legal": bool: no recorded command is forbidden by the pivot rule}."""
    from beebot.vla.voxel import reverse_live as L

    ep = L.LiveEpisode(row)
    draw = PhysicsDraw.from_dict(ep.meta["physics"])
    physics = DrawnPhysics(draw, ep.world(0))
    world = physics.start(ep.world(0))
    legal, first_bad = True, -1
    for t in range(ep.steps):
        cmd = ep.codes(t).unsqueeze(0)
        legal &= bool((_legal(world, cmd) == cmd).all())
        world, _ = physics.step(world, cmd, t)
        want = ep.world(t + 1)
        same = torch.equal(world.mass > 0.5, want.mass > 0.5) and torch.equal(world.cheese_pos, want.cheese_pos) and torch.equal(world.cheese_alive, want.cheese_alive)
        stuck = torch.zeros_like(world.mass, dtype=torch.bool) if world.stuck is None else world.stuck
        same = same and torch.equal(stuck, torch.zeros_like(stuck) if want.stuck is None else want.stuck)
        if not same and first_bad < 0:
            first_bad = t
    last = ep.occupancy(ep.steps)
    goal = row.get("goal_mass")
    reached = True if goal is None else bool((last == np.unpackbits(goal)[: last.size].reshape(last.shape).astype(bool)).all())
    return {"passed": first_bad < 0 and legal and reached, "first_bad": first_bad, "legal": legal}


def placement_of(meta, t):
    """views.Placement for step t of an episode with this meta (None: perfect cells)."""
    from beebot.vla.voxel.views import Placement

    p = meta.get("placement")
    if not p:
        return None
    return Placement(seed=int(meta["seed"]) & 0x7FFFFFFF, offset=p["offset"], yaw=p["yaw"], t=float(t))
