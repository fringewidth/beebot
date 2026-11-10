"""An expert for cubes that cannot leave a bond (world.pivot_moves, rule "free_cube"), and the
comparison of experts and rules on the suite's tasks.

Under the rule a cube that touches another cube face to face can only pivot on that cube, and
every pivot ends face to face with the support or the surface it continues on. So joining a
group cannot be undone: a cube leaves only the floor and terrain behind. The planner therefore
treats lone cubes as its material and groups as scenery that can crawl:

1. Mobile cubes are the lone ones (no cube on any face) and the ones this planner has brought
   to a structure and is still moving (climbing). Every other cube is part of the scenery.
1. Each step one breadth-first search per scene runs backward from the empty target voxels
   through the rule's own moves, as a lone cube would make them among the scenery. It yields
   every voxel's number of pivots to the nearest target it can reach: rolling, climbing a wall
   of cubes (concave up, then convex up from hanging), walking along a structure's top and
   stepping off its end all come out of the search.
1. Mobile cubes are handled nearest first. A cube takes an allowed move that lowers its
   distance, unless the landing cell is taken or touches a face of another mobile cube (where
   it stands or where it is going): two cubes that meet in transit are stuck together for good.
   So cubes travel with gaps and arrive one at a time.
1. Targets that would be walled in are served first (a head start for voxels whose four level
   neighbours are targets too), and a tower is built one column after the other.
1. Cubes that sit in a group, off the targets, crawl after the mobile cubes have chosen: each
   takes an allowed pivot that lowers the same distance (a pair leapfrogs across the floor at
   half a lone cube's pace) or, with no way to a target, the pivot that brings it nearest to
   one in a straight line, as the attractor controller would. Never next to a mobile cube.
1. Of a cube and the cube it pivots on, only one moves in a step (world.refuse).

Tried and dropped (48 scenes of another seed, 72 steps): keeping crawling groups from joining
each other (fewer shell voxels covered), no straight-line fallback (0.45 against 0.49 mean
score), columns of props under a formation's voxels in the air (no gain: the limit is how many
cubes can still move, not what they can reach).

With rule "switch_cube" (bonds can be released) every cube off a target is mobile and the gap
keeping is off.

    OMP_NUM_THREADS=4 python -m beebot.vla.voxel.free_cube --scenes 96 --steps 72 144 --out runs/free_cube/compare.jsonl
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from beebot.vla.voxel import suite as S
from beebot.vla.voxel.controller import act, direction_code
from beebot.vla.voxel.world import CUBE_MOVES, DIR_TABLE, DIRS, STAY, _at, _FACES, _padded, _SIDES, cheese_id_map, failure_rates, flat_index, pivot_moves, rule_table

FAR = 30000  # a distance no path has
SEARCH_STEPS = 96
INNER_HEAD = 3  # pivots of head start for a target whose four level neighbours are targets
COLUMN_HEAD = 4  # and of a tower's column over the next one
CUBE_CODES = tuple(c for c in range(STAY) if bool(CUBE_MOVES[c]))


def distances(paths, seeds, head, passable, limit=SEARCH_STEPS):
    """(B, D, H, W) int16: the fewest pivots from each voxel to a seed voxel, counted from the
    seed's head (B, D, H, W) (a seed with a larger head is further by that much), through moves
    of `paths` (pivot_moves) whose landing voxels are passable. FAR where there is no way."""
    dist = torch.full(seeds.shape, FAR, dtype=torch.int16)
    last_head = int(head[seeds].max()) if seeds.any() else -1
    k = 0
    while k < limit:
        dist[seeds & (head == k) & (dist == FAR)] = k
        frontier = (dist == k) & passable
        if k > last_head and not frontier.any():
            break
        ahead = _padded(frontier, False)
        reach = torch.zeros_like(seeds)
        for code in CUBE_CODES:
            reach |= paths[:, code] & _at(ahead, DIRS[code])
        dist[reach & (dist == FAR)] = k + 1
        k += 1
    return dist


class FreeCubeExpert(S.Expert):
    """The suite's expert with the planner of this module in place of the attractor controller:
    the same plans, segments and sub-goal records. No jitter: a random move can bond a cube."""

    def __init__(self, tasks, seed=None, rule="free_cube", crawl=True):
        super().__init__(tasks, moves=CUBE_MOVES, seed=seed, rule=rule)
        self.crawl = crawl  # cubes in groups move too, along their group
        self.wander = True  # crawlers with no way to a target still head for the nearest one
        self.stats = None  # of the last step, per scene: [on a target, mobile, crawling, moved, held (no pivot), no way (pivots, none nearer a target)]
        self.transit = None  # (B, D, H, W) bool: cubes this planner is moving

    def heads(self, i, seg, world, shells):
        """Each target's head start (pivots; lower is served first), in targets_of order."""
        if seg["kind"] == "union":
            out = [value for part in seg["parts"] for value in self.heads(i, part, world, shells)]
        else:
            points = [tuple(p) for p in self.targets_of(i, seg, world, shells)[0]]
            cells = set(points)
            out = [0 if all((z, y + dy, x + dx) in cells for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0))) else INNER_HEAD for z, y, x in points]
            if seg["kind"] == "targets" and seg.get("label", {}).get("kind") == "tower":
                columns = sorted({(y, x) for _, y, x in points})
                out = [value + COLUMN_HEAD * columns.index((y, x)) for value, (_, y, x) in zip(out, points)]
        return out

    def __call__(self, world, t, generator=None):
        lists, presses, _, segs = self.segments(world, t)
        targets, valid, press_cheese, press_dir = self.tensors(lists, presses)
        shells = S.shell(world)
        head = torch.zeros(valid.shape, dtype=torch.int16)
        for i, seg in enumerate(segs):
            if seg is not None and lists[i]:
                head[i, : len(lists[i])] = torch.tensor(self.heads(i, seg, world, shells), dtype=torch.int16)
        return self.plan(world, targets, valid, head, press_cheese, press_dir, self.generator if generator is None else generator)

    def plan(self, world, targets, valid, head, press_cheese, press_dir, generator=None):
        """Action codes (B, D, H, W) int8 for target voxels (B, S, 3) with heads (B, S)."""
        b, d, h, w = world.mass.shape
        occupied = world.mass > 0.5
        ids = cheese_id_map(world)
        cheese = ids >= 0
        release = self.rule == "switch_cube"
        flat, inside = flat_index(targets, (d, h, w))
        wanted = torch.zeros(b, d * h * w, dtype=torch.bool)
        wanted.scatter_(1, flat, valid & inside)
        wanted = wanted.view(b, d, h, w)
        head_grid = torch.zeros(b, d * h * w, dtype=torch.int16)
        head_grid.scatter_(1, flat, torch.where(valid & inside, head, torch.zeros_like(head)))
        head_grid = head_grid.view(b, d, h, w)
        on_target = occupied & wanted
        others = _padded(occupied, False)
        lone = occupied & (sum(_at(others, f).to(torch.int8) for f in _FACES) == 0)
        if release:
            mobile = occupied & ~on_target
        else:
            mobile = (lone | (self.transit & occupied if self.transit is not None else torch.zeros_like(occupied))) & ~on_target
        scenery = occupied & ~mobile
        paths = pivot_moves(scenery, world.walls, cheese, release=release)
        dist = distances(paths, wanted & ~occupied & ~world.walls & ~cheese, head_grid, ~scenery)
        if not release:
            mobile &= lone | (dist < FAR)  # a cube that joined a group and has no way on is scenery from now

        codes = torch.full((b, d, h, w), STAY, dtype=torch.int8)
        crawlers = occupied & ~mobile & ~on_target if self.crawl else torch.zeros_like(occupied)
        scene, z, y, x = (mobile | crawlers).nonzero(as_tuple=True)
        transit = torch.zeros_like(occupied)
        if scene.numel():
            table, side = rule_table(world, self.rule, sides=True)
            truth = table[scene, :, z, y, x]  # (M, 27)
            here = torch.stack([z, y, x], dim=1)
            there = here.unsqueeze(1) + DIR_TABLE  # (M, 27, 3)
            flat_there, inside_there = flat_index(there, (d, h, w))
            beyond = torch.where(inside_there, dist.view(b, -1)[scene.unsqueeze(1), flat_there], FAR).long()
            own = dist[scene, z, y, x].long()
            lower = truth & (beyond < own.unsqueeze(1))
            lower[:, STAY] = False
            is_mobile = mobile[scene, z, y, x]
            lost = ~is_mobile & (own >= FAR) & self.wander  # crawlers with no way to a target head for the nearest one as the attractor would
            if lost.any():
                greedy = act(world, targets, valid, jitter=0.0, generator=generator, allowed=table.permute(0, 2, 3, 4, 1)[occupied] & CUBE_MOVES)[scene, z, y, x].long()
                lower[lost, greedy[lost]] = greedy[lost] != STAY
            cost = torch.where(lower, beyond.clamp(max=FAR).float(), 2.0 * FAR) + torch.rand(lower.shape, generator=generator)
            ranking = cost.argsort(dim=1)[:, :4].tolist()
            n_lower = lower.sum(dim=1).tolist()
            order = (own.float() + torch.rand(own.shape, generator=generator) + 2.0 * FAR * (~is_mobile)).argsort().tolist()  # mobile cubes first, each kind nearest first
            sides = side[scene, :, z, y, x].tolist()
            cells = here.tolist()
            scenes = scene.tolist()
            mobiles = is_mobile.tolist()
            near = np.zeros((b, d + 2, h + 2, w + 2), dtype=np.int16)  # how many mobile cubes (or their landing cells) touch each voxel
            taken = np.zeros((b, d + 2, h + 2, w + 2), dtype=bool)  # landing cells
            busy = np.zeros((b, d + 2, h + 2, w + 2), dtype=bool)  # cubes that move this step
            leant = np.zeros((b, d + 2, h + 2, w + 2), dtype=bool)  # voxels a moving cube pivots on
            for i, (cz, cy, cx), free in zip(scenes, cells, mobiles):
                if free:
                    for fz, fy, fx in _FACES:
                        near[i, cz + fz + 1, cy + fy + 1, cx + fx + 1] += 1
            chosen = [STAY] * len(cells)
            for m in order:
                i, (cz, cy, cx) = scenes[m], cells[m]
                if leant[i, cz + 1, cy + 1, cx + 1]:
                    continue
                for code in ranking[m][: n_lower[m]]:
                    dz, dy, dx = DIRS[code]
                    tz, ty, tx = cz + dz + 1, cy + dy + 1, cx + dx + 1
                    quarter = abs(dz) + abs(dy) + abs(dx) == 1
                    own_touch = 1 if quarter and mobiles[m] else 0
                    sz, sy, sx = _SIDES[sides[m][code]]
                    support = (i, cz + sz + 1, cy + sy + 1, cx + sx + 1)
                    rest = (i, tz + sz, ty + sy, tx + sx) if quarter else support
                    if taken[i, tz, ty, tx] or busy[support] or busy[rest] or (not release and near[i, tz, ty, tx] > own_touch):
                        continue
                    chosen[m] = code
                    taken[i, tz, ty, tx] = True
                    busy[i, cz + 1, cy + 1, cx + 1] = True
                    leant[support] = leant[rest] = True
                    if mobiles[m]:
                        for fz, fy, fx in _FACES:
                            near[i, tz + fz, ty + fy, tx + fx] += 1
                        transit[i, tz - 1, ty - 1, tx - 1] = True
                    break
            codes[scene, z, y, x] = torch.tensor(chosen, dtype=torch.int8)
            moved = torch.tensor(chosen) != STAY
            can_pivot = truth[:, :STAY].any(dim=1)
            columns = [torch.ones_like(moved), is_mobile, ~is_mobile, moved, ~can_pivot, can_pivot & ~lower.any(dim=1)]
            stats = torch.zeros(b, len(columns), dtype=torch.long).index_add_(0, scene, torch.stack(columns, dim=1).long())
            stats[:, 0] = on_target.sum(dim=(1, 2, 3))
            self.stats = stats
            transit[scene[is_mobile], z[is_mobile], y[is_mobile], x[is_mobile]] = True
        self.transit = transit

        # robots whose step along the press direction lands on the pressed cheese take it (as controller.act)
        scene, z, y, x = occupied.nonzero(as_tuple=True)
        landing = torch.stack([z, y, x], dim=1) + press_dir[scene]
        flat_landing, inside_landing = flat_index(landing, (d, h, w))
        pressed = press_cheese[scene]
        on = inside_landing & (pressed >= 0) & (ids.view(b, -1)[scene, flat_landing] == pressed)
        codes[scene[on], z[on], y[on], x[on]] = direction_code(press_dir)[scene[on]].to(torch.int8)
        return codes


# ---------------------------------------------------------------------------------------------
# Comparison: the same tasks and seeds under each rule and expert

ARMS = {
    # name: (expert, rule, gravity on everywhere, robots re-placed apart at the start)
    "cube": ("attractor", None, False, False),
    "cube, gravity on": ("attractor", None, True, False),
    "switch_cube": ("attractor", "switch_cube", True, False),
    "switch_cube, planner": ("planner", "switch_cube", True, False),
    "free_cube": ("attractor", "free_cube", True, False),
    "free_cube, lone planner": ("lone", "free_cube", True, False),
    "free_cube, planner": ("planner", "free_cube", True, False),
    "free_cube, apart": ("attractor", "free_cube", True, True),
    "free_cube, planner, apart": ("planner", "free_cube", True, True),
}
BUCKETS = (("under 15", 0, 14), ("15 to 40", 15, 40), ("over 40", 41, 10**6))


def make_expert(name, tasks, rule, seed):
    """The expert of an arm: "attractor" (suite.Expert), "planner" (FreeCubeExpert) or "lone"
    (FreeCubeExpert that leaves grouped cubes where they are)."""
    if name in ("planner", "lone"):
        expert = FreeCubeExpert(tasks, seed=seed, rule=rule, crawl=name == "planner")
    else:
        expert = S.Expert(tasks, moves=CUBE_MOVES, seed=seed, rule=rule)
    return expert


def apart(world, rng):
    """The scenes with every swarm set down again on the terrain so that no two robots touch
    (as a swarm would be laid out by hand): random ground cells, none beside another and none on
    or beside a cheese's footprint. Robots that do not fit that way (a very dense scene) go on
    any free ground cell."""
    b, d, h, w = world.mass.shape
    mass = torch.zeros_like(world.mass)
    cheese = (cheese_id_map(world) >= 0).any(dim=1)  # (B, H, W)
    ground = world.walls.sum(dim=1)  # terrain is a heightmap
    for i in range(b):
        n = int(world.mass[i].sum())
        keep_clear = cheese[i].clone()
        keep_clear[1:] |= cheese[i, :-1]
        keep_clear[:-1] |= cheese[i, 1:]
        keep_clear[:, 1:] |= cheese[i, :, :-1]
        keep_clear[:, :-1] |= cheese[i, :, 1:]
        full = keep_clear | (ground[i] >= d)
        crowded = full.clone()
        spare = []
        for flat in rng.permutation(h * w).tolist():
            y, x = divmod(flat, w)
            if n == 0 or full[y, x]:
                continue
            if crowded[y, x]:
                spare.append((y, x))
                continue
            mass[i, int(ground[i, y, x]), y, x] = 1.0
            crowded[max(y - 1, 0) : y + 2, x] = True
            crowded[y, max(x - 1, 0) : x + 2] = True
            n -= 1
        for y, x in spare[:n]:
            mass[i, int(ground[i, y, x]), y, x] = 1.0
    return replace(world, mass=mass)


def lone_share(world):
    """(B,) the share of each scene's robots with no robot on any face."""
    occupied = world.mass > 0.5
    others = _padded(occupied, False)
    lone = occupied & (sum(_at(others, f).to(torch.int8) for f in _FACES) == 0)
    return lone.sum(dim=(1, 2, 3)).float() / occupied.sum(dim=(1, 2, 3)).clamp(min=1)


def compare(n_scenes, steps_list, arms, seed=0, chunk=24, out=None, failure=None):
    """Play demos.py's cube sample (tasks fitted to swarms of suite.ROBOTS, counts log-uniform,
    one instruction per family a scene allows) under every arm, from the same scenes with the
    same seeds (the "apart" arms re-place each swarm first, apart). Returns rows {"arm", "steps", "family", "kind", "robots", "gravity", "lone",
    "score", "unasked", "success"}; out: a .jsonl path to append them to as they come."""
    from beebot.vla.voxel.demos import SUCCESS, contrast  # demos imports this module

    rng = np.random.default_rng(seed)
    failure = failure_rates() if failure is None else failure
    rows = []
    sink = open(out, "a") if out else None
    for c in range(0, n_scenes, chunk):
        world, metas, tasks = S.sample_tasks(rng, min(chunk, n_scenes - c), "train", robots=S.ROBOTS, check=True, counts="log")
        world, metas, tasks = contrast(rng, world, metas, tasks, check=True)
        run_seed = seed * 1_000_003 + c
        lone = lone_share(world).tolist()
        for steps in steps_list:
            for arm in arms:
                expert_name, rule, gravity_on, spaced = ARMS[arm]
                start = replace(world, gravity=torch.ones_like(world.gravity)) if gravity_on else world
                if spaced:
                    start = apart(start, np.random.default_rng(run_seed))
                began = time.time()
                results, _ = S.run(start, metas, tasks, make_expert(expert_name, tasks, rule, run_seed), steps=steps, seed=run_seed, failure=failure, rule=rule)
                for i, ((score, unasked), task) in enumerate(zip(results, tasks)):
                    row = {
                        "arm": arm,
                        "steps": steps,
                        "family": task.family,
                        "kind": S.task_kind(task),
                        "robots": int(world.mass[i].sum()),
                        "gravity": bool(world.gravity[i]),
                        "lone": round(lone[i], 3),
                        "score": round(float(score), 4),
                        "unasked": bool(unasked),
                        "success": bool(not unasked and score >= SUCCESS.get(task.family, S.PASS_MARK)),
                    }
                    rows.append(row)
                    if sink:
                        sink.write(json.dumps(row) + "\n")
                if sink:
                    sink.flush()
                print(f"scenes {min(c + chunk, n_scenes)}: {arm}, {steps} steps, {len(tasks)} episodes, {time.time() - began:.0f} s", flush=True)
    if sink:
        sink.close()
    return rows


def table(rows, key):
    """Markdown rows: for each value of key(row), each arm's mean score and success share."""
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    groups = sorted({key(r) for r in rows})
    lines = ["| | n | " + " | ".join(arms) + " |", "| --- | --- | " + " | ".join("---" for _ in arms) + " |"]
    for group in groups:
        cells = []
        for arm in arms:
            part = [r for r in rows if r["arm"] == arm and key(r) == group]
            cells.append(f"{np.mean([0.0 if r['unasked'] else r['score'] for r in part]):.2f} / {np.mean([r['success'] for r in part]):.2f}" if part else "")
        n = len([r for r in rows if r["arm"] == arms[0] and key(r) == group])
        lines.append(f"| {group} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def bucket(robots):
    return next(f"{k} {name}" for k, (name, low, high) in enumerate(BUCKETS) if low <= robots <= high)


def report(rows):
    """The tables of a comparison: score / success by family, swarm size, task kind."""
    parts = []
    for steps in sorted({r["steps"] for r in rows}):
        part = [r for r in rows if r["steps"] == steps]
        parts.append(f"{steps} steps, mean score (0 when a forbidden cheese was eaten) / success share\n")
        parts.append(table(part, lambda r: "all"))
        parts.append(table(part, lambda r: r["family"]))
        parts.append(table(part, lambda r: bucket(r["robots"])))
        parts.append(table(part, lambda r: r["kind"]))
    return "\n\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scenes", type=int, default=96)
    parser.add_argument("--steps", type=int, nargs="+", default=[S.CUBE_STEPS, 2 * S.CUBE_STEPS])
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=24)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", default=None, help="a .jsonl file to append the per-episode rows to")
    parser.add_argument("--report", default=None, help="print the tables of an existing .jsonl instead of playing")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.report:
        rows = [json.loads(line) for line in open(args.report)]
    else:
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        rows = compare(args.scenes, args.steps, args.arms, args.seed, args.chunk, args.out)
    print(report(rows))


if __name__ == "__main__":
    main()
