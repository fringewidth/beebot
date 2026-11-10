"""Score a pivoting-cube design (hardware/cube.py) on the moves the voxel policy issues, and
hill-climb designs.

Per primitive, the controller's knob, the flywheel speed, is tuned the way a real module would be
calibrated. A geometric grid of speeds, TUNE_TRIALS jittered trials each, finds the band where the
move works (the speed whose success rate, averaged with its grid neighbours, is highest). Between
that speed's grid neighbours, REFINE start speeds are then compared on shared tuning trials (more
trials for the better ones, LADDER_STAGES), and the one with the highest value (success rate minus
Control.harm_weight times the harmful rate) is kept; if none has a positive value the move is not
attempted. The primitive is then scored on VALIDATE_TRIALS fresh jittered trials. Trials
use common random numbers: the same seeds for every design. Primitives with neighbours are scored
twice, once with the neighbours anchored (part of a large lattice) and once free (lone modules),
and the two are averaged.

A move may be a ladder of attempts (Control.attempts): whenever the module finds itself still
aligned in its start cell, it tries again, at the same speed or a step faster (STEPS; the tuning
picks the step along with the start speed). A ladder that starts low crosses the threshold of the
scene it is in from below, instead of at one speed chosen for the average scene.

The design's move score weights each primitive by how often the voxel suite's expert issues that
move to a crawler swarm (MOVE_MIX). The policy issues single pivots only: the expert's horizontal
diagonals (two pivots each) are left out and the remaining shares renormalised.

    uv run python -m beebot.hardware.cube_search eval --side 0.03
    uv run python -m beebot.hardware.cube_search climb --side 0.03 --out runs/cube/climb-3cm.jsonl
    uv run python -m beebot.hardware.cube_search eval --side 0.05 --brake-scale 2.56 --steel-floor 0.3 \\
        --harm-weight 3 --attempts 6 --phase test --trials 120 --first-trial 120
"""

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace

import numpy as np

from beebot.hardware.cube import G, PRIMITIVES, Cube, Jitter, trial

SPEEDS = 20  # grid points from MIN_RPM to the motor's max
MIN_RPM = 1000.0
TUNE_TRIALS = 5
REFINE = 9  # speeds between the best coarse speed's neighbours
LADDER_STAGES = ((8, None), (24, 6), (64, 2))  # (further tuning trials, candidates kept for them; None: all)
STEPS = (1.0, 1.06)  # speed ratio from one attempt to the next
VALIDATE_TRIALS = 40
OUTCOMES = ("success", "stay", "off", "knock")
PHASES = ("tune", "validate", "test")

# Share of the moves the voxel suite's expert issues to a 3 cm permanent-magnet crawler swarm
# (L3cm-magnet-c0.12-a0.25-f0, 192 training tasks, seed 0), by pivot primitive. Diagonals are
# horizontal moves along two axes at once.
EXPERT_MIX = {
    "convex_up": 0.3319,
    "diagonal_up": 0.1967,
    "diagonal_floor": 0.1516,
    "convex_down": 0.1103,
    "wall_crawl": 0.087,
    "floor_roll": 0.0481,
    "floor_peel": 0.0331,
    "concave_down": 0.0201,
    "concave_up": 0.0162,
    "traverse": 0.005,
}
# The policy issues single pivots only (decided 2026-11-10): the expert's mix without the diagonal
# rows, renormalised. The mix of an expert that plans without diagonals has not been measured.
SINGLE_PIVOTS = sum(share for move, share in EXPERT_MIX.items() if move in PRIMITIVES)
MOVE_MIX = {move: share / SINGLE_PIVOTS for move, share in EXPERT_MIX.items() if move in PRIMITIVES}

# Reference: the 3D M-Blocks (Romanishin et al., ICRA 2015) as calibrated in HARDWARE.md.
REF_SIDE = 0.05
REF_DENSITY = 1200.0  # kg/m^3: 150 g in a 50 mm cube
REF_FLYWHEEL_SHARE = 0.155
REF_MAGNET_SHARE = 0.10  # assumed share of the module's mass in magnets
REF = Cube()


@dataclass(frozen=True)
class Build:
    """A buildable design, scaled from the calibrated 3D M-Blocks. Brake torque scales as L^3 (the
    band brake's size), magnet pull as L^2 (face area); flywheel radius stays 0.38 L."""

    side: float = 0.03
    flywheel_share: float = REF_FLYWHEEL_SHARE
    max_rpm: float = 20000.0
    brake_scale: float = 1.0  # brake torque relative to the reference's, scaled to this size
    magnet_scale: float = 1.0  # magnet volume relative to the reference's, scaled to this size
    steel_floor: float = 0.0  # grip of a steel floor on each magnet, relative to its twin's (0: no steel; thinner steel or a spacer lowers it)

    def cube(self):
        ref_mass = REF_DENSITY * self.side**3
        mass = ref_mass * (1 + self.flywheel_share - REF_FLYWHEEL_SHARE + REF_MAGNET_SHARE * (self.magnet_scale - 1))
        ref_weight_here = REF_DENSITY * REF_SIDE**3 * G * (self.side / REF_SIDE) ** 2  # pull scales with area
        pull = self.magnet_scale * ref_weight_here / (mass * G)
        return Cube(
            side=self.side,
            mass=mass,
            flywheel_share=self.flywheel_share * ref_mass / mass,
            max_rpm=self.max_rpm,
            brake_torque=REF.brake_torque * self.brake_scale * (self.side / REF_SIDE) ** 3,
            edge_pull=REF.edge_pull * pull,
            face_pull=REF.face_pull * pull,
            floor_pull=self.steel_floor,
        )

    def label(self):
        return f"L{self.side * 100:g}cm-fw{self.flywheel_share:.3g}-rpm{self.max_rpm:.0f}-brake{self.brake_scale:.3g}-mag{self.magnet_scale:.3g}-steel{self.steel_floor:g}"


@dataclass(frozen=True)
class Control:
    """How a module makes a move. The defaults are one attempt at the speed with the highest
    success rate, as in rounds 2 and 2b."""

    harm_weight: float = 0.0  # tuning maximises success rate - harm_weight * harmful rate
    attempts: int = 1  # most attempts per move; a further attempt follows only a harmless failure
    trials: int = VALIDATE_TRIALS  # scored moves per primitive and neighbour kind
    first_trial: int = 0  # seed index of the first scored move (to score on seeds not used before)


BOUNDS = {"flywheel_share": (0.05, 0.35), "max_rpm": (8000.0, 40000.0), "brake_scale": (0.25, 3.0), "magnet_scale": (0.2, 3.0), "steel_floor": (0.05, 1.0)}
STEEL_START = 0.3  # floor grip when the climb first lays steel


def rng_for(phase, primitive, index, k):
    return np.random.default_rng([PHASES.index(phase), list(PRIMITIVES).index(primitive), index, k])


def speed_grid(cube):
    return np.geomspace(MIN_RPM, cube.max_rpm, SPEEDS) * 2 * math.pi / 60


def tune_rates(cube, primitive, anchored, jitter, speeds, first_index):
    """(success rate, harmful rate) per speed, each (len(speeds),), over TUNE_TRIALS tuning trials."""
    outcomes = [[trial(cube, primitive, omega, rng_for("tune", primitive, first_index + i, k), jitter, anchored)["outcome"] for k in range(TUNE_TRIALS)] for i, omega in enumerate(speeds)]
    success = np.array([np.mean([o == "success" for o in row]) for row in outcomes])
    harm = np.array([np.mean([o in ("off", "knock") for o in row]) for row in outcomes])
    return success, harm


def smooth(values):
    return np.convolve(np.pad(values, 1, mode="edge"), np.ones(3) / 3, mode="valid")


def ladder_from(start, step, attempts, cube):
    """Flywheel speeds (rad/s) of a move's attempts: `start`, then `step` times faster each time, up
    to the motor's top speed."""
    return [min(start * step**i, cube.max_omega()) for i in range(attempts)]


def run_ladder(cube, primitive, anchored, jitter, ladder, phase, index, trials):
    """Outcome counts and total attempts of the moves made with `ladder` on `phase` seeds
    `trials` (a range)."""
    counts = dict.fromkeys(OUTCOMES, 0)
    attempts = 0
    for k in trials:
        result = trial(cube, primitive, ladder[0], rng_for(phase, primitive, index, k), jitter, anchored, retries=ladder[1:])
        counts[result["outcome"]] += 1
        attempts += result["attempts"]
    return counts, attempts


def pick_ladder(cube, primitive, anchored, jitter, candidates, harm_weight, idle):
    """The candidate ladder with the highest value (successes minus harm_weight times harmful
    failures) on shared tuning seeds, by successive halving over LADDER_STAGES: each stage keeps
    the best candidates so far and runs them on more trials. Ties go to the earlier candidate.
    When even the best one harms more than it helps, the move is not worth making: `idle`."""
    values = np.zeros(len(candidates))
    alive = list(range(len(candidates)))
    count = 0
    for trials, keep in LADDER_STAGES:
        alive = sorted(sorted(alive, key=lambda i: -values[i])[:keep])  # stable: ties keep the earlier candidate
        for i in alive:
            counts, _ = run_ladder(cube, primitive, anchored, jitter, candidates[i], "tune", SPEEDS, range(count, count + trials))
            values[i] += counts["success"] - harm_weight * (counts["off"] + counts["knock"])
        count += trials
    best = max(alive, key=lambda i: values[i])
    return candidates[best] if values[best] > 0 else idle


def score_primitive(args):
    """Tune the speed ladder for one primitive, then score it on `phase` seeds. Returns a dict of
    rates per move (after all its attempts)."""
    cube, primitive, anchored, phase, jitter, control = args
    grid = speed_grid(cube)
    rates, harms = tune_rates(cube, primitive, anchored, jitter, grid, 0)
    coarse = int(smooth(rates).argmax())
    ladder = [grid[0]]  # a move that never works is not attempted: one pulse too slow to move anything
    if rates.any():
        # refine between the best grid point's neighbours, where narrow windows live
        fine = np.geomspace(grid[max(coarse - 1, 0)], grid[min(coarse + 1, SPEEDS - 1)], REFINE)
        candidates = [ladder_from(start, step, control.attempts, cube) for start in fine for step in (STEPS if control.attempts > 1 else STEPS[:1])]
        ladder = pick_ladder(cube, primitive, anchored, jitter, candidates, control.harm_weight, ladder)
    counts, attempts = run_ladder(cube, primitive, anchored, jitter, ladder, phase, 0, range(control.first_trial, control.first_trial + control.trials))
    return {
        "primitive": primitive,
        "anchored": anchored,
        "rpm": round(float(ladder[0]) * 60 / (2 * math.pi)),
        "last_rpm": round(float(ladder[-1]) * 60 / (2 * math.pi)),
        "attempts": attempts / control.trials,
        "trials": control.trials,
        "tune_curve": [round(float(r), 2) for r in rates],
        "tune_harm": [round(float(r), 2) for r in harms],
        **{o: counts[o] / control.trials for o in OUTCOMES},
    }


def jobs(cube, phase="validate", jitter=Jitter(), control=Control()):
    out = []
    for p, spec in PRIMITIVES.items():
        for anchored in (True, False) if spec["substrate"] else (False,):
            out.append((cube, p, anchored, phase, jitter, control))
    return out


def combine(parts):
    """Per-primitive success and harm (off or knock), averaged over anchored and free neighbours,
    and the move score weighted by MOVE_MIX."""
    rates = {}
    for p in PRIMITIVES:
        rows = [r for r in parts if r["primitive"] == p]
        rates[p] = {"success": float(np.mean([r["success"] for r in rows])), "harm": float(np.mean([r["off"] + r["knock"] for r in rows]))}
    score = sum(w * rates[m]["success"] for m, w in MOVE_MIX.items())
    harm = sum(w * rates[m]["harm"] for m, w in MOVE_MIX.items())
    return round(score, 4), round(harm, 4), {p: {k: round(v, 3) for k, v in r.items()} for p, r in rates.items()}


def evaluate_many(cubes, phase="validate", workers=5, control=Control()):
    """[(score, harm, rates, parts)] for each cube, all primitives run in one pool."""
    tasks = [(i, job) for i, c in enumerate(cubes) for job in jobs(c, phase, control=control)]
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(score_primitive, [job for _, job in tasks], chunksize=1))
    out = []
    for i in range(len(cubes)):
        parts = [r for (j, _), r in zip(tasks, results) if j == i]
        out.append((*combine(parts), parts))
    return out


def neighbours(build, step):
    """Designs one step away: each continuous knob up and down by a factor `step` (clipped to
    BOUNDS; the floor's grip only once there is steel), and the steel floor laid or removed."""
    out = []
    for field, (lo, hi) in BOUNDS.items():
        if getattr(build, field) == 0:
            continue
        for factor in (step, 1 / step):
            value = float(np.clip(getattr(build, field) * factor, lo, hi))
            if not math.isclose(value, getattr(build, field)):
                out.append(replace(build, **{field: value}))
    out.append(replace(build, steel_floor=0.0 if build.steel_floor > 0 else STEEL_START))
    return out


def record(path, row):
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")


def row_for(build, result, phase, control=Control()):
    score, harm, rates, parts = result
    return {"design": build.label(), "build": asdict(build), "cube": asdict(build.cube()), "control": asdict(control), "phase": phase, "score": score, "harm": harm, "rates": rates, "parts": parts}


def climb(start, out, rounds=12, step=1.6, min_step=1.08, margin=0.01, workers=5, control=Control()):
    """Greedy hill-climb from `start` on move score - control.harm_weight * harm: move to the best
    neighbour when it beats the current design by `margin`, else shrink the step. Every evaluation
    is appended to `out`."""
    current = start
    (best,) = evaluate_many([current.cube()], workers=workers, control=control)
    record(out, {**row_for(current, best, "validate", control), "round": 0, "accepted": True})
    print(f"start {current.label()} score {best[0]:.3f} harm {best[1]:.3f}", flush=True)
    for r in range(1, rounds + 1):
        if step < min_step:
            break
        candidates = neighbours(current, step)
        results = evaluate_many([c.cube() for c in candidates], workers=workers, control=control)
        top = int(np.argmax([res[0] - control.harm_weight * res[1] for res in results]))
        accept = results[top][0] - control.harm_weight * results[top][1] > best[0] - control.harm_weight * best[1] + margin
        for i, (c, res) in enumerate(zip(candidates, results)):
            record(out, {**row_for(c, res, "validate", control), "round": r, "accepted": accept and i == top})
        print(f"round {r} step {step:.3f}: best neighbour {candidates[top].label()} {results[top][0]:.3f} harm {results[top][1]:.3f} vs {best[0]:.3f} harm {best[1]:.3f} -> {'move' if accept else 'shrink'}", flush=True)
        if accept:
            current, best = candidates[top], results[top]
        else:
            step = math.sqrt(step)
    (test,) = evaluate_many([current.cube()], phase="test", workers=workers, control=control)
    record(out, {**row_for(current, test, "test", control), "round": "final"})
    print(f"final {current.label()} validate {best[0]:.3f} test {test[0]:.3f} harm {test[1]:.3f}", flush=True)
    print(json.dumps(test[2], indent=1), flush=True)
    return current


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["eval", "climb"])
    parser.add_argument("--out")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--phase", default="validate", choices=PHASES[1:])
    for field, value in asdict(Build()).items():
        parser.add_argument(f"--{field.replace('_', '-')}", type=float, default=value)
    parser.add_argument("--harm-weight", type=float, default=Control().harm_weight)
    parser.add_argument("--attempts", type=int, default=Control().attempts)
    parser.add_argument("--trials", type=int, default=Control().trials)
    parser.add_argument("--first-trial", type=int, default=Control().first_trial)
    args = parser.parse_args()
    build = Build(**{f: getattr(args, f) for f in asdict(Build())})
    control = Control(
        harm_weight=args.harm_weight,
        attempts=args.attempts,
        trials=args.trials,
        first_trial=args.first_trial,
    )
    start = time.time()
    if args.command == "eval":
        (result,) = evaluate_many([build.cube()], phase=args.phase, workers=args.workers, control=control)
        record(args.out, row_for(build, result, args.phase, control))
        print(f"{build.label()} score {result[0]:.3f} harm {result[1]:.3f}")
        for part in result[3]:
            print(f"  {part['primitive']:13s} {'anchored' if part['anchored'] else 'free    '} rpm {part['rpm']:6d} to {part['last_rpm']:6d}  attempts {part['attempts']:.2f}  success {part['success']:.3f}  stay {part['stay']:.3f}  off {part['off']:.3f}  knock {part['knock']:.3f}")
    else:
        climb(build, args.out, rounds=args.rounds, workers=args.workers, control=control)
    print(f"({time.time() - start:.0f} s)")


if __name__ == "__main__":
    main()
