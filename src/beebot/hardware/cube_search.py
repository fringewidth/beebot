"""Score a pivoting-cube design (hardware/cube.py) on the moves the voxel policy issues, and
hill-climb designs.

Per primitive, the controller's one knob, the flywheel speed, is tuned the way a real module would
be calibrated: a geometric grid of speeds, TUNE_TRIALS jittered trials each, and the speed whose
success rate (averaged with its grid neighbours, so a lucky lone speed is not picked) is highest.
The primitive is then scored on VALIDATE_TRIALS fresh jittered trials at that speed. Trials use
common random numbers: the same seeds for every design. Primitives with neighbours are scored
twice, once with the neighbours anchored (part of a large lattice) and once free (lone modules),
and the two are averaged.

The design's move score weights each primitive by how often the voxel suite's expert issues that
move to a crawler swarm (MOVE_MIX). A horizontal diagonal is not one pivot but two, so it scores
the square of the matching straight move.

    uv run python -m beebot.hardware.cube_search eval --side 0.03
    uv run python -m beebot.hardware.cube_search climb --side 0.03 --out runs/cube/climb-3cm.jsonl
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
VALIDATE_TRIALS = 40
OUTCOMES = ("success", "stay", "off", "knock")
PHASES = ("tune", "validate", "test")

# Share of the moves the voxel suite's expert issues to a 3 cm permanent-magnet crawler swarm
# (L3cm-magnet-c0.12-a0.25-f0, 192 training tasks, seed 0), by pivot primitive. Diagonals are
# horizontal moves along two axes at once.
MOVE_MIX = {
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


BOUNDS = {"flywheel_share": (0.05, 0.35), "max_rpm": (8000.0, 40000.0), "brake_scale": (0.25, 3.0), "magnet_scale": (0.2, 3.0), "steel_floor": (0.05, 1.0)}
STEEL_START = 0.3  # floor grip when the climb first lays steel


def rng_for(phase, primitive, index, k):
    return np.random.default_rng([PHASES.index(phase), list(PRIMITIVES).index(primitive), index, k])


def speed_grid(cube):
    return np.geomspace(MIN_RPM, cube.max_rpm, SPEEDS) * 2 * math.pi / 60


def score_primitive(args):
    """Tune the speed for one primitive, then score it on `phase` seeds. Returns a dict of rates."""
    cube, primitive, anchored, phase, jitter = args
    grid = speed_grid(cube)
    rates = np.array([np.mean([trial(cube, primitive, omega, rng_for("tune", primitive, i, k), jitter, anchored)["outcome"] == "success" for k in range(TUNE_TRIALS)]) for i, omega in enumerate(grid)])
    smooth = np.convolve(np.pad(rates, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    coarse = int(smooth.argmax())
    # refine between the best grid point's neighbours, where narrow windows live
    fine = np.geomspace(grid[max(coarse - 1, 0)], grid[min(coarse + 1, SPEEDS - 1)], REFINE)
    fine_rates = np.array([np.mean([trial(cube, primitive, omega, rng_for("tune", primitive, SPEEDS + i, k), jitter, anchored)["outcome"] == "success" for k in range(TUNE_TRIALS)]) for i, omega in enumerate(fine)])
    fine_smooth = np.convolve(np.pad(fine_rates, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    best = int(fine_smooth.argmax())
    omega = fine[best]
    counts = dict.fromkeys(OUTCOMES, 0)
    for k in range(VALIDATE_TRIALS):
        counts[trial(cube, primitive, omega, rng_for(phase, primitive, 0, k), jitter, anchored)["outcome"]] += 1
    return {
        "primitive": primitive,
        "anchored": anchored,
        "rpm": round(float(omega) * 60 / (2 * math.pi)),
        "tune_curve": [round(float(r), 2) for r in rates],
        **{o: counts[o] / VALIDATE_TRIALS for o in OUTCOMES},
    }


def jobs(cube, phase="validate", jitter=Jitter()):
    out = []
    for p, spec in PRIMITIVES.items():
        for anchored in (True, False) if spec["substrate"] else (False,):
            out.append((cube, p, anchored, phase, jitter))
    return out


def combine(parts):
    """Per-primitive success and harm (off or knock), averaged over anchored and free neighbours,
    and the move score weighted by MOVE_MIX."""
    rates = {}
    for p in PRIMITIVES:
        rows = [r for r in parts if r["primitive"] == p]
        rates[p] = {"success": float(np.mean([r["success"] for r in rows])), "harm": float(np.mean([r["off"] + r["knock"] for r in rows]))}
    floor_weight = MOVE_MIX.get("floor_roll", 0) + MOVE_MIX.get("floor_peel", 0)
    floor = {k: (MOVE_MIX.get("floor_roll", 0) * rates["floor_roll"][k] + MOVE_MIX.get("floor_peel", 0) * rates["floor_peel"][k]) / max(floor_weight, 1e-9) for k in ("success", "harm")}
    moves = {p: rates[p] for p in PRIMITIVES}
    moves["diagonal_floor"] = {"success": floor["success"] ** 2, "harm": 1 - (1 - floor["harm"]) ** 2}
    moves["diagonal_up"] = {"success": rates["traverse"]["success"] ** 2, "harm": 1 - (1 - rates["traverse"]["harm"]) ** 2}
    total = sum(MOVE_MIX.values())
    score = sum(w * moves[m]["success"] for m, w in MOVE_MIX.items()) / total
    harm = sum(w * moves[m]["harm"] for m, w in MOVE_MIX.items()) / total
    return round(score, 4), round(harm, 4), {p: {k: round(v, 3) for k, v in r.items()} for p, r in rates.items()}


def evaluate_many(cubes, phase="validate", workers=5):
    """[(score, harm, rates, parts)] for each cube, all primitives run in one pool."""
    tasks = [(i, job) for i, c in enumerate(cubes) for job in jobs(c, phase)]
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


def row_for(build, result, phase):
    score, harm, rates, parts = result
    return {"design": build.label(), "build": asdict(build), "cube": asdict(build.cube()), "phase": phase, "score": score, "harm": harm, "rates": rates, "parts": parts}


def climb(start, out, rounds=12, step=1.6, min_step=1.08, margin=0.01, workers=5):
    """Greedy hill-climb from `start`: move to the best neighbour when it beats the current design
    by `margin`, else shrink the step. Every evaluation is appended to `out`."""
    current = start
    (best,) = evaluate_many([current.cube()], workers=workers)
    record(out, {**row_for(current, best, "validate"), "round": 0, "accepted": True})
    print(f"start {current.label()} score {best[0]:.3f} harm {best[1]:.3f}", flush=True)
    for r in range(1, rounds + 1):
        if step < min_step:
            break
        candidates = neighbours(current, step)
        results = evaluate_many([c.cube() for c in candidates], workers=workers)
        top = int(np.argmax([res[0] for res in results]))
        accept = results[top][0] > best[0] + margin
        for i, (c, res) in enumerate(zip(candidates, results)):
            record(out, {**row_for(c, res, "validate"), "round": r, "accepted": accept and i == top})
        print(f"round {r} step {step:.3f}: best neighbour {candidates[top].label()} {results[top][0]:.3f} vs {best[0]:.3f} -> {'move' if accept else 'shrink'}", flush=True)
        if accept:
            current, best = candidates[top], results[top]
        else:
            step = math.sqrt(step)
    (test,) = evaluate_many([current.cube()], phase="test", workers=workers)
    record(out, {**row_for(current, test, "test"), "round": "final"})
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
    args = parser.parse_args()
    build = Build(**{f: getattr(args, f) for f in asdict(Build())})
    start = time.time()
    if args.command == "eval":
        (result,) = evaluate_many([build.cube()], phase=args.phase, workers=args.workers)
        record(args.out, row_for(build, result, args.phase))
        print(f"{build.label()} score {result[0]:.3f} harm {result[1]:.3f}")
        for part in result[3]:
            print(f"  {part['primitive']:13s} {'anchored' if part['anchored'] else 'free    '} rpm {part['rpm']:6d}  success {part['success']:.2f}  stay {part['stay']:.2f}  off {part['off']:.2f}  knock {part['knock']:.2f}")
    else:
        climb(build, args.out, rounds=args.rounds, workers=args.workers)
    print(f"({time.time() - start:.0f} s)")


if __name__ == "__main__":
    main()
