"""Machines of the chosen SMA cube (hardware/src/beebot/hardware/sma_cube.py) with its latching bonds: the
arm, walker, crane and crank of hardware/scripts/small_joint_machines.py (not edited; its `setup` and
`GearedRig` are swapped at import), plus a new machine layout B could not hold: a five-cube reach
boom that pitches up and slews.

What changes against the joint_gear14 run (hardware/runs/module_phys/joint_gear14/, magnet bonds, 4.8 g cube):
1. Cube mass from Design.mass (gear route, LiPo, legs and bolts), not the envelope fit.
1. Bonds: a plain bond between cubes with parallel x axes is latched (edge bolts: Design.bonds()
   'plain'); a plain bond between cubes with crossed x axes cannot latch (the bolts' hooks are on
   x-parallel edges only) and is magnetic only ('mag_plain'); every disc bond is latched by the
   bayonet ('disc'). The Monte Carlo 'bond' scale (0.75-1.10) multiplies latch capacities too.
1. Joint: geared_joint.nominal_design(L) through JOINT_ROUTE=gear, as joint_gear14.

    .venv/bin/python hardware/scripts/module_phys/sma_cube_machines.py run arm
    .venv/bin/python hardware/scripts/module_phys/sma_cube_machines.py mc --n 30 --workers 2
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "hardware" / "runs" / "module_phys" / "sma_cube" / "machines"
os.environ["JOINT_ROUTE"] = "gear"
os.environ["SJ_OUT"] = str(OUT)
sys.path.insert(0, str(ROOT / "hardware" / "scripts"))
sys.path.insert(0, str(ROOT / "hardware" / "scripts" / "module_phys"))

import numpy as np  # noqa: E402

import small_joint_machines as sjm  # noqa: E402
from beebot.hardware import geared_joint as gj  # noqa: E402
from sma_cube_sim import chosen  # noqa: E402

D = chosen()
CAPS = D.bonds()
_ORIG_SETUP = sjm.setup


def setup(L, mass=None, **over):
    return _ORIG_SETUP(L, mass=D.mass, **over)


class LatchRig(gj.GearedRig):
    """GearedRig whose bond capacities follow the latch rules above."""

    def _bond_setup(self):
        super()._bond_setup()
        rots, _ = gj.poses(self.cubes, self.spec.pitch)
        m = self.model
        kinds = []
        for i, c in enumerate(self.cubes):  # same order as GearedRig._bond_setup
            if c.parent is None:
                continue
            anchor = gj.part_of(c.mate)
            if m.body_weldid[m.body(f"c{i}{anchor}").id] == 0:
                continue
            disc = anchor != "b" or gj.part_of(c.face) != "b"
            parallel = abs(float(rots[i][:, 0] @ rots[c.parent][:, 0])) > 0.99
            kinds.append("disc" if disc else ("plain" if parallel else "mag_plain"))
        self.latch_kind = kinds
        self.cap_t = np.array([CAPS[k]["tension_N"] for k in kinds])
        self.cap_m = np.array([CAPS[k]["peel_Nm"] for k in kinds])
        self.cap_tw = np.array([CAPS[k]["twist_Nm"] for k in kinds])
        self.cap_t_al = self.cap_t.copy()
        self.cap_m_al = self.cap_m.copy()

    compliance: dict = {}

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        for name, stiffness in LatchRig.compliance.items():
            self.set_compliance(name, stiffness)

    def summary(self):
        s = super().summary()
        s["bond_kinds"] = {k: self.latch_kind.count(k) for k in set(self.latch_kind)}
        s["reoriented_cubes"] = sorted(set(REORIENTED))
        return s


REORIENTED = []
_ORIG_LATTICE = gj.lattice


def latch_lattice(cells, pitch):
    """Turn passive cubes so their plain bonds latch: a cube hanging on a parent's plain face with a
    crossed x axis is turned to the parent's axis when none of its own discs carries a child (so no
    joint the machine uses moves). Layout B oriented these cubes for its magnet bonds, which do not
    care about x."""
    REORIENTED.clear()
    cells = [dict(c) for c in cells]
    axis = [np.array(gj.unit(c["axis"])) for c in cells]
    for i, c in enumerate(cells):
        p = c.get("parent")
        if p is None:
            continue
        step = np.subtract(c["at"], cells[p]["at"])
        if abs(float(axis[p] @ step)) > 0.5 or abs(float(axis[i] @ axis[p])) > 0.99:
            continue  # a disc bond, or already parallel
        disc_kids = [k for k, ck in enumerate(cells) if ck.get("parent") == i and abs(float(axis[i] @ np.subtract(ck["at"], c["at"]))) > 0.5]
        if disc_kids:
            # the child's discs carry a joint: turn the parent instead, so the bond sits on the parent's
            # disc (a bayonet-latched disc bond held by that joint's self-locking worm), if the parent's
            # own bond stays disc-to-plain and no other child of the parent sits on its discs
            pp = cells[p].get("parent")
            new_axis = step / np.linalg.norm(step)
            sibs = [k for k, ck in enumerate(cells) if ck.get("parent") == p and k != i and abs(float(new_axis @ np.subtract(ck["at"], cells[p]["at"]))) > 0.5]
            own_ok = pp is None or abs(float(axis[pp] @ np.subtract(cells[p]["at"], cells[pp]["at"]))) > 0.5 and abs(float(new_axis @ np.subtract(cells[p]["at"], cells[pp]["at"]))) < 0.5
            if pp is not None and own_ok and not sibs and not any(k for k, ck in enumerate(cells) if ck.get("parent") == p and abs(float(axis[p] @ np.subtract(ck["at"], cells[p]["at"]))) > 0.5):
                cells[p]["axis"] = ("+" if new_axis[np.argmax(np.abs(new_axis))] > 0 else "-") + "xyz"[int(np.argmax(np.abs(new_axis)))]
                cells[p].pop("up", None)
                axis[p] = new_axis
                REORIENTED.append(p)
            continue
        c["axis"] = cells[p]["axis"] if abs(float(axis[i] @ axis[p])) < 0.99 else c["axis"]
        c.pop("up", None)
        axis[i] = axis[p]
        REORIENTED.append(i)
    return _ORIG_LATTICE(cells, pitch)


sjm.setup = setup
sjm.GearedRig = LatchRig
sjm.lattice = latch_lattice


def boom(L, over=None, cam=None):
    """New machine: a yaw root on the floor, a tower cube whose disc pitches a boom of five latched
    cubes (x axes parallel, so every boom bond is latched) reaching four cells out. Lift the boom
    30 deg, slew 90 deg, lower it. Layout B's magnet bonds peel at the boom root (moment about
    4 mN m against 3-4 mN m magnetic peel); here the root is a latched disc bond (133 mN m)."""
    over = over or {}
    spec, joint, bonds, bscale = sjm.setup(L, **over)
    cells = [
        {"at": (0, 0, 0), "axis": "+z", "mount": "n", "z0": L / 2},
        {"at": (0, 0, 1), "axis": "+y", "parent": 0},  # tower: its +y disc pitches the boom
        {"at": (0, 1, 1), "axis": "+z", "parent": 1},  # boom root
        {"at": (1, 1, 1), "axis": "+z", "parent": 2},
        {"at": (2, 1, 1), "axis": "+z", "parent": 3},
        {"at": (3, 1, 1), "axis": "+z", "parent": 4},
        {"at": (4, 1, 1), "axis": "+z", "parent": 5},
    ]
    rig = LatchRig(gj.lattice(cells, spec.pitch), spec, joint, bonds, bond_scale=bscale, timestep=sjm.DT)
    watch = cam.bind(rig) if cam else None
    if cam:
        cam.caption(f"New machine: a five-cube latched boom ({L * 1e3:.0f} mm cubes) reaching four cells out, pitched by\nthe tower's disc and slewed by the base. Magnet-only bonds peel at the boom root.")
    rig.run(1.0 * sjm.TIME, watch)
    z0 = rig.position("c6b")[2]
    lift = math.radians(30)
    # which sense of the tower disc raises the boom: from the lattice frames (the controller knows its
    # own geometry), checked by the tip's height after a 2 deg probe
    rig.command(targets={"1p": math.radians(2)}, seconds=0.5 * sjm.TIME)
    sjm.run_to(rig, {"1p": math.radians(2)}, 0.6 * sjm.TIME, watch)
    sign = 1.0 if rig.position("c6b")[2] > z0 else -1.0
    rig.command(targets={"1p": sign * lift}, seconds=3.0 * sjm.TIME)
    sjm.run_to(rig, {"1p": sign * lift}, 3.2 * sjm.TIME, watch)
    raised = rig.position("c6b")[2] - z0
    a0 = math.atan2(*rig.position("c6b")[[1, 0]])
    rig.command(targets={"0n": math.pi / 2}, seconds=3.0 * sjm.TIME)  # slew on the bolted base disc, as the crane
    sjm.run_to(rig, {"0n": math.pi / 2}, 3.2 * sjm.TIME, watch)
    a1 = math.atan2(*rig.position("c6b")[[1, 0]])
    rig.command(targets={"1p": 0.0}, seconds=3.0 * sjm.TIME)
    sjm.run_to(rig, {"1p": 0.0}, 3.2 * sjm.TIME, watch)
    slewed = math.degrees((a1 - a0 + math.pi) % (2 * math.pi) - math.pi)
    want = 4 * spec.pitch * math.sin(lift) * 0.8
    ok = raised >= want and abs(abs(slewed) - 90) <= 10
    measured = {"raised_mm": round(raised * 1e3, 2), "slewed_deg": round(slewed, 1), "sim_s": round(rig.data.time, 1)}
    return sjm.verdict(rig, ok, measured, "boom", L, {"criterion": "boom tip raised >= 0.8 x 4 pitches x sin 30 deg, slewed 90 +/- 10 deg"})


sjm.MACHINES["boom"] = boom


def arm(L, over=None, cam=None):
    """Layout B's arm (sjm.arm) with the flexure finger on the wrist joint for the geared route too:
    sjm.arm adds it for the SMA stepper only, and a stiff worm closing on the block dumps its
    reflected rotor inertia into one tooth (joint_gear14: 5p broke at 38 mN m in every trial). The
    finger is a printed flexure either way (A: FINGER_K at the tip)."""
    pitch = sjm.setup(L)[0].pitch
    LatchRig.compliance = {"5p": sjm.FINGER_K * (1.5 * pitch) ** 2}
    try:
        return sjm.arm(L, over, cam)
    finally:
        LatchRig.compliance = {}


sjm.MACHINES["arm"] = arm


def walker(L, over=None, cam=None):
    """Layout B's walker with a 20 s gait period. The default period comes from the gear's top speed
    (about 7.8 s at 17.84 mm). That makes the teeter at each leg lift dynamic: joint 0.92 of the tooth
    limit nominal, and it fails at 1.5x. At 20 s the peak is 0.36 of the limit and bonds are at 0.08."""
    return sjm.walker(L, over, cam, period=20.0)


sjm.MACHINES["walker"] = walker


def _trial(args):  # lives here so spawned workers import this module (and its patches) first
    return sjm._trial(args)


def monte_carlo(names, n, workers=2, seed=1):
    path = OUT / f"mc_{D.L * 1e3:.2f}mm.jsonl"
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done.setdefault(r["machine"], set()).add(r["trial"])
    jobs = [(name, D.L, k, abs(int(seed * 100003 + k * 7919 + zlib.crc32(name.encode()) % 1000))) for k in range(n) for name in names if k not in done.get(name, set())]
    print(f"{len(jobs)} machine trials on {workers} processes", flush=True)
    with mp.get_context("spawn").Pool(workers) as pool, path.open("a") as fh:
        for row in pool.imap_unordered(_trial, jobs):
            fh.write(json.dumps(row, default=float) + "\n")
            fh.flush()
            print(row["machine"], row["trial"], row.get("ok"), row.get("ok_1.5x"), row.get("peak_joint_over_capacity"), row.get("peak_bond_utilisation"), row.get("wall_s"), flush=True)
    return sjm.summarise(path)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run")
    a.add_argument("machine")
    a.add_argument("--duty", type=float, default=None)
    a = sub.add_parser("mc")
    a.add_argument("--n", type=int, default=30)
    a.add_argument("--workers", type=int, default=2)
    a.add_argument("--machines", default="arm,walker,crane,crank,boom")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cmd == "run":
        t = time.time()
        kw = {"duty": args.duty} if args.machine == "walker" and args.duty else {}
        row = sjm.MACHINES[args.machine](D.L, **kw)
        row["wall_s"] = round(time.time() - t, 1)
        with (OUT / "nominal.jsonl").open("a") as f:
            f.write(json.dumps(row, default=float) + "\n")
        print(json.dumps(row, default=float))
    else:
        print(json.dumps(monte_carlo(args.machines.split(","), args.n, workers=min(args.workers, 6)), default=float, indent=1))


if __name__ == "__main__":
    main()
