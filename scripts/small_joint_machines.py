"""Machines of geared small cubes (src/beebot/hardware/geared_joint.py) in MuJoCo: the 11-cube arm,
the 14-cube walker, the 5-cube crane and the 7-cube crank and slider of layout B
(scripts/module_demos.py), rebuilt at cube side L with the geared joint, its lock, backlash,
compliance and tooth strength, and bonds judged against the sphere-magnet model.

    .venv/bin/python scripts/small_joint_machines.py run arm --L 12        # one nominal run, prints JSON
    .venv/bin/python scripts/small_joint_machines.py mc --L 12 --n 150     # Monte Carlo, all four machines
    .venv/bin/python scripts/small_joint_machines.py video --L 12          # captioned arm + walker MP4

Writes runs/module_phys/joint/machines_<L>mm.jsonl (nominal), mc_<L>mm.jsonl (one line per trial),
and runs/handoff/media/small-joint.mp4. CPU only; Monte Carlo uses at most 5 processes (nice 10).
Memo: runs/handoff/small-joint.md.

Time is slower than layout B's demos: the joint's reflected rotor inertia (I_rotor N^2, about
0.01 kg m2 at 12 mm) limits its acceleration to about 1 rad/s^2, so every move takes TIME x the
layout B duration and the walker's gait period is 16 s instead of 4 s.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time
import zlib
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

from beebot.hardware.geared_joint import G, BondModel, CubeSpec, GearedRig, JointParams, lattice, nominal_bonds, nominal_design, sma_design, stepper_params
from beebot.hardware.mmsim.opt import wilson_interval

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "module_phys" / "joint"
MEDIA = ROOT / "runs" / "handoff" / "media"
ROUTE = os.environ.get("JOINT_ROUTE", "sma")  # 'sma' (chosen) or 'gear'
TIME = 1.5 if ROUTE == "sma" else 2.5  # slow-down against layout B's move durations
DT = float(os.environ.get("SJ_DT", "1e-4"))  # s, simulation step
DT_WALKER = float(os.environ.get("SJ_DT_WALKER", "2e-4"))  # s: the 4-cycle walker is 250 s of sim time; at 8 mm, 2e-4 s gave peak bond load 1.31 against 1.06 at 1e-4 s (conservative)


def joint_speed(joint):
    """Top joint speed (rad/s): one tooth per step period (stepper) or the no-load speed (gear)."""
    if joint.kind == "stepper":
        return joint.step / joint.step_period
    return joint.omega_max / joint.ratio
REF = 0.050  # m, layout B cube side
FINGER_K = 50.0  # N/m at the finger tip: a flexure finger (A), so a ratchet step squeezes the block instead of stalling on it


def cube_mass(L: float) -> float:
    """Whole-cube mass (kg) from the envelope layout of scripts/module_phys/small_joint.py, fitted
    as m = a L^3 + b L^2 (E). Overridden by the layout's own number when that has been run."""
    path = OUT / "mass_fit.json"
    if path.exists():
        fit = json.loads(path.read_text())
        return fit["a"] * L**3 + fit["b"] * L**2
    return 2400.0 * L**3 + 4.0 * L**2


def setup(L: float, mass: float | None = None, **over):
    """CubeSpec, JointParams and BondModel at side L with Monte Carlo overrides: mu (worm friction),
    eps (split-ring loss per unit ratio), eta_spur, backlash (scale), bond (strength scale),
    floor and body (friction coefficients)."""
    mass = cube_mass(L) if mass is None else mass
    spec = CubeSpec(L=L, mass=mass, floor_friction=over.get("floor", 0.8), body_friction=over.get("body", 0.6), disc_radius_frac=0.45)
    bonds = nominal_bonds(L)
    if ROUTE == "sma":
        joint = stepper_params(sma_design(L), bonds, force_scale=over.get("force", 1.0), cool_scale=over.get("cool", 1.0), eta_pawl=over.get("eta_pawl"), backlash_scale=over.get("backlash", 1.0))
    else:
        design = nominal_design(L)
        if "eps" in over:
            design.train.meshes[-1] = replace(design.train.meshes[-1], eps=over["eps"])
        joint = JointParams.from_design(design, bonds, mu=over.get("mu"), eta_spur=over.get("eta_spur"), backlash_scale=over.get("backlash", 1.0))
    return spec, joint, bonds, over.get("bond", 1.0)


def box(name, pos, size, mass, rgba="0.80 0.30 0.25 1", friction=0.8, free=True, extra=""):
    joint = "<freejoint/>" if free else ""
    p = " ".join(f"{v:.6g}" for v in pos)
    s = " ".join(f"{v:.6g}" for v in size)
    return f'<body name="{name}" pos="{p}">{joint}<geom name="{name}" type="box" size="{s}" mass="{mass:.6g}" rgba="{rgba}" friction="{friction} 0.005 0.0001" {extra}/></body>\n'


def mark(pos, size, rgba="0.2 0.6 0.9 1"):
    return f'<geom type="box" pos="{pos[0]:.6g} {pos[1]:.6g} {size[2] if len(size) > 2 else 0.0001:.6g}" size="{size[0]:.6g} {size[1]:.6g} 0.0001" rgba="{rgba}" contype="0" conaffinity="0"/>\n'


def verdict(rig, criterion_ok, measured, name, L, extra=None):
    s = rig.summary()
    # ok: the task done, no tooth overloaded, no bond opened (Newmark opening model at full strength).
    # ok_1.5x: also every bond holds at strength / 1.5 and every joint stays under its tooth limit / 1.5.
    # peak_bond_utilisation (load over strength, 5 ms average) is the stricter, quasi-static view.
    ok = bool(criterion_ok) and s["joint_broken"] is None and s["bond_opened"] is None
    ok15 = ok and s["bond_opened_at_1.5x"] is None and s["peak_joint_over_capacity"] <= 1 / 1.5
    row = {"machine": name, "L_mm": round(L * 1e3, 2), "ok": ok, "ok_1.5x": ok15, "criterion_ok": bool(criterion_ok), "mass_g": round(rig.spec.mass * 1e3, 3)}
    row.update(s)
    row["measured"] = measured
    if extra:
        row.update(extra)
    return row


# ------------------------------------------------------------------------------------------- arm


def arm(L, over=None, cam=None, counterweight=False):
    """Layout B's swing arm (module_demos.arm) at side L: bolted base yaw, a two-cube tower, a lift
    joint swinging a seven-cube hand, a jaw of one swung and one fixed finger; block a third of a
    cube's mass, sized x L / 50 mm, to be carried a quarter turn of yaw."""
    over = over or {}
    spec, joint, bonds, bscale = setup(L, **over)
    s = L / REF
    cells = [
        {"at": (0, 0, 0), "axis": "+z", "mount": "n", "z0": L / 2 + 0.12 * L},
        {"at": (0, 0, 1), "axis": "+x", "parent": 0},
        {"at": (0, 0, 2), "axis": "+x", "parent": 1},
        {"at": (0, 1, 2), "axis": "+x", "parent": 2},  # 3 lift: its +x disc swings everything below
        {"at": (1, 1, 2), "axis": "+y", "parent": 3},
        {"at": (1, 1, 1), "axis": "+y", "parent": 4},  # 5 wrist: its +y disc swings the finger
        {"at": (1, 2, 1), "axis": "+x", "parent": 5},
        {"at": (1, 2, 0), "axis": "+x", "parent": 6},  # 7 moving finger
        {"at": (2, 1, 1), "axis": "+y", "parent": 5},  # 8 fixed finger, three cubes round; turned so it bonds to the wrist by a plain face, not its disc (layout B used the disc: 1.3x the disc bond's strength at the grip)
        {"at": (2, 1, 0), "axis": "+x", "parent": 8},
        {"at": (2, 2, 0), "axis": "+x", "parent": 9},
    ]
    if counterweight:  # load-reducing: two cubes behind the tower, against the hand's overhang on the base disc
        cells += [{"at": (-1, 0, 2), "axis": "+x", "parent": 2}, {"at": (-1, -1, 2), "axis": "+x", "parent": len(cells)}]
    P = spec.pitch
    w = 0.014 * s
    pad = 2 * P - L / 2
    bx = pad - 0.003 * s - w / 2
    pick = np.array([-2 * P, bx])
    block_mass = spec.mass / 3
    scene = box("block", (pick[0], pick[1], 0.010 * s + 0.0002 * s), (0.015 * s, w / 2, 0.010 * s), block_mass, rgba="0.85 0.25 0.20 1", friction=0.9)
    scene += mark((bx, 2 * P), (0.012 * s, 0.02 * s))
    rig = GearedRig(lattice(cells, spec.pitch), spec, joint, bonds, scene=scene, bond_scale=bscale, timestep=DT)
    if joint.kind == "stepper":
        rig.set_compliance("5p", FINGER_K * (1.5 * spec.pitch) ** 2)  # tip about 1.5 pitches from the wrist axis
    watch = cam.bind(rig) if cam else None
    mgl = spec.mass * G * L
    lift, open_, yaw = math.radians(24), math.radians(20), math.pi / 2
    squeeze = 0.3 / (0.120 * G * REF) * mgl  # layout B's 0.3 N m grip, in units of m g L
    i_grip = squeeze / (joint.ratio * joint.eta_f * joint.k_t)
    log = {}

    def go(targets, seconds, caption=None, extra=0.6, currents=None):
        if caption and cam:
            cam.caption(caption)
        rig.command(targets=targets, seconds=seconds * TIME, currents=currents)
        if currents:  # a grip stalls on the block by design
            rig.run(seconds * TIME + extra, watch)
        else:
            run_to(rig, targets, seconds * TIME + extra, watch)

    if cam:
        cam.caption(f"Eleven {L * 1e3:.2f} mm cubes: a bolted base that turns, a tower, a lift joint that swings the hand, and a jaw.\n" + ("Every joint is a ratchet stepped by nitinol wires, about 4 deg a step; a lock pawl holds it with no power." if joint.kind == "stepper" else "Every joint is the cube's flywheel motor through a worm and a printed split-ring gear."))
    rig.run(0.5, watch)
    go({"3p": lift}, 1.5, "Lift and open the jaw.")
    go({"5p": open_}, 1.0)
    go({"0n": -yaw}, 3.0, "Turn a quarter to the block.")
    go({"3p": 0.0}, 1.5, "Lower around the block.")
    go({"5p": -0.3}, 1.2, "Close at a set wire force: the lock pawl keeps the squeeze with no power." if joint.kind == "stepper" else "Close at a set motor current: the self-locking worm keeps the squeeze after the current stops.", currents={"5p": i_grip})
    log["grip_z"] = rig.position("block")[2]
    go({"3p": lift}, 1.5, "Lift.")
    log["lifted"] = rig.position("block")[2] - log["grip_z"]
    go({"0n": 0.0}, 3.0, "Turn back a quarter, carrying the block.")
    go({"3p": 0.03}, 1.5, "Lower to just above the table and let go.")
    go({"5p": open_}, 1.0, currents={"5p": joint.i_max})
    go({"3p": lift}, 1.2, "Lift clear.")
    end = rig.position("block")
    moved = float(np.linalg.norm(end[:2] - pick))
    miss = float(np.linalg.norm(end[:2] - np.array([bx, 2 * P])))
    upright = rig.data.xmat[rig.model.body("block").id].reshape(3, 3)[2, 2] > 0.95
    ok = log["lifted"] >= 0.01 * s and moved >= 0.15 * s and miss <= 0.02 * s and upright and end[2] < 0.03 * s
    measured = {"lifted_mm": round(log["lifted"] * 1e3, 2), "moved_mm": round(moved * 1e3, 1), "miss_mm": round(miss * 1e3, 2), "upright": bool(upright), "sim_s": round(rig.data.time, 1)}
    row = verdict(rig, ok, measured, "arm", L, {"criterion": "block lifted >= 1 cm, moved >= 15 cm, set within 2 cm of the mark, upright (layout B's, times L / 50 mm)"})
    if cam:
        cam.caption(f"Lifted {measured['lifted_mm']:.1f} mm, carried {moved * 1e3:.0f} mm, set {miss * 1e3:.1f} mm from the mark. Peak joint {row['peak_joint_over_capacity']:.2f} of\nits tooth limit, peak bond load {row['peak_bond_utilisation']:.2f} of its strength (measured, nominal parameters).")
        cam.pause(rig, 2.5)
    return row


# ---------------------------------------------------------------------------------------- walker


def walker(L, over=None, cam=None, period=None, cycles=4, duty=None):
    """Layout B's 14-cube quadruped (module_demos.walker): four legs that turn all the way round on
    the body cubes' discs, wave gait (duty 0.75) on a smoothed Buehler clock. With three feet down
    the body's centre of mass sits near the diagonal between two of them (McGhee and Frank 1968: the
    wave gait's stability margin goes to zero at duty 3/4), so the body teeters for about 0.1 s
    each time a leg lifts; that is where the peak bond loads are. Tried and dropped: duty 0.85
    (worse at 1.5x), a 19-cube tripod hexapod (its five-cube body sags: bond loads 4.7x)."""
    over = over or {}
    spec, joint, bonds, bscale = setup(L, **over)
    s = L / REF
    duty = 0.75 if duty is None else duty
    n_body = 4
    cells = [
        {"at": (0, 0, 0), "axis": "+y", "z0": 1.048 * L + 0.5 * L + 0.01 * L},
        {"at": (-1, 0, 0), "axis": "+y", "parent": 0},
        {"at": (-2, 0, 0), "axis": "+y", "parent": 1},
        {"at": (-3, 0, 0), "axis": "+y", "parent": 2},
        {"at": (0, 1, 0), "axis": "+x", "parent": 0},
        {"at": (0, 1, -1), "axis": "+y", "parent": 4},
        {"at": (0, -1, 0), "axis": "+x", "parent": 0},
        {"at": (0, -1, -1), "axis": "+y", "parent": 6},
        {"at": (-3, 1, 0), "axis": "+x", "parent": 3},
        {"at": (-3, 2, 0), "axis": "+x", "parent": 8},
        {"at": (-3, 2, -1), "axis": "+y", "parent": 9},
        {"at": (-3, -1, 0), "axis": "+x", "parent": 3},
        {"at": (-3, -2, 0), "axis": "+x", "parent": 11},
        {"at": (-3, -2, -1), "axis": "+y", "parent": 12},
    ]
    legs = {"0p": 0.25, "0n": 0.75, "3p": 0.0, "3n": 0.5}
    rig = GearedRig(lattice(cells, spec.pitch), spec, joint, bonds, bond_scale=bscale, timestep=DT_WALKER)
    if period is None:  # the swing (about 320 deg in the swing fraction of a cycle) at 85 % of the joint's top speed
        period = (2 * math.pi - 2 * math.radians(20)) / ((1 - duty) * 0.85 * joint_speed(joint)) * 1.15
    watch = cam.bind(rig) if cam else None
    sweep, edge = math.radians(20), 0.15
    grid = np.linspace(0.0, 1.0, 4001)

    def bump(u):
        rise = np.clip((u - (duty - edge / 2)) / edge, 0, 1)
        fall = np.clip((u - (1 - edge / 2)) / edge, 0, 1)
        rise, fall = rise * rise * (3 - 2 * rise), fall * fall * (3 - 2 * fall)
        early = np.clip((u + edge / 2) / edge, 0, 1)
        early = 1 - early * early * (3 - 2 * early)
        return np.clip(rise - fall + np.where(u < edge / 2, early, 0.0), 0, 1)

    slow = 2 * sweep / duty
    weight = bump(grid)
    fast = (2 * math.pi - slow) / np.trapezoid(weight, grid)
    rate = slow + fast * weight
    angle = np.concatenate([[0.0], np.cumsum((rate[1:] + rate[:-1]) / 2 * np.diff(grid))])
    angle -= np.interp(duty / 2, grid, angle)

    def clock(t, phase):
        u = t / period + phase
        k, u = math.floor(u), u - math.floor(u)
        return 2 * math.pi * k + float(np.interp(u, grid, angle)), float(np.interp(u, grid, rate)) / period

    if cam:
        cam.caption(f"Fourteen {L * 1e3:.2f} mm cubes: a body of four, each leg two or three cubes on a body disc that turns\nall the way round. Wave gait, one leg swinging at a time, {period:.0f} s a cycle.")
    start = {leg: clock(0.0, phase)[0] for leg, phase in legs.items()}
    for leg, a0 in start.items():  # assembled in its starting pose (legs at their phases), then gravity ramps in
        a = rig.index[leg]
        rig.data.qpos[rig.qadr[a]] = a0
        rig.phi[a] = rig.q0[a] = rig.q1[a] = a0
    mujoco.mj_forward(rig.model, rig.data)
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * L / 2
    low = min(float((rig.data.xpos[rig.model.body(f"c{k}b").id] + corners @ rig.data.xmat[rig.model.body(f"c{k}b").id].reshape(3, 3).T)[:, 2].min()) for k in range(len(cells)))
    rig.data.qpos[2] += 0.01 * L - low  # lowest foot corner 0.01 L above the table, whatever the start angles
    mujoco.mj_forward(rig.model, rig.data)
    rig.run(1.0, watch)
    t0 = rig.data.time
    x0 = rig.position("c1b")[0]
    for leg, phase in legs.items():
        rig.follow(leg, lambda t, phase=phase: clock(t - t0, phase))
    low = [1.0]
    track = []

    def look(r):
        if watch:
            watch(r)
        if r.steps % 50 == 0:
            low[0] = min(low[0], float(min(r.position(f"c{k}b")[2] for k in range(n_body))))
            track.append(float(np.max(np.abs(r.data.qpos[r.qadr[[r.index[k] for k in legs]]] - np.array([clock(r.data.time - t0, p)[0] for p in legs.values()])))))

    if cam:
        cam.caption(f"Walking: {cycles} gait cycles.")
    rig.run(cycles * period, look)
    for leg in legs:
        rig.follow(leg, None)
    rig.run(0.5, look)
    walked = rig.position("c1b")[0] - x0
    ok = walked >= 0.15 * s and low[0] > L / 2 + 0.04 * L
    measured = {"walked_mm": round(walked * 1e3, 1), "body_lowest_mm": round(low[0] * 1e3, 2), "max_tracking_error_rad": round(max(track) if track else 0.0, 3), "period_s": period, "sim_s": round(rig.data.time, 1)}
    row = verdict(rig, ok, measured, "walker", L, {"criterion": "body forward >= 15 cm x L / 50 mm in four cycles, body cubes stay > 0.54 L up"})
    if cam:
        cam.caption(f"Walked {walked * 1e3:.1f} mm in {cycles} cycles, body never down. Peak joint {row['peak_joint_over_capacity']:.2f} of its tooth limit,\npeak bond load {row['peak_bond_utilisation']:.2f} of its strength (measured, nominal parameters).")
        cam.pause(rig, 2.5)
    return row


# ----------------------------------------------------------------------------------------- crane


def run_to(rig, targets, planned, watch=None):
    """Run the planned time, then on until every joint is within a step of its target (the
    controller counts steps, so it waits for a slow, hot-ambient joint), at most 4x the plan."""
    rig.run(planned, watch)
    step = getattr(rig.joint, "step", 0.0) or 0.01
    extra = 0.0
    while extra < 3 * planned and any(abs(rig.angle(n) - a) > 0.6 * step for n, a in targets.items()):
        rig.run(0.2, watch)
        extra += 0.2


def crane(L, over=None, cam=None):
    over = over or {}
    spec, joint, bonds, bscale = setup(L, **over)
    s = L / REF
    cells = [
        {"at": (0, 0, 0), "axis": "+z", "mount": "n", "z0": L / 2},
        {"at": (0, 0, 1), "axis": "+y", "parent": 0},
        {"at": (0, 1, 1), "axis": "+z", "parent": 1},
        {"at": (1, 1, 1), "axis": "+y", "parent": 2},
        {"at": (1, 2, 1), "axis": "+z", "parent": 3},
    ]
    P = spec.pitch
    plate = L / 2 + P + L / 2
    load = 0.1 / 0.120 * spec.mass
    scene = box("load", (P, 2 * P, plate + 0.02 * s + 0.0005 * s), (0.02 * s, 0.02 * s, 0.02 * s), load, rgba="0.30 0.35 0.75 1")
    rig = GearedRig(lattice(cells, spec.pitch), spec, joint, bonds, scene=scene, bond_scale=bscale, timestep=DT)
    rig.command(targets={"1p": math.radians(20), "3p": -math.radians(20)}, seconds=0.8 * TIME)
    rig.run(1.2 * TIME)
    z0 = rig.position("load")[2]
    lift = math.radians(40)
    rig.command(targets={"1p": -lift, "3p": lift}, seconds=2.5 * TIME)
    run_to(rig, {"1p": -lift, "3p": lift}, 2.8 * TIME)
    high = rig.position("load")[2]
    a0 = math.atan2(*rig.position("load")[[1, 0]])
    rig.command(targets={"0n": math.pi / 2}, seconds=2.5 * TIME)
    run_to(rig, {"0n": math.pi / 2}, 2.8 * TIME)
    a1 = math.atan2(*rig.position("load")[[1, 0]])
    rig.command(targets={"1p": math.radians(20), "3p": -math.radians(20)}, seconds=2.5 * TIME)
    rig.run(2.9 * TIME)
    raised = high - z0
    slewed = math.degrees((a1 - a0 + math.pi) % (2 * math.pi) - math.pi)
    on = abs(rig.position("load")[2] - z0) < 0.005 * s and np.linalg.norm(rig.position("load")[:2] - rig.position("c4b")[:2]) < 0.02 * s
    ok = raised >= 0.04 * s and abs(abs(slewed) - 90) <= 10 and on
    measured = {"raised_mm": round(raised * 1e3, 2), "slewed_deg": round(slewed, 1), "load_on": bool(on)}
    return verdict(rig, ok, measured, "crane", L, {"criterion": "load raised >= 4 cm x L / 50 mm, slewed 90 +/- 10 deg, still on the platform"})


# ----------------------------------------------------------------------------------------- crank


def crank_b(L, over=None, cam=None, speed=None, seconds=None):
    """Layout B's crank and slider as it was (module_demos.crank): the whole linkage hangs off the
    frame's disc as a cantilever. Kept for comparison; it overloads that disc bond at small L."""
    over = over or {}
    spec, joint, bonds, bscale = setup(L, **over)
    s = L / REF
    cells = [
        {"at": (0, 0, 2), "axis": "+y", "mount": "b", "z0": L / 2},
        {"at": (0, 1, 2), "axis": "+z", "parent": 0},
        {"at": (1, 1, 2), "axis": "+y", "parent": 1},
        {"at": (1, 2, 2), "axis": "+z", "parent": 2},
        {"at": (2, 2, 2), "axis": "+z", "parent": 3},
        {"at": (3, 2, 2), "axis": "+y", "parent": 4},
        {"at": (3, 3, 2), "axis": "+z", "parent": 5},
    ]
    P = spec.pitch
    zc = L / 2 + 2 * P
    stand = 2 * P
    rail_t = 0.003 * s
    scene = box("stand", (0, 0, stand / 2), (0.3 * L, 0.3 * L, stand / 2), 1, rgba="0.55 0.55 0.58 1", free=False)
    for sz in (1, -1):
        scene += box(f"rail{sz}", (2.5 * P, 3 * P, zc + sz * (L / 2 + 0.0004 * s + rail_t)), (2.4 * P, L / 2, rail_t), 1, rgba="0.55 0.55 0.58 0.9", free=False)
    for px in (0.0, 5.0):
        scene += box(f"rail_post{px:.0f}", (px * P, 3 * P, zc / 2), (0.08 * L, 0.08 * L, zc / 2 + L / 2), 1, rgba="0.55 0.55 0.58 1", free=False)
    rig = GearedRig(lattice(cells, spec.pitch), spec, joint, bonds, scene=scene, bond_scale=bscale, timestep=DT)
    top = joint_speed(joint)
    speed = 0.85 * top if speed is None else speed
    seconds = 3.3 * 2 * math.pi / speed if seconds is None else seconds
    rig.command(free=["2p", "5p"])
    rig.run(0.5)
    xs = []
    angle0 = rig.angle("0p")
    rig.command(speeds={"0p": speed}, ramp=1.5)

    def look(r):
        if r.steps % 20 == 0:
            xs.append(r.position("c6b")[0])

    rig.run(seconds, look)
    rig.command(speeds={"0p": 0.0}, ramp=1.0)
    rig.run(1.5, look)
    turns = (rig.angle("0p") - angle0) / (2 * math.pi)
    stroke = max(xs) - min(xs)
    ok = turns >= 3 and stroke >= 0.09 * s
    measured = {"crank_turns": round(turns, 2), "stroke_mm": round(stroke * 1e3, 2)}
    return verdict(rig, ok, measured, "crank", L, {"criterion": "crank >= 3 turns, piston stroke >= 90 mm x L / 50 mm; crank pin and piston pin are discs with the selector at neutral"})


def crank(L, over=None, cam=None, speed=None, seconds=None):
    """Crank and slider redesigned so the linkage rests on the table instead of hanging off a
    bond: the frame cube is bolted overhead and turns the hub below it about a vertical axis; the
    crank arm swings the crank pin (its lower disc, free) on a circle of one pitch; a four-cube rod
    (pins three pitches apart) lies on the table; the piston rides on the rod end's upper disc
    (free, the piston pin) between two rails on the line through the crank axis, so it strokes
    two pitches, from 2 to 4 pitches out, and never reaches the crank arm's circle (1.68 pitches).
    Assembled with the crank at a quarter turn (rod along the rails). The rod is pushed at its top
    while it slides on its bottom, so on a table with friction above 0.5 it tips over instead of
    sliding (measured: 180 mN edge contacts and bonds opening at 0.8); the crank therefore stands on
    a smooth sheet with smooth rails (friction 0.25 nominal, 0.15-0.35 in Monte Carlo, A).
    The crank pin has axial play so the frame, hub, crank arm and rod do not form a rigid loop."""
    over = dict(over or {})
    sheet = over.pop("sheet", 0.25)
    over["floor"] = sheet
    spec, joint, bonds, bscale = setup(L, **over)
    s = L / REF
    P = spec.pitch
    cells = [
        {"at": (0, 0, 2), "axis": "+z", "mount": "b", "z0": L / 2},  # 0 frame, bolted overhead; its lower disc (0n) turns the hub
        {"at": (0, 0, 1), "axis": "+z", "parent": 0},  # 1 hub
        {"at": (0, 1, 1), "axis": "+z", "parent": 1, "play": "n"},  # 2 crank arm; its lower disc (2n) is the crank pin, with axial play so the rod rests on the table
        {"at": (0, 1, 0), "axis": "+z", "parent": 2},  # 3 rod, on the table
        {"at": (0, 2, 0), "axis": "+z", "parent": 3},  # 4
        {"at": (0, 3, 0), "axis": "+z", "parent": 4},  # 5
        {"at": (0, 4, 0), "axis": "+z", "parent": 5},  # 6 rod end; its upper disc (6p) is the piston pin
        {"at": (0, 4, 1), "axis": "+z", "parent": 6},  # 7 piston
    ]
    zc = L / 2 + P
    rail_t = 0.003 * s
    y0, y1 = 1.75 * P, 4.6 * P
    scene = box("mount", (0, 0, 2 * P + L + 0.2 * L), (0.4 * L, 0.4 * L, 0.2 * L), 1, rgba="0.55 0.55 0.58 1", free=False)
    for sx in (1, -1):
        scene += box(f"rail{sx}", (sx * (L / 2 + 0.0004 * s + rail_t), (y0 + y1) / 2, zc), (rail_t, (y1 - y0) / 2, L / 2), 1, rgba="0.55 0.55 0.58 0.9", friction=sheet, free=False, extra='priority="1"')
    rig = GearedRig(lattice(cells, spec.pitch), spec, joint, bonds, scene=scene, bond_scale=bscale, timestep=DT)
    watch = cam.bind(rig) if cam else None
    top = joint_speed(joint)
    speed = 0.85 * top if speed is None else speed
    seconds = 3.3 * 2 * math.pi / speed if seconds is None else seconds
    if cam:
        cam.caption(f"Eight {L * 1e3:.2f} mm cubes as a crank and slider: a bolted frame turns the hub, the crank arm swings\na four-cube rod lying on the table, the piston slides between rails. Crank pin and piston pin are free discs.")
    rig.command(free=["2n", "6p"])
    rig.run(0.5, watch)
    ys = []
    angle0 = rig.angle("0n")
    rig.command(speeds={"0n": speed}, ramp=1.5)

    def look(r):
        if watch:
            watch(r)
        if r.steps % 20 == 0:
            ys.append(r.position("c7b")[1])

    rig.run(seconds, look)
    rig.command(speeds={"0n": 0.0}, ramp=1.0)
    rig.run(1.5, look)
    turns = abs(rig.angle("0n") - angle0) / (2 * math.pi)
    stroke = max(ys) - min(ys)
    ok = turns >= 3 and stroke >= 0.09 * s
    measured = {"crank_turns": round(turns, 2), "stroke_mm": round(stroke * 1e3, 2), "sim_s": round(rig.data.time, 1)}
    row = verdict(rig, ok, measured, "crank", L, {"criterion": "crank >= 3 turns, piston stroke >= 90 mm x L / 50 mm; crank pin and piston pin are discs with the joint free (lock latched up)"})
    if cam:
        cam.caption(f"{turns:.1f} turns, stroke {stroke * 1e3:.1f} mm. Peak joint {row['peak_joint_over_capacity']:.2f} of its tooth limit (measured, nominal parameters).")
        cam.pause(rig, 2.0)
    return row


MACHINES = {"arm": arm, "walker": walker, "crane": crane, "crank": crank}


# ----------------------------------------------------------------------------------- Monte Carlo

RANGES_SMA = {  # uniform, all varied together in every trial
    "eta_pawl": (0.55, 0.90),  # pawl and lever efficiency (the stepper's 'gear efficiency', E)
    "force": (0.80, 1.10),  # wire pull against 172 MPa nominal: wire diameter tolerance, temperature (E)
    "cool": (0.8, 2.0),  # cooling time against nominal (enclosure, ambient 20-40 C)
    "backlash": (0.5, 2.0),  # times the nominal tooth clearance
    "bond": (0.75, 1.10),  # sphere-bond strength times the derated model
    "floor": (0.5, 1.0),
    "body": (0.4, 0.8),
    "sheet": (0.15, 0.35),  # the crank's base sheet and rail liners (PTFE or glass, A); only the crank reads it
}
RANGES = {  # uniform, all four varied together in every trial
    "mu": (0.15, 0.35),  # worm-wheel friction, POM on POM dry to lightly greased (A); self-locking needs > 0.083
    "eps": (0.015, 0.040),  # split-ring loss per unit ratio (E): efficiency 0.40-0.72 at ratio 34.8
    "eta_spur": (0.80, 0.93),  # Didel spur mesh with pin bearings (Didel claims 0.9, V claim)
    "backlash": (0.5, 2.0),  # times the nominal output backlash
    "bond": (0.75, 1.10),  # sphere-bond strength times the derated model (published/model 0.64-0.80 for other magnets)
    "floor": (0.5, 1.0),  # floor friction
    "body": (0.4, 0.8),  # cube-cube and cube-object friction
}


def _trial(args):
    name, L, k, seed = args
    rng = np.random.default_rng(seed)
    over = {key: float(rng.uniform(*span)) for key, span in (RANGES_SMA if ROUTE == "sma" else RANGES).items()}
    t0 = time.time()
    try:
        row = MACHINES[name](L, over)
    except Exception as exc:  # an unstable run is a failed trial, recorded
        row = {"machine": name, "L_mm": round(L * 1e3, 2), "ok": False, "error": repr(exc)}
    row["trial"] = k
    row["params"] = {key: round(v, 4) for key, v in over.items()}
    row["wall_s"] = round(time.time() - t0, 1)
    return row


def monte_carlo(L, names, n, workers=5, seed=1):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"mc_{L * 1e3:.2f}mm.jsonl"
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done.setdefault(r["machine"], set()).add(r["trial"])
    order = sorted(names, key=lambda x: ("walker", "crank", "arm", "crane").index(x) if x in ("walker", "crank", "arm", "crane") else 9)  # longest first
    jobs = [(name, L, k, seed * 100003 + k * 7919 + zlib.crc32(name.encode()) % 1000) for k in range(n) for name in order if k not in done.get(name, set())]  # crc32: seeds repeat across runs
    jobs = [(a, b, c, abs(int(d))) for a, b, c, d in jobs]
    print(f"{len(jobs)} trials to run at L = {L * 1e3:.1f} mm on {workers} processes", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=os.nice, initargs=(10,)) as pool, path.open("a") as fh:
        for row in pool.imap_unordered(_trial, jobs):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(row["machine"], row["trial"], row.get("ok"), row.get("ok_1.5x"), row.get("bond_opening_at_1.5x"), row.get("peak_joint_over_capacity"), row.get("peak_bond_utilisation"), row.get("wall_s"), flush=True)
    return summarise(path)


def summarise(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    out = {}
    for name in MACHINES:
        mine = [r for r in rows if r["machine"] == name]
        if not mine:
            continue
        k = sum(r.get("ok", False) for r in mine)
        lo, hi = wilson_interval(k, len(mine))
        k15 = sum(r.get("ok_1.5x", False) for r in mine)
        lo15, hi15 = wilson_interval(k15, len(mine))
        op = [min(r["bond_opening_at_1.5x"], 99.0) for r in mine if "bond_opening_at_1.5x" in r]  # past 1 the bond has failed; the integrator keeps going
        tj = [r["peak_joint_over_capacity"] for r in mine if "peak_joint_over_capacity" in r]
        tb = [r["peak_bond_utilisation"] for r in mine if "peak_bond_utilisation" in r]
        causes = {"criterion": sum(1 for r in mine if not r.get("criterion_ok", False)), "joint_broken": sum(1 for r in mine if r.get("joint_broken")), "bond_opened": sum(1 for r in mine if r.get("bond_opened")), "bond_opened_at_1.5x": sum(1 for r in mine if r.get("bond_opened_at_1.5x")), "joint_over_1/1.5": sum(1 for r in mine if r.get("peak_joint_over_capacity", 0) > 1 / 1.5), "error": sum(1 for r in mine if "error" in r)}
        out[name] = {
            "n": len(mine), "success": k, "rate": round(k / len(mine), 3), "wilson95": [round(lo, 3), round(hi, 3)],
            "success_1.5x": k15, "rate_1.5x": round(k15 / len(mine), 3), "wilson95_1.5x": [round(lo15, 3), round(hi15, 3)],
            "bond_opening_at_1.5x_p95": round(float(np.percentile(op, 95)), 3) if op else None,
            "joint_peak_over_capacity_median": round(float(np.median(tj)), 3) if tj else None,
            "joint_peak_over_capacity_p95": round(float(np.percentile(tj, 95)), 3) if tj else None,
            "joint_peak_over_capacity_max": round(float(np.max(tj)), 3) if tj else None,
            "bond_utilisation_median": round(float(np.median(tb)), 3) if tb else None,
            "bond_utilisation_p95": round(float(np.percentile(tb, 95)), 3) if tb else None,
            "bond_utilisation_max": round(float(np.max(tb)), 3) if tb else None,
            "failures_by_cause": causes,
        }
    return out


# ----------------------------------------------------------------------------------------- video


class Cam:
    """Wraps module_demos' Reel and Camera for a GearedRig, framing scaled to L."""

    def __init__(self, reel, lookat, distance, azimuth, elevation, follow=None, slow=1.0):
        self.reel, self.args = reel, (lookat, distance, azimuth, elevation, follow, slow)
        self.camera = None

    def bind(self, rig):
        sys.path.insert(0, str(ROOT / "scripts"))
        from module_demos import Camera

        lookat, distance, azimuth, elevation, follow, slow = self.args
        self.camera = Camera(rig, self.reel, lookat, distance, azimuth, elevation, follow=follow, slow=slow)
        return self.camera

    def caption(self, text):
        self.reel.caption = text

    def pause(self, rig, seconds):
        self.camera.pause(rig, seconds)


def video(L):
    sys.path.insert(0, str(ROOT / "scripts"))
    from module_demos import Reel

    MEDIA.mkdir(parents=True, exist_ok=True)
    s = L / REF
    parts = []
    for name, fn, cam_args, kw in (
        ("arm", arm, ([-0.03 * s, 0.06 * s, 0.05 * s], 0.65 * s, 340, -22, None, 1.0 / TIME), {}),
        ("walker", walker, ([-0.03 * s, 0.0, 0.05 * s], 0.80 * s, 140, -20, "c1b", 1.0 / 8.0), {}),
        ("crank", crank, ([0.0, 0.02 * s * 2, 0.02 * s], 0.75 * s, 300, -35, None, 1.0 / 4.0), {}),
    ):
        path = OUT / f"{name}_{L * 1e3:.2f}mm.mp4"
        title = {"arm": "Arm: pick and place", "walker": "Walker: four rotating legs", "crank": "Crank and slider"}[name]
        reel = Reel(path, f"{title}, {L * 1e3:.2f} mm cubes with SMA ratchet joints" if ROUTE == "sma" else f"{title}, {L * 1e3:.0f} mm geared cubes")
        cam = Cam(reel, *cam_args)
        row = fn(L, {}, cam, **kw)
        reel.close()
        parts.append(path)
        print(json.dumps(row))
    listing = OUT / "video_list.txt"
    listing.write_text("".join(f"file '{p}'\n" for p in parts))
    out = MEDIA / "small-joint.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(out)], check=True)
    print("wrote", out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["run", "mc", "video", "summary"])
    parser.add_argument("names", nargs="*", default=list(MACHINES))
    parser.add_argument("--L", type=float, default=12.0, help="cube side, mm")
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    L = args.L * 1e-3
    OUT.mkdir(parents=True, exist_ok=True)
    if args.what == "run":
        path = OUT / f"machines_{args.L:.2f}mm.jsonl"
        rows = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
        for name in args.names:
            t0 = time.time()
            row = MACHINES[name](L)
            row["wall_s"] = round(time.time() - t0, 1)
            print(json.dumps(row), flush=True)
            rows = [r for r in rows if r["machine"] != name] + [row]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    elif args.what == "mc":
        print(json.dumps(monte_carlo(L, args.names, args.n, args.workers), indent=1))
    elif args.what == "summary":
        print(json.dumps(summarise(OUT / f"mc_{args.L:.2f}mm.jsonl"), indent=1))
    else:
        video(L)


if __name__ == "__main__":
    main()
