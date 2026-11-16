"""Behaviours, Monte Carlo, outcome model, machines, assembly, optimisation and media for the SMA cube
(src/beebot/hardware/sma_cube.py). Memo: runs/handoff/sma-cube.md.

    .venv/bin/python scripts/module_phys/sma_cube_sim.py run tip_book           # one nominal trial
    .venv/bin/python scripts/module_phys/sma_cube_sim.py mc --moves all --n 200   # Monte Carlo
    .venv/bin/python scripts/module_phys/sma_cube_sim.py confirm                  # 2,000 trials for moves above 0.99
    .venv/bin/python scripts/module_phys/sma_cube_sim.py machines --n 30          # arm, walker, crane, crank, lift
    .venv/bin/python scripts/module_phys/sma_cube_sim.py assembly --seeds 10
    .venv/bin/python scripts/module_phys/sma_cube_sim.py opt                      # NSGA-II
    .venv/bin/python scripts/module_phys/sma_cube_sim.py outcomes                 # outcomes.json
    .venv/bin/python scripts/module_phys/sma_cube_sim.py media

CPU only, at most 6 processes. Results append to runs/module_phys/sma_cube/results.jsonl.
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
from dataclasses import asdict, replace
from pathlib import Path

import mujoco
import numpy as np

from beebot.hardware.sma_cube import (CARRIERS, LATTICE_ROTATIONS, LEG_PARTNER, LEGS, Design, Params, World, bfs_lone, key, leg_world, moves, rot90, snap_rotation)
from beebot.hardware.mmsim.opt import wilson_interval

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "module_phys" / "sma_cube"
MEDIA = ROOT / "runs" / "handoff" / "media"
WORKERS = 6
Zv = np.array([0, 0, 1])


def chosen() -> Design:
    path = OUT / "chosen.json"
    if path.exists():
        return Design(**json.loads(path.read_text())["design"])
    return Design(L=10.5e-3, storage="cera3")


# ------------------------------------------------------------------------------------- helpers


def orient(**faces) -> np.ndarray:
    """Rotation (body -> world) from two face assignments, e.g. orient(y='-Z', z='+X')."""
    W = {"X": np.array([1, 0, 0]), "Y": np.array([0, 1, 0]), "Z": np.array([0, 0, 1])}
    B = {"x": 0, "y": 1, "z": 2}
    R = np.zeros((3, 3))
    items = list(faces.items())
    for b, w in items:
        s = -1 if w[0] == "-" else 1
        R[:, B[b]] = s * W[w[1]]
    missing = [k for k in "xyz" if k not in faces][0]
    i = B[missing]
    j, k = (i + 1) % 3, (i + 2) % 3
    R[:, i] = np.cross(R[:, j], R[:, k])
    return R


def place(d: Design, cells, rots, z_extra=0.0):
    """World cube list from lattice cells (pitch L; L + tread along X, where x-parallel cubes meet disc
    to disc; floor at z = 0) and rotations; disc-down cubes sit on their tread."""
    out = []
    for c, R in zip(cells, rots):
        c = np.asarray(c, float)
        lift = d.tread if abs(R[2, 0]) > 0.5 and c[2] == 0 else 0.0
        out.append((np.array([c[0] * (d.L + d.tread), c[1] * d.L, c[2] * d.L + d.h + lift + 1e-5 + z_extra]), np.asarray(R, float)))
    return out


def sample_params(rng, seed) -> Params:
    """Monte Carlo variables, all uniform (the brief's ranges)."""
    return Params(
        derate=rng.uniform(0.70, 0.90), moment_spread=0.10, k_insert=None, tip_k=None,
        floor_mu=rng.uniform(0.2, 0.8), cube_mu=rng.uniform(0.2, 0.6), tread_mu=rng.uniform(0.5, 1.0),
        damp=rng.uniform(0.3, 1.0), pocket=rng.uniform(0.03e-3, 0.05e-3), size=rng.uniform(-0.05e-3, 0.05e-3),
        leg_len=rng.uniform(-0.05e-3, 0.05e-3), hinge=rng.uniform(0.0, 0.05e-3), wire=rng.uniform(0.8, 1.2),
        cool=rng.uniform(0.8, 2.0), eta=rng.uniform(0.55, 0.90), seed=seed)


def lattice_state(w: World, i):
    """(cell, rotation, position error m, angle error rad) of cube i."""
    pos, R = w.pose(i)
    d = w.d
    lift = d.tread if abs(R[2, 0]) > 0.5 else 0.0
    cell = np.round(np.array([pos[0] / d.L, pos[1] / d.L, (pos[2] - d.h - lift) / d.L])).astype(int)
    ideal = np.array([cell[0] * d.L, cell[1] * d.L, cell[2] * d.L + d.h + lift])
    Rs, ang = snap_rotation(R)
    return tuple(int(v) for v in cell), Rs, float(np.linalg.norm(pos - ideal)), float(ang)


def snapshot(w: World):
    return [w.pose(i) for i in range(w.n)]


def harm(w: World, before, movers, tol_pos=None, tol_ang=math.radians(10)):
    """Neighbours moved more than 0.1 L or turned more than 10 deg (cube.DISTURBED convention)."""
    tol_pos = 0.1 * w.d.L if tol_pos is None else tol_pos
    worst = 0.0
    hit = False
    for i in range(w.n):
        if i in movers:
            continue
        p0, R0 = before[i]
        p1, R1 = w.pose(i)
        dp = float(np.linalg.norm(p1 - p0))
        ang = math.acos(max(-1, min(1, (np.trace(R0.T @ R1) - 1) / 2)))
        worst = max(worst, dp / tol_pos, ang / tol_ang)
        hit |= dp > tol_pos or ang > tol_ang
    return hit, worst


def settle(w, t=0.25):
    w.run(t)


def result(name, ok, cls, harmful, w, extra=None, t0=None):
    r = {"move": name, "ok": bool(ok), "class": cls, "harmful": bool(harmful), "sim_s": round(w.data.time, 2),
         "energy_J": round(w.steppers.energy + w.rel_energy, 4), "failed_steps": int(w.steppers.failed.sum()),
         "broken": bool(w.steppers.broken.any()), "skipped_s": round(getattr(w, "skipped", 0.0), 2)}
    if extra:
        r.update(extra)
    return r


def drive_leg(w, i, leg, angle, frac=1.0, timeout=90.0):
    w.leg(i, leg, angle, frac)
    return w.wait_actuators(timeout=timeout)


def tilt_from(w, i, R0):
    """Angle (rad) between the cube's current and starting attitude (IMU)."""
    _, R = w.pose(i)
    return math.acos(max(-1.0, min(1.0, (np.trace(R0.T @ R) - 1) / 2)))


def drive_leg_until(w, i, leg, max_angle, cond, frac=1.0, timeout=150.0):
    """Step a leg one tooth at a time toward max_angle until cond(w) (the controller reads the IMU
    between steps)."""
    a = w.aidx[f"c{i}{leg}"]
    w.steppers.drive_frac[a] = frac
    t_end = w.data.time + timeout
    sgn = 1 if max_angle > w.steppers.phi[a] else -1
    while w.data.time < t_end:
        if cond(w):
            return True
        if sgn * (w.steppers.phi[a] - max_angle) >= -0.5 * w.steppers.step[a]:
            w.run(0.3)
            return cond(w)
        w.steppers.target[a] = w.steppers.phi[a] + sgn * w.steppers.step[a]
        w.run_until(lambda: not w.steppers.busy(a), 5.0)
        w.run(0.03)
    return cond(w)


def pivot(w, i, leg, turn_deg, frac=1.0, timeout=200.0):
    """Open a leg a tooth at a time until the IMU shows the body turned turn_deg (within 3 deg) or the
    leg reaches turn_deg + 6 deg: the controller stops on the landing instead of grinding the leg
    into the support it has landed on."""
    _, R0 = w.pose(i)
    goal = math.radians(turn_deg - 3)
    return drive_leg_until(w, i, leg, math.radians(turn_deg + 6), lambda w: tilt_from(w, i, R0) > goal, frac, timeout)


def open_others(w, i, keep=()):
    """Fully open every latched face of cube i except those to `keep` (both bolts on the face free the
    hooks and retract the spheres), as execute() does before a move."""
    d = w.d
    for j in range(w.n):
        if j == i or j in keep or not w.latched(i, j):
            continue
        face, _, _ = w.face_between(i, j)
        bolts = World.FACE_BOLTS.get(face, ())
        for c in bolts:
            w.release(i, [c], True, level="full")
        w.run_until(lambda: all(w.rel[i, list(CARRIERS).index(c)] >= d.hook_free - 1e-9 for c in bolts), 20)
        w.unlatch(i, j)


def relieve(w, i, leg, teeth=3):
    """After a landing, back the leg off a few teeth so its lock preload (the leg still pressing the
    support or the floor) is released before the bolts relatch; a pressed leg pushes the structure
    across a slippery floor."""
    a = w.aidx[f"c{i}{leg}"]
    st = w.steppers
    if abs(st.phi[a]) < math.radians(5):
        return  # folded: nothing presses
    w.steppers.target[a] = st.phi[a] - np.sign(st.phi[a]) * teeth * st.step[a]
    w.run_until(lambda: not st.busy(a), 20.0)
    w.run(0.2)


# ------------------------------------------------------------------------------------- lone floor moves


def lone_pick(R, kind, catch=None):
    """First available grid move of the given kind for a lone cube with rotation R."""
    for mv in moves(R, (0, 0, 0), set()):
        if mv.kind == kind and (catch is None or (mv.catch is not None) == catch):
            return mv
    return None


def floor_tip(d, p, R, kind, use_catch=False, record=None):
    """One lone tip on a bare floor: BOOK (leg in the bottom face opens over its hinge edge) or KICK (a
    leg in a vertical face, hinged at the bottom edge, swings down and lifts the cube over the far edge);
    with use_catch, the landing face's far-edge leg is opened first and lowers the cube after the
    balance point. Success: the cube in the planned cell and orientation within 0.15 L and 10 deg, legs
    closed again."""
    mv = lone_pick(R, kind, catch=True if use_catch else None)
    if mv is None:
        return None
    w = World(d, place(d, [(0, 0, 0)], [R]), p)
    if record:
        w.watch = record
    settle(w, 0.2)
    execute_tip(w, 0, mv, d.floor_frac, use_catch)
    cell, Rs, perr, aerr = lattice_state(w, 0)
    want_cell = tuple(int(v) for v in mv.delta)
    ok_cell = cell == want_cell
    ok_rot = key(Rs.round().astype(int)) == mv.R_new
    ok = ok_cell and ok_rot and perr < 0.15 * d.L and aerr < math.radians(10) and not w.steppers.broken.any()
    if ok:
        cls = "reached"
    elif cell == (0, 0, 0) and key(Rs.round().astype(int)) == key(np.asarray(R).round().astype(int)):
        cls = "stayed"
    elif not ok_rot or not ok_cell:
        cls = "wrong_cell_or_face"
    else:
        cls = "off_lattice"
    return result(f"tip_{kind.lower()}{'_catch' if use_catch else ''}", ok, cls, False, w,
                  {"pos_err_mm": round(perr * 1e3, 3), "ang_err_deg": round(math.degrees(aerr), 2), "leg": mv.leg, "catch": mv.catch if use_catch else None})


def execute_tip(w, i, mv, frac, use_catch=False):
    """Controller of one lone tip (BOOK or KICK): book opens the bottom leg to flat; kick swings the
    leg down and lifts until the IMU shows the balance point passed (tilt > 48 deg), then stops; with a
    catch, the landing face's far-edge leg is first opened to 150 deg so its tip lands beyond the far
    edge, and after the balance it opens to flat, lowering the cube. Then every leg closes."""
    _, R0 = w.pose(i)
    if mv.kind == "BOOK":
        pivot(w, i, mv.leg, 90, frac)
    else:
        if use_catch and mv.catch:
            drive_leg(w, i, mv.catch, math.radians(CATCH_DEG), frac)
        drive_leg_until(w, i, mv.leg, math.radians(250), lambda w: tilt_from(w, i, R0) > math.radians(48), frac)
        if use_catch and mv.catch:
            drive_leg(w, i, mv.catch, math.radians(184), frac)
    w.run(0.3)
    drive_leg(w, i, mv.leg, 0.0, frac, timeout=150)
    if use_catch and mv.catch:
        drive_leg(w, i, mv.catch, 0.0, frac, timeout=150)
    settle(w, 0.3)


CATCH_DEG = 166.0  # catch leg pre-opened so its tip lands beyond the far edge only after the balance point

HOME_STATES = {
    "book_x": orient(y="-Z", z="+X"),  # XP in the bottom face: book over (+y,+z)
    "book_d": orient(z="-Z", x="-X"),  # DN in the bottom face: book onto -x
    "kick_x": orient(z="-Z", y="+X"),  # XN hinge at the bottom edge (-y,-z): kick toward +y
    "kick_d": orient(x="-Z", z="+X"),  # DP hinge at the bottom: kick off the disc face
}


def tip_move(kind, catch=False):
    """A random lone orientation that offers a tip of this kind (with a catch leg if asked), random
    heading."""
    def f(d, p, record=None):
        rng = np.random.default_rng(p.seed + 17)
        cands = [R for R in LATTICE_ROTATIONS if lone_pick(R, kind, catch=True if catch else None) is not None]
        R = cands[int(rng.integers(len(cands)))]
        return floor_tip(d, p, R, kind, use_catch=catch, record=record)
    return f


def yaw_move(d, p, record=None, quarter=1):
    """Disc face down on the floor: the joint turns the body on the disc (tread grips) a quarter turn;
    the controller learns the joint-to-heading ratio from the first stroke (IMU) and corrects (yaw_exec);
    success within 3 deg of the target heading."""
    R = orient(x="-Z", z="+X")
    w = World(d, place(d, [(0, 0, 0)], [R]), p)
    if record:
        w.watch = record
    settle(w, 0.2)
    _, R0 = w.pose(0)
    ok = yaw_exec(w, 0, quarter)
    _, R1 = w.pose(0)
    yaw = math.atan2(R1[1, 2], R1[0, 2]) - math.atan2(R0[1, 2], R0[0, 2])
    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
    err = quarter * math.pi / 2 - yaw
    pos, _ = w.pose(0)
    return result("yaw", ok, "reached" if ok else ("short" if abs(err) > math.radians(30) else "off_target"), False, w,
                  {"yaw_err_deg": round(math.degrees(err), 2), "drift_mm": round(float(np.linalg.norm(pos[:2])) * 1e3, 3)})


def self_right(d, p, record=None, face=None):
    """From a resting face (random heading), the BFS-planned sequence of tips to the home orientation
    (R = I: +z up, x along X); each tip is a physical trial; a failed tip is re-planned from the state
    reached, at most 10 tips."""
    rng = np.random.default_rng(p.seed + 5)
    faces = ["+x", "-x", "+y", "-y", "+z", "-z"]
    f = faces[face] if face is not None else faces[rng.integers(6)]
    down = {"+x": orient(x="-Z", y="+X"), "-x": orient(x="+Z", y="+X"), "+y": orient(y="-Z", z="+X"), "-y": orient(y="+Z", z="+X"), "+z": orient(z="-Z", x="+X"), "-z": orient(z="+Z", x="+X")}[f]
    R = rot90(Zv, int(rng.integers(4))) @ down
    w = World(d, place(d, [(0, 0, 0)], [R]), p)
    if record:
        w.watch = record
    settle(w, 0.2)
    home = key(np.eye(3, dtype=int))
    n_tips = 0
    n_yaws = 0
    from collections import deque
    while n_tips < 10:
        cell, Rs, perr, aerr = lattice_state(w, 0)
        Rk = key(Rs.round().astype(int))
        if Rk == home:
            break
        # BFS over orientation only (cells free)
        prev = {Rk: None}
        dq = deque([Rk])
        while dq:
            r = dq.popleft()
            if r == home:
                break
            for mv in moves(np.array(r).reshape(3, 3), (0, 0, 0), set()):
                if mv.R_new not in prev:
                    prev[mv.R_new] = (r, mv)
                    dq.append(mv.R_new)
        path = []
        r = home
        while prev.get(r):
            r0, mv = prev[r]
            path.append(mv)
            r = r0
        if not path:
            break
        mv = path[-1]
        frac = d.floor_frac
        if mv.kind == "YAW":
            q = 1 if key(rot90(Zv, 1) @ Rs.round().astype(int)) == mv.R_new else -1
            yaw_exec(w, 0, q)
            n_yaws += 1
            if n_yaws > 6:
                break
        else:
            execute_tip(w, 0, mv, frac)  # no catch leg: the kick-with-catch variant props the cube at its balance point (recorded failure)
            n_tips += 1
    cell, Rs, perr, aerr = lattice_state(w, 0)
    ok = key(Rs.round().astype(int)) == home and aerr < math.radians(10)
    return result("self_right", ok, "reached" if ok else "not_home", False, w, {"start_face": f, "tips": n_tips, "yaws": n_yaws})


AXES_W = {"+x": np.array([1.0, 0, 0]), "-x": np.array([-1.0, 0, 0]), "+y": np.array([0, 1.0, 0]), "-y": np.array([0, -1.0, 0]), "+z": np.array([0, 0, 1.0]), "-z": np.array([0, 0, -1.0])}


def fall(d, p, record=None):
    """20 cm drop in a random orientation onto the floor; contact stiffened to a corner Hertz time
    constant (0.15 ms) to read the peak impact. Survival: no lock overloaded (legs, joints), legs still
    closed, and the peak deceleration's loads under the part limits (computed after the run)."""
    rng = np.random.default_rng(p.seed + 23)
    from scipy.spatial.transform import Rotation as Rot
    R = Rot.random(random_state=int(rng.integers(1 << 30))).as_matrix()
    w = World(d, [(np.array([0, 0, 0.20 + d.L]), R)], replace(p, damp=p.damp), dt=2e-5, mag_every=50)
    w.model.geom_solref[:, 0] = 1.5e-4
    if record:
        w.watch = record
    v_peak, a_peak = 0.0, 0.0
    vel_prev = None
    lin = np.zeros(6)
    while w.data.time < 0.30:
        w.step()
        mujoco.mj_objectVelocity(w.model, w.data, mujoco.mjtObj.mjOBJ_BODY, w.body[0], lin, 0)
        v = lin[3:].copy()
        if vel_prev is not None:
            acc = float(np.linalg.norm(v - vel_prev)) / w.dt
            if acc > a_peak:
                a_peak = acc
                R_hit = w.pose(0)[1].copy()
                a_dir = (v - vel_prev) / max(np.linalg.norm(v - vel_prev), 1e-12)
        vel_prev = v
        v_peak = max(v_peak, float(np.linalg.norm(v)))
    w.dt_restore = True
    settle(w, 0.0)
    legs_closed = all(abs(w.angle(f"c0{lg}") if False else w.data.qpos[w.qadr[w.aidx[f"c0{lg}"]]]) < math.radians(8) for lg in LEGS)
    g = a_peak / 9.81
    # loads at the peak deceleration (E): leg plate as a cantilever on its lock, corner sphere on its cover
    m_leg = d.leg_len * d.leg_w_frac * d.L * d.leg_t * 1150
    # a closed leg sits in its face pocket: only the inertial load that pulls it out of the pocket
    # loads its lock (the deceleration a_dir pushes the body; the plate lags along -a_dir)
    out = max(max(0.0, -float((R_hit @ AXES_W[LEGS[lg][0]]) @ a_dir)) for lg in LEGS) if a_peak > 0 else 0.0
    leg_torque = m_leg * a_peak * d.leg_len / 2 * out
    lock_ok = leg_torque < d.leg_actuator().hold * 45.0 / 30.0  # hold is at 30 MPa design stress; the tooth breaks at the resin's strength (A: 45 MPa, ABS-like resin 40-50 MPa)
    m_sph = math.pi / 6 * d.d_corner**3 * 7500
    f_sph = m_sph * a_peak
    # Hertz contact stress of the sphere on its 0.08 mm cover (resin E 2 GPa, sphere 200 GPa)
    E_star = 1 / ((1 - 0.35**2) / 2.0e9 + (1 - 0.3**2) / 160e9)
    a_c = (3 * f_sph * (d.d_corner / 2) / (4 * E_star)) ** (1 / 3)
    p_max = 3 * f_sph / (2 * math.pi * a_c**2)
    cover_ok = p_max < 90e6
    # LiPo / cell: mass x a on a 10 mm2 bond area, adhesive 5 MPa (A)
    ok = (not w.steppers.broken.any()) and legs_closed and lock_ok and cover_ok
    return result("fall_20cm", ok, "survived" if ok else "damaged", False, w,
                  {"peak_g": round(g), "leg_torque_over_hold": round(leg_torque / d.leg_actuator().hold, 3),
                   "sphere_cover_MPa": round(p_max / 1e6, 1), "legs_closed": legs_closed, "impact_speed": round(v_peak, 2)})


# ------------------------------------------------------------------------------------- structure moves

CARRIER_SIGNS = {k: v for k, v in CARRIERS.items()}


def edge_carrier(face_a, face_b):
    """Carrier on the x-parallel edge between two plain faces (body labels like '+y', '-z')."""
    sy = sz = None
    for f in (face_a, face_b):
        if f[1] == "y":
            sy = 1 if f[0] == "+" else -1
        if f[1] == "z":
            sz = 1 if f[0] == "+" else -1
    if sy is None or sz is None:
        return None
    return next(k for k, v in CARRIERS.items() if v == (sy, sz))


def body_label(R, wvec):
    v = np.asarray(R).T @ np.asarray(wvec, float)
    ax = int(np.argmax(np.abs(v)))
    return ("+" if v[ax] > 0 else "-") + "xyz"[ax]


def build_scene(d, p, cells, rots, record=None):
    """World with cubes at lattice cells, every neighbouring pair that can latch latched."""
    w = World(d, place(d, cells, rots), p)
    if record:
        w.watch = record
    w.run(0.25)
    for a in range(w.n):
        for b in range(a + 1, w.n):
            if np.sum(np.abs(np.subtract(cells[a], cells[b]))) == 1:
                w.latch(a, b)
    w.run(0.1)
    return w


def pick_rotation(rng, cell, occ, kind, delta, braced=None):
    cands = []
    for R in LATTICE_ROTATIONS:
        Ri = R.round().astype(int)
        for mv in moves(Ri, cell, occ, supports_ok=braced):
            if mv.kind == kind and tuple(int(v) for v in mv.delta) == tuple(delta):
                cands.append((Ri, mv))
    if not cands:
        return None, None
    return cands[int(rng.integers(len(cands)))]


def support_rots(rng, cells):
    """Supports with x axes all along world X (parallel, so their plain faces latch), random quarter
    turns about X."""
    out = []
    for _ in cells:
        out.append(rot90(np.array([1, 0, 0]), int(rng.integers(4))))
    return out


def rigid_shift(w: World, before, movers):
    """How much of the supports' disturbance is one rigid shift of the whole support structure (bonds
    intact, lattice kept relative to itself): the mean displacement of the non-movers, and the largest
    change of a pairwise offset between them over 0.1 L (harm relative to the structure). Reported next
    to harm_ratio; the class still uses the absolute convention."""
    idx = [i for i in range(w.n) if i not in movers]
    d = {i: w.pose(i)[0] - before[i][0] for i in idx}
    mean = np.mean([d[i] for i in idx], axis=0) if idx else np.zeros(3)
    rel = max((float(np.linalg.norm(d[a] - d[b])) for a in idx for b in idx if a < b), default=0.0)
    return {"support_shift_mm": round(float(np.linalg.norm(mean)) * 1e3, 3), "rel_harm_ratio": round(rel / (0.1 * w.d.L), 3)}


def finish_structure_move(name, w, d, mover, mv, start_cell, before, latch_to, extra=None):
    w.run(0.3)
    cell, Rs, perr, aerr = lattice_state(w, mover)
    want = tuple(int(a + b) for a, b in zip(start_cell, mv.delta))
    ok_place = cell == want and key(Rs.round().astype(int)) == mv.R_new and perr < 0.15 * d.L and aerr < math.radians(10)
    latched = [w.latched(mover, j) for j in latch_to]
    hit, worst = harm(w, before, {mover})
    ok = ok_place and all(latched) and not hit and not w.steppers.broken.any()
    if ok:
        cls = "reached"
    elif hit:
        cls = "knocked_support" if worst > 1 else "harm"
    elif w.steppers.broken.any():
        cls = "broken"
    elif cell == tuple(start_cell):
        cls = "stayed"
    elif ok_place and not all(latched):
        cls = "unlatched"
    else:
        cls = "off_lattice"
    ex = {"pos_err_mm": round(perr * 1e3, 3), "ang_err_deg": round(math.degrees(aerr), 2), "latched": latched, "harm_ratio": round(worst, 3), "leg": mv.leg}
    ex.update(rigid_shift(w, before, {mover}))
    if extra:
        ex.update(extra)
    return result(name, ok, cls, hit, w, ex)


def relatch(w, mover, targets, retries=2, reseats=0):
    """Let the bolts back out (shared unlock) and close every target latch; a latch outside the lead-in
    is retried after a short settle. With reseats > 0, a face whose hooks do not engage is re-seated:
    the controller retracts that face's corner spheres (the bolts' full stroke), so a cube the magnets
    hold off-level drops onto the floor, then lets them back so the magnets pull it in at floor level,
    and tries the hooks again (the bolt rack counts whether the hooks closed: a contact reading)."""
    for c in CARRIERS:
        w.release(mover, [c], retract=False)
    w.run_until(lambda: np.all(w.rel[mover] <= 1e-9), 15.0)
    for k in range(retries + 1 + reseats):
        if all(w.latched(mover, j) or w.latch(mover, j) for j in targets):
            return True
        if k >= retries:  # re-seat every face still open
            for j in targets:
                if w.latched(mover, j):
                    continue
                face, _, _ = w.face_between(mover, j)
                fj, _, _ = w.face_between(j, mover)
                for c in World.FACE_BOLTS.get(face, ()):
                    w.release(mover, [c], True, level="full")
                held = [c for c in World.FACE_BOLTS.get(fj, ()) if not any(w.latched(j, k) for k in range(w.n) if k != j and k != mover and w.face_between(j, k)[0] == fj)]
                for c in held:  # the neighbour retracts its spheres on the facing side too (told over IR)
                    w.release(j, [c], True, level="full")
                w.run_until(lambda: not w._rel_moving(), 20.0)
                w.run(0.5)
                for c in CARRIERS:
                    w.release(mover, [c], retract=False)
                    w.release(j, [c], retract=False)
                w.run_until(lambda: np.all(w.rel[mover] <= 1e-9) and np.all(w.rel[j] <= 1e-9), 15.0)
                w.run(0.3)
                w.reseats = getattr(w, "reseats", 0) + 1
        else:
            w.run(0.3)
    return all(w.latched(mover, j) for j in targets)


def climb(config="row", brake=False):
    """CLIMB: the mover beside a support on the floor swings 180 deg about the support's top edge onto
    its top. config 'row': the support has a latched neighbour behind it; 'lone': the support is alone;
    'top': mover on a 1-high row climbs onto a cube that sits on the row (level 1 to 2).

    brake=True: the climb leg's partner (in the mover's top face, on the same hinge edge) is opened to
    41 deg first; the climb leg pushes to 139 deg, where the partner rests on the support's top, and the
    partner then steps shut, lowering the last 41 deg instead of letting the cube flop onto the top."""
    def f(d, p, record=None):
        rng = np.random.default_rng(p.seed + 31)
        if config == "row":
            m_cell, sup = (0, 0, 0), [(0, 1, 0), (0, 2, 0)]
        elif config == "lone":
            m_cell, sup = (0, 0, 0), [(0, 1, 0)]
        else:
            m_cell, sup = (0, 0, 1), [(0, 0, 0), (0, 1, 0), (0, 2, 0), (0, 1, 1), (0, 2, 1)]
        occ = set(sup)
        braced = (lambda s: True) if config == "lone" else None
        R, mv = pick_rotation(rng, m_cell, occ, "CLIMB", (0, 1, 1), braced)
        name = f"climb_{config}" + ("_braked" if brake else "")
        if R is None:
            return {"move": name, "ok": False, "class": "no_move", "harmful": False}
        cells = [m_cell] + sup
        w = build_scene(d, p, cells, [R] + support_rots(rng, sup), record)
        before = snapshot(w)
        s0 = cells.index((m_cell[0], m_cell[1] + 1, m_cell[2]))
        wl, wo = leg_world(R, mv.leg)
        Q = body_label(R, wl)
        B = body_label(R, -Zv)
        far = edge_carrier(Q, B)
        if brake:
            bleg = LEG_PARTNER[mv.leg]
            drive_leg(w, 0, bleg, math.radians(41), 1.0)
        open_others(w, 0, keep=(s0,))  # e.g. the cube the mover stands on (climb from a level)
        if not w.open_face(0, s0, far_edge=far):
            return finish_structure_move(name, w, d, 0, mv, m_cell, before, [s0], {"stage": "bolt_stall"})
        w.run(0.2)
        if brake:
            _, R0 = w.pose(0)
            pivot(w, 0, mv.leg, 142, 1.0)  # stops at 139 deg, resting on the partner leg
            drive_leg_until(w, 0, bleg, 0.0, lambda w: tilt_from(w, 0, R0) > math.radians(177), 1.0, timeout=300)
            drive_leg(w, 0, bleg, 0.0, 1.0)
        else:
            pivot(w, 0, mv.leg, 180, 1.0)
        w.run(0.4)
        target = tuple(int(a + b) for a, b in zip(m_cell, mv.delta))
        below = cells.index((target[0], target[1], target[2] - 1))
        relieve(w, 0, mv.leg)
        relatch(w, 0, [below], reseats=2)
        drive_leg(w, 0, mv.leg, 0.0, 1.0, timeout=200)
        return finish_structure_move(name, w, d, 0, mv, m_cell, before, [below])
    return f


def brake_rots(dvec):
    """Support orientations (x along +-X, so it latches the row) whose leg lies in the face toward dvec
    and is hinged on that face's top edge: the CLIMB leg pointing off the row's end."""
    out = []
    for Rr in LATTICE_ROTATIONS:
        Ri = Rr.round().astype(int)
        if abs(Ri[0, 0]) != 1:
            continue
        for leg in ("XP", "XN"):
            wl, wo = leg_world(Ri, leg)
            if (wl == np.asarray(dvec)).all() and (wo == np.array([0, 0, 1])).all():
                out.append((Ri, leg))
    return out


def roll_top(kind="BOOK", brake=None):
    """Roll one cell along the top of a 3-cube row (BOOK: leg in the bottom face), or BOOK_DESCEND off
    the end of a 2-cube row down to the floor (180 deg, leg on the row's top).

    brake (descend only) keeps the whole 180 deg quasi-static. Without it the lift leg can only push
    the cube up to the balance point (45 deg); from there it swings 135 deg about the magnetic hinge
    and the hinge reaction drags the row across the floor (book_descend's knocked_support).
    "self": the mover's second x-leg (LEG_PARTNER, in the face that ends against the support, on the
    same hinge edge) is opened to 131 deg first; the lift leg raises the cube to 49 deg, where that leg
    comes to rest on the support's end face, and the leg then steps shut a tooth at a time, lowering
    the cube onto the floor. "end": the row's end cube does the braking with its CLIMB leg (in its end
    face, hinged on its top edge = the hinge line), told over IR; needs the end cube in one of the
    orientations brake_rots() returns, so it is the planner's fallback. In both, the brake lowers the cube
    to 160 deg and folds; the lift leg then swings over onto the support's top and presses the cube flat
    (the original descent's last step), driving its floor edge in against friction."""
    def f(d, p, record=None):
        rng = np.random.default_rng(p.seed + 41)
        if kind == "BOOK":
            m_cell, sup, delta, name = (0, 0, 1), [(0, 0, 0), (0, 1, 0), (0, -1, 0)], None, "roll_top"
        else:
            m_cell, sup, delta, name = (0, 1, 1), [(0, 1, 0), (0, 2, 0)], (0, -1, -1), "book_descend"
        occ = set(sup)
        cands = []
        for Rr in LATTICE_ROTATIONS:
            Ri = Rr.round().astype(int)
            for mv in moves(Ri, m_cell, occ):
                if mv.kind == kind and (delta is None or tuple(int(v) for v in mv.delta) == delta):
                    if brake == "self" and mv.leg not in LEG_PARTNER:
                        continue
                    cands.append((Ri, mv))
        if not cands:
            return {"move": name, "ok": False, "class": "no_move", "harmful": False}
        R, mv = cands[int(rng.integers(len(cands)))]
        cells = [m_cell] + sup
        rots = [R] + support_rots(rng, sup)
        if brake == "end":
            name = "descend_endbrake" if kind != "BOOK" else "roll_top_endbrake"
            opts = brake_rots((0, -1, 0))
            rots[1], bleg = opts[int(rng.integers(len(opts)))]
            bcube = 1
        elif brake == "self":
            name = "descend_braked" if kind != "BOOK" else "roll_top_braked"
            bleg, bcube = LEG_PARTNER[mv.leg], 0
        total = 90 if kind == "BOOK" else 180
        w = build_scene(d, p, cells, rots, record)
        before = snapshot(w)
        s0 = 1
        wl, wo = leg_world(R, mv.leg)
        B = body_label(R, -Zv)
        far = edge_carrier(B, body_label(R, -wo))
        if brake:
            # the brake leg meets the support at 49 deg of the turn: open it to (total - 49) deg
            drive_leg(w, bcube, bleg, math.radians(total - 49), 1.0)
        if not w.open_face(0, s0, far_edge=far):
            return finish_structure_move(name, w, d, 0, mv, m_cell, before, [s0], {"stage": "bolt_stall"})
        if brake:
            _, R0 = w.pose(0)
            pivot(w, 0, mv.leg, 52, 1.0)  # stops at 49 deg: past balance, resting on the brake leg
        if brake and kind == "BOOK":  # roll on top: lower the last 41 deg onto the next support
            drive_leg_until(w, bcube, bleg, 0.0, lambda w: tilt_from(w, 0, R0) > math.radians(87), 1.0, timeout=300)
            drive_leg(w, bcube, bleg, 0.0, 1.0)
            drive_leg(w, 0, mv.leg, 0.0, 1.0)
        elif brake:
            # lower to 160 deg; below that the cube's floor edge touches down (the hinge rolls on the
            # rounded edges and sags 0.3-0.5 mm, so the edge lands at 164-168 deg, not at 180)
            drive_leg_until(w, bcube, bleg, 0.0, lambda w: tilt_from(w, 0, R0) > math.radians(160), 1.0, timeout=300)
            end = os.environ.get("SMA_DESCEND_END", "press")
            if end == "retract":
                # touchdown with no magnet across the hinge: the cube finishes about its floor edge and
                # may stand off the support by up to ~2 mm; relatch() then lets the spheres back and
                # the magnets pull the two together (re-seat if the hooks miss)
                hinge = next(c for c in World.FACE_BOLTS.get(B, ()) if c != far)
                w.release(0, [hinge], True, level="full")
                f1, _, _ = w.face_between(1, 0)
                for c in World.FACE_BOLTS.get(f1, ()):
                    w.release(1, [c], True, level="full")  # the support retracts its facing spheres (IR)
                w.run_until(lambda: not w._rel_moving(), 20.0)
                drive_leg(w, bcube, bleg, 0.0, 1.0)
                drive_leg(w, 0, mv.leg, 0.0, 1.0)
            else:  # "press"
                drive_leg(w, bcube, bleg, 0.0, 1.0)  # brake folds; the cube settles on its floor edge
                # the lift leg swings over (through the air above the support), lands on the support's
                # top and presses the cube flat about the hinge, driving the floor edge in
                drive_leg_until(w, 0, mv.leg, math.radians(186), lambda w: tilt_from(w, 0, R0) > math.radians(177), 1.0, timeout=300)
        else:
            pivot(w, 0, mv.leg, 90 if kind == "BOOK" else 180, 1.0)
        w.run(0.4)
        target = tuple(int(a + b) for a, b in zip(m_cell, mv.delta))
        nb = [cells.index(c) for c in [(target[0], target[1], target[2] - 1), (target[0], target[1] + 1, target[2]), (target[0], target[1] - 1, target[2])] if c in cells and cells.index(c) != 0]
        relieve(w, 0, mv.leg)
        relatch(w, 0, nb, reseats=2)
        drive_leg(w, 0, mv.leg, 0.0, 1.0, timeout=200)
        extra = {"brake_leg": bleg, "brake_cube": bcube} if brake else None
        return finish_structure_move(name, w, d, 0, mv, m_cell, before, nb, extra)
    return f


def split(config="pair"):
    """SPLIT: a cube latched to a neighbour (pair) or at the end of a row opens the latch, retracts both
    edges' corner spheres and tips away on the floor."""
    def f(d, p, record=None):
        rng = np.random.default_rng(p.seed + 43)
        m_cell = (0, 0, 0)
        sup = [(0, 1, 0)] if config == "pair" else [(0, 1, 0), (0, 2, 0)]
        occ = set(sup)
        cands = []
        for Rr in LATTICE_ROTATIONS:
            Ri = Rr.round().astype(int)
            if abs(Ri[0, 0]) != 1:
                continue  # latched to the row: x parallel to the row's x (world X), plain face toward it
            for mv in moves(Ri, m_cell, occ):
                if mv.kind in ("BOOK", "KICK") and tuple(int(v) for v in mv.delta) == (0, -1, 0):
                    cands.append((Ri, mv))
        R, mv = cands[int(rng.integers(len(cands)))]
        cells = [m_cell] + sup
        w = build_scene(d, p, cells, [R] + support_rots(rng, sup), record)
        before = snapshot(w)
        Q = body_label(R, np.array([0, 1, 0]))
        bolts = World.FACE_BOLTS.get(Q, ())
        for c in bolts:
            w.release(0, [c], True, level="full")
        w.run_until(lambda: np.all([w.rel[0, list(CARRIERS).index(c)] >= d.hook_free for c in bolts]), 20)
        w.unlatch(0, 1)
        w.run_until(lambda: np.all([abs(w.rel[0, list(CARRIERS).index(c)] - d.rel_stroke) < 1e-9 for c in bolts]), 20)
        execute_tip(w, 0, mv, d.floor_frac)
        for c in CARRIERS:
            w.release(0, [c], retract=False)
        w.run(1.0)
        return finish_structure_move(f"split_{config}", w, d, 0, mv, m_cell, before, [])
    return f


DOCK_SENSE_SIGMA = 0.25e-3  # m, IR range error of the gap reading before the last tip (A)
DOCK_TOO_CLOSE = 0.8e-3  # m: a reading this much closer than nominal triggers the back-off (A, set from the ride-up cases)


def landing_errors():
    """Measured landing errors of single floor tips (BOOK and KICK trials of this geometry; the sharp-edge
    batch if none yet): position error magnitudes (m) and heading errors (rad)."""
    pos, ang = [], []
    for base in (OUT / "trials_sharp_edges", TRIALS):  # fixed sample: the sharp-edge batch (32 BOOK and KICK landings)
        for m in ("tip_book", "tip_kick"):
            f = base / f"{m}.jsonl"
            if f.exists():
                for l in f.open():
                    r = json.loads(l)
                    if r.get("ok"):
                        pos.append(r["pos_err_mm"] * 1e-3)
                        ang.append(math.radians(r["ang_err_deg"]))
        if pos:
            break
    return np.array(pos), np.array(ang)


def dock_variant(err="measured", backoff=True):
    """DOCK: a lone cube one cell from a latched 2-cube row tips toward it; the magnets pull it in and the
    bolts latch (re-seat on a failed hook). Approach error: 'assumed' = uniform +-0.15 L in both floor
    axes and +-6 deg heading (the first model, A); 'measured' = one preceding floor tip's landing error,
    drawn from the BOOK/KICK trials (magnitude, random direction; heading with random sign). With
    backoff, the cube reads the gap to the row by IR before the last tip (sigma 0.25 mm, A); if it reads
    more than 0.8 mm too close it tips away one cell and plans back (BFS) to the same cell and
    orientation, then reads again (at most 2 back-offs, counted as retries, their time and energy in
    the trial)."""
    def f(d, p, record=None):
        rng = np.random.default_rng(p.seed + 47)
        sup = [(0, 2, 0), (0, 3, 0)]
        occ = set(sup)
        cands = []
        for Rr in LATTICE_ROTATIONS:
            Ri = Rr.round().astype(int)
            if abs(Ri[1, 0]) == 1 or abs(Ri[2, 0]) == 1:
                continue  # x must lie along world X to latch onto the row (planner rule)
            for mv in moves(Ri, (0, 0, 0), occ):
                if mv.kind in ("BOOK", "KICK") and tuple(int(v) for v in mv.delta) == (0, 1, 0):
                    R1 = np.array(mv.R_new).reshape(3, 3)
                    if abs(R1[0, 0]) == 1:
                        cands.append((Ri, mv))
        if not cands:
            return {"move": "dock", "ok": False, "class": "no_move", "harmful": False}
        R, mv = cands[int(rng.integers(len(cands)))]
        if err == "assumed":
            off = np.array([rng.uniform(-0.15, 0.15) * d.L, rng.uniform(-0.15, 0.15) * d.L, 0.0])
            yaw = math.radians(rng.uniform(-6, 6))
        else:
            pe, ae = landing_errors()
            k = int(rng.integers(len(pe)))
            th = rng.uniform(0, 2 * math.pi)
            off = np.array([pe[k] * math.cos(th), pe[k] * math.sin(th), 0.0])
            yaw = ae[int(rng.integers(len(ae)))] * rng.choice([-1, 1])
        Ry = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        cubes = place(d, [(0, 0, 0)] + sup, [Ry @ R] + support_rots(rng, sup))
        cubes[0] = (cubes[0][0] + off, cubes[0][1])
        w = World(d, cubes, p)
        if record:
            w.watch = record
        w.run(0.25)
        w.latch(1, 2)
        before = snapshot(w)
        backoffs = 0
        cells = [(0, 0, 0)] + sup
        while backoffs <= 2:
            pos, Rn = w.pose(0)
            gap_err = pos[1]  # + = closer to the row than nominal (the tip lands one cell on)
            reading = gap_err + rng.normal(0, DOCK_SENSE_SIGMA)
            if not backoff or reading < DOCK_TOO_CLOSE or backoffs == 2:
                break
            backoffs += 1
            cell, Rs, _, _ = lattice_state(w, 0)
            away = next((m for m in moves(Rs.round().astype(int), cell, occ) if m.kind in ("BOOK", "KICK") and tuple(int(v) for v in m.delta) == (0, -1, 0)), None)
            if away is None:
                away = next((m for m in moves(Rs.round().astype(int), cell, occ) if m.kind in ("BOOK", "KICK") and m.delta[1] <= 0 and m.delta[2] == 0), None)
            if away is None:
                break
            execute_tip(w, 0, away, d.floor_frac)
            for _ in range(8):  # BFS back to the dock cell and orientation
                cell, Rs, _, _ = lattice_state(w, 0)
                if cell == (0, 0, 0) and key(Rs.round().astype(int)) == key(R):
                    break
                plan = plan_back(cell, Rs, (0, 0, 0), key(R), occ)
                if not plan:
                    break
                execute_tip(w, 0, plan[0], d.floor_frac) if plan[0].kind != "YAW" else yaw_exec(w, 0, 1 if plan[0].R_new == key(rot90(Zv, 1) @ Rs.round().astype(int)) else -1)
        cell, Rs, _, _ = lattice_state(w, 0)
        if cell != (0, 0, 0) or key(Rs.round().astype(int)) != key(R):
            return finish_structure_move("dock", w, d, 0, mv, (0, 0, 0), before, [1], {"backoffs": backoffs, "stage": "back_off_lost", "err_model": err})
        execute_tip(w, 0, mv, d.floor_frac)
        w.run(0.5)
        first = relatch(w, 0, [1], retries=3)
        ok_latch = first or relatch(w, 0, [1], retries=0, reseats=2)
        return finish_structure_move("dock", w, d, 0, mv, (0, 0, 0), before, [1], {"latched_first_try": first, "reseats": getattr(w, "reseats", 0), "backoffs": backoffs, "err_model": err, "start_offset_mm": [round(v * 1e3, 3) for v in off[:2]]})
    return f


def plan_back(cell, R, goal_cell, goal_key, occ):
    """BFS over floor moves to an exact cell and orientation (floor cells y in -3..0, |x| <= 2)."""
    from collections import deque
    s0 = (tuple(cell), key(np.asarray(R).round().astype(int)))
    prev = {s0: None}
    dq = deque([s0])
    while dq:
        c, Rk = dq.popleft()
        if c == tuple(goal_cell) and Rk == goal_key:
            out = []
            st = (c, Rk)
            while prev[st] is not None:
                pst, mv = prev[st]
                out.append(mv)
                st = pst
            return out[::-1]
        for mv in moves(np.array(Rk).reshape(3, 3), c, occ):
            nc = tuple(int(a + b) for a, b in zip(c, mv.delta))
            if nc[2] != 0 or abs(nc[0]) > 2 or nc[1] > 0 or nc[1] < -3:
                continue
            st = (nc, mv.R_new)
            if st not in prev:
                prev[st] = ((c, Rk), mv)
                dq.append(st)
    return None


dock = dock_variant("assumed", backoff=False)  # the first model, kept as the batch's 'dock'


# ------------------------------------------------------------------------------------- assembly


def plan_cube(start, R0, goal, occupied, allowed, max_states=20000):
    """Shortest move sequence (BFS over cell and orientation) taking one cube from (start, R0) to the
    goal cell with its x axis along world X (so it latches), others fixed in `occupied`; cells limited
    by allowed(cell). Returns [(Move, cell_before, R_before)] or None."""
    from collections import deque

    s0 = (tuple(start), key(np.asarray(R0).round().astype(int)))
    prev = {s0: None}
    dq = deque([s0])
    while dq and len(prev) < max_states:
        cell, Rk = dq.popleft()
        R = np.array(Rk).reshape(3, 3)
        if cell == tuple(goal) and abs(R[0, 0]) == 1:
            out = []
            st = (cell, Rk)
            while prev[st] is not None:
                pst, mv = prev[st]
                out.append((mv, pst[0], np.array(pst[1]).reshape(3, 3)))
                st = pst
            return out[::-1]
        for mv in moves(R, cell, occupied):
            nc = tuple(int(a + b) for a, b in zip(cell, mv.delta))
            if not allowed(nc):
                continue
            st = (nc, mv.R_new)
            if st not in prev:
                prev[st] = ((cell, Rk), mv)
                dq.append(st)
    return None


def yaw_exec(w, i, quarter, tol=math.radians(3), tries=4):
    """Turn a disc-down cube a quarter on its bottom disc: the controller learns the joint-to-heading
    ratio from the first stroke (IMU) and corrects."""
    _, R0 = w.pose(i)
    part = "p" if R0[2, 0] < 0 else "n"
    a = w.aidx[f"c{i}j{part}"]
    h = 2 if abs(R0[2, 2]) < 0.5 else 1  # a horizontal body axis for the heading
    head0 = math.atan2(R0[1, h], R0[0, h])
    target = quarter * math.pi / 2
    gain = -1.0
    phi0 = w.steppers.phi[a]
    corr = target
    for k in range(tries):
        w.joint(i, part, w.steppers.phi[a] + gain * corr)
        w.wait_actuators(timeout=60)
        _, R1 = w.pose(i)
        yaw = (math.atan2(R1[1, h], R1[0, h]) - head0 + math.pi) % (2 * math.pi) - math.pi
        if k == 0 and abs(yaw) > 0.2:
            gain = (w.steppers.phi[a] - phi0) / yaw
        corr = target - yaw
        if abs(corr) < tol:
            return True
    return False


def execute(w, i, mv, cell, cells):
    """Carry out one grid move of cube i at `cell` in a world whose cubes sit at `cells` (list, index =
    cube): unlatch from every neighbour (the face the cube pivots on keeps its hinge edge), drive the
    leg, let the bolts back and latch every new neighbour that can latch, close the leg."""
    d = w.d
    R = np.asarray(w.pose(i)[1]).round().astype(int)
    occ = {tuple(c): j for j, c in enumerate(cells) if j != i}
    nbrs = [occ[tuple(np.add(cell, o))] for o in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)) if tuple(np.add(cell, o)) in occ]
    wl, wo = leg_world(R, mv.leg) if mv.leg else (None, None)
    hinge_nb = None
    far = None
    if mv.kind == "CLIMB":
        hinge_nb = occ[tuple(np.add(cell, wl))]
        far = edge_carrier(body_label(R, wl), body_label(R, -Zv))
    elif mv.kind in ("BOOK", "BOOK_DESCEND") and cell[2] > 0:
        hinge_nb = occ[tuple(np.add(cell, (0, 0, -1)))]
        far = edge_carrier(body_label(R, -Zv), body_label(R, -wo))
    for j in nbrs:  # every other neighbour: open the whole face
        if j == hinge_nb:
            continue
        face, _, _ = w.face_between(i, j)
        for c in World.FACE_BOLTS.get(face, ()):
            w.release(i, [c], True, level="full")
        w.run_until(lambda: all(w.rel[i, list(CARRIERS).index(c)] >= d.hook_free - 1e-9 for c in World.FACE_BOLTS.get(face, ())), 20)
        w.unlatch(i, j)
    if hinge_nb is not None:
        w.open_face(i, hinge_nb, far_edge=far)
    w.run_until(lambda: not w._rel_moving(), 20)
    if mv.kind == "YAW":
        yaw_exec(w, i, 1 if mv.R_new == key(rot90(Zv, 1) @ R) else -1)
    elif mv.kind in ("BOOK", "KICK") and cell[2] == 0:
        execute_tip(w, i, mv, d.floor_frac)
    else:
        pivot(w, i, mv.leg, 90 if mv.kind == "BOOK" else 180, 1.0)
        w.run(0.4)
    new = tuple(int(a + b) for a, b in zip(cell, mv.delta))
    cells2 = list(cells)
    cells2[i] = new
    occ2 = {tuple(c): j for j, c in enumerate(cells2) if j != i}
    targets = [occ2[tuple(np.add(new, o))] for o in ((0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)) if tuple(np.add(new, o)) in occ2]
    if mv.leg and not (mv.kind in ("BOOK", "KICK") and cell[2] == 0):
        relieve(w, i, mv.leg)
    latched = relatch(w, i, [j for j in targets if w.can_latch(i, j) or w.latched(i, j)], retries=2, reseats=2) if targets else True
    if mv.kind not in ("YAW",) and not (mv.kind in ("BOOK", "KICK") and cell[2] == 0):
        drive_leg(w, i, mv.leg, 0.0, 1.0, timeout=200)
    w.run(0.3)
    return new, latched


def assembly(d, p, record=None, log=print):
    """Eight loose cubes build a 2 x 2 x 2: two columns (x = 0 and x = 1), each a 2 x 2 square in the
    y-z plane built from four cubes: a floor pair (the second cube tips and docks to the first),
    then two cubes that tip to the pair, CLIMB onto it and ROLL_TOP across. Each column passes
    through a three-cube step on the way. Cubes start in random orientations and floor cells; every
    move is planned by BFS over the grid moves and executed in MuJoCo; after each move the planner
    reads the cube's lattice state and replans from where it actually is (at most 3 replans a cube).
    Columns meet disc to disc: magnetic only, not latched (the bayonet needs a held neighbour)."""
    rng = np.random.default_rng(p.seed + 101)
    goal_order = [(0, 0), (1, 0), (0, 1), (1, 1)]  # (y, z) build order within a column: pair, step, top
    starts = {}
    cells, rots, goals = [], [], []
    # each column keeps to its own side (x <= 0 or x >= 1) so the two never pass disc to disc
    region = lambda c, x: (-2 <= c[0] <= 0 if x == 0 else 1 <= c[0] <= 3) and -9 <= c[1] <= 7 and c[2] <= 1
    rejected = 0
    for x in (0, 1):
        ys = [0, 3, -3, -6] if x == 0 else [0, 5, -4, -8]  # no two loose cubes touch at the start (only the anchors, disc to disc)
        for k, (gy, gz) in enumerate(goal_order):
            cells.append((x, ys[k], 0))
            goals.append((x, gy, gz))
    for i in range(8):
        # a random orientation whose class can reach its goal latched (x along X): the grid moves tie
        # orientation to cell parity, so about half of all loose cubes can fill a given cell; the
        # planner takes such a cube (rejections counted)
        x = goals[i][0]
        occ = {c for j, c in enumerate(cells) if j > i} | {goals[j] for j in range(i)}
        while True:
            R = LATTICE_ROTATIONS[int(rng.integers(24))].round().astype(int)
            if goals[i] == cells[i] and abs(R[0, 0]) == 1 or plan_cube(cells[i], R, goals[i], occ - {cells[i]}, lambda c, x=x: region(c, x)):
                break
            rejected += 1
        rots.append(R)
    w = World(d, place(d, cells, rots), p)
    if record:
        w.watch = record
    w.run(0.3)
    cur = [tuple(c) for c in cells]
    rows = []
    t0 = w.data.time
    ok_all = True
    for i in range(8):
        x = goals[i][0]
        if i % 4 == 0:  # the column's anchor: self-right in place if needed
            pass
        fails = 0
        for attempt in range(40):  # planned steps; at most 3 that do not land as planned
            cell, Rs, perr, aerr = lattice_state(w, i)
            if perr > 0.25 * d.L or aerr > math.radians(15):
                rows.append({"cube": i, "event": "off_lattice", "pos_err_mm": perr * 1e3})
                ok_all = False
                break
            cur[i] = cell
            if cell == goals[i] and abs(Rs[0, 0]) > 0.5:
                break
            occ = {c for j, c in enumerate(cur) if j != i}
            plan = plan_cube(cell, Rs, goals[i], occ, lambda c, x=x: region(c, x))
            if plan is None:
                rows.append({"cube": i, "event": "no_plan", "cell": cell})
                ok_all = False
                break
            mv, c0, _ = plan[0]
            before = snapshot(w)
            new, latched = execute(w, i, mv, c0, cur)
            hit, worst = harm(w, before, {i})
            got, Rg, pe, ae = lattice_state(w, i)
            rows.append({"cube": i, "move": mv.kind, "leg": mv.leg, "from": c0, "to": new, "got": got, "ok": got == new and key(Rg.round().astype(int)) == mv.R_new, "latched": latched, "harm": bool(hit), "t": round(w.data.time - t0, 1)})
            log(json.dumps(rows[-1], default=int))
            if hit:
                ok_all = False
            cur[i] = got
            if got == goals[i] and abs(Rg[0, 0]) > 0.5:
                break
            if got == new:
                continue  # planned step done: plan the next one
            fails += 1
            if fails > 3:
                ok_all = False
                break
        else:
            ok_all = False
        cur[i] = lattice_state(w, i)[0]
    final = [lattice_state(w, i)[0] for i in range(8)]
    built = sorted(final) == sorted(goals)
    return {"move": "assembly_2x2x2", "ok": bool(built and ok_all), "built": bool(built), "class": "built" if built else "not_built", "harmful": any(r.get("harm") for r in rows), "sim_s": round(w.data.time, 1),
            "moves": sum(1 for r in rows if "move" in r), "failed_moves": sum(1 for r in rows if "move" in r and not r["ok"]), "final": final, "log": rows,
            "energy_J": round(w.steppers.energy + w.rel_energy, 3), "start_orientations_rejected": rejected}


# ------------------------------------------------------------------------------------- NSGA-II

WIRES = [0.025e-3, 0.038e-3, 0.050e-3, 0.076e-3]
STORES = ["cera1", "cera2", "cera3", "lipo301010", "lipo401012", "lipo401020", "lipo402020", "lipo502020"]
OPT_LO = [16.0, 0.0, 0.0, 1.0, 0.80, 0.06, 1.2, 0.0, 6.0]  # leg >= 0.8 L: a KICK lifts the hinge edge to asin(l / L), which must pass the 45 deg balance with margin (0.67 L stayed at 42 deg in the World)
OPT_HI = [25.4, 1.0, 3.999, 8.999, 0.90, 0.12, 3.6, 7.999, 40.0]
# measured requirements in m g L: peak leg lock torque in a nominal CLIMB (1.97) and KICK (1.19) at 17.72 mm with
# a 6-strand leg (logs/sma_cube/need.py; the first search used the static 0.71 and 0.5, which let a 1.18 mgL leg
# through that then stalled in the World), walker and arm peak joint and walker peak bond with the latch cube
NEED = {"climb_lift_mgL": 1.97, "floor_lift_mgL": 1.19, "joint_peak_mgL": 31.75, "arm_drive_mgL": 17.8, "bond_peak_mgL": 70.0}  # walker peak joint, arm peak joint (both measured with the latch cube), walker peak bond
MIN_LONE_TIPS = 30  # a cube cut off from the chain must be able to tip back to it (user: every cube has a battery); 30 tips is about 40 cm
RETRACT_GAIN = 0.91  # residual pull after retraction = 4 pair_force(d, 2 cover + 0.91 (stroke - hook_free)); fitted to 3 World runs


def design_from(x) -> Design:
    from beebot.hardware.sma_cube import pair_force  # noqa: F401
    return replace(chosen(), L=x[0] * 1e-3, joint_route="gear" if x[1] >= 0.5 else "sma", leg_wire=WIRES[int(x[2])],
                   leg_strands=int(x[3]), leg_l_frac=float(x[4]), corner_frac=float(x[5]), rel_stroke=x[6] * 1e-3,
                   storage=STORES[int(x[7])], joint_strands=int(x[8]))


def margins(d: Design) -> dict:
    """Ratios of capacity to the measured need (1.0 = just enough, the spec wants 1.5): the surrogate
    for reliability in the search (the chosen point's reliability is the Monte Carlo)."""
    from beebot.hardware.sma_cube import pair_force
    mgl = d.mass * 9.81 * d.L
    leg = d.leg_actuator()
    js = d.joint_summary()
    b = d.bonds()
    resid = 4 * pair_force(d.d_corner, 2 * d.cover + RETRACT_GAIN * max(d.rel_stroke - d.hook_free, 0.0), d.derate)
    return {
        "climb": leg.drive / (NEED["climb_lift_mgL"] * mgl),
        "floor": leg.drive * d.floor_frac / (NEED["floor_lift_mgL"] * mgl),
        "split": d.mass * 9.81 / max(resid, 1e-9),
        "joint": min(js["hold_Nm"] / (NEED["joint_peak_mgL"] * mgl), js["drive_Nm"] / (NEED["arm_drive_mgL"] * mgl)),
        "latch": b["disc"]["peel_Nm"] / (NEED["bond_peak_mgL"] * mgl),
    }


def opt_eval(x):
    d = design_from(x)
    try:
        t = d.table()
    except Exception:
        return [x[0], 1e3, 0.0, 0.0, 0.0]
    feasible = t["packing"] >= 1.5 and t["layout"]["fits"] and (d.joint_route == "sma" or d.L >= 11e-3) and t["power"]["lone_tips_per_charge"] >= MIN_LONE_TIPS
    if not feasible:
        return [x[0], 1e3, 0.0, 0.0, 0.0]
    m = min(margins(d).values())
    return [x[0], t["cost_usd"]["1M"], -min(m, 3.0), -t["power"]["lone_speed_mm_s"], -js_drive(d)]


def js_drive(d):
    return d.joint_summary()["drive_Nm"] * 1e3


def run_opt(pop=48, gen=40, workers=2):
    from beebot.hardware.mmsim.opt import run_search
    res = run_search(opt_eval, OPT_LO, OPT_HI, n_obj=5, algo="nsga2", pop=pop, n_gen=gen, workers=workers, seed=7)
    rows = []
    for x, f in zip(res.X, res.F):
        if f[1] >= 1e3:
            continue
        d = design_from(x)
        t = d.table()
        rows.append({"L_mm": round(d.L * 1e3, 2), "joint_route": d.joint_route, "leg_wire_um": round(d.leg_wire * 1e6), "leg_strands": d.leg_strands,
                     "leg_l_frac": round(d.leg_l_frac, 3), "corner_frac": round(d.corner_frac, 3), "rel_stroke_mm": round(d.rel_stroke * 1e3, 2),
                     "storage": d.storage, "joint_strands": d.joint_strands, "cost_usd": t["cost_usd"], "mass_g": t["mass_g"],
                     "worst_margin": round(-f[2], 3), "margins": {k: round(v, 3) for k, v in margins(d).items()},
                     "lone_speed_mm_s": t["power"]["lone_speed_mm_s"], "lone_run_min": t["power"]["lone_run_min"], "joint_drive_mNm": round(-f[4], 2),
                     "packing": t["packing"], "layout": t["layout"], "wires": t["wires"]["wires"], "lone_tips": t["power"]["lone_tips_per_charge"]})
    rows.sort(key=lambda r: r["L_mm"])
    out = {"n_evals": res.n_evals, "pop": pop, "gen": gen, "objectives": ["L (min)", "cost at 1M (min)", "worst margin, capped at 3 (max)", "lone speed (max)", "joint drive (max)"],
           "front": rows, "need": NEED, "retract_gain": RETRACT_GAIN, "min_lone_tips": MIN_LONE_TIPS,
           "pick_rule": "smallest L whose worst margin >= 1.5; then lowest cost at 1M",
           "pick": next((r for r in sorted(rows, key=lambda r: (r["L_mm"], r["cost_usd"]["1M"])) if r["worst_margin"] >= 1.5), None)}
    (OUT / "pareto.json").write_text(json.dumps(out, indent=1, default=float))
    return out


# ------------------------------------------------------------------------------------- Monte Carlo

MOVES = {
    "tip_book": tip_move("BOOK"),
    "tip_kick": tip_move("KICK"),
    "yaw": yaw_move,
    "self_right": self_right,
    "fall": fall,
    "dock": dock,
    "dock_measured": dock_variant("measured", backoff=False),
    "dock_measured_backoff": dock_variant("measured", backoff=True),
    "dock_assumed_backoff": dock_variant("assumed", backoff=True),
    "climb_row": climb("row"),
    "climb_top": climb("top"),
    "climb_lone": climb("lone"),
    "roll_top": roll_top("BOOK"),
    "book_descend": roll_top("BOOK_DESCEND"),
    "descend_braked": roll_top("BOOK_DESCEND", brake="self"),
    "descend_endbrake": roll_top("BOOK_DESCEND", brake="end"),
    "roll_top_braked": roll_top("BOOK", brake="self"),
    "climb_row_braked": climb("row", brake=True),
    "split_pair": split("pair"),
    "split_row": split("row"),
}
SLOW = {"assembly": assembly}
# failures a controller can retry from without help (the cube is where it started, or placed but unlatched)
RECOVERABLE = {"stayed", "unlatched", "bolt_stall"}
TRIALS = OUT / "trials"


def trial(args):
    move, seed, design_json = args
    d = Design(**json.loads(design_json)) if design_json else chosen()
    p = sample_params(np.random.default_rng(seed), seed)
    t0 = time.time()
    try:
        r = MOVES[move](d, p)
    except Exception as e:  # a crash is recorded as a failed trial, never dropped
        r = {"move": move, "ok": False, "class": f"error:{type(e).__name__}", "harmful": False, "error": str(e)[:200]}
    r.update({"move": move, "seed": seed, "wall_s": round(time.time() - t0, 1),
              "params": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in asdict(p).items()}})
    return r


def run_mc(moves_, n, workers=WORKERS, seed0=1000, design=None):
    TRIALS.mkdir(parents=True, exist_ok=True)
    dj = json.dumps(asdict(design), default=float) if design is not None else None
    jobs = []
    have = {}
    for m in moves_:
        f = TRIALS / f"{m}.jsonl"
        have[m] = {json.loads(l)["seed"] for l in f.open()} if f.exists() else set()
    jobs = [(m, seed0 + k, dj) for k in range(n) for m in moves_ if seed0 + k not in have[m]]  # seed-major: every move fills evenly
    print(f"{len(jobs)} trials on {workers} workers", flush=True)
    with mp.get_context("spawn").Pool(workers) as pool:
        for k, r in enumerate(pool.imap_unordered(trial, jobs)):
            with (TRIALS / f"{r['move']}.jsonl").open("a") as f:
                f.write(json.dumps(r, default=float) + "\n")
            if k % 10 == 0:
                print(f"{k + 1}/{len(jobs)} {r['move']} {r['class']} {r.get('wall_s')}s", flush=True)


def summarise(move):
    rows = [json.loads(l) for l in (TRIALS / f"{move}.jsonl").open()]
    n = len(rows)
    ok = sum(r["ok"] for r in rows)
    harm_n = sum(r["harmful"] for r in rows)
    rec = sum((not r["ok"]) and (not r["harmful"]) and r["class"] in RECOVERABLE for r in rows)
    p1 = ok / n
    # with one retry from a recoverable failure, assuming the retry succeeds at the single-try rate
    p2 = p1 + (rec / n) * p1
    classes = {}
    for r in rows:
        classes[r["class"]] = classes.get(r["class"], 0) + 1
    lo, hi = wilson_interval(ok, n)
    hl, hh = wilson_interval(harm_n, n)
    e = [r.get("energy_J", 0) for r in rows if r.get("energy_J") is not None]
    t = [r.get("sim_s", 0) for r in rows if r.get("sim_s") is not None]
    return {"move": move, "n": n, "single_try": round(p1, 4), "single_try_ci95": [round(lo, 4), round(hi, 4)],
            "with_retry": round(p2, 4), "harmful": round(harm_n / n, 4), "harmful_ci95": [round(hl, 4), round(hh, 4)],
            "classes": dict(sorted(classes.items(), key=lambda kv: -kv[1])),
            "energy_J_mean": round(float(np.mean(e)), 3) if e else None, "time_s_mean": round(float(np.mean(t)), 2) if t else None}


def write_results(tag="chosen"):
    d = chosen()
    rows = [summarise(m) for m in MOVES if (TRIALS / f"{m}.jsonl").exists()]
    worst = min(rows, key=lambda r: r["single_try"])
    with (OUT / "results.jsonl").open("a") as f:
        for r in rows:
            r.update({"design": tag, "L_mm": d.L * 1e3, "joint_route": d.joint_route, "storage": d.storage,
                      "worst_behaviour_success": worst["single_try"], "worst_behaviour": worst["move"],
                      "time": time.strftime("%Y-%m-%d %H:%M")})
            f.write(json.dumps(r) + "\n")
    return rows


ACTIONS = {  # grid action -> Monte Carlo moves pooled for it
    "TIP_BOOK": ["tip_book"],
    "TIP_KICK": ["tip_kick"],
    "YAW": ["yaw"],
    "DOCK": ["dock"],
    "CLIMB": ["climb_row", "climb_top"],
    "ROLL_TOP": ["roll_top"],
    "DESCEND": ["book_descend"],
    "SPLIT": ["split_pair", "split_row"],
    "SELF_RIGHT": ["self_right"],
    "FALL_20CM": ["fall"],
}


def write_outcomes():
    """outcomes.json: per grid action, outcome class probabilities (with Wilson intervals), the harm
    rate, mean energy and time; BOND (latch closes after a move) and RELEASE (bolts open) pooled
    from every structure move; TURN_JOINT from the machines' Monte Carlo."""
    out = {"design": asdict(chosen()), "convention": "class probabilities per attempt; 'reached' is success; harmful = a neighbour moved > 0.1 L or turned > 10 deg; with_retry assumes one retry from a recoverable failure (stayed, unlatched, bolt_stall) succeeds at the single-try rate", "actions": {}}
    for act, ms in ACTIONS.items():
        rows = []
        for m in ms:
            f = TRIALS / f"{m}.jsonl"
            if f.exists():
                rows += [json.loads(l) for l in f.open()]
        if not rows:
            continue
        n = len(rows)
        cls = {}
        for r in rows:
            cls[r["class"]] = cls.get(r["class"], 0) + 1
        ok = sum(r["ok"] for r in rows)
        harm_n = sum(r["harmful"] for r in rows)
        rec = sum((not r["ok"]) and (not r["harmful"]) and r["class"] in RECOVERABLE for r in rows)
        out["actions"][act] = {
            "moves": ms, "n": n, "p_success": round(ok / n, 4), "ci95": [round(v, 4) for v in wilson_interval(ok, n)],
            "p_success_with_retry": round(ok / n + rec / n * ok / n, 4), "p_harmful": round(harm_n / n, 4), "harm_ci95": [round(v, 4) for v in wilson_interval(harm_n, n)],
            "classes": {k: round(v / n, 4) for k, v in sorted(cls.items(), key=lambda kv: -kv[1])},
            "energy_J": round(float(np.mean([r.get("energy_J", 0) or 0 for r in rows])), 3), "time_s": round(float(np.mean([r.get("sim_s", 0) or 0 for r in rows])), 2)}
    lat = [x for m in ("dock", "climb_row", "climb_top", "roll_top", "book_descend") if (TRIALS / f"{m}.jsonl").exists() for r in (json.loads(l) for l in (TRIALS / f"{m}.jsonl").open()) if r.get("latched") for x in r["latched"]]
    if lat:
        k = sum(bool(x) for x in lat)
        out["actions"]["BOND"] = {"n": len(lat), "p_success": round(k / len(lat), 4), "ci95": [round(v, 4) for v in wilson_interval(k, len(lat))], "note": "latch closes after a structure move or dock (pooled)"}
    rel = [r for m in ("climb_row", "climb_top", "roll_top", "book_descend", "split_pair", "split_row") if (TRIALS / f"{m}.jsonl").exists() for r in (json.loads(l) for l in (TRIALS / f"{m}.jsonl").open())]
    if rel:
        k = sum(r["class"] != "bolt_stall" for r in rel)
        out["actions"]["RELEASE"] = {"n": len(rel), "p_success": round(k / len(rel), 4), "ci95": [round(v, 4) for v in wilson_interval(k, len(rel))], "note": "edge bolts open the latch (no stall), pooled over structure moves"}
    mfile = OUT / "machines" / f"mc_{chosen().L * 1e3:.2f}mm.jsonl"
    if mfile.exists():
        mr = [json.loads(l) for l in mfile.open()]
        k = sum(r.get("joint_broken") is None and "error" not in r for r in mr)
        out["actions"]["TURN_JOINT"] = {"n": len(mr), "p_success": round(k / len(mr), 4), "ci95": [round(v, 4) for v in wilson_interval(k, len(mr))], "note": "no tooth broken over whole machine runs (arm, walker, crane, crank, boom), each run many joint moves"}
    (OUT / "outcomes.json").write_text(json.dumps(out, indent=1, default=float))
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run")
    a.add_argument("move")
    a.add_argument("--seed", type=int, default=1)
    a.add_argument("--nominal", action="store_true")
    a = sub.add_parser("mc")
    a.add_argument("--moves", default="all")
    a.add_argument("--n", type=int, default=200)
    a.add_argument("--workers", type=int, default=WORKERS)
    a.add_argument("--seed0", type=int, default=1000)
    sub.add_parser("summary")
    sub.add_parser("outcomes")
    a = sub.add_parser("opt")
    a.add_argument("--pop", type=int, default=48)
    a.add_argument("--gen", type=int, default=40)
    a.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    if args.cmd == "run":
        p = Params(seed=args.seed) if args.nominal else sample_params(np.random.default_rng(args.seed), args.seed)
        t = time.time()
        fn = MOVES.get(args.move) or SLOW[args.move]
        print(json.dumps(fn(chosen(), p), default=float), round(time.time() - t, 1), flush=True)
    elif args.cmd == "mc":
        ms = list(MOVES) if args.moves == "all" else args.moves.split(",")
        run_mc(ms, args.n, min(args.workers, WORKERS), args.seed0)
    elif args.cmd == "opt":
        out = run_opt(args.pop, args.gen, min(args.workers, WORKERS))
        print(len(out["front"]), "front points")
    elif args.cmd == "outcomes":
        print(json.dumps(write_outcomes()["actions"], indent=1))
    elif args.cmd == "summary":
        for r in write_results():
            print(json.dumps(r))


if __name__ == "__main__":
    main()
