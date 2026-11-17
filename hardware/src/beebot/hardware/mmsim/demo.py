"""Two 5 mm cubes with edge magnets in MuJoCo: pulled together, seat flush. Also a momentum test model."""
from __future__ import annotations

import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from .magnetics import Coil, Magnet
from .mjcoupling import Attached, CylPairTable, MagneticCoupling, SurfaceForces, rot_z_to, total_momentum

CUBE = 5e-3
MAG_D, MAG_H, MAG_BR = 1e-3, 1e-3, 1.45
MASS = 3.0e-4  # kg, cube with magnets and electronics (illustrative)


def cube_xml(name, x, yaw_deg=0.0, y=0.0, free=True, solref="4e-4 1"):
    half = CUBE / 2
    inertia = MASS * CUBE**2 / 6
    return f"""
    <body name="{name}" pos="{x} {y} {half}" euler="0 0 {yaw_deg}">
      {'<freejoint/>' if free else ''}
      <geom type="box" size="{half} {half} {half}" mass="{MASS}" friction="0 0 0" condim="1"
            solref="{solref}" solimp="0.9 0.95 1e-5"/>
    </body>"""


def build_model(gap=1.5e-3, offset_y=0.3e-3, yaw_deg=2.0, dt=2e-5, gravity=True, floor=True):
    SOLREF = f"{max(2.2 * dt, 1e-5):.3g} 1"
    g = "0 0 -9.81" if gravity else "0 0 0"
    floor_xml = (f'<geom name="floor" type="plane" size="0.1 0.1 0.01" friction="0 0 0" condim="1" '
                 f'solref="{SOLREF}" solimp="0.9 0.95 1e-5"/>') if floor else ""
    xa = 0.0
    xb = CUBE + gap
    xml = f"""<mujoco><option timestep="{dt}" gravity="{g}" integrator="implicitfast"/>
    <worldbody>{floor_xml}{cube_xml("A", xa, solref=SOLREF)}{cube_xml("B", xb, yaw_deg, offset_y, solref=SOLREF)}</worldbody></mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def edge_magnets(face_sign: int, pattern_flip=False):
    """Four magnets near the corners of the +x (face_sign=+1) or -x (face_sign=-1) face, axes along x.

    Polarity checkerboard sign(y z) so that matched faces attract and a 90 degree twist repels.
    """
    out = []
    x = face_sign * (CUBE / 2 - MAG_H / 2)
    off = CUBE / 2 - MAG_D / 2 - 0.3e-3  # 0.3 mm from the edge
    for sy in (-1, 1):
        for sz in (-1, 1):
            pol = sy * sz * (-1 if pattern_flip else 1)
            axis = np.array([pol, 0.0, 0.0])
            out.append(Magnet("cyl", (MAG_D, MAG_H), MAG_BR, (x, sy * off, sz * off), rot_z_to(axis)))
    return out


def attached_pair():
    att = []
    for m in edge_magnets(+1):
        att.append(Attached("A", m))
    for m in edge_magnets(-1):
        att.append(Attached("B", m))
    return att


def run_seating(dt=2e-5, t_end=0.04, mode="direct", table=None, surface=None, gap=1.5e-3, offset_y=0.3e-3,
                yaw_deg=2.0, mag_mesh=None, record_every=50):
    model = build_model(gap, offset_y, yaw_deg, dt)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cpl = MagneticCoupling(model, data, attached_pair(), mode=mode, table=table, surface=surface, mag_mesh=mag_mesh)
    ia = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "A")
    ib = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "B")
    n = int(t_end / dt)
    log = []
    t0 = time.time()
    vmax = 0.0
    for k in range(n):
        cpl.step()
        if k % record_every == 0:
            rel = data.xpos[ib] - data.xpos[ia]
            yaw = R.from_matrix(data.xmat[ib].reshape(3, 3)).inv() * R.from_matrix(data.xmat[ia].reshape(3, 3))
            ang = np.degrees(np.linalg.norm(yaw.as_rotvec()))
            log.append((data.time, rel[0] - CUBE, rel[1], rel[2], ang))
        for b in (ia, ib):
            vmax = max(vmax, float(np.linalg.norm(data.cvel[b][3:])))
    wall = time.time() - t0
    rel = data.xpos[ib] - data.xpos[ia]
    yaw = R.from_matrix(data.xmat[ib].reshape(3, 3)).inv() * R.from_matrix(data.xmat[ia].reshape(3, 3))
    return {
        "dt": dt, "mode": mode, "gap_mm": (rel[0] - CUBE) * 1e3, "lateral_y_mm": rel[1] * 1e3,
        "lateral_z_mm": rel[2] * 1e3, "twist_deg": float(np.degrees(np.linalg.norm(yaw.as_rotvec()))),
        "speed_max": vmax, "wall_s": wall, "steps": n, "log": np.array(log),
        "ms_per_step": wall / n * 1e3, "finite": bool(np.isfinite(data.qpos).all()),
    }


# ---------------------------------------------------------------- momentum test (free space, magnets and a coil)

def run_momentum_test(n_steps=1500, dt=2e-5, mode="direct", table=None):
    model = build_model(gap=1.0e-3, offset_y=0.5e-3, yaw_deg=8.0, dt=dt, gravity=False, floor=False)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    att = attached_pair()
    # a small current coil on cube A's +y face region interacting with B's magnets, to include coil forces
    from .coil import Spiral, spiral_vertices
    sp = Spiral(4, 3e-3, 100e-6, 100e-6, 18e-6)
    v = spiral_vertices(sp)
    coil = Coil(v, 2.0, (CUBE / 2 + 0.05e-3, 0.0, 0.0), rot_z_to([1.0, 0, 0]))
    att.append(Attached("A", coil))
    cpl = MagneticCoupling(model, data, att, mode=mode, table=table)
    p0, L0 = total_momentum(model, data)
    pmax = 0.0
    dp_max = 0.0
    dL_max = 0.0
    ke_max = 0.0
    for _ in range(n_steps):
        cpl.step()
        p, L = total_momentum(model, data)
        dp_max = max(dp_max, float(np.linalg.norm(p - p0)))
        dL_max = max(dL_max, float(np.linalg.norm(L - L0)))
        ia = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "A")
        ib = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "B")
        vv = np.linalg.norm(data.cvel[ia][3:]) * MASS
        pmax = max(pmax, float(vv))
    return {"dp_max": dp_max, "p_scale": pmax, "dL_max": dL_max, "n_steps": n_steps}
