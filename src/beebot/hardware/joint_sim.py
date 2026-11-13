"""MuJoCo ground truth for joint_strength: ramp the load on a cube structure until it gives.

Cubes are cube_swarm's (cube_swarm.DESIGN: 5 cm, 150 g, its magnets, contacts and steel floor),
as plain boxes: no flywheel (the hinge cube's motor and latch are idle in a standing structure and
their mass is part of the 150 g). Anchored cubes are welded to the world (a wall in a test); their
magnets still pull. Cubes start exactly in their cells, so every face is closed.

The load factor rises linearly from `start` at `rate` per second (quasi-static: the cantilever's
natural period is milliseconds): in mode "gravity" gravity is factor x g and the extra load is
factor x load; in mode "live" gravity stays at g and the extra load is factor x load. The
structure fails at the first factor at which a face joint has opened `opening` L at a corner (at 0.02 L, 1 mm, the magnets there pull 0.6 of
their peak and the joint cannot close again under load), slid `slip` L, or a cube has moved `fall`
L (rigid motion with no joint opening, e.g. a whole tower sliding). The load is then held there for `develop` s so the collapse shows; the
cubes that by then moved 0.1 L or turned 5 degrees are the ones that went. Returns that factor
(None if it held up to `stop`), those cubes and the jitter drawn (magnet strength and mass per
cube, free cubes first then anchored ones; friction), so the check can be fed the same cubes.

    from beebot.hardware.joint_sim import ramp
    ramp([(1, 0, x) for x in range(7)], anchored=[(1, 0, -1), (0, 0, -1)])
"""

import math

import mujoco
import numpy as np

from beebot.hardware.cube import DT, G, Jitter, NOMINAL
from beebot.hardware.cube_swarm import DESIGN, SwarmMagnets


def scene(cube, xyz, free, masses, friction):
    h = cube.side / 2
    bodies = []
    for i, (c, m) in enumerate(zip(xyz, masses)):
        joint = f'<freejoint name="c{i}"/>' if free[i] else ""
        bodies.append(f'<body name="c{i}" pos="{c[0]} {c[1]} {c[2]}">{joint}<geom type="box" size="{h} {h} {h}" mass="{m}"/></body>')
    return f"""<mujoco>
  <option timestep="{DT}" integrator="implicitfast" cone="elliptic" impratio="10" gravity="0 0 {-G}"/>
  <default><geom friction="{friction} 0.005 0.0001" condim="4" solref="0.002 1" solimp="0.95 0.99 0.0005"/></default>
  <worldbody><geom type="plane" size="3 3 0.1"/>{''.join(bodies)}</worldbody>
</mujoco>"""


def face_pairs(every, n):
    """Face contacts to watch: (i, j or -1 for the floor, local normal from i toward j (3,), the
    face's four corners in i's frame (4, 3)), for pairs with at least one free cube (index < n)."""
    index = {tuple(c): k for k, c in enumerate(every)}
    I, J, N, C = [], [], [], []
    for i, c in enumerate(every):
        for axis in range(3):
            for sign in (-1, 1):
                step = np.zeros(3, dtype=int)
                step[2 - axis] = sign
                other = tuple(c + step)
                floor = c[0] == 0 and axis == 2 and sign == -1
                j = index.get(other, -1 if floor else None)
                if j is None or (j >= 0 and (j < i or (i >= n and j >= n))) or (j == -1 and i >= n):
                    continue
                nrm = np.zeros(3)
                nrm[axis] = sign
                t1 = np.eye(3)[(axis + 1) % 3]
                t2 = np.eye(3)[(axis + 2) % 3]
                I.append(i)
                J.append(j)
                N.append(nrm)
                C.append([0.5 * nrm + a * 0.5 * t1 + b * 0.5 * t2 for a in (-1, 1) for b in (-1, 1)])
    return np.array(I, dtype=int), np.array(J, dtype=int), np.array(N).reshape(-1, 3), np.array(C).reshape(-1, 4, 3)


class Watch:
    """Corner points of the watched faces, fixed in both cubes (the floor for j = -1) at the start;
    opening (L) is how far a corner of i has come off j along the face normal, slide (L) how far
    it has moved along the face."""

    def __init__(self, pairs, xpos, xmat, side):
        self.I, self.J, N, C = pairs
        self.side = side
        self.N = N
        pi = xpos[self.I][:, None, :] + np.einsum("pij,pkj->pki", xmat[self.I], C * side)
        self.a = C * side
        Jc = np.where(self.J < 0, 0, self.J)
        xj = np.where((self.J < 0)[:, None], 0.0, xpos[Jc])
        Rj = np.where((self.J < 0)[:, None, None], np.eye(3), xmat[Jc])
        self.b = np.einsum("pji,pkj->pki", Rj, pi - xj[:, None, :])

    def state(self, xpos, xmat):
        Jc = np.where(self.J < 0, 0, self.J)
        xj = np.where((self.J < 0)[:, None], 0.0, xpos[Jc])
        Rj = np.where((self.J < 0)[:, None, None], np.eye(3), xmat[Jc])
        pi = xpos[self.I][:, None, :] + np.einsum("pij,pkj->pki", xmat[self.I], self.a)
        pj = xj[:, None, :] + np.einsum("pij,pkj->pki", Rj, self.b)
        d = pi - pj
        n = np.einsum("pij,pj->pi", xmat[self.I], self.N)[:, None, :]
        along = (d * n).sum(-1)
        opening = -along / self.side
        slide = np.linalg.norm(d - along[..., None] * n, axis=-1) / self.side
        return opening.max(1), slide.max(1)


def ramp(cells, anchored=(), load=None, mode="gravity", seed=None, jitter=Jitter(), start=0.05, stop=2.5, rate=0.25, opening=0.02, slip=0.05, fall=0.25, hold=0.2, develop=0.3, cube=None):
    """cells, anchored: grid (z, y, x). load: (n, 3) physical force per free cube in module
    weights. seed None: nominal cubes (no jitter, friction 0.5); else jittered as cube_swarm
    (magnet strength and mass per cube, friction per scene). Returns a dict."""
    cube = cube or DESIGN.cube()
    cells = np.asarray(cells, dtype=int).reshape(-1, 3)
    anchored = np.asarray(anchored, dtype=int).reshape(-1, 3)
    every = np.concatenate([cells, anchored])
    n, total = len(cells), len(every)
    jit = NOMINAL if seed is None else jitter
    rng = np.random.default_rng(seed if seed is not None else 0)
    lo, hi = jit.friction
    friction = float(rng.uniform(lo, hi)) if hi > lo else lo
    masses = cube.mass * (1 + rng.uniform(-jit.mass, jit.mass, size=total) if jit.mass > 0 else np.ones(total))
    strength = 1 + rng.uniform(-jit.magnet, jit.magnet, size=total) if jit.magnet > 0 else np.ones(total)
    xyz = (every[:, ::-1] + [0, 0, 0.5]) * cube.side
    model = mujoco.MjModel.from_xml_string(scene(cube, xyz, [True] * n + [False] * len(anchored), masses, friction))
    data = mujoco.MjData(model)
    magnets = SwarmMagnets(cube, total, strength)
    bodies = np.array([model.body(f"c{i}").id for i in range(total)])
    extra = np.zeros((total, 3))
    if load is not None:
        extra[:n] = np.asarray(load, dtype=float).reshape(n, 3) * cube.mass * G
    mujoco.mj_forward(model, data)
    x0 = data.xpos[bodies[:n]].copy()
    r0 = data.xmat[bodies[:n]].reshape(n, 3, 3).copy()
    watch = Watch(face_pairs(every, n), data.xpos[bodies], data.xmat[bodies].reshape(total, 3, 3), cube.side)
    t, factor = 0.0, start
    failed = None
    why = None
    check_every = 25
    step = 0
    hold_at = None
    while factor <= stop or failed is not None:
        factor = start + rate * max(0.0, t - hold) if failed is None else hold_at
        g_scale = factor if mode == "gravity" else 1.0
        l_scale = factor
        model.opt.gravity[2] = -G * g_scale
        wrench = magnets.wrench(data.xpos[bodies], data.xmat[bodies].reshape(total, 3, 3))
        wrench[:, :3] += extra * l_scale
        data.xfrc_applied[bodies] = wrench
        mujoco.mj_step(model, data)
        t += DT
        step += 1
        if step % check_every == 0 and failed is None:
            moved = np.abs(data.xpos[bodies[:n]] - x0).max(1) / cube.side
            gap, slid = watch.state(data.xpos[bodies], data.xmat[bodies].reshape(total, 3, 3))
            flags = {"opened": gap.max() > opening, "slid": slid.max() > slip, "fell": moved.max() > fall}
            if any(flags.values()):
                failed, after = factor, t
                hold_at = factor
                why = [k for k, v in flags.items() if v]
                worst = int(np.argmax(gap)) if flags["opened"] else int(np.argmax(slid))
                first = [every[watch.I[worst]].tolist(), every[watch.J[worst]].tolist() if watch.J[worst] >= 0 else "floor"]
        if failed is not None:
            factor = hold_at  # let the collapse develop at the failing load for `develop` s
            if t - after > develop:
                break
    moved = np.abs(data.xpos[bodies[:n]] - x0).max(1) / cube.side
    rot = data.xmat[bodies[:n]].reshape(n, 3, 3)
    turned = (np.einsum("nij,nij->n", rot, r0) - 1) / 2 < math.cos(math.radians(5.0))
    return {
        "factor": failed,
        "why": why,
        "first_joint": first if failed is not None else None,
        "moved": cells[(moved > 0.1) | turned].tolist(),
        "strength": strength.tolist(),
        "masses": (masses / cube.mass).tolist(),
        "friction": round(friction, 3),
        "sim_seconds": round(t, 3),
    }
