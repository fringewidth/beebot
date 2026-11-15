"""MuJoCo coupling: pairwise magnetic force and torque between sources attached to bodies.

Every step: world poses of all sources are rebuilt from the body poses, the wrench on body b from the sources
of body a is evaluated once per unordered pair (direct magpylib evaluation or a lookup table), and the reaction
on a is derived from it so that linear momentum and angular momentum about any point are conserved by
construction. Forces go through data.xfrc_applied (world frame, at the body's centre of mass).

Surface forces at small scale (SurfaceForces): constant adhesion per contact point plus Coulomb friction with
stiction (regularised Stribeck law), applied from MuJoCo contacts as equal and opposite pairs. Set the geoms'
own MuJoCo friction to zero when SurfaceForces is on, or friction is counted twice.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.transform import Rotation as R

from .magnetics import Coil, Magnet, wrench


@dataclass(frozen=True)
class Attached:
    """A magnet or coil fixed in a MuJoCo body. The source's pos and rot are in the body frame."""

    body: str
    source: object  # Magnet | Coil


def rot_z_to(v) -> R:
    """Rotation taking +z onto the unit vector v."""
    v = np.asarray(v, float)
    v = v / np.linalg.norm(v)
    if np.allclose(v, [0, 0, 1]):
        return R.identity()
    if np.allclose(v, [0, 0, -1]):
        return R.from_rotvec([np.pi, 0, 0])
    axis = np.cross([0, 0, 1], v)
    ang = np.arccos(np.clip(v[2], -1, 1))
    return R.from_rotvec(axis / np.linalg.norm(axis) * ang)


# ---------------------------------------------------------------- lookup table for identical cylinder pairs

class CylPairTable:
    """Wrench on a target cylinder magnet from a source cylinder magnet (identical d, h, Br), tabulated.

    Source at origin, axis +z, polarisation +z. Target at (rho, 0, z) with unit axis n = (nx, ny, +-sqrt(..)).
    Table axes: rho, z, nx, ny, per hemisphere sign. Valid for rho <= rho_max, |z| <= z_max and tilt of the target
    axis from +-z of at most tilt_max_deg. Outside, callers must fall back to direct evaluation. Penetration
    (|z| < h with rho < d) is clamped to contact.
    """

    def __init__(self, d: float, h: float, br: float, n_rho: int = 14, n_gap: int = 16, n_tilt: int = 5,
                 rho_max_d: float = 4.5, tilt_max_deg: float = 30.0, mag_mesh=None, z_max_d: float = 3.0):
        self.d, self.h, self.br = d, h, br
        self.mag_mesh = mag_mesh
        self.rho_max = rho_max_d * d
        self.tilt_max = np.sin(np.deg2rad(tilt_max_deg))
        self.z_max = h + z_max_d * d
        rho = np.linspace(0.0, self.rho_max, n_rho)
        gaps = np.geomspace(0.02 * h, z_max_d * d, n_gap)
        inner = np.linspace(-h, h, 7)[1:-1]  # side-by-side region for rho > d
        zg = np.unique(np.concatenate([-(h + gaps[::-1]), [-h, h], inner, h + gaps]))
        self.rho, self.z = rho, zg
        self.nxy = np.linspace(-self.tilt_max, self.tilt_max, n_tilt)
        self.vals = {+1: None, -1: None}
        self.build_seconds = 0.0
        self._build()

    def _eval_point(self, rho, z, nx, ny, sgn):
        if rho < self.d and abs(z) < self.h:
            z = np.sign(z if z != 0 else 1.0) * self.h
        nz = sgn * np.sqrt(max(1 - nx * nx - ny * ny, 0.0))
        src = Magnet("cyl", (self.d, self.h), self.br)
        tgt = Magnet("cyl", (self.d, self.h), self.br, (rho, 0.0, z), rot_z_to([nx, ny, nz]))
        F, T = wrench([src], [tgt], pivot=tgt.pos, mag_mesh=self.mag_mesh)
        return np.concatenate([F, T])

    def _build(self):
        import time

        t0 = time.time()
        shape = (len(self.rho), len(self.z), len(self.nxy), len(self.nxy), 6)
        for sgn in (+1, -1):
            arr = np.zeros(shape)
            for a, rho in enumerate(self.rho):
                for b, z in enumerate(self.z):
                    for c, nx in enumerate(self.nxy):
                        for e, ny in enumerate(self.nxy):
                            arr[a, b, c, e] = self._eval_point(rho, z, nx, ny, sgn)
            self.vals[sgn] = arr
        self.interp = {
            s: RegularGridInterpolator((self.rho, self.z, self.nxy, self.nxy), self.vals[s], method="linear")
            for s in (+1, -1)
        }
        self.build_seconds = time.time() - t0

    def in_range(self, rho, z, nx, ny, nzsign) -> np.ndarray:
        return (rho <= self.rho_max) & (np.abs(z) <= self.z_max) & (np.hypot(nx, ny) <= self.tilt_max)

    def query(self, src_pos, src_rot: R, tgt_pos, tgt_rot: R):
        """Wrench on target (about its centre) from source, world frame. Vectorised over leading dim.

        Returns (F (n,3), T (n,3), ok (n,)). Entries with ok False must be evaluated directly.
        """
        src_pos, tgt_pos = np.atleast_2d(src_pos), np.atleast_2d(tgt_pos)
        rel = src_rot.inv().apply(tgt_pos - src_pos)
        n_rel = (src_rot.inv() * tgt_rot).apply([0, 0, 1.0])
        psi = np.arctan2(rel[:, 1], rel[:, 0])
        rho = np.hypot(rel[:, 0], rel[:, 1])
        rz = R.from_rotvec(np.column_stack([np.zeros_like(psi), np.zeros_like(psi), -psi]))
        n_loc = rz.apply(n_rel)
        z = rel[:, 2].copy()
        pen = (rho < self.d) & (np.abs(z) < self.h)
        z[pen] = np.where(z[pen] >= 0, self.h, -self.h)
        sgn = np.where(n_loc[:, 2] >= 0, 1, -1)
        ok = self.in_range(rho, z, n_loc[:, 0], n_loc[:, 1], sgn)
        out = np.zeros((len(rho), 6))
        for s in (+1, -1):
            m = ok & (sgn == s)
            if m.any():
                pts = np.column_stack([rho[m], z[m], n_loc[m, 0], n_loc[m, 1]])
                out[m] = self.interp[s](pts)
        back = src_rot * rz.inv()
        F = back.apply(out[:, :3])
        T = back.apply(out[:, 3:])
        return F, T, ok


# ---------------------------------------------------------------- surface forces

@dataclass(frozen=True)
class SurfaceForces:
    """Adhesion and Coulomb friction with stiction from MuJoCo contacts.

    adhesion: normal pull per contact point (N), always on while a contact exists (dist < dist_on).
    mu_s, mu_k: static and kinetic friction coefficients; v_stick: speed (m/s) below which the static value
    applies (Stribeck width); v_eps: regularisation speed, creep scale in the stuck regime.
    """

    adhesion: float = 0.0
    mu_s: float = 0.5
    mu_k: float = 0.3
    v_stick: float = 1e-3
    v_eps: float = 1e-5
    dist_on: float = 1e-6

    def apply(self, model, data, xfrc):
        """Add surface forces for current data.contact into xfrc (nbody, 6)."""
        f6 = np.zeros(6)
        vel = np.zeros(6)
        for i in range(data.ncon):
            c = data.contact[i]
            if c.dist > self.dist_on:
                continue
            b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
            if b1 == b2:
                continue
            n = c.frame[:3]
            pt = c.pos
            mujoco.mj_contactForce(model, data, i, f6)
            normal = max(f6[0], 0.0) + self.adhesion
            v = []
            for b in (b1, b2):
                if b == 0:
                    v.append(np.zeros(3))
                else:
                    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, b, vel, 0)
                    v.append(vel[3:] + np.cross(vel[:3], pt - data.xipos[b]))
            vrel = v[1] - v[0]
            vt = vrel - np.dot(vrel, n) * n
            sp = np.linalg.norm(vt)
            mu = self.mu_k + (self.mu_s - self.mu_k) * np.exp(-((sp / self.v_stick) ** 2))
            ft = -mu * normal * vt / np.sqrt(sp * sp + self.v_eps**2)  # force on body 2
            fn = -self.adhesion * n  # pulls body 2 toward body 1
            f2 = ft + fn
            if b2 != 0:
                xfrc[b2, :3] += f2
                xfrc[b2, 3:] += np.cross(pt - data.xipos[b2], f2)
            if b1 != 0:
                xfrc[b1, :3] -= f2
                xfrc[b1, 3:] -= np.cross(pt - data.xipos[b1], f2)


def hybrid_wrench(table: CylPairTable, S, T, com, mag_mesh=None, coil_mesh=40, info: dict | None = None):
    """Wrench on target group T (about com) from sources S: table for in-range cylinder pairs, direct otherwise."""
    def ok_type(s):
        return (isinstance(s, Magnet) and s.kind == "cyl" and tuple(s.dim) == (table.d, table.h)
                and s.br == table.br)

    i_s = [i for i, s in enumerate(S) if ok_type(s)]
    i_t = [i for i, t in enumerate(T) if ok_type(t)]
    F = np.zeros(3)
    Tq = np.zeros(3)
    direct_pairs = []  # (source index, target index) evaluated directly
    if i_s and i_t:
        ii, jj = np.meshgrid(i_s, i_t, indexing="ij")
        ii, jj = ii.ravel(), jj.ravel()
        sp = np.array([S[i].pos for i in ii])
        tp = np.array([T[j].pos for j in jj])
        srot = R.from_quat(np.array([S[i].rot.as_quat() for i in ii]))
        trot = R.from_quat(np.array([T[j].rot.as_quat() for j in jj]))
        Fp, Tp, ok = table.query(sp, srot, tp, trot)
        F += Fp[ok].sum(0)
        Tq += Tp[ok].sum(0) + np.cross(tp[ok] - com, Fp[ok]).sum(0)
        direct_pairs = [(int(i), int(j)) for i, j in zip(ii[~ok], jj[~ok])]
        if info is not None:
            info['table_fraction'] = float(ok.mean())
    for i, j in direct_pairs:
        f, t_ = wrench([S[i]], [T[j]], pivot=com, mag_mesh=mag_mesh, coil_mesh=coil_mesh)
        F += f
        Tq += t_
    covered = set((i, j) for i in i_s for j in i_t)
    rest_s = [i for i in range(len(S)) if i not in set(i_s)]
    rest_t = [j for j in range(len(T)) if j not in set(i_t)]
    if rest_s:
        f, t_ = wrench([S[i] for i in rest_s], T, pivot=com, mag_mesh=mag_mesh, coil_mesh=coil_mesh)
        F += f
        Tq += t_
    if rest_t and i_s:
        f, t_ = wrench([S[i] for i in i_s], [T[j] for j in rest_t], pivot=com, mag_mesh=mag_mesh, coil_mesh=coil_mesh)
        F += f
        Tq += t_
    return F, Tq


# ---------------------------------------------------------------- coupling

class MagneticCoupling:
    """Applies pairwise magnetic wrenches for sources attached to bodies. mode 'direct' or 'lookup'."""

    def __init__(self, model, data, attached: list, mode: str = "direct", table: CylPairTable | None = None,
                 mag_mesh=None, coil_mesh: int = 40, surface: SurfaceForces | None = None,
                 min_pair_distance: float = 0.0):
        self.model, self.data = model, data
        self.attached = attached
        self.mode = mode
        self.table = table
        self.mag_mesh = mag_mesh
        self.coil_mesh = coil_mesh
        self.surface = surface
        self.bodies = sorted({mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, a.body) for a in attached})
        self.by_body = {b: [a for a in attached if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, a.body) == b]
                        for b in self.bodies}
        self.last_wrenches = {}
        self.eval_count = 0

    # world pose of a source attached to body b
    def _world(self, b, src):
        pos_b = self.data.xpos[b]
        rot_b = R.from_matrix(self.data.xmat[b].reshape(3, 3))
        wpos = pos_b + rot_b.apply(np.asarray(src.pos))
        wrot = rot_b * src.rot
        return src.with_pose(wpos, wrot)

    def world_sources(self, b):
        return [self._world(b, a.source) for a in self.by_body[b]]

    def _tab_ok(self, s) -> bool:
        tb = self.table
        return (isinstance(s, Magnet) and s.kind == "cyl" and tuple(s.dim) == (tb.d, tb.h) and s.br == tb.br)

    def wrench_on(self, b_src, b_tgt):
        """Force (3), torque about COM of b_tgt (3) on body b_tgt from the sources of b_src."""
        S = self.world_sources(b_src)
        T = self.world_sources(b_tgt)
        com = self.data.xipos[b_tgt]
        self.eval_count += 1
        if self.mode == "lookup" and self.table is not None:
            return hybrid_wrench(self.table, S, T, com, self.mag_mesh, self.coil_mesh)
        return wrench(S, T, pivot=com, mag_mesh=self.mag_mesh, coil_mesh=self.coil_mesh)

    def apply(self):
        d = self.data
        d.xfrc_applied[:] = 0.0
        self.last_wrenches = {}
        for ia, a in enumerate(self.bodies):
            for b in self.bodies[ia + 1:]:
                Fb, Tb = self.wrench_on(a, b)
                Fa = -Fb
                Ta = -Tb - np.cross(d.xipos[b] - d.xipos[a], Fb)
                d.xfrc_applied[b, :3] += Fb
                d.xfrc_applied[b, 3:] += Tb
                d.xfrc_applied[a, :3] += Fa
                d.xfrc_applied[a, 3:] += Ta
                self.last_wrenches[(a, b)] = (Fb, Tb)
        if self.surface is not None:
            self.surface.apply(self.model, d, d.xfrc_applied)

    def step(self, n: int = 1):
        for _ in range(n):
            self.apply()
            mujoco.mj_step(self.model, self.data)


def total_momentum(model, data):
    """Linear momentum (3) and angular momentum about the system centre of mass (3) of all free bodies."""
    mass = model.body_mass
    p = np.zeros(3)
    com = np.zeros(3)
    for b in range(1, model.nbody):
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, b, vel := np.zeros(6), 0)
        p += mass[b] * vel[3:]
        com += mass[b] * data.xipos[b]
    com /= mass[1:].sum()
    L = np.zeros(3)
    for b in range(1, model.nbody):
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, b, vel := np.zeros(6), 0)
        I = data.ximat[b].reshape(3, 3) @ np.diag(model.body_inertia[b]) @ data.ximat[b].reshape(3, 3).T
        L += mass[b] * np.cross(data.xipos[b] - com, vel[3:]) + I @ vel[:3]
    return p, L
