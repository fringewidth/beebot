"""Fixed face magnets of the cube (S = 60 mm): layout, exact forces, fields at the AS5600s.

Every number of the layout lives in params.json["magnets"]; change it there (one edit) and the
test (hardware/tests/test_cube50_magnets.py), the sizing script (magnet_study.py) and the MuJoCo
pivots (cube50_sim.py magnets) all follow.

Frames. The cube frame is cube50's: discs on +-z, hinges on the +x edges. Each face has an outward
normal n and an in-plane basis (e1, e2) with e1 x e2 = n, so angles are counted counterclockwise as
seen from outside the face. A site at angle phi with pole +1 has N out (magnetisation along +n).

Force model (exact for rigid, uniformly magnetised cylinders): the magnetic charge model. A uniform
magnetisation M has no volume charge, only surface charge sigma = M . n_surface (the end caps of an
axial magnet, the curved side of a diametric one). The force on a target magnet is the integral of
sigma * B_source over its charged surfaces, the torque the integral of r x (sigma B dA). B_source is
magpylib's closed-form field of a uniformly magnetised cylinder (exact everywhere outside the
source). The only approximation is the quadrature over the target's surfaces (Gauss-Legendre in
radius or height, uniform in angle); its error is checked against magpylib.getFT and against a finer
quadrature in the test (under 0.5 % at contact). No dipole approximation anywhere.

Not modelled: recoil permeability (NdFeB mu_r about 1.05 is taken as 1), soft-magnetic parts (steel
gears, bearing, screws, motor cans and the motors' own magnets), temperature, eddy currents.
"""

import json
from functools import lru_cache
from pathlib import Path

import magpylib
import numpy as np
from magpylib.func import cylinder_field
from scipy.spatial.transform import Rotation

PARAMS = Path(__file__).parent / "params.json"
MU0 = 4e-7 * np.pi


def layout():
    """The "magnets" section of params.json (lengths in mm there, returned in m where named _m)."""
    p = json.loads(PARAMS.read_text())["magnets"]
    lay = p["layout"]
    return {
        "R": lay["radius_mm"] / 1000,
        "delta": lay["delta_deg"],
        "D": p["part"]["diameter_mm"] / 1000,
        "T": p["part"]["thickness_mm"] / 1000,
        "skin": lay["skin_mm"] / 1000,
        "Br": p["part"]["Br_T"] * p["force_model"]["scale"],
        "half": p["cube_size_mm"] / 2000,
        "poles": lay["poles"],
        "encoders": p["encoders"],
    }


def sites(delta, poles=None):
    """[(angle deg, pole)] of the 8 sites: +delta + 90k with pole a, -delta + 90k with pole -a."""
    a = 1 if poles is None or poles.get("plus_delta", "N") == "N" else -1
    return [(delta + 90 * k, a) for k in range(4)] + [(-delta + 90 * k, -a) for k in range(4)]


def face_frames():
    """{name: (n, e1, e2)} with e1 x e2 = n for the six faces."""
    out = {}
    eye = np.eye(3)
    for k, ax in enumerate("xyz"):
        for s in (1, -1):
            n, e1, e2 = s * eye[k], eye[(k + 1) % 3], s * eye[(k + 2) % 3]
            assert np.allclose(np.cross(e1, e2), n)
            out[("+" if s > 0 else "-") + ax] = (n, e1, e2)
    return out


FACES = face_frames()
DISC_FACES = ("+z", "-z")


def face_magnets(lay, face, disc_angle=0.0, half=None):
    """Centres (8, 3), magnetisation unit vectors (8, 3) of one face's magnets in the cube frame."""
    n, e1, e2 = FACES[face]
    half = lay["half"] if half is None else half
    depth = half - lay["skin"] - lay["T"] / 2
    pos, mag = [], []
    for phi, pole in sites(lay["delta"], lay["poles"]):
        a = np.radians(phi + disc_angle)
        pos.append(depth * n + lay["R"] * (np.cos(a) * e1 + np.sin(a) * e2))
        mag.append(pole * n)
    return np.array(pos), np.array(mag)


def cube_magnets(lay, disc_angles=None, faces=None):
    """All face magnets of one cube (48 by default): centres, magnetisation units, face labels."""
    disc_angles = disc_angles or {}
    P, M, F = [], [], []
    for face in faces or FACES:
        p, m = face_magnets(lay, face, disc_angles.get(face, 0.0))
        P.append(p)
        M.append(m)
        F += [face] * len(p)
    return np.concatenate(P), np.concatenate(M), F


# ------------------------------------------------------------------ magnets as arrays
class Magnets:
    """A rigid set of cylinders in one body frame: centres c (n, 3), magnetisation units u (n, 3),
    kind "axial" (u along the cylinder axis) or "diam" (u across it; axis a), D, T, Br per magnet."""

    def __init__(self, c, u, D, T, Br, axis=None):
        self.c = np.atleast_2d(np.asarray(c, float))
        self.u = np.atleast_2d(np.asarray(u, float))
        n = len(self.c)
        self.D = np.broadcast_to(np.asarray(D, float), (n,)).copy()
        self.T = np.broadcast_to(np.asarray(T, float), (n,)).copy()
        self.Br = np.broadcast_to(np.asarray(Br, float), (n,)).copy()
        self.axis = self.u.copy() if axis is None else np.atleast_2d(np.asarray(axis, float))
        self.diam = np.abs(np.einsum("ij,ij->i", self.axis, self.u)) < 0.5
        # rotation local -> body: local z = cylinder axis, local x = magnetisation if diametric
        self.rot = []
        for a, u, d in zip(self.axis, self.u, self.diam):
            x = u if d else _any_perp(a)
            self.rot.append(np.column_stack([x, np.cross(a, x), a]))
        self.rot = np.array(self.rot)

    def __len__(self):
        return len(self.c)

    def __add__(self, other):
        out = Magnets.__new__(Magnets)
        for k in ("c", "u", "D", "T", "Br", "axis", "diam", "rot"):
            setattr(out, k, np.concatenate([getattr(self, k), getattr(other, k)]))
        return out

    def subset(self, idx):
        out = Magnets.__new__(Magnets)
        for k in ("c", "u", "D", "T", "Br", "axis", "diam", "rot"):
            setattr(out, k, getattr(self, k)[idx])
        return out

    def moved(self, R, x):
        """The same magnets in another frame: body pose (R, x)."""
        out = Magnets.__new__(Magnets)
        out.c = self.c @ R.T + x
        out.u = self.u @ R.T
        out.axis = self.axis @ R.T
        out.rot = np.einsum("ij,njk->nik", R, self.rot)
        for k in ("D", "T", "Br", "diam"):
            setattr(out, k, getattr(self, k))
        return out


def _any_perp(a):
    t = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    p = np.cross(a, t)
    return p / np.linalg.norm(p)


@lru_cache(maxsize=16)
def _cap_rule(nr, nphi):
    """Points (q, 2) and weights (q,) on the unit disc (weights sum to pi)."""
    x, w = np.polynomial.legendre.leggauss(nr)
    r = (x + 1) / 2  # on [0, 1]
    wr = w / 2 * r  # r dr
    phi = (np.arange(nphi) + 0.5) * 2 * np.pi / nphi
    pts = np.array([[ri * np.cos(p), ri * np.sin(p)] for ri in r for p in phi])
    wts = np.array([wi * 2 * np.pi / nphi for wi in wr for _ in phi])
    return pts, wts


@lru_cache(maxsize=16)
def _side_rule(nh, nphi):
    """Points (q, 2) as (phi, h in [-1/2, 1/2]) and weights (q,) for the curved side (dphi dh)."""
    x, w = np.polynomial.legendre.leggauss(nh)
    phi = (np.arange(nphi) + 0.5) * 2 * np.pi / nphi
    pts = np.array([[p, xi / 2] for xi in x for p in phi])
    wts = np.array([wi / 2 * 2 * np.pi / nphi for wi in w for _ in phi])
    return pts, wts


def charges(mag, fine=(6, 24)):
    """Surface charge quadrature of each magnet in its body frame.
    Returns points (n, q, 3) and charge weights sigma dA (n, q) in A m (sigma = Br / mu0 . n)."""
    n = len(mag)
    nr, nphi = fine
    cap_p, cap_w = _cap_rule(nr, nphi)
    side_p, side_w = _side_rule(nr, nphi)
    q = max(2 * len(cap_w), len(side_w))
    P = np.zeros((n, q, 3))
    W = np.zeros((n, q))
    for i in range(n):
        a_ = mag.D[i] / 2
        h = mag.T[i]
        M = mag.Br[i] / MU0
        R = mag.rot[i]
        if not mag.diam[i]:
            loc = np.concatenate([
                np.column_stack([a_ * cap_p, np.full(len(cap_p), h / 2)]),
                np.column_stack([a_ * cap_p, np.full(len(cap_p), -h / 2)]),
            ])
            wt = np.concatenate([M * a_ * a_ * cap_w, -M * a_ * a_ * cap_w])
        else:  # magnetisation along local x: sigma = M cos(phi) on the side
            phi, hh = side_p[:, 0], side_p[:, 1] * h
            loc = np.column_stack([a_ * np.cos(phi), a_ * np.sin(phi), hh])
            wt = M * np.cos(phi) * a_ * h * side_w
        P[i, : len(wt)] = loc @ R.T + mag.c[i]
        W[i, : len(wt)] = wt
    return P, W


def field(src, obs):
    """Exact B (T) at observer points obs (m, 3) from all magnets in src, summed."""
    obs = np.atleast_2d(obs)
    return field_pairs(src, np.arange(len(src)).repeat(len(obs)), np.tile(obs, (len(src), 1))).reshape(len(src), len(obs), 3).sum(0)


def field_pairs(src, si, pts):
    """B at pts (k, 3) from source magnet si (k,) each: one source per row (vectorised)."""
    if len(si) == 0:
        return np.zeros((0, 3))
    R = src.rot[si]
    loc = np.einsum("kji,kj->ki", R, pts - src.c[si])  # R^T (p - c)
    pol = np.zeros((len(si), 3))
    pol[:, 2] = np.where(src.diam[si], 0.0, src.Br[si])
    pol[:, 0] = np.where(src.diam[si], src.Br[si], 0.0)
    dim = np.column_stack([src.D[si], src.T[si]])
    B = cylinder_field("B", observers=loc, dimensions=dim, polarizations=pol, squeeze=False)
    return np.einsum("kij,kj->ki", R, B)


def wrench(src, tgt, pivot, cutoff=0.045, fine=(6, 24), coarse=(3, 12), near=0.010, tgt_q=None):
    """Force (N) and torque (N m, about `pivot`) on all magnets of `tgt` from all magnets of `src`,
    both in the same frame. Pairs with centres further than `cutoff` apart are left out (each such
    pair is below 1e-4 of a contact pair's force; see the test). Pairs closer than `near` use the
    fine quadrature, the rest the coarse one."""
    d = np.linalg.norm(tgt.c[:, None, :] - src.c[None, :, :], axis=2)
    F = np.zeros(3)
    tau = np.zeros(3)
    for rule, mask in ((fine, d < near), (coarse, (d >= near) & (d < cutoff))):
        ti, si = np.nonzero(mask)
        if not len(ti):
            continue
        P, W = charges(tgt, rule) if tgt_q is None else tgt_q[rule]
        q = P.shape[1]
        pts = P[ti].reshape(-1, 3)
        B = field_pairs(src, np.repeat(si, q), pts)
        dF = B * W[ti].reshape(-1)[:, None]
        F += dF.sum(0)
        tau += np.cross(pts - pivot, dF).sum(0)
    return F, tau


def wrench_ft(src, tgt, pivot, meshing=200):
    """Cross-check with magpylib.getFT (dipole-meshed target, finite differences)."""
    s = [_object(src, i) for i in range(len(src))]
    t = magpylib.Collection(*[_object(tgt, i) for i in range(len(tgt))])
    FT = np.asarray(magpylib.getFT(s, t, pivot=np.asarray(pivot), meshing=meshing, squeeze=False))  # (F/T, source, path, target, 3)
    return FT[0].sum(axis=(0, 1, 2)), FT[1].sum(axis=(0, 1, 2))


def _object(mag, i):
    pol = (mag.Br[i], 0, 0) if mag.diam[i] else (0, 0, mag.Br[i])
    return magpylib.magnet.Cylinder(position=mag.c[i], orientation=Rotation.from_matrix(mag.rot[i]), dimension=(mag.D[i], mag.T[i]), polarization=pol)


# ------------------------------------------------------------------ whole cubes
def cube_set(lay=None, disc_angles=None, encoders=True, shaft_angles=None):
    """Magnets of one cube in the cube frame: 48 face magnets, then the 4 AS5600 magnets."""
    lay = lay or layout()
    c, u, _ = cube_magnets(lay, disc_angles)
    mags = Magnets(c, u, lay["D"], lay["T"], lay["Br"])
    if encoders:
        mags = mags + encoder_magnets(lay, disc_angles, shaft_angles)
    return mags


def encoder_magnets(lay, disc_angles=None, shaft_angles=None):
    """The four diametric AS5600 magnets: top and bottom disc shafts (turn with their discs), top
    and bottom hinge motor gears. Positions from params.json["magnets"]["encoders"]."""
    disc_angles = disc_angles or {}
    shaft_angles = shaft_angles or {}
    C, U, A, D, T, Br = [], [], [], [], [], []
    for e in lay["encoders"]["magnets"]:
        a = np.array(e["axis"], float)
        ang = shaft_angles.get(e["name"], disc_angles.get(e.get("turns_with", ""), 0.0))
        x0 = np.array(e["magnetisation_at_zero"], float)
        u = Rotation.from_rotvec(np.radians(ang) * a).apply(x0)
        C.append(np.array(e["centre_mm"]) / 1000)
        U.append(u)
        A.append(a)
        D.append(e["diameter_mm"] / 1000)
        T.append(e["thickness_mm"] / 1000)
        Br.append(e["Br_T"])
    return Magnets(np.array(C), np.array(U), D, T, Br, axis=np.array(A))


def chips(lay=None):
    """{name: (die centre m, die normal, its magnet's name)} for the four AS5600s."""
    lay = lay or layout()
    return {c["name"]: (np.array(c["die_centre_mm"]) / 1000, np.array(c["normal"], float), c["magnet"]) for c in lay["encoders"]["chips"]}


def as5600_error(B_fn, die, normal, ring=0.001, n=32):
    """AS5600 reading: its Hall plates sit on a 1 mm radius circle in the die plane and see Bz
    (along `normal`); the angle comes from the first harmonic of Bz round the circle, so a uniform
    field cancels. Returns (first-harmonic amplitude T, phase rad, mean Bz T) of B_fn on that ring."""
    e1 = _any_perp(normal)
    e2 = np.cross(normal, e1)
    phi = np.arange(n) * 2 * np.pi / n
    pts = die + ring * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2)
    bz = B_fn(pts) @ normal
    c1 = 2 / n * (bz * np.exp(-1j * phi)).sum()
    return abs(c1), float(np.angle(c1)), float(bz.mean())


# ------------------------------------------------------------------ fast rigid bodies (for MuJoCo)
class Body:
    """A rigid magnet set with its surface charges cached in the body frame, for repeated wrenches
    at changing poses. Quadrature: `fine` for pairs closer than `near`, `coarse` beyond, both checked
    in the test against the reference rule."""

    def __init__(self, mags, fine=(4, 12), coarse=(2, 6), near=0.010, cutoff=0.045):
        self.mags = mags
        self.rules = {"fine": fine, "coarse": coarse}
        self.q = {k: charges(mags, r) for k, r in self.rules.items()}
        self.near, self.cutoff = near, cutoff

    def wrench_from(self, src, R, x, pivot):
        """Force and torque (about `pivot`, world) on this body at pose (R, x) from the magnets
        `src` (a Magnets already in the world frame)."""
        tgt_c = self.mags.c @ R.T + x
        d = np.linalg.norm(tgt_c[:, None, :] - src.c[None, :, :], axis=2)
        F = np.zeros(3)
        tau = np.zeros(3)
        for key, mask in (("fine", d < self.near), ("coarse", (d >= self.near) & (d < self.cutoff))):
            ti, si = np.nonzero(mask)
            if not len(ti):
                continue
            P, W = self.q[key]
            q = P.shape[1]
            pts = (P[ti] @ R.T + x).reshape(-1, 3)
            B = field_pairs(src, np.repeat(si, q), pts)
            dF = B * W[ti].reshape(-1)[:, None]
            F += dF.sum(0)
            tau += np.cross(pts - pivot, dF).sum(0)
        return F, tau
