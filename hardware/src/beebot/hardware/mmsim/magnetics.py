"""Analytic magnetics with magpylib: force and torque as functions of pose.

All quantities SI (metres, tesla, ampere, newton). A pose is (position (3,), scipy Rotation).
Magnets are magnetised along their local +z axis (thickness axis for cuboids).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import magpylib as mp
import numpy as np
from scipy.spatial.transform import Rotation as R

MU0 = 4e-7 * np.pi

# Remanence by grade, tesla. Mid-range of the grade standard, chosen before running any
# validation and not adjusted afterwards. K&J quotes Brmax 14800 G for N52 (1.48 T).
BR_BY_GRADE = {"N52": 1.45, "N50": 1.40, "N48": 1.40, "N42": 1.30, "N40": 1.27, "N35": 1.20}


@dataclass(frozen=True)
class Magnet:
    """Permanent magnet. kind 'cyl' dim=(diameter, height); 'cuboid' dim=(a, b, c) with c along magnetisation."""

    kind: str
    dim: tuple
    br: float
    pos: tuple = (0.0, 0.0, 0.0)
    rot: R = field(default_factory=lambda: R.identity())

    def with_pose(self, pos, rot) -> "Magnet":
        return Magnet(self.kind, self.dim, self.br, tuple(np.asarray(pos, float)), rot)

    def volume(self) -> float:
        if self.kind == "cyl":
            return np.pi * (self.dim[0] / 2) ** 2 * self.dim[1]
        return float(np.prod(self.dim))

    def moment_axis(self) -> np.ndarray:
        return self.rot.apply([0, 0, 1.0])


@dataclass(frozen=True)
class Coil:
    """Current-carrying polyline. vertices are local coordinates (n,3); current in amperes (all turns in series)."""

    vertices: np.ndarray
    current: float
    pos: tuple = (0.0, 0.0, 0.0)
    rot: R = field(default_factory=lambda: R.identity())

    def with_pose(self, pos, rot) -> "Coil":
        return Coil(self.vertices, self.current, tuple(np.asarray(pos, float)), rot)


def magnet_obj(m: Magnet):
    kw = dict(polarization=(0, 0, m.br), position=m.pos, orientation=m.rot)
    if m.kind == "cyl":
        return mp.magnet.Cylinder(dimension=m.dim, **kw)
    return mp.magnet.Cuboid(dimension=m.dim, **kw)


def coil_obj(c: Coil):
    return mp.current.Polyline(
        current=c.current, vertices=np.asarray(c.vertices), position=c.pos, orientation=c.rot
    )


def to_obj(s):
    return magnet_obj(s) if isinstance(s, Magnet) else coil_obj(s)


def _mesh(targets, mag_mesh, coil_mesh):
    out = {}
    for t, o in targets:
        out[o] = mag_mesh if isinstance(t, Magnet) else coil_mesh
    return out


def wrench(sources, targets, pivot=None, mag_mesh=None, coil_mesh=40):
    """Force (3,) and torque (3,) on the rigid group `targets` from `sources`.

    Torque about `pivot` (default: centroid of the group, magpylib convention).
    mag_mesh: cell count per target magnet; default is 4x4x4-ish from magpylib int meaning (see tests).
    """
    src = [to_obj(s) for s in sources]
    tobjs = [to_obj(t) for t in targets]
    mesh_map = {}
    for t, o in zip(targets, tobjs):
        if isinstance(t, Magnet):
            mesh_map[o] = mag_mesh if mag_mesh is not None else _default_mesh(t)
        else:
            mesh_map[o] = coil_mesh
    group = mp.Collection(tobjs)
    kw = {}
    if pivot is not None:
        kw["pivot"] = np.asarray(pivot, float)
    F, T = mp.getFT(src, group, meshing=mesh_map, **kw)
    return np.asarray(F).reshape(-1, 3).sum(0), np.asarray(T).reshape(-1, 3).sum(0)


def _default_mesh(m: Magnet):
    if m.kind == "cyl":
        return (12, 3, 4)
    return (4, 4, 4)


def force_magnet_magnet(a: Magnet, b: Magnet, mag_mesh=None):
    """Force and torque on b (about its centre) from a."""
    return wrench([a], [b], pivot=b.pos, mag_mesh=mag_mesh)


def force_coil_magnet(c: Coil, m: Magnet, mag_mesh=None):
    return wrench([c], [m], pivot=m.pos, mag_mesh=mag_mesh)


def force_magnet_coil(m: Magnet, c: Coil, pivot=None, coil_mesh=40):
    piv = pivot if pivot is not None else np.asarray(c.pos)
    return wrench([m], [c], pivot=piv, coil_mesh=coil_mesh)


def force_coil_coil(a: Coil, b: Coil, pivot=None, coil_mesh=40):
    piv = pivot if pivot is not None else np.asarray(b.pos)
    return wrench([a], [b], pivot=piv, coil_mesh=coil_mesh)


# ---------------------------------------------------------------- steel plate by images

def image_magnet(m: Magnet, plane_point=(0, 0, 0), plane_normal=(0, 0, 1), k=1.0) -> Magnet:
    """Image of magnet m in a plane of relative permeability mu_r; k = (mu_r-1)/(mu_r+1) (1 for infinite).

    Valid for linear, unsaturated iron half-space. Geometry is mirrored; the polarisation is the mirrored
    polarisation, negated and scaled by k (normal component kept, tangential reversed).
    """
    n = np.asarray(plane_normal, float)
    n = n / np.linalg.norm(n)
    p0 = np.asarray(plane_point, float)

    def mirror(v):
        v = np.asarray(v, float)
        return v - 2 * np.dot(v, n) * n

    pos = np.asarray(m.pos, float)
    pos_i = p0 + mirror(pos - p0)
    mat = m.rot.as_matrix()
    cols = [mirror(mat[:, 0]) * -1.0, mirror(mat[:, 1]), mirror(mat[:, 2])]
    rot_i = R.from_matrix(np.column_stack(cols))
    # polarisation of the image along local -z of the mirrored frame, scaled by k
    return Magnet(m.kind, m.dim, -k * m.br, tuple(pos_i), rot_i)


def force_magnet_steel(m: Magnet, plane_point=(0, 0, 0), plane_normal=(0, 0, 1), mu_r=None, mag_mesh=None):
    """Force and torque (about magnet centre) on magnet m from a flat iron half-space (image method).

    mu_r None means infinite permeability. Saturation is neglected: the result is an upper bound when the
    magnet's flux density in the iron exceeds about 1.5 T.
    """
    k = 1.0 if mu_r is None else (mu_r - 1.0) / (mu_r + 1.0)
    img = image_magnet(m, plane_point, plane_normal, k)
    return wrench([img], [m], pivot=m.pos, mag_mesh=mag_mesh)


def pull_force_to_steel(m_kind, dim, br, gap=0.0, mu_r=None, mag_mesh=None):
    """Normal pull force (N, positive = attraction) of a magnet whose axis is normal to the plate, face gap `gap`."""
    h = dim[1] if m_kind == "cyl" else dim[2]
    m = Magnet(m_kind, dim, br, (0, 0, h / 2 + gap), R.identity())
    F, _ = force_magnet_steel(m, mu_r=mu_r, mag_mesh=mag_mesh)
    return -F[2]


def pull_force_magnet_magnet(m_kind, dim, br, gap, mag_mesh=None, lateral=0.0):
    """Pull force (N) between two identical magnets facing each other, N pole to S pole, with face gap."""
    h = dim[1] if m_kind == "cyl" else dim[2]
    a = Magnet(m_kind, dim, br, (0, 0, 0), R.identity())
    b = Magnet(m_kind, dim, br, (lateral, 0, h + gap), R.identity())
    F, _ = force_magnet_magnet(a, b, mag_mesh)
    return -F[2]
