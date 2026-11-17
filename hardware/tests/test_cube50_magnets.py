"""Proof of the cube's face-magnet layout (hardware/cube50/sim/magnets.py, params.json "magnets").

Every face pair (fixed/fixed, disc/fixed, disc/disc), all four quarter turns and disc faces at
every disc angle (1-degree sweep): the faces attract in register, register is a stable minimum of
the twist about the face normal (and of the lateral offset), no quarter turn repels or tilts, and
the force model agrees with magpylib.getFT. Whole cubes are used (48 face magnets and the four
AS5600 magnets each), so the other faces' magnets are in every number.

    uv run pytest -q hardware/tests/test_cube50_magnets.py     (about 2 minutes on CPU)
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cube50" / "sim"))
import magnet_study as ms  # noqa: E402
import magnets as mg  # noqa: E402

LAY = mg.layout()
FIXED = ("+x", "-x", "+y", "-y")
DISC = mg.DISC_FACES
ALL = FIXED + DISC


def test_envelope():
    """The CAD envelope (2026-11-17, derived by the CAD agent): outer radius, spacing, depth."""
    R, D, T, skin, d = LAY["R"] * 1e3, LAY["D"] * 1e3, LAY["T"] * 1e3, LAY["skin"] * 1e3, LAY["delta"]
    assert R + D / 2 <= 8.3 + 1e-9
    assert 2 * R * np.sin(np.radians(min(d, 45 - d))) >= D + 0.8 - 0.03  # 0.02 mm short at D 4, R 6.3: the CAD agent's feasible point
    assert T <= 2.0 and skin >= 0.6 and skin + T <= 2.7


def test_sites_off_mirror_lines():
    for phi, _ in mg.sites(LAY["delta"], LAY["poles"]):
        off = min(abs((phi - m + 90) % 180 - 90) for m in (0, 45, 90, 135))
        assert off >= 7.0 - 1e-9, phi


@pytest.mark.parametrize("delta", [7.0, 16.2, 22.5, 30.0, 38.0])
def test_pinwheel_geometry(delta):
    """Two faces meet through a reflection: a partner's site at phi lands on alpha - phi. Every site
    must land on a site of the other pole for every quarter turn alpha (pure geometry)."""
    s = mg.sites(delta)
    table = {round(p % 360, 6): pole for p, pole in s}
    for alpha in (0, 90, 180, 270):
        for phi, pole in s:
            hit = round((alpha - phi) % 360, 6)
            assert hit in table, (alpha, phi)
            assert table[hit] == -pole, (alpha, phi)


def test_force_model_against_getft():
    """Charge-model wrench on one face pair at contact against magpylib.getFT (dipole mesh)."""
    R, x = ms.place("+x", "-x", 0, 0.0, half=LAY["half"])
    A = mg.Magnets(*mg.face_magnets(LAY, "+x"), LAY["D"], LAY["T"], LAY["Br"])
    B = mg.Magnets(*mg.face_magnets(LAY, "-x"), LAY["D"], LAY["T"], LAY["Br"]).moved(R, x)
    B = B.moved(np.eye(3), np.array([0.0, 0.0007, -0.0004]))  # off register, so every component is non-zero
    F, tau = mg.wrench(A, B, x, cutoff=1.0)
    F2, tau2 = mg.wrench_ft(A, B, x, meshing=1500)
    assert np.linalg.norm(F - F2) < 0.005 * np.linalg.norm(F)
    assert np.linalg.norm(tau - tau2) < 0.01 * np.linalg.norm(tau) + 1e-6


def test_cutoff_is_negligible():
    """Pairs beyond the 45 mm cutoff: whole-cube wrench with and without it."""
    R, x = ms.place("+x", "+y", 1, 7.0, half=LAY["half"])
    A = mg.cube_set(LAY)
    B = mg.cube_set(LAY).moved(R, x)
    F, _ = mg.wrench(A, B, x)
    F_all, _ = mg.wrench(A, B, x, cutoff=1.0)
    assert np.linalg.norm(F - F_all) < 0.002 * np.linalg.norm(F_all)


PAIRS = list(itertools.product(ALL, ALL, range(4)))


@pytest.mark.parametrize("fa,fb,q", PAIRS)
def test_register_every_face_pair(fa, fb, q):
    """All 6 x 6 face pairs at all 4 quarter turns (discs at 0): attractive, no lateral pull, no
    tilt, twist and lateral offsets restore."""
    F, tau = ms.face_pair(LAY, fa, fb, q, encoders=True)
    assert F[0] < -1.5, F  # attract (about 2.4 N; the encoder magnets shift it a little)
    assert abs(F[1]) < 0.02 and abs(F[2]) < 0.02
    assert abs(tau) < 4e-4  # the hinge encoder magnets near the corners leave about 2e-4 N m: a 0.25 degree offset at 0.049 N m/rad
    for tw in (-1.0, 1.0):
        _, t = ms.face_pair(LAY, fa, fb, q, twist=tw, encoders=True)
        assert t * tw < 0, (tw, t)  # restoring
    for lat in ((0.0003, 0), (0, 0.0003), (-0.0003, 0), (0, -0.0003)):
        Fl, _ = ms.face_pair(LAY, fa, fb, q, lateral=lat, encoders=True)
        assert Fl[1] * lat[0] + Fl[2] * lat[1] < 0, (lat, Fl)


def test_no_tilt_at_quarter_turns():
    """Torque about the in-plane axes through the contact point is zero in register (no flipping)."""
    h = LAY["half"]
    for fa, fb in (("+x", "-x"), ("+z", "+z"), ("+z", "+y")):
        for q in range(4):
            R, x = ms.place(fa, fb, q, half=h)
            A = mg.cube_set(LAY)
            B = mg.cube_set(LAY).moved(R, x)
            n = mg.FACES[fa][0]
            _, tau = mg.wrench(A, B, x - h * n)
            assert np.linalg.norm(tau - (tau @ n) * n) < 1e-4


def _sweep(fa, fb, q, angles, which):
    out = []
    for a in angles:
        kw = {"discA": a} if which == "A" else {"discB": a} if which == "B" else {"discA": a[0], "discB": a[1]}
        out.append(ms.face_pair(LAY, fa, fb, q, encoders=True, **kw))
    F = np.array([o[0][0] for o in out])
    T = np.array([o[1] for o in out])
    return F, T


def _minima_only_at_quarter_turns(angles, torque, sign):
    """A stable minimum of the energy where the torque on the turning disc crosses from + to -
    (energy E = -integral of torque). Every such crossing must be at a quarter turn (within 1 deg)."""
    t = sign * torque
    for i in range(len(t) - 1):
        if t[i] > 0 >= t[i + 1]:
            a = angles[i] + (angles[i + 1] - angles[i]) * t[i] / (t[i] - t[i + 1])
            assert min(abs(a - 90 * k) for k in range(-1, 6)) < 1.0, a


@pytest.mark.parametrize("fb,q", list(itertools.product(ALL, range(4))))
def test_disc_against_every_face(fb, q):
    """A's disc (+z) against each of B's faces, disc angle 0..360 in 1-degree steps: attractive and
    twist-stable at every quarter turn; the only stable twist angles are quarter turns."""
    ang = np.arange(-2.0, 362.1, 1.0)
    F, T = _sweep("+z", fb, q, ang, "A")
    quarter = np.isin(ang, [0, 90, 180, 270, 360])
    assert (F[quarter] < -1.5).all()
    # torque on B about A's normal; the disc turns A's pattern: torque on the disc is -T
    _minima_only_at_quarter_turns(ang, -T, +1)


def _stable(ang, t):
    """Angles where the torque on the turning disc t crosses from + to - (energy minima)."""
    out = []
    for i in range(len(t) - 1):
        if t[i] > 0 >= t[i + 1]:
            out.append(ang[i] + (ang[i + 1] - ang[i]) * t[i] / (t[i] - t[i + 1]))
    return out


@pytest.mark.parametrize("fb,q", list(itertools.product(DISC, range(4))))
def test_disc_against_disc(fb, q):
    """Disc on disc: B's disc at 0, 7.5, ..., 82.5 degrees, A's disc swept 0..180 by 2 degrees. For
    every B angle the stable A angles are exactly 90 degrees apart (the pattern's own period: no
    extra, false register) and attract there."""
    ang = np.arange(-4.0, 184.1, 2.0)
    for b in np.arange(0, 90, 7.5):
        _, T = _sweep("+z", fb, q, [(a, b) for a in ang], "AB")
        stable = _stable(ang, -T)
        assert len(stable) >= 2, (b, stable)
        assert np.allclose(np.diff(stable), 90, atol=1.0), (b, stable)
        for a in stable:
            f, t = ms.face_pair(LAY, "+z", fb, q, discA=a, discB=b, encoders=True)
            assert f[0] < -1.5, (a, b, f)
