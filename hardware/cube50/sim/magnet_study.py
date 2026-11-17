"""Sizing of the face magnets: the two bounds (capture and peel), the twist and slide curves and the
disc cogging, for the layout in params.json or for candidate parts.

    nice -n 10 .venv/bin/python hardware/cube50/sim/magnet_study.py [sweep] [point] [cogging] [as5600]

Writes one JSON line per result to hardware/cube50/sim/magnet_results.jsonl.

Geometry (cube frame of magnets.py; S = 60 mm): the support A sits at the origin, the partner B at
(S, 0, 0), so A's +x face meets one of B's faces. The pivot edge of a climb is the shared edge line
x = S/2, z = S/2 (along y), in the contact plane, S/2 from the face centre. Opening the pair by an
angle theta turns B about that line away from A. A side face slides past a neighbour when the mover
turns about an axis normal to that face, through the face's corner (S/sqrt 2 from its centre).
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent))
import magnets as mg  # noqa: E402

OUT = Path(__file__).parent / "magnet_results.jsonl"
G = 9.81


def emit(row):
    print(json.dumps(row), flush=True)
    with OUT.open("a") as f:
        f.write(json.dumps(row) + "\n")


def lay_with(**over):
    lay = mg.layout()
    lay.update(over)
    return lay


def facing(face):
    """Rotation of B (cube frame to world) that puts B's `face` against A's +x face, quarter turn 0."""
    n, e1, e2 = mg.FACES[face]
    # B's face normal must be -x; map (n, e1) -> (-x, y): a reflection of A's (+x: e1 = y, e2 = z) frame
    src = np.column_stack([n, e1, np.cross(n, e1)])
    dst = np.column_stack([[-1.0, 0, 0], [0, 1.0, 0], np.cross([-1.0, 0, 0], [0, 1.0, 0])])
    return dst @ src.T


def pose_B(face="-x", quarter=0, twist=0.0, gap=0.0, lateral=(0.0, 0.0), open_deg=0.0, half=None):
    """World pose (R, x) of B: its `face` against A's +x, turned by quarter * 90 + twist degrees about
    the contact normal, pushed out by `gap` and sideways by `lateral` (y, z) in m, then opened by
    open_deg about the edge line x = half, z = half."""
    half = half or mg.layout()["half"]
    R = Rotation.from_rotvec(math.radians(90 * quarter + twist) * np.array([1.0, 0, 0])).as_matrix() @ facing(face)
    x = np.array([2 * half + gap, lateral[0], lateral[1]])
    if open_deg:
        Ro = Rotation.from_rotvec(-math.radians(open_deg) * np.array([0, 1.0, 0])).as_matrix()
        p = np.array([half, 0, half])
        R, x = Ro @ R, Ro @ (x - p) + p
    return R, x


def pair(lay, poseB, discA=None, discB=None, encoders=False, pivot=None):
    """Force on B (N) and torque on B about `pivot` (default B's centre), from A's magnets."""
    A = mg.cube_set(lay, discA, encoders=encoders)
    B = mg.cube_set(lay, discB, encoders=encoders).moved(*poseB)
    return mg.wrench(A, B, poseB[1] if pivot is None else np.asarray(pivot))


def peel_curve(lay, face="-x", quarter=0, angles=np.arange(0, 30.1, 0.5)):
    """Magnetic torque (N m) resisting the opening of a face pair about the pivot edge, per angle."""
    h = lay["half"]
    out = []
    for a in angles:
        _, tau = pair(lay, pose_B(face, quarter, open_deg=a, half=h), pivot=[h, 0, h])
        out.append(tau[1])  # +y torque closes (B opens with a negative turn about +y)
    return np.array(out)


def slide_curve(lay, angles=np.arange(0, 40.1, 0.5)):
    """B is a side neighbour of A (B's -x on A's +x). The mover A turns about the axis along x through
    the corner (y, z) = (-h, h) of that face, as a cube does when its hinge edge runs along x.
    Returns the magnetic torque on A about that axis resisting the turn, per angle (N m)."""
    h = lay["half"]
    A0 = mg.cube_set(lay, encoders=False)
    B = mg.cube_set(lay, encoders=False).moved(*pose_B("-x", 0, half=h))
    p = np.array([0, -h, h])
    out = []
    for a in angles:
        R = Rotation.from_rotvec(math.radians(a) * np.array([1.0, 0, 0])).as_matrix()
        x = R @ (np.zeros(3) - p) + p
        F, tau = mg.wrench(B, A0.moved(R, x), p)
        out.append(-tau[0])
    return np.array(out)


def place(faceA, faceB, quarter=0, twist=0.0, gap=0.0, lateral=(0.0, 0.0), half=None):
    """World pose (R, x) of B with its `faceB` against A's `faceA` (A at the origin, unrotated): B's
    face e1 lands at angle 90 * quarter + twist (deg, counterclockwise seen from outside A's face)
    from A's e1. `lateral` (m) is along A's (e1, e2), `gap` (m) along A's normal."""
    half = half or mg.layout()["half"]
    nA, e1A, e2A = mg.FACES[faceA]
    nB, e1B, e2B = mg.FACES[faceB]
    b = math.radians(90 * quarter + twist)
    v1 = math.cos(b) * e1A + math.sin(b) * e2A
    R = np.column_stack([-nA, v1, np.cross(-nA, v1)]) @ np.column_stack([nB, e1B, e2B]).T
    x = (2 * half + gap) * nA + lateral[0] * e1A + lateral[1] * e2A
    return R, x


def face_pair(lay, faceA, faceB, quarter=0, twist=0.0, discA=0.0, discB=0.0, gap=0.0, lateral=(0.0, 0.0), encoders=False):
    """Force on B (N, in A's face frame: normal, e1, e2; normal < 0 attracts) and torque on B about
    the contact point's normal (N m, counterclockwise seen from outside A positive), with disc angles
    discA on A's disc faces and discB on B's."""
    h = lay["half"]
    R, x = place(faceA, faceB, quarter, twist, gap, lateral, h)
    A = mg.cube_set(lay, {f: discA for f in mg.DISC_FACES}, encoders=encoders)
    B = mg.cube_set(lay, {f: discB for f in mg.DISC_FACES}, encoders=encoders).moved(R, x)
    nA, e1A, e2A = mg.FACES[faceA]
    F, tau = mg.wrench(A, B, x - h * nA)
    return np.array([F @ nA, F @ e1A, F @ e2A]), float(tau @ nA)


def capture(lay, open_deg, lateral=(0.0, 0.0), twist=0.0):
    """Closing torque about the pivot edge (N m, positive closes) with B open by open_deg, shifted and
    twisted; also the normal force and twist torque about B's face normal."""
    h = lay["half"]
    pB = pose_B("-x", 0, twist=twist, lateral=lateral, open_deg=open_deg, half=h)
    F, tau_edge = pair(lay, pB, pivot=[h, 0, h])
    _, tau_c = pair(lay, pB)
    return tau_edge[1], F, tau_c


def point(lay, label):
    h = lay["half"]
    F0, _ = pair(lay, pose_B("-x", 0, half=h))
    peel = peel_curve(lay)
    slide = slide_curve(lay)
    row = {"check": "magnet_point", "label": label, "D_mm": lay["D"] * 1e3, "T_mm": lay["T"] * 1e3, "Br_T": lay["Br"], "R_mm": lay["R"] * 1e3, "delta_deg": lay["delta"], "skin_mm": lay["skin"] * 1e3,
           "normal_force_register_n": round(-F0[0], 3), "peel_torque_start_nm": round(peel[0], 4), "peel_torque_max_nm": round(peel.max(), 4),
           "slide_torque_max_nm": round(slide.max(), 4), "slide_peak_deg": float(np.arange(0, 40.1, 0.5)[slide.argmax()])}
    for b in (1.0, 2.0, 3.0):
        t, F, _ = capture(lay, b)
        row[f"capture_torque_at_{b:g}deg_nm"] = round(t, 4)
    return row


if __name__ == "__main__":
    args = sys.argv[1:] or ["point"]
    if "point" in args:
        emit(point(mg.layout(), "params.json"))
    if "sweep" in args:
        for D, T, Br, grade in ((3.0, 1.0, 1.21, "N35"), (3.0, 1.0, 1.45, "N52"), (4.0, 1.0, 1.21, "N35"), (3.0, 1.5, 1.21, "N35"), (2.0, 1.0, 1.21, "N35"), (3.0, 0.5, 1.21, "N35")):
            R = min(6.8 if D <= 3 else 6.3, 8.3 - D / 2)
            lay = lay_with(D=D / 1000, T=T / 1000, Br=Br, R=R / 1000, delta=22.5)
            emit(point(lay, f"{D:g}x{T:g} {grade}"))


# ------------------------------------------------------------------ disc cogging
def _split(mags, idx):
    keep = np.zeros(len(mags), bool)
    keep[idx] = True
    return mags.subset(np.nonzero(keep)[0]), mags.subset(np.nonzero(~keep)[0])


def cogging(lay, angles=np.arange(0, 90.1, 1.0)):
    """Torque about the disc axis (N m) the magnets put on A's +z disc drive, per disc angle.
    bonded: the disc carries a partner cube (its -z disc face on A's +z, quarter 0) and turns with it;
    the rest of A stays. neighbour: no bond; a cube C sits on A's +z face and stays, the disc turns
    under it. Also the forces on C in the neighbour case (does it get dragged or lifted?)."""
    h = lay["half"]
    names = [e["name"] for e in lay["encoders"]["magnets"]]
    out = {"bonded": [], "neighbour": [], "C_normal_n": [], "C_lateral_n": [], "C_twist_nm": []}
    R, x = place("+z", "-z", 0, half=h)
    z = np.array([0, 0, 1.0])
    for a in angles:
        A = mg.cube_set(lay, {"+z": a})
        disc = list(range(32, 40)) + [48 + k for k, n in enumerate(names) if n == "enc_disc_top"]  # face order: +x -x +y -y +z -z
        D, rest = _split(A, disc)
        Rz = Rotation.from_rotvec(math.radians(a) * z).as_matrix()
        B = mg.cube_set(lay).moved(Rz @ R, Rz @ x)
        _, t1 = mg.wrench(rest, D + B, np.zeros(3))
        out["bonded"].append(t1 @ z)
        C = mg.cube_set(lay).moved(R, x)
        _, t2 = mg.wrench(rest + C, D, np.zeros(3))
        out["neighbour"].append(t2 @ z)
        Fc, tc = mg.wrench(A, C, x)
        out["C_normal_n"].append(Fc[2])
        out["C_lateral_n"].append(float(np.hypot(Fc[0], Fc[1])))
        out["C_twist_nm"].append(tc @ z)
    return {k: np.array(v) for k, v in out.items()}


# ------------------------------------------------------------------ AS5600 stray field (static)
def as5600_static(lay, faces=mg.FACES, quarters=range(4)):
    """For each AS5600: its own magnet's first harmonic of Bz on the Hall circle (the signal), and
    the largest first harmonic from (a) the cube's own face magnets, (b) own face magnets plus a
    partner in register on any face at any quarter turn (discs at 0). Angle error bound
    asin(stray / signal)."""
    A = mg.cube_set(lay)
    names = [e["name"] for e in lay["encoders"]["magnets"]]
    face_only = A.subset(np.arange(48))
    out = {}
    for chip, (die, nrm, mag) in mg.chips(lay).items():
        own = A.subset([48 + names.index(mag)])
        sig, _, _ = mg.as5600_error(lambda p: mg.field(own, p), die, nrm)
        bz = float(mg.field(own, die + 0.001 * mg._any_perp(nrm))[0] @ nrm)
        h_own, _, m_own = mg.as5600_error(lambda p: mg.field(face_only, p), die, nrm)
        others = A.subset(np.array([48 + k for k in range(len(names)) if names[k] != mag], int))
        h_enc = mg.as5600_error(lambda p: mg.field(others, p), die, nrm)[0] if len(others) else 0.0
        worst, where = 0.0, None
        for fa in faces:
            for fb in ("-x", "+z"):
                for q in quarters:
                    R, x = place(fa, fb, q, half=lay["half"])
                    src = face_only + others + mg.cube_set(lay).moved(R, x)
                    h, _, _ = mg.as5600_error(lambda p: mg.field(src, p), die, nrm)
                    if h > worst:
                        worst, where = h, f"partner on {fa} ({fb}, quarter {q})"
        out[chip] = {"signal_h1_mt": round(sig * 1e3, 2), 
                     "stray_h1_own_faces_mt": round(h_own * 1e3, 4), "stray_mean_own_faces_mt": round(abs(m_own) * 1e3, 3),
                     "stray_h1_other_encoders_mt": round(h_enc * 1e3, 4),
                     "stray_h1_with_partner_mt": round(worst * 1e3, 4), "worst_partner": where,
                     "error_bound_deg_own": round(math.degrees(math.asin(min(1, (h_own + h_enc) / sig))), 3),
                     "error_bound_deg_with_partner": round(math.degrees(math.asin(min(1, worst / sig))), 3)}
    return out


def summary(lay=None):
    """Every number the DESIGN.md section quotes, for the layout in params.json."""
    lay = lay or mg.layout()
    ang = np.arange(0, 20.1, 0.25)
    peel = peel_curve(lay, angles=ang)
    slide = slide_curve(lay, angles=ang)
    both = peel + 2 * slide
    row = point(lay, "params.json")
    row.update({"check": "magnet_summary", "peel_plus_two_slides_max_nm": round(both.max(), 4), "peel_plus_two_slides_at_deg": float(ang[both.argmax()]),
                "peel_curve_nm": {f"{a:g}": round(p, 4) for a, p in zip(ang[::8], peel[::8])},
                "slide_curve_nm": {f"{a:g}": round(p, 4) for a, p in zip(ang[::4], slide[::4])}})
    for br in (1.17, 1.21, 1.24):
        lb = dict(lay, Br=br)
        row[f"Br_{br}"] = {"peel_nm": round(peel_curve(lb, angles=[0])[0], 4), "capture_2deg_nm": round(capture(lb, 2)[0], 4), "capture_3deg_nm": round(capture(lb, 3)[0], 4),
                           "capture_2deg_lat0.5_twist1.5_nm": round(capture(lb, 2, lateral=(0.0005, 0.0005), twist=1.5)[0], 4)}
    for s in (0.9, 1.0, 1.1, 1.2, 1.3):
        ls = dict(lay, skin=s / 1000)
        row[f"skin_{s}"] = {"peel_nm": round(peel_curve(ls, angles=[0])[0], 4), "capture_2deg_nm": round(capture(ls, 2)[0], 4)}
    # misalignment basin: lateral and twist restoring at the 2 degree lead-in
    _, F, tau = capture(lay, 2, lateral=(0.0005, 0.0005), twist=1.5)
    row["lead_in_2deg_lat0.5_twist1.5"] = {"force_n": [round(v, 3) for v in F], "torque_about_B_centre_nm": [round(v, 5) for v in tau]}
    tw = np.arange(0, 45.1, 2.5)
    fn = [face_pair(lay, "+x", "-x", 0, t)[0][0] for t in tw]
    row["normal_force_vs_twist_n"] = {f"{t:g}": round(f, 3) for t, f in zip(tw, fn)}
    c = cogging(lay)
    row["cogging"] = {"bonded_max_nm": round(float(np.abs(c["bonded"]).max()), 5), "neighbour_max_nm": round(float(np.abs(c["neighbour"]).max()), 4),
                      "neighbour_normal_on_C_n": [round(float(c["C_normal_n"].min()), 3), round(float(c["C_normal_n"].max()), 3)],
                      "neighbour_twist_on_C_max_nm": round(float(np.abs(c["C_twist_nm"]).max()), 4)}
    row["as5600_static"] = as5600_static(lay)
    emit(row)
    return row


if __name__ == "__main__" and "summary" in sys.argv[1:]:
    summary()


def bench(lay=None, gaps=(0.0, 0.2, 0.5, 1.0, 1.5, 2.0)):
    """Predictions for the step-0 bench test (ASSEMBLY.md): printed test faces (plates) with the
    design skin, no cube around them. single: one magnet in each of two coupons, coaxial, pole N on
    S; face: two 8-magnet plates in register (quarter 0); both against an extra air gap (shims, mm).
    peel: the two plates hinged on a line 30 mm from the pattern centre (tape hinge at the plate
    edge for a 60 mm plate), torque to open; twist: plate turned about the common normal."""
    lay = lay or mg.layout()
    h = lay["half"]
    depth = lay["skin"] + lay["T"] / 2
    one = mg.Magnets([[0, 0, -depth]], [[0, 0, 1]], lay["D"], lay["T"], lay["Br"])
    plateA = mg.Magnets(*mg.face_magnets(lay, "+x"), lay["D"], lay["T"], lay["Br"])
    out = {"check": "magnet_bench", "skin_mm": lay["skin"] * 1e3, "Br_T": lay["Br"], "single_n": {}, "face_n": {}}
    for g in gaps:
        two = mg.Magnets([[0, 0, depth + g / 1000]], [[0, 0, 1]], lay["D"], lay["T"], lay["Br"])
        out["single_n"][f"{g:g}"] = round(-mg.wrench(one, two, two.c[0])[0][2], 4)
        R, x = place("+x", "-x", 0, gap=g / 1000, half=h)
        plateB = mg.Magnets(*mg.face_magnets(lay, "-x"), lay["D"], lay["T"], lay["Br"]).moved(R, x)
        out["face_n"][f"{g:g}"] = round(-mg.wrench(plateA, plateB, x)[0][0], 4)
    plateB0 = mg.Magnets(*mg.face_magnets(lay, "-x"), lay["D"], lay["T"], lay["Br"])
    for a in (0.0, 2.0, 5.0):
        R, x = pose_B("-x", 0, open_deg=a, half=h)
        _, tau = mg.wrench(plateA, plateB0.moved(R, x), np.array([h, 0, h]))
        out[f"peel_{a:g}deg_nm"] = round(tau[1], 4)
    for t in (5.0, 10.0, 20.0):
        R, x = place("+x", "-x", 0, t, half=h)
        _, tau = mg.wrench(plateA, plateB0.moved(R, x), x)
        out[f"twist_{t:g}deg_nm"] = round(tau[0], 5)
    emit(out)
    return out


if __name__ == "__main__" and "bench" in sys.argv[1:]:
    bench()
