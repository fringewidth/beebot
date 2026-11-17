"""Packing study: the smallest cube size S at which the cube closes with the hinge on a sled.

Run from the repo root:  .venv/bin/python hardware/cube50/cad/packing_study.py [S ...]
For each S it builds the parts that fight for the core as simple solids (boxes, cylinders, and the
volumes the sled parts sweep over the deploy stroke) and prints every pair that overlaps by more
than 0.05 mm3, plus every core part that leaves the core cavity.

Fixed by the brief (2026-11-17): the 50 mm disc drive (150:1 N20, 6700 bearing, m0.5 external
train of about 6.67:1, at least 1.13 N m at the disc); the hinge on a rigid sled (motor, train,
axle, carriers) slid diagonally (3.0 mm in x and z) by a DFRobot FIT0577 lead-screw gearmotor; the
FIT0577 lies along x (along the stroke it needs about 36 mm of diagonal core height, more than these
cubes have), its nut in a carrier that drives the sled plate through a pin in a slot; the stowed hinge at
least 0.2 mm inside every face; the bond scaled with the face. The bottom foot's parts are the
top foot's turned 180 degrees about x, as in cube50_cad.py.
"""

import math
import sys
from dataclasses import dataclass, field

import cadquery as cq

N20_W, N20_T = 12.0, 10.0  # bought: N20 gearmotor cross-section (source: Pololu drawing)
HINGE_MOTOR_L = 26.5  # 1000:1 HP N20 with its long gearbox (guess from the generic listing)
DISC_MOTOR_L = 24.2  # 150:1 HP N20 (Pololu drawing, from memory)
DEPLOY_L = 25.0  # FIT0577 body with gearbox (DFRobot drawing gives about 24; allow 25)
DEPLOY_SCREW_R = 1.5  # M3 lead screw
NUT_AF, NUT_T = 5.5, 2.4  # M3 brass hex nut, DIN 934
KR = 2.8  # knuckle and carrier radius about the 2 mm axle (50 mm design)
RETRACT = KR + 0.2  # stowed, the axle sits this far inside the disc plane and the +x face
CELL_DEFAULT = (3.0, 25.0, 35.0)  # 302535 pouch: thickness, width, length


def box(x0, x1, y0, y1, z0, z1):
    return cq.Workplane("XY").box(x1 - x0, y1 - y0, z1 - z0).translate(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))


def cyl_z(r, z0, z1, x=0.0, y=0.0):
    return cq.Workplane("XY").circle(r).extrude(z1 - z0).translate((x, y, z0))


def cyl_x(r, x0, x1, y=0.0, z=0.0):
    return cq.Workplane("YZ").circle(r).extrude(x1 - x0).translate((x0, y, z))


def cyl_y(r, y0, y1, x=0.0, z=0.0):
    return cq.Workplane("XZ").circle(r).extrude(-(y1 - y0)).translate((x, y0, z))


def swept_box(x0, x1, y0, y1, z0, z1, dx, dz):
    """The volume a box sweeps when it slides by (dx, dz) in the xz plane (dx, dz >= 0): a hexagonal prism."""
    pts = [(x0, z0), (x1, z0), (x1 + dx, z0 + dz), (x1 + dx, z1 + dz), (x0 + dx, z1 + dz), (x0, z1)]
    return cq.Workplane("XZ").polyline(pts).close().extrude(-(y1 - y0)).translate((0, y0, 0))


def swept_cyl_y(r, y0, y1, x, z, dx, dz, steps=6):
    return union([cyl_y(r, y0, y1, x + dx * k / steps, z + dz * k / steps) for k in range(steps + 1)])


def union(shapes):
    out = shapes[0]
    for s in shapes[1:]:
        out = out.union(s)
    return out


def flip(shape):
    return shape.rotate((0, 0, 0), (1, 0, 0), 180)


def volume(shape):
    try:
        return shape.val().Volume()
    except Exception:
        return float("nan")


def hull_xz(points, y0, y1):
    """Convex hull of (x, z) points, extruded over y0..y1."""
    pts = sorted(set(points))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    ring = lower[:-1] + upper[:-1]
    return cq.Workplane("XZ").polyline(ring).close().extrude(-(y1 - y0)).translate((0, y0, 0))


def circle_pts(c, r, n=16):
    return [(round(c[0] + r * math.cos(2 * math.pi * k / n), 4), round(c[1] + r * math.sin(2 * math.pi * k / n), 4)) for k in range(n)]


def meet(a, ra, b, rb, side=1):
    """A point at distance ra from a and rb from b (side picks one of the two)."""
    d = math.dist(a, b)
    ux, uz = (b[0] - a[0]) / d, (b[1] - a[1]) / d
    p = (ra * ra - rb * rb + d * d) / (2 * d)
    h = math.sqrt(max(ra * ra - p * p, 0.0))
    return (a[0] + p * ux - side * h * uz, a[1] + p * uz + side * h * ux)


@dataclass
class Layout:
    S: float = 50.0
    # hinge train, module 0.4 hardened steel: axle gear, two idlers, motor gear
    m_h: float = 0.4
    n_axle: int = 12
    n_idler: int = 24
    n_motor: int = 18
    gear_face: float = 2.5
    plate_t: float = 1.5  # sled end plates, laser-cut stainless
    # disc train (top foot): motor pinion 12T, stage-1 gear n_b, compound small 12T, centre gear n_c (m0.5)
    n_b: int = 30
    n_c: int = 32
    face_c: float = 4.0
    comp_deg: float = 255.0  # direction of the compound shaft from the disc axis (top disc)
    motor_deg: float = 175.0  # direction of the disc motor from the compound
    s1: float = None  # stage-1 band bottom (z), default just under the core top
    s2: float = None  # stage-2 band bottom (z)
    # deploy motor lane (top sled): body along x, 10 mm in y, 12 mm in z
    deploy_y: float = None
    deploy_z: float = None
    deploy_x0: float = None
    screw_len: float = 20.0
    deploy_l: float = DEPLOY_L
    carrier_gap: float = 0.5  # gearbox face to the stowed nut carrier
    motor_rot: bool = False  # disc motor box turned 90 degrees about z (12 mm along x)
    # cell and main board
    cell: tuple = CELL_DEFAULT
    cell_at: tuple = None  # (x0, y0, z0) of the cell box, cell lying flat (thickness along z)
    cell_axis: str = "z"  # which axis the cell's thickness runs along
    board: tuple = (1.6, 33.0, 26.0)  # main board: thickness, and its two sides
    board_at: tuple = None  # (x0, y0, z0) corner of the board box (board plus 1.5 mm of parts)
    board_axis: str = "x"  # which axis the board's thickness runs along
    bolt_xy: tuple = None
    bolt_len: float = 3.3  # stow bolt guide below the core top
    notes: list = field(default_factory=list)

    @property
    def K(self):
        return self.S / 50.0

    @property
    def H(self):
        return self.S / 2

    @property
    def TF(self):  # foot: floor 1.0 + 6700 bearing 4.0 + 0.05 + disc floor 0.35 + bond bay 5.2 (scaled)
        return 1.0 + 4.0 + 0.05 + 0.35 + 5.2 * self.K

    @property
    def ZC(self):
        return self.H - self.TF - 0.4

    @property
    def XW(self):  # inner face of the x walls: the face's bayonet pockets (3.2, scaled) and a skin
        return self.H - 3.4 * self.K

    @property
    def PY(self):
        return self.H - 3.4 * self.K

    @property
    def YA(self):  # inner face of the +y sled plate; the train lies beyond it and ends 5 mm short of the corner
        return self.H - 5.0 - (0.3 + self.plate_t + 0.3 + self.gear_face)


def polar(r, deg, c=(0.0, 0.0)):
    return (c[0] + r * math.cos(math.radians(deg)), c[1] + r * math.sin(math.radians(deg)))


def build(L):
    """Parts of the top foot's drives and sled (stowed, with their deploy sweeps), their bottom
    copies, the cell and the board. Returns {name: (shape, where)}, where = 'core' (must stay in the
    core cavity) or 'cell' (may use the walls, must stay in the cell and 5 mm short of the corners)."""
    H, R, ZC, XW, YA = L.H, RETRACT, L.ZC, L.XW, L.YA
    rp = lambda n: L.m_h * n / 2  # hinge train pitch radius
    rt = lambda n: L.m_h * (n + 2) / 2
    parts = {}
    # --- sled (top hinge): axle, motor, train, plates; everything sweeps (R, R)
    A = (H - R, H - R)
    mx1, mz1 = XW - 0.3 - R, ZC - 0.3 - R  # stowed motor box: +x and +z faces
    mot = (mx1 - N20_W, mx1, YA - HINGE_MOTOR_L, YA, mz1 - N20_T, mz1)
    G2 = (mx1 - N20_W / 2, mz1 - N20_T / 2)
    G1a = (A[0] - (rp(L.n_axle) + rp(L.n_idler)) / math.sqrt(2), A[1] - (rp(L.n_axle) + rp(L.n_idler)) / math.sqrt(2))
    G1b = meet(G1a, 2 * rp(L.n_idler), G2, rp(L.n_idler) + rp(L.n_motor), side=-1)
    L.notes.append(f"hinge train (x, z) stowed: axle {A[0]:.2f},{A[1]:.2f}; idlers {G1a[0]:.2f},{G1a[1]:.2f} and {G1b[0]:.2f},{G1b[1]:.2f}; motor gear {G2[0]:.2f},{G2[1]:.2f}; reach left {2 * rp(L.n_idler) + rp(L.n_idler) + rp(L.n_motor) - math.dist(G1a, G2):.2f}")
    parts["hinge_motor"] = (swept_box(mot[0], mot[1], mot[2], mot[3], mot[4], mot[5], R, R), "core")
    gy0 = YA + L.plate_t + 0.3
    gy1 = gy0 + L.gear_face
    for name, c, n in (("g_axle", A, L.n_axle), ("g_idler_a", G1a, L.n_idler), ("g_idler_b", G1b, L.n_idler), ("g_motor", G2, L.n_motor)):
        parts[name] = (swept_cyl_y(rt(n), gy0, gy1, c[0], c[1], R, R), "cell")
    plate_pts = circle_pts(A, KR) + circle_pts(G1a, 2.0) + circle_pts(G1b, 2.0) + circle_pts(G2, 3.0)
    parts["plate_py"] = (union([hull_xz(plate_pts, YA, YA + L.plate_t).translate((R * k / 4, 0, R * k / 4)) for k in range(5)]), "cell")
    # --- deploy (top sled): FIT0577 body along x in the lane beside the hinge motor's tail (top half),
    # its nut in a carrier that slides along x only; a pin on the carrier drives the -y plate through a
    # slot along z (the plate's diagonal guides turn the push into the stroke)
    dy = L.deploy_y if L.deploy_y is not None else (-L.PY + YA - HINGE_MOTOR_L) / 2
    dz = L.deploy_z if L.deploy_z is not None else (0.2 + ZC) / 2
    x0 = L.deploy_x0 if L.deploy_x0 is not None else H - 1.0 - L.screw_len - L.deploy_l
    parts["deploy_motor"] = (box(x0, x0 + L.deploy_l, dy - N20_T / 2, dy + N20_T / 2, dz - N20_W / 2, dz + N20_W / 2), "core")
    parts["deploy_screw"] = (cyl_x(DEPLOY_SCREW_R, x0 + L.deploy_l, x0 + L.deploy_l + L.screw_len, dy, dz), "cell")
    xn = x0 + L.deploy_l + L.carrier_gap  # stowed nut carrier, just past the gearbox face
    carrier = (xn, xn + NUT_T + 2.0, dy - 4.0, dy + 4.0, dz - 4.0, dz + 4.0)  # the nut in 1 mm walls
    parts["nut_carrier"] = (swept_box(*carrier, R, 0.0), "core")
    L.notes.append(f"deploy lane y {dy - 5:.2f}..{dy + 5:.2f}, z {dz - 6:.2f}..{dz + 6:.2f}; body x {x0:.2f}..{x0 + L.deploy_l:.2f}; screw to x {x0 + L.deploy_l + L.screw_len:.2f}; nut used {carrier[1] - x0 - L.deploy_l + R:.1f} of {L.screw_len} mm")
    pin = (carrier[1] - 2.0, dz)
    stem = (A[0] - 4.0, A[1] - 4.0)  # the plate leaves the axle boss along the stroke, then widens inside the cell
    plate_n = circle_pts(stem, 2.5) + circle_pts(pin, 2.5) + circle_pts((pin[0], pin[1] + R + 2.0), 2.5)
    slot = union([cyl_x(DEPLOY_SCREW_R + 0.5, x0 + L.deploy_l, x0 + L.deploy_l + L.screw_len, dy, dz).translate((-R * k / 4, 0, -R * k / 4)) for k in range(5)])
    boss = hull_xz(circle_pts(A, KR) + circle_pts(stem, KR), -YA - L.plate_t, -YA)
    plate0 = hull_xz(plate_n, -YA - L.plate_t, -YA).union(boss).cut(slot)  # the screw passes the plate through a slot along the stroke
    parts["plate_ny"] = (union([plate0.translate((R * k / 4, 0, R * k / 4)) for k in range(5)]), "cell")
    # --- disc drive (top disc)
    s1 = L.s1 if L.s1 is not None else ZC - 0.3 - 3.0
    s2 = L.s2 if L.s2 is not None else s1 - 0.5 - L.face_c
    d1 = 0.25 * L.n_c + 3.0
    d2 = 0.25 * L.n_b + 3.0
    Cc = polar(d1, L.comp_deg)
    Mc = polar(d2, L.motor_deg, Cc)
    L.notes.append(f"disc: compound at {Cc[0]:.2f},{Cc[1]:.2f}; motor at {Mc[0]:.2f},{Mc[1]:.2f}; stage 1 z {s1:.2f}..{s1 + 3:.2f}, stage 2 z {s2:.2f}..{s2 + L.face_c:.2f}; ratio {L.n_b / 12 * L.n_c / 12:.2f}")
    parts["centre_gear"] = (cyl_z(0.25 * L.n_c + 0.5, s2, s2 + L.face_c), "core")
    parts["compound"] = (cyl_z(0.25 * L.n_b + 0.5, s1, s1 + 3.0, *Cc).union(cyl_z(3.5, s2, s1, *Cc)).union(cyl_z(1.0, s2 - 1.5, min(s1 + 4.5, ZC), *Cc)), "core")
    hx, hy = (N20_W / 2, N20_T / 2) if L.motor_rot else (N20_T / 2, N20_W / 2)
    parts["disc_motor"] = (box(Mc[0] - hx, Mc[0] + hx, Mc[1] - hy, Mc[1] + hy, s1 - 0.2 - DISC_MOTOR_L, s1 - 0.2).union(cyl_z(3.5, s1 - 0.2, s1 + 3.0, *Mc)), "core")
    parts["centre_shaft"] = (cyl_z(2.0, s2 - 0.1, ZC), "core")
    parts["enc_disc"] = (cyl_z(3.0, s2 - 2.8, s2 - 0.3).union(box(-5, 5, -5, 5, s2 - 3.8, s2 - 3.0)), "core")
    bx, by = L.bolt_xy if L.bolt_xy is not None else (-15.4 * L.K, 0.0)
    parts["bolt_guide"] = (cyl_z(2.6, ZC - L.bolt_len, ZC, bx, by), "core")
    out = {}
    for k, v in parts.items():
        out[k + "_t"] = v
        out[k + "_b"] = (flip(v[0]), v[1])
    # --- cell and board (one each)
    t, w, l = L.cell
    if L.cell_at is not None:
        cx, cy, cz = L.cell_at
        dims = {"z": (l, w, t), "x": (t, w, l), "y": (l, t, w)}[L.cell_axis]
        out["cell"] = (box(cx, cx + dims[0], cy, cy + dims[1], cz, cz + dims[2]), "core")
    bt, bw, bh = L.board
    bx0, by0, bz0 = L.board_at if L.board_at is not None else (-XW, -bw / 2, -bh / 2)
    bd = {"x": (bt + 1.5, bw, bh), "z": (bw, bh, bt + 1.5), "y": (bw, bt + 1.5, bh)}[L.board_axis]
    out["board"] = (box(bx0, bx0 + bd[0], by0, by0 + bd[1], bz0, bz0 + bd[2]), "core")
    return out


INTENDED = {frozenset(p) for s in ("_t", "_b") for p in (
    ("g_axle" + s, "g_idler_a" + s), ("g_idler_a" + s, "g_idler_b" + s), ("g_idler_b" + s, "g_motor" + s),
    ("compound" + s, "centre_gear" + s), ("compound" + s, "disc_motor" + s), ("centre_shaft" + s, "centre_gear" + s),
    ("centre_shaft" + s, "enc_disc" + s), ("deploy_screw" + s, "nut_carrier" + s), ("deploy_screw" + s, "deploy_motor" + s),
    ("nut_carrier" + s, "plate_ny" + s),
)}


def check(L, verbose=True):
    parts = build(L)
    H, ZC, XW, PY = L.H, L.ZC, L.XW, L.PY
    core = box(-XW, XW, -PY, PY, -ZC, ZC)
    core_clear = box(-XW + 0.2, XW - 0.2, -PY + 0.2, PY - 0.2, -ZC + 0.2, ZC - 0.2)  # 0.2 mm off the walls
    # sled parts may leave the cell only into the edge channel (inscribed radius C / sqrt 2) of their own hinge
    ch = 7.5 * L.K / math.sqrt(2) - 0.1
    channel = cyl_y(ch, -(H - 5.0), H - 5.0, H, H)
    cell = box(-H + 0.2, H - 0.2, -(H - 5.0), H - 5.0, -H + 0.2, H - 0.2).union(channel).union(flip(channel))
    rows = []
    for name, (sh, where) in parts.items():
        flush = name in ("board", "cell") or name.startswith(("centre_shaft", "bolt_guide", "compound"))  # meant to reach a wall or the core top
        v = volume(sh.cut((core if flush else core_clear) if where == "core" else cell))
        if v > 0.05:
            rows.append((f"{name} outside the {where}", v))
    names = sorted(parts)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if frozenset((a, b)) in INTENDED:
                continue
            sa, sb = parts[a][0], parts[b][0]
            p, q = sa.val().BoundingBox(), sb.val().BoundingBox()
            if p.xmax < q.xmin or q.xmax < p.xmin or p.ymax < q.ymin or q.ymax < p.ymin or p.zmax < q.zmin or q.zmax < p.zmin:
                continue
            v = volume(sa.intersect(sb))
            if v > 0.05:
                rows.append((f"{a} x {b}", v))
    if verbose:
        print(f"S = {L.S:.1f}: H {H:.2f}, core x +-{XW:.2f}, y +-{PY:.2f}, z +-{ZC:.2f}; foot {L.TF:.2f}; sled plate at y {L.YA:.2f}")
        for n in L.notes:
            print("  " + n)
        for r, v in rows:
            print(f"  CLASH {r}: {v:.2f} mm3")
        print(f"  {len(rows)} clashes")
    return rows


# The layout that closes at 56 mm (0 overlaps, smallest clearance 0.30 mm); run at another S it
# shows what blocks there. 50, 52 and 54 fail (see the report printed for each).
LAYOUT_56 = dict(
    comp_deg=180, motor_deg=110,  # compound at (-11, 0), disc motor at (-14.6, 9.9): the disc train itself is the 50 mm one
    n_idler=18, n_motor=16,  # hinge train 12T axle gear, two 18T idlers, 16T motor gear (m0.4)
    deploy_x0=-9.2, deploy_z=8.2, screw_len=11.0,  # FIT0577 along x, screw cut to 11 mm
    cell=(3, 35, 25), cell_at=(-8.7, -17.5, -1.5),  # the 302535 cell flat in the mid-plane
    bolt_xy=(-21.2, -9.5), bolt_len=3.0,
)


def gaps(L, near=2.0, show=10):
    """Smallest clearances between parts that are not meant to touch (pairs within `near` mm)."""
    parts = build(L)
    names = sorted(parts)
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if frozenset((a, b)) in INTENDED:
                continue
            p, q = parts[a][0].val().BoundingBox(), parts[b][0].val().BoundingBox()
            if p.xmax + near < q.xmin or q.xmax + near < p.xmin or p.ymax + near < q.ymin or q.ymax + near < p.ymin or p.zmax + near < q.zmin or q.zmax + near < p.zmin:
                continue
            d = parts[a][0].val().distance(parts[b][0].val())
            if d < near:
                out.append((d, a, b))
    out.sort()
    for d, a, b in out[:show]:
        print(f"  gap {a} - {b}: {d:.2f} mm")
    return out


if __name__ == "__main__":
    for s in [float(a) for a in sys.argv[1:] if not a.startswith("-")] or [56.0]:
        layout = Layout(S=s, **LAYOUT_56)
        check(layout)
        if "--gaps" in sys.argv:
            gaps(Layout(S=s, **LAYOUT_56), near=1.0, show=40)
