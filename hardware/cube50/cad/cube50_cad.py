"""Cube60: one modular cube (S = 60 mm, one parameter) with two hinged, turning feet. Every part,
checks, mass. The directory keeps its cube50 name; the size lives in S below.

Run from the repo root:  nice -n 10 .venv/bin/python hardware/cube50/cad/cube50_cad.py [--quick]
Writes into hardware/cube50/cad/:
  print/<part>.stl and .step   every printed part, in its print orientation
  bought/<part>.step           envelopes of the bought parts (motors, gears, bearing, cells, boards)
  bought/*.dxf                 laser-cutting outlines (lip rings, sled plates)
  cube50_assembly.step         the whole cube, feet stowed
  cube50_open90.step           the same with the top foot open 90 degrees (a floor pivot)
  report.txt                   interference, clearance, sled-travel, assembly-order and driver-path
                               checks, mass and inertia
--quick skips the STEP/STL export (checks and report only).

Coordinates: the cube spans x, y, z = -H..H, centre at the origin. +z is the top foot, -z the bottom
foot (the same parts turned 180 degrees about x). Both feet hinge on the two y-direction edges of the
+x face: the top foot about the line x = H, z = H, the bottom about x = H, z = -H. Every edge carries
a chamfer C; the four chamfers that meet at a lattice edge leave a square channel round the edge
line (inscribed radius C / sqrt 2).

Mechanisms (argued in hardware/cube50/DESIGN.md):
1. Pivot (retractable hinge on a sled, 2026-11-17): the hinge motor, its train, the axle and two
   stainless end plates form one rigid sled that slides RETRACT in x and in z (45 degrees). Stowed,
   every hinge part is at least 0.2 mm inside every face; deployed, the axle lies on the edge line.
   The foot knuckle slides in a guide in the foot and is keyed to the axle by a flat. A FIT0577
   lead-screw gearmotor along x pushes a nut carrier; a pin on the carrier drives the -y plate
   through a slot along z, and the plates' pockets in the body (45 degree guides) turn that push
   into the diagonal stroke. Two tact switches per sled (IN at the gearbox face, OUT in the +x wall).
   The hinge encoder reads a magnet on the axle end (the foot angle, 1:1).
2. Rotation: a disc in each foot on a 6700 bearing, driven when the foot is stowed by a vertical N20
   150:1 in the core through 12T-30T and 12T-32T hardened m0.5 stages (6.67:1) and a centre shaft
   whose spigot and dog enter the disc hub. When the foot opens, a spring pin locks the disc at the
   nearest quarter turn.
3. Bond: four M2.5 pan-head pins rise from the disc and turn 12 degrees under the partner face's
   stainless lip ring (bistable cam ring, two SMA wires). Every face carries the same four slots;
   face magnets (sized by the magnet design) sit in pockets inside the lip ring.
4. Stow latch: a spring bolt in the core enters each stowed foot; an SMA wire pulls it.
"""

import functools
import math
import sys
from pathlib import Path

import cadquery as cq

HERE = Path(__file__).parent
QUICK = "--quick" in sys.argv

# ---------------------------------------------------------------- size (mm)
S = 60.0
H = S / 2
K = S / 50.0  # face-relative dimensions are the 50 mm design's times K
C = 0.15 * S  # edge chamfer: 9.0, channel inscribed radius 6.36
GAP = 0.6  # printed part to printed part, static
EDGE_END = H - 5.0  # every hinge part ends at least 5 mm short of the cube corners

# ---------------------------------------------------------------- foot and disc stack (top foot)
BEARING = (10.0, 19.0, 5.0)  # 6800ZZ (bought): C0 0.65-0.93 kN by maker (6700: 0.435 kN, too small for the bond's pull-out)
LIP_T = 1.0 * K  # printed skin + stainless lip ring
SKIN = 0.5 * K
PLATE_T = LIP_T - SKIN  # 0.6 mm stainless lip ring
RISE = 2.2 * K
SLIDER_T = 1.6  # the pin slider is an M2.5 thin nut (DIN 439, 1.6 thick, 5 AF)
RAMP_BASE = 0.8 * K
BAY_Z0 = H - LIP_T - RISE - SLIDER_T - RAMP_BASE
TF = BEARING[2] + 0.05 + 0.35 + (H - BAY_Z0)  # bearing (pressed in from the foot's inner face, no floor), gap, disc floor, bond bay
ZF = H - TF  # inner face of the top foot
ZC = ZF - GAP  # top of the core
BEARING_Z = (ZF, ZF + BEARING[2])
BEARING_LIP = (8.3, 9.9, 0.6)  # foot collar lip over the outer race (inner r, outer r, thickness): stops the bond pulling the disc out
DISC_Z0 = BEARING_Z[1] + 0.05
DISC_D = 34.0
PIN_R = 10.0 * K
PIN_ANGLES = [22.5 + 90 * k for k in range(4)]
SLOT_ANGLES = [67.5 + 90 * k for k in range(4)]
SLOT_ARC = 14.0  # degrees of the slot under its lip
POCKET_DEPTH = 3.2 * K  # passive face: lip + room for the partner's pin heads
PLATE_R = (9.1, 16.2)  # lip ring; inner radius raised from 6.6 to leave the face magnets room
BAY_R = (9.1, 15.5)  # mechanism bay in the disc
CAM_R = (10.4, 13.8)  # narrowed 2026-11-17: clears the bearing lip (inside) and the lock pin (outside)
HEAD_R = 2.25  # M2.5 pan head (4.5 mm), the pin
STEM_R = 1.25
CONTACT_R = (12.2, 13.8)  # ring tracks under the disc plate, sprung contacts in the foot
LOCK_R = 15.6  # quarter-turn lock: spring pin in the foot, SMA-released, into notches of a steel ring under the disc rim
LOCK_RING = (15.0, 17.0, 0.8)  # 304 stainless, laser cut: inner r, outer r, thickness; 4 notches open to the inside
LOCK_PIN_D, LOCK_STROKE = 2.5, 1.1  # hardened dowel; engaged 0.75 into the ring

# face magnets (provisional, set by the magnet design): 8 sites at +-MAG_DELTA + 90k degrees
MAG_R, MAG_DELTA, MAG_D, MAG_T, MAG_SKIN = 6.3, 22.5, 4.0, 1.0, 1.1  # N35 4 x 1 axial discs (magnet design, provisional until bench step 0; skin may move 0.9-1.3)
MAG_ANGLES = [s * MAG_DELTA + 90 * k for k in range(4) for s in (1, -1)]

# ---------------------------------------------------------------- body
XW = H - 3.4 * K  # inner face of the x walls (the face pockets and a skin)
PY = XW  # inner face of the y walls
FY = 19.0  # foot half-width in y (19.0: the deploy motor drops past the slot edge)
PYF = FY + GAP  # the body's foot slot
KNUCKLE_Y = 15.0  # foot knuckle |y| <= this

# ---------------------------------------------------------------- gears
M = 0.5  # every gear in the cube is module 0.5 hardened steel


def pitch_r(t, m=M):
    return m * t / 2


def tip_r(t, m=M):
    return m * (t + 2) / 2


def meet(a, ra, b, rb, side=1):
    """A point at distance ra from a and rb from b (side picks one of the two)."""
    d = math.dist(a, b)
    ux, uz = (b[0] - a[0]) / d, (b[1] - a[1]) / d
    p = (ra * ra - rb * rb + d * d) / (2 * d)
    h = math.sqrt(max(ra * ra - p * p, 0.0))
    return (a[0] + p * ux - side * h * uz, a[1] + p * uz + side * h * ux)


# ---------------------------------------------------------------- hinge sled (top hinge)
KR = 2.8  # knuckle radius about the axle
AXLE_D = 2.0
N_MOTOR, N_BIG, N_SMALL, N_IDLER, N_AXLE = 12, 25, 14, 16, 14  # motor gear; compound big/small; idler; axle gear
HINGE_RATIO = (N_BIG / N_MOTOR) * (N_AXLE / N_SMALL)  # 2.083: foot turns 2.08x slower than the motor output (2026-11-17: 1.5 -> 2.08 for side-neighbour drag)
GEAR_FACE = 2.8
RETRACT = max(KR, tip_r(N_AXLE)) + 0.2  # 4.2: stowed, everything 0.2 inside the disc plane and the +x face
P = (H, H)  # deployed axle: on the edge line
PR = (H - RETRACT, H - RETRACT)  # stowed axle
STROKE = RETRACT * math.sqrt(2)
SLIDE = (RETRACT, RETRACT)  # stroke vector in (x, z)
PLATE_S = 1.2  # sled end plates, laser-cut 1.2 mm 304 stainless
YA = 19.9  # inner face of the +y plate (it carries the motor and the train)
YB = 19.6  # inner face of the -y plate (pin slot, encoder board)
L1_Y = (YA + PLATE_S + 0.2, YA + PLATE_S + 0.2 + GEAR_FACE)  # outside the +y plate: motor gear, compound big
L2_Y = (YA - 0.3 - GEAR_FACE, YA - 0.3)  # inside: compound small, idler, axle gear
NOTCH_Y = L2_Y[0] - 0.9  # the foot is pocketed beyond this for the inner gears
AXLE_Y = (-YB - PLATE_S - 0.3, YA + PLATE_S)  # 2 mm ground rod with a full-length D flat
N20_W, N20_H = 12.0, 10.0  # N20 gearmotor cross-section (Pololu drawing)
HINGE_MOTOR_L = 26.5  # 1000:1 HP N20 (long gearbox)
HINGE_SHAFT_L = 4.6  # output shaft shortened from 9 to 4.6 mm (ends flush with the motor gear)
MX1 = XW - 0.3 - RETRACT  # stowed hinge motor: +x face
MZ1 = ZC - 0.3 - RETRACT  # stowed hinge motor: +z face
G2 = (MX1 - N20_W / 2, MZ1 - N20_H / 2)  # motor gear centre (x, z), stowed
_DI = (pitch_r(N_IDLER) + pitch_r(N_AXLE)) / math.sqrt(2)
GI = (PR[0] - _DI, PR[1] - _DI)  # idler straight in from the axle: deployed, it stays inside the cell
GC = meet(GI, pitch_r(N_SMALL) + pitch_r(N_IDLER), G2, pitch_r(N_MOTOR) + pitch_r(N_BIG), side=-1)  # compound
HENC_MAG = (2.0, 1.5)  # hinge encoder magnet on the axle end: radius, thickness (4 x 1.5 diametric N52, about 47 mT at the die, derived; 4 x 1 N35 gave under the AS5600's 30 mT)
HENC_Y = (AXLE_Y[0] - HENC_MAG[1], AXLE_Y[0])
HENC_GAP = 0.3  # running gap, magnet face to chip top (the magnet turns with the axle, the board rides on the same sled)
HENC_BOARD_Y = (HENC_Y[0] - HENC_GAP - 1.5 - 0.6, HENC_Y[0] - HENC_GAP - 1.5)  # 0.6 mm board, 1.5 mm SOIC-8 toward the magnet

# ---------------------------------------------------------------- deploy drive (top sled)
FIT_L = 25.0  # DFRobot FIT0577 body with gearbox (allow 25)
SCREW_L = 13.0  # its M3 lead screw, cut from 20
DEP_Y, DEP_Z = -14.4, 9.2  # axis of the screw (along x)
DEP_X0 = -10.3  # motor body x DEP_X0 .. DEP_X0 + FIT_L
SW = (2.0, 4.0, 3.0)  # tact switch body (x along its plunger, y, z): 3 x 4 x 2 SMD
SW_YZ = (DEP_Y - 2.0, DEP_Y + 2.0, DEP_Z + 1.8, DEP_Z + 4.8)  # both switches sit off the screw axis
INB_X = (DEP_X0 + FIT_L, DEP_X0 + FIT_L + 0.8)  # IN switch board, on the gearbox face round the screw
CAR_X = (INB_X[1] + SW[0] + 0.2, INB_X[1] + SW[0] + 0.2 + 4.0)  # nut carrier (stowed): M3 nut in 0.8 walls
OUT_X = (CAR_X[1] + RETRACT + 0.05, CAR_X[1] + RETRACT + 0.05 + SW[0])  # OUT switch, in the +x wall
PIN_D = 2.0  # carrier pin (2 x 6 dowel) into the -y plate's slot
PIN_XZ = ((CAR_X[0] + CAR_X[1]) / 2, DEP_Z - 3.4)

# ---------------------------------------------------------------- disc drive (the 50 mm design's)
# The motor and the three stages are parameters: swap the motor class or the counts here and the
# layout below follows (positions come from the meshes; check the packing with the CAD checks).
DISC_MOTOR = {  # Pololu #3079 (298:1 HPCB 6 V, extended back shaft) with a custom in-outline encoder; figures: Pololu
    "free_rpm": 110.0, "stall_Nm": 0.333, "stall_A": 1.5, "free_A": 0.15, "rating_Nm": 0.196,
    "L": 25.6, "enc_L": 5.0, "shaft_L": 10.0}  # OL max 25.6 (Pololu drawing); encoder: 6-pole disc on the back shaft and a 10 x 12 mm two-Hall board inside the motor's outline, 5 mm (Pololu #4761 adds 5 mm but its 14.2 x 11.6 board pokes 4.2 mm out of one side, which fits nowhere here)
DT = (12, 40, 14, 36, 12, 20)  # motor pinion; C1 big, C1 small; C2 big, C2 small; centre gear
DT_M = (0.5, 0.5, 0.8)  # module of stages 1, 2, 3
DISC_RATIO = (DT[1] / DT[0]) * (DT[3] / DT[2]) * (DT[5] / DT[4])  # 14.29
CHOP_A = 0.94  # DRV8833 hardware current limit: the gearbox's 2 kg cm
BRIDGE_Z = (3.3, 4.6)  # shelf: pin presses, centre shaft bushing, both motor slots
LA = (4.9, 9.4)  # stage 3: C2 small and the centre gear (m0.8, 4.5 face)
LB = (9.7, 13.7)  # stage 2: C1 small and C2 big (m0.5, 4.0 face)
LC = (14.0, 16.0)  # stage 1: motor pinion and C1 big (m0.5, 2.0 face)
PLATE_Z = (16.4, ZC)  # 1.2 mm 304 drive plate over the train, on the deploy straps
DISC_MOTOR_XY = (-17.0, 19.0)  # against the +y wall; the bottom drive's is at -y
C2_XY = tuple((pitch_r(DT[4], DT_M[2]) + pitch_r(DT[5], DT_M[2])) * f(math.radians(170.0)) for f in (math.cos, math.sin))
C1_XY = max((meet(C2_XY, pitch_r(DT[2], DT_M[1]) + pitch_r(DT[3], DT_M[1]), DISC_MOTOR_XY, pitch_r(DT[0], DT_M[0]) + pitch_r(DT[1], DT_M[0]), side=sd) for sd in (1, -1)), key=lambda c: c[1])
DRIVE_POST = (7.0, 14.0)  # the drive plate's third screw, on a post from the bridge
MOTOR_FACE_Z = LB[1]
SHAFT_D = 5.0  # hardened centre shaft
SHAFT_Z = (BRIDGE_Z[0] - 1.0, BEARING_Z[0] + 2.4)  # below the bridge (E-clip) to the stainless hub
CROSS_PIN = (2.5, 8.0)  # hardened dowel through the shaft, in the hub's slot
HUB = (5.0, 7.6, 1.2)  # 303 stainless hub: bore-seat radius, flange radius, flange thickness


# ---------------------------------------------------------------- cell, board, bolts
CELL = (3.0, 35.0, 25.0)  # 302535 LiPo, 1S 220 mAh: thickness (z), length (along x), width (along y)
CELL_X0 = -10.8  # slides in along +x through the -x opening before the cover goes on
BOARD = (-XW, -XW + 3.1, -18.5, 18.5, -16.8, 16.8)  # main board + parts on its inner side, screwed to the -x cover
USB_YZ = (17.0, 11.5)  # USB-C opening in the -x cover, long side along z (corner outside the lip ring)
COVER = (19.6, 17.1)  # -x cover half-extents (y, z); the body opening is 0.5 larger all round
COVER_SCREWS = ((-16.5, 10.0), (-16.5, -10.0), (16.5, -10.0))  # (y, z): M2 x 6 countersunk A2 into brass inserts in body tabs
BOLT_XY = (-19.8, -6.8)  # moved from -9.5: the other drive's motor encoder clears its guide


# ---------------------------------------------------------------- helpers
def box(x0, x1, y0, y1, z0, z1):
    return cq.Workplane("XY").box(x1 - x0, y1 - y0, z1 - z0).translate(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))


def cyl_z(r, z0, z1, x=0.0, y=0.0):
    return cq.Workplane("XY").circle(r).extrude(z1 - z0).translate((x, y, z0))


def cyl_y(r, y0, y1, x=0.0, z=0.0):
    return cq.Workplane("XZ").circle(r).extrude(-(y1 - y0)).translate((x, y0, z))


def cyl_x(r, x0, x1, y=0.0, z=0.0):
    return cq.Workplane("YZ").circle(r).extrude(x1 - x0).translate((x0, y, z))


def ring_z(r0, r1, z0, z1, x=0.0, y=0.0):
    return cyl_z(r1, z0, z1, x, y).cut(cyl_z(r0, z0 - 1, z1 + 1, x, y))


def polar(r, deg):
    return (r * math.cos(math.radians(deg)), r * math.sin(math.radians(deg)))


def flip(shape):
    """The bottom-foot copy of a top-foot part: 180 degrees about the x axis."""
    return shape.rotate((0, 0, 0), (1, 0, 0), 180)


def union(shapes):
    out = shapes[0]
    for s in shapes[1:]:
        out = out.union(s)
    return out


def outline():
    return cq.Workplane("XY").box(S, S, S).edges().chamfer(C)


def arc_slot(r, a0, a1, width, z0, z1, steps=6):
    """A slot of the given width along a circle of radius r from angle a0 to a1 (degrees)."""
    return union([cyl_z(width / 2, z0, z1, *polar(r, a0 + (a1 - a0) * k / steps)) for k in range(steps + 1)])


def hull_xy(points, z0, z1):
    """Convex hull of (x, y) points, extruded over z0..z1."""
    return cq.Workplane("XY").polyline(_hull(points)).close().extrude(z1 - z0).translate((0, 0, z0))


def _hull(points):
    pts = sorted(set((round(p[0], 4), round(p[1], 4)) for p in points))

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
    return lower[:-1] + upper[:-1]


def hull_xz(points, y0, y1):
    """Convex hull of (x, z) points, extruded over y0..y1."""
    pts = sorted(set((round(p[0], 4), round(p[1], 4)) for p in points))

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
    return cq.Workplane("XZ").polyline(lower[:-1] + upper[:-1]).close().extrude(-(y1 - y0)).translate((0, y0, 0))


def circle_pts(c, r, n=20):
    return [(c[0] + r * math.cos(2 * math.pi * k / n), c[1] + r * math.sin(2 * math.pi * k / n)) for k in range(n)]


def slid(shape, f):
    """A sled part moved by the fraction f of the stroke (0 stowed, 1 deployed)."""
    return shape.translate((SLIDE[0] * f, 0, SLIDE[1] * f)) if f else shape


def swept(shape, steps=4, back=False):
    """The volume a sled part sweeps over the stroke (back: the hole a fixed part needs in a sled part)."""
    sgn = -1 if back else 1
    return union([slid(shape, sgn * k / steps) for k in range(steps + 1)])


# ---------------------------------------------------------------- face pattern
def slots_cut(z_face, inward):
    """Bayonet slots of one face (at height z_face, material toward -z if inward=-1): for each slot
    angle an entry hole for the partner's pin head and an arc for its stem under the lip, then a
    pocket below the lip for the head."""
    lip_bottom = z_face + inward * LIP_T
    deep = z_face + inward * POCKET_DEPTH
    z0, z1 = min(z_face, lip_bottom) - 0.01, max(z_face, lip_bottom) + 0.01
    cuts = []
    for a in SLOT_ANGLES:
        cuts.append(cyl_z(2.2 * K, z0, z1, *polar(PIN_R, a)))  # head entry
        cuts.append(arc_slot(PIN_R, a, a - SLOT_ARC, 2.4 * K, z0, z1))  # stem
        cuts.append(arc_slot(PIN_R, a + 4, a - SLOT_ARC - 4, 4.8 * K, min(lip_bottom, deep), max(lip_bottom, deep)))  # head pocket
    return union(cuts)


def mag_pockets(z_face, inward):
    """The face-magnet pockets of one face: MAG_D (+0.1) wide, from MAG_SKIN under the face to the
    magnet's back (+0.1); the magnets drop in at a print pause."""
    a = z_face + inward * MAG_SKIN
    b = z_face + inward * (MAG_SKIN + MAG_T + 0.1)
    return union([cyl_z(MAG_D / 2 + 0.05, min(a, b), max(a, b), *polar(MAG_R, t)) for t in MAG_ANGLES])


@functools.cache
def magnets(z_face, inward):
    a = z_face + inward * MAG_SKIN
    b = z_face + inward * (MAG_SKIN + MAG_T)
    return union([cyl_z(MAG_D / 2, min(a, b), max(a, b), *polar(MAG_R, t)) for t in MAG_ANGLES])


def on_face(shape, axis, sign):
    """Move a feature built on the +z face (z = H outward) to another face."""
    if axis == "z":
        return shape if sign > 0 else shape.rotate((0, 0, 0), (1, 0, 0), 180)
    if axis == "x":
        s = shape.rotate((0, 0, 0), (0, 1, 0), 90)
        return s if sign > 0 else s.rotate((0, 0, 0), (0, 0, 1), 180)
    s = shape.rotate((0, 0, 0), (1, 0, 0), -90)
    return s if sign > 0 else s.rotate((0, 0, 0), (0, 0, 1), 180)


def plate_seat():
    """The slot for the stainless lip ring under the face's printed skin (filled at a print pause)."""
    return ring_z(PLATE_R[0] - 0.1, PLATE_R[1] + 0.1, H - SKIN - PLATE_T, H - SKIN)


@functools.cache
def lip_plate(active):
    """0.6 mm stainless ring (laser cut): the four slot lips the partner's pin heads hook under, and on
    a disc also the four arcs its own pins pass through."""
    r = ring_z(PLATE_R[0], PLATE_R[1], H - SKIN - PLATE_T, H - SKIN).cut(slots_cut(H, -1))
    if active:
        for a in PIN_ANGLES:
            r = r.cut(arc_slot(PIN_R, a, a + 12, 2 * HEAD_R + 0.2, H - LIP_T - 0.01, H + 0.01))
    return r


PASSIVE_FACES = (("x", 1), ("x", -1), ("y", 1), ("y", -1))


# ---------------------------------------------------------------- hinge sled (top hinge, stowed; slid() moves it)
U = (-1 / math.sqrt(2), -1 / math.sqrt(2))  # from the axle back along the stroke, into the cube


def _back(c, d):
    return (c[0] + U[0] * d, c[1] + U[1] * d)


SLOT_XZ = (PIN_XZ[0], PIN_XZ[1] - RETRACT / 2)  # the pin slot's centre in the -y plate (stowed frame)
HENC_STEM = (_back(PR, 4.0), _back(PR, 8.5))  # the hinge encoder board's stem: two screw points


def _left_half(c, r):
    """The half of a circle toward -x and its bottom point: a deployed plate's hull then stays inside the +x face."""
    return [p for p in circle_pts(c, r, 40) if p[0] <= c[0] + 1e-9]


def plate_py():
    """+y end plate (1.5 mm 304 stainless, laser cut): axle bearing hole, idler pin, compound bush,
    motor shaft hole and the motor's two M1.6 countersunk screws."""
    pts = _left_half(PR, KR) + circle_pts(GI, 2.5) + circle_pts(GC, 2.5) + circle_pts((G2[0] - 4.5, G2[1]), 1.8) + circle_pts((G2[0] + 4.5, G2[1]), 1.8)
    p = hull_xz(pts, YA, YA + PLATE_S).union(cyl_y(KR, YA, YA + PLATE_S, *PR))
    for c, r in ((PR, AXLE_D / 2 + 0.03), (GI, 0.8), (GC, 1.05), (G2, 1.6), ((G2[0] - 4.5, G2[1]), 0.85), ((G2[0] + 4.5, G2[1]), 0.85)):
        p = p.cut(cyl_y(r, YA - 1, YA + PLATE_S + 1, *c))
    return p


def plate_ny():
    """-y end plate (1.5 mm 304 stainless): axle bearing hole, the slot along z the carrier pin drives,
    two M1.6 tapped holes for the hinge encoder board."""
    y0, y1 = -YB - PLATE_S, -YB
    pts = _left_half(PR, KR) + circle_pts(HENC_STEM[1], 2.2) + circle_pts((SLOT_XZ[0], SLOT_XZ[1] - RETRACT / 2), 2.6) + circle_pts((SLOT_XZ[0], SLOT_XZ[1] + RETRACT / 2), 2.6)
    p = hull_xz(pts, y0, y1).union(cyl_y(KR, y0, y1, *PR))
    p = p.cut(cyl_y(AXLE_D / 2 + 0.03, y0 - 1, y1 + 1, *PR))
    p = p.cut(box(SLOT_XZ[0] - PIN_D / 2 - 0.05, SLOT_XZ[0] + PIN_D / 2 + 0.05, y0 - 1, y1 + 1, SLOT_XZ[1] - RETRACT / 2 - PIN_D / 2 - 0.05, SLOT_XZ[1] + RETRACT / 2 + PIN_D / 2 + 0.05))
    for c in HENC_STEM:
        p = p.cut(cyl_y(0.65, y0 - 1, y1 + 1, *c))
    return p


def knuckle():
    """Foot knuckle (printed, 100 % infill): a boss round the axle, a stem back along the stroke that
    runs in the foot's guide, the bore with the axle's flat."""
    s = cyl_y(KR, -KNUCKLE_Y, KNUCKLE_Y, *PR)
    stem = box(-6.0, 0.0, -KNUCKLE_Y, KNUCKLE_Y, -1.5, 1.5).rotate((0, 0, 0), (0, 1, 0), -45).translate((PR[0], 0, PR[1]))
    s = s.union(stem)
    bore = cyl_y(AXLE_D / 2 + 0.03, -KNUCKLE_Y - 1, KNUCKLE_Y + 1, *PR).cut(box(PR[0] + AXLE_D / 2 - 0.3, PR[0] + 2, -H, H, PR[1] - 2, PR[1] + 2))
    return s.cut(bore)


def axle():
    a = cyl_y(AXLE_D / 2 - 0.01, AXLE_Y[0], AXLE_Y[1], *PR)
    return a.cut(box(PR[0] + AXLE_D / 2 - 0.3, PR[0] + 2, -H, H, PR[1] - 2, PR[1] + 2))  # the D flat (0.3 deep)


def gear_y(teeth, c, y0, y1, bore=0.0):
    g = cyl_y(tip_r(teeth), y0, y1, *c)
    return g.cut(cyl_y(bore, y0 - 1, y1 + 1, *c)) if bore else g


def hinge_motor():
    m = box(MX1 - N20_W, MX1, YA - HINGE_MOTOR_L, YA, MZ1 - N20_H, MZ1)
    return m.union(cyl_y(1.5, YA, YA + HINGE_SHAFT_L, *G2))


@functools.cache
def sled_parts():
    """{name: (shape, kind, material)} of the top sled, stowed."""
    def e(c, y):  # E-clip for a 2 mm shaft (DIN 6799 size 1.5)
        return cyl_y(1.9, y, y + 0.4, *c).cut(cyl_y(0.75, y - 1, y + 1, *c))

    clips = union([e(GC, L2_Y[0] - 0.5), e(GC, L1_Y[1] + 0.1), e(GI, L2_Y[0] - 0.5)])  # the axle gear is trapped between the plate and the foot
    board = cyl_y(3.9, HENC_BOARD_Y[0], HENC_BOARD_Y[1], *PR).union(hull_xz(circle_pts(_back(PR, 2.0), 2.6) + circle_pts(HENC_STEM[1], 2.4), HENC_BOARD_Y[0], HENC_BOARD_Y[1]))
    chip = box(PR[0] - 2.45, PR[0] + 2.45, HENC_BOARD_Y[1], HENC_BOARD_Y[1] + 1.5, PR[1] - 1.95, PR[1] + 1.95).rotate((PR[0], 0, PR[1]), (PR[0], 1, PR[1]), 45)
    spacers = union([cyl_y(1.0, HENC_BOARD_Y[1], -YB - PLATE_S, *c).cut(cyl_y(0.45, -H, 0, *c)) for c in HENC_STEM])
    return {
        "axle": (axle(), "bought", "steel"),
        "knuckle": (knuckle(), "print", "PETG"),
        "plate_py": (plate_py(), "bought", "steel"),
        "plate_ny": (plate_ny(), "bought", "steel"),
        "hinge_motor": (hinge_motor(), "bought", "motor"),
        "g_motor": (gear_y(N_MOTOR, G2, *L1_Y, 1.5), "bought", "steel"),
        "g_compound": (gear_y(N_BIG, GC, *L1_Y).union(gear_y(N_SMALL, GC, *L2_Y)).union(cyl_y(1.0, L2_Y[0] - 0.5, L1_Y[1] + 0.5, *GC)), "bought", "steel"),
        "g_idler": (gear_y(N_IDLER, GI, *L2_Y, 0.8).union(cyl_y(0.75, L2_Y[0] - 0.5, YA + PLATE_S, *GI)), "bought", "steel"),
        "g_axle": (gear_y(N_AXLE, PR, *L2_Y, AXLE_D / 2), "bought", "steel"),
        "eclips": (clips, "bought", "steel"),
        "enc_board_hinge": (board.union(chip), "bought", "pcb"),
        "enc_magnet_hinge": (cyl_y(HENC_MAG[0], HENC_Y[0], HENC_Y[1], *PR), "bought", "magnet"),
        "enc_spacers": (spacers, "print", "PETG"),
    }


# ---------------------------------------------------------------- deploy drive (top sled; fixed to the body except carrier, nut, pin)
CAR_Y = (DEP_Y - 4.0, DEP_Y + 4.0)
CAR_Z = (DEP_Z - 5.0, DEP_Z + 5.0)
CRADLE = (DEP_X0 - 0.3, INB_X[0] - 0.1, DEP_Y - N20_H / 2 - 1.6, DEP_Y + N20_H / 2 + 1.6, DEP_Z - N20_W / 2 - 1.4, DEP_Z + N20_W / 2)
STRAP_XS = ((DEP_X0 + 1.0, DEP_X0 + 5.0), (DEP_X0 + 15.0, DEP_X0 + 19.0))  # two printed straps over the open cradle: each hooks under a ledge on the -y wall, one M2 self-tapping screw into the +y wall
PIN_Y = (-YB - PLATE_S - 0.2, CAR_Y[0] + 1.0)


@functools.cache
def deploy_fixed():
    fit = box(DEP_X0, DEP_X0 + FIT_L, DEP_Y - N20_H / 2, DEP_Y + N20_H / 2, DEP_Z - N20_W / 2, DEP_Z + N20_W / 2)
    fit = fit.union(cyl_x(1.5, INB_X[0], INB_X[0] + SCREW_L, DEP_Y, DEP_Z))
    inb = box(INB_X[0], INB_X[1], DEP_Y - 5.0, DEP_Y + 5.0, DEP_Z - 5.0, DEP_Z + 5.0).cut(cyl_x(1.7, INB_X[0] - 1, INB_X[1] + 1, DEP_Y, DEP_Z))
    sw_in = box(INB_X[1], INB_X[1] + SW[0], *SW_YZ)
    sw_out = box(*OUT_X, *SW_YZ)
    y0, y1, z1 = CRADLE[2], CRADLE[3], CRADLE[5]
    strap = union([box(a, b, y0 + 0.3, y1, z1, z1 + 1.2).cut(cyl_z(1.1, z1 - 1, z1 + 2, (a + b) / 2, y1 - 0.8)) for a, b in STRAP_XS])
    return {
        "deploy_motor": (fit, "bought", "motor"),
        "deploy_in_board": (inb, "bought", "pcb"),
        "switch_in": (sw_in, "bought", "pcb"),
        "switch_out": (sw_out, "bought", "pcb"),
        "deploy_strap": (strap, "print", "PETG"),
    }


@functools.cache
def deploy_moving():
    """Carrier (printed, holds the M3 brass nut), the nut and the 2 mm pin: these move along x only."""
    car = box(CAR_X[0], CAR_X[1], *CAR_Y, *CAR_Z).cut(cyl_x(1.65, CAR_X[0] - 1, CAR_X[1] + 1, DEP_Y, DEP_Z))
    xm = (CAR_X[0] + CAR_X[1]) / 2
    nut = cq.Workplane("YZ").polygon(6, NUT_AF / math.cos(math.radians(30))).extrude(NUT_T).translate((xm - NUT_T / 2, DEP_Y, DEP_Z))
    car = car.cut(nut.union(box(xm - NUT_T / 2 - 0.05, xm + NUT_T / 2 + 0.05, DEP_Y - NUT_AF / 2 - 0.05, DEP_Y + NUT_AF / 2 + 0.05, DEP_Z, CAR_Z[1] + 1)))  # nut drops in from the top
    car = car.cut(cyl_y(PIN_D / 2 - 0.02, CAR_Y[0] - 1, PIN_Y[1], *PIN_XZ))  # pin press fit (0.02)
    car = car.cut(box(CAR_X[0] - 1, CAR_X[1] + 1, DEP_Y - RAIL_W / 2 - 0.25, DEP_Y + RAIL_W / 2 + 0.25, CAR_Z[0] - 1, CAR_Z[0] + 1.05))  # groove over the tongue
    nut = nut.cut(cyl_x(1.5, xm - 2, xm + 2, DEP_Y, DEP_Z))
    return {
        "carrier": (car, "print", "PETG"),
        "nut": (nut, "bought", "brass"),
        "pin": (cyl_y(PIN_D / 2, *PIN_Y, *PIN_XZ), "bought", "steel"),
    }


NUT_AF, NUT_T = 5.5, 2.4  # M3 brass hex nut, DIN 934
RAIL_W = 2.0  # the carrier's anti-rotation tongue


# ---------------------------------------------------------------- body
def _swept_hull(pts, y0, y1):
    """A prism over y0..y1 whose xz section is the hull of pts and pts slid over the stroke."""
    return hull_xz(pts + [(x + SLIDE[0], z + SLIDE[1]) for x, z in pts], y0, y1)


def sled_pockets():
    """Room the top sled needs in the body's top layer (|y| > PYF above ZC): each part's xz outline
    swept over the stroke plus clearance, open toward the foot slot so the plate subassemblies slide
    in along y. The pockets also guide the plates along the 45 degree stroke."""
    yo = PYF - 0.5
    plate_p = _left_half(PR, KR + 0.3) + circle_pts(GI, 2.8) + circle_pts(GC, 2.8) + circle_pts((G2[0] - 4.5, G2[1]), 2.1) + circle_pts((G2[0] + 4.5, G2[1]), 2.1)
    cuts = [_swept_hull(plate_p, yo, YA + PLATE_S + 0.3)]
    for c, t in ((G2, N_MOTOR), (GC, N_BIG)):
        cuts.append(_swept_hull(circle_pts(c, tip_r(t) + 0.4), yo, L1_Y[1] + 0.5))
    for c, t in ((GC, N_SMALL), (GI, N_IDLER), (PR, N_AXLE)):
        cuts.append(_swept_hull(circle_pts(c, tip_r(t) + 0.4), yo, L2_Y[1] + 0.3))
    ny = _left_half(PR, KR + 0.3) + circle_pts(HENC_STEM[1], 2.5) + circle_pts((SLOT_XZ[0], SLOT_XZ[1] - RETRACT / 2), 2.9) + circle_pts((SLOT_XZ[0], SLOT_XZ[1] + RETRACT / 2), 2.9)
    cuts.append(_swept_hull(ny, -YB - PLATE_S - 0.3, -(yo - 1.6)))  # reaches 1.6 inside the slot: the plate drops in 1.4 mm short and steps out
    cuts += [_swept_hull(circle_pts(PR, KR + 0.3), yo, YA + PLATE_S + 0.3), _swept_hull(circle_pts(PR, KR + 0.3), -YB - PLATE_S - 0.3, -(yo - 1.6))]
    enc = circle_pts(PR, 4.3) + circle_pts(_back(PR, 2.0), 2.9) + circle_pts(HENC_STEM[1], 2.7)
    cuts.append(_swept_hull(enc, HENC_BOARD_Y[0] - 0.3, -yo))
    return union(cuts)


def bridge():
    """Shelf under each disc drive: the C1 and C2 pin presses, the centre shaft bushing, both motors'
    slots (the motor and its pinion drop in straight through the strip's opening), and the
    post that carries the drive plate's third screw."""
    b = box(-22.5, 8.8, -PY, PY, *BRIDGE_Z)
    b = b.cut(box(CRADLE[0] - 2.0, 9.0, -PY - 1, CRADLE[3] + 0.6, BRIDGE_Z[0] - 1, BRIDGE_Z[1] + 1))  # the deploy cradle (and room to drop the deploy module in 1.5 mm short of home)
    b = b.cut(cyl_z(SHAFT_D / 2 + 0.05, BRIDGE_Z[0] - 1, BRIDGE_Z[1] + 1))  # centre shaft bushing (ream 5.0)
    b = b.cut(cyl_z(1.48, BRIDGE_Z[0] - 1, BRIDGE_Z[1] + 1, *C1_XY))  # 3 mm C1 pin, 0.04 press
    b = b.cut(cyl_z(1.98, BRIDGE_Z[0] - 1, BRIDGE_Z[1] + 1, *C2_XY))  # 4 mm C2 pin, 0.04 press
    mx, my = DISC_MOTOR_XY
    for sg in (1, -1):  # crush-rib slots (ribs 0.1 into the motor, not drawn)
        y0, y1 = my - N20_W / 2 - 0.2, my + N20_W / 2 + 0.2
        b = b.cut(box(mx - N20_H / 2 - 0.2, mx + N20_H / 2 + 0.2, min(sg * y0, sg * y1), max(sg * y0, sg * y1), BRIDGE_Z[0] - 1, BRIDGE_Z[1] + 1))
    b = b.union(cyl_z(1.75, BRIDGE_Z[1] - 0.01, PLATE_Z[0], *DRIVE_POST).cut(cyl_z(0.8, PLATE_Z[0] - 6, PLATE_Z[0] + 1, *DRIVE_POST)))
    return b


STRIP = (-22.2, 5.3, PYF, 25.4, LC[1] + 0.4)  # removable strip of the +y wall over the top drive: x0, x1, y0, y1, z0 (to the top face)
STRIP_SCREWS = (-19.0, 2.5)  # x of its two M2 x 8 countersunk A2 screws, from the +y face into brass inserts in the strip
STRIP_SCREW_Z = 18.6


@functools.cache
def strip():
    """Printed strip that closes the +y wall over the top drive (the motor and C1 go in straight down
    through its opening); 0.5 mm clear of the body all round, two screws through the +y face."""
    x0, x1, y0, y1, z0 = STRIP
    t = box(x0, x1, y0, y1, z0, H).intersect(outline())
    for x in STRIP_SCREWS:
        t = t.cut(cyl_y(1.6, y1 - 4.0, y1 + 1, x, STRIP_SCREW_Z))  # M2 x 3 heat-set insert
    return t


def strip_opening():
    x0, x1, y0, y1, z0 = STRIP
    o = box(x0 - 0.5, x1 + 0.5, y0 - 0.01, y1 + 0.5, z0 - 0.01, H + 1)
    for x in STRIP_SCREWS:
        o = o.union(cyl_y(1.1, y1, H + 1, x, STRIP_SCREW_Z)).union(cq.Workplane("XY").add(cq.Solid.makeCone(1.9, 0.0, 1.9, cq.Vector(x, H, STRIP_SCREW_Z), cq.Vector(0, -1, 0))))
    return o


def deploy_body():
    """Cradle (an open-topped channel the FIT0577 drops into), its rib to the -y wall, the carrier's
    rails from the +x wall, the OUT switch pocket and the screw's hole in the +x wall."""
    x0, x1, y0, y1, z0, z1 = CRADLE
    cr = box(x0, x1, -PY - 0.01, y1, z0, z1).cut(box(x0 - 1, x1 + 1, DEP_Y - N20_H / 2 - 0.1, DEP_Y + N20_H / 2 + 0.1, DEP_Z - N20_W / 2 - 0.1, z1 + 1))
    for a, b in STRAP_XS:  # pilot hole for each strap's M2 self-tapping screw, and the ledge its other end hooks under
        cr = cr.cut(cyl_z(0.8, z1 - 5, z1 + 1, (a + b) / 2, y1 - 0.8))
        cr = cr.union(box(a - 0.5, b + 0.5, -PY, y0 + 1.4, z1 + 1.4, z1 + 2.4))
    rails = box(CAR_X[0] + 1.0, XW + 0.01, DEP_Y - RAIL_W / 2, DEP_Y + RAIL_W / 2, CAR_Z[0] - 1.45, CAR_Z[0] + 0.8)  # one tongue under the carrier, in its groove: stops it turning
    holes = box(OUT_X[0] - 0.05, OUT_X[1] + 0.05, SW_YZ[0] - 0.1, SW_YZ[1] + 0.1, SW_YZ[2] - 0.1, SW_YZ[3] + 0.1)
    holes = holes.union(cyl_x(1.9, XW - 0.5, INB_X[0] + SCREW_L + 0.6, DEP_Y, DEP_Z))
    return cr.union(rails), holes


@functools.cache
def body():
    b = outline()
    for z0, z1 in ((ZC, H + 1), (-H - 1, -ZC)):  # foot slots
        b = b.cut(box(-H - 1, H + 1, -PYF, PYF, z0, z1))
    b = b.cut(box(-XW, XW, -PY, PY, -ZC - 0.01, ZC + 0.01).edges("|Z").chamfer(3.0))  # core cavity, open to both feet; its vertical corners chamfered so the wall stays 1.5 mm under the edge chamfers
    for f in (lambda s: s, flip):
        add, holes = deploy_body()
        b = b.union(f(add)).union(f(bridge()))
        b = b.cut(f(holes)).cut(f(sled_pockets()))
        b = b.cut(f(strip_opening()))
        b = b.cut(f(cyl_z(tip_r(DT[5], DT_M[2]) + 0.5, LA[0] - 0.3, CRADLE[5] + 1)))  # centre gear's room (and its way down): a window in the deploy cradle's +y wall
        b = b.cut(f(box(9.0, CRADLE[1] + 1, DEP_Y + N20_H / 2 - 0.1, CRADLE[3] + 0.1, 3.0, CRADLE[5] + 1)))  # cradle +y wall cut down beyond x 9: the hinge motor drops past it
        b = b.cut(f(cyl_y(PIN_D / 2 + 0.2, -H - 1, -YB + 0.01, *PIN_XZ)))  # access hole in the -y face: the carrier pin goes in last, along +y
        b = b.cut(f(cyl_y(AXLE_D / 2 + 0.3, YA, H + 1, *PR)))  # the axle's way in along -y, through the chamfer at the +y end
    b = b.cut(box(-H - 1, -XW + 0.01, -COVER[0] - 0.5, COVER[0] + 0.5, -ZC - 0.01, ZC + 0.01))  # -x opening for the cover
    for yy, zz in COVER_SCREWS:  # tabs from the y walls behind the cover, each with an M2 x 3 brass heat-set insert (3.2 hole through the 3.3 tab)
        s = 1 if yy > 0 else -1
        tab = box(-XW, -22.6, min(s * 14.0, s * (PY + 0.01)), max(s * 14.0, s * (PY + 0.01)), zz - 2.5, zz + 2.5)
        b = b.union(tab.cut(cyl_x(1.6, -XW - 1, -XW + 4.0, yy, zz)))
    b = b.cut(wire_channels())
    for axis, sign in PASSIVE_FACES:
        b = b.cut(on_face(slots_cut(H, -1), axis, sign)).cut(on_face(plate_seat(), axis, sign)).cut(on_face(mag_pockets(H, -1), axis, sign))
    return b


@functools.cache
def cover():
    """-x face cover (printed): the passive face's lip ring seat, slots and magnet pockets, the USB-C
    receptacle pocket, the board's four inserts and three countersunk screws into the body tabs."""
    cy, cz = COVER
    c = box(-H, -XW, -cy, cy, -cz, cz)
    c = c.cut(on_face(slots_cut(H, -1), "x", -1)).cut(on_face(plate_seat(), "x", -1)).cut(on_face(mag_pockets(H, -1), "x", -1))
    uy, uz = USB_YZ
    c = c.cut(box(-H + 0.3, -XW + 0.01, uy - 2.0, uy + 2.0, uz - 5.0, uz + 5.0))  # receptacle pocket, 0.3 skin
    c = c.cut(box(-H - 1, -H + 0.31, uy - 1.7, uy + 1.7, uz - 4.5, uz + 4.5))  # plug opening
    for yy, zz in BOARD_SCREWS:
        c = c.cut(cyl_x(1.6, -XW - 3.0, -XW + 0.01, yy, zz))  # M2 brass heat-set insert, 3.2 hole 3 deep
    for yy, zz in COVER_SCREWS:  # M2 countersunk (90 deg, 3.8 head) clearance holes
        c = c.cut(cyl_x(1.1, -H - 1, -XW + 1, yy, zz))
        c = c.cut(cq.Solid.makeCone(1.9, 0.0, 1.9, cq.Vector(-H, yy, zz), cq.Vector(1, 0, 0)))
    return c


BOARD_SCREWS = ((12.0, 14.0), (-12.0, 14.0), (12.0, -14.0), (-12.0, -14.0))  # (y, z): M2 x 4 A2 into brass inserts in the cover, outside the lip ring


def wire_channels():
    """Grooves (1.2 deep) in the inner walls that keep the leads off the moving sleds: along both y
    walls at the mid-plane (2.0 wide, stopping 3.5 short of the corners), and up the -x wall beside
    the cover (1.4 wide at y +-21.3, outside the cover opening); the walls keep 0.8 under the chamfers."""
    g = []
    for s in (1, -1):
        y0, y1 = (PY - 0.01, PY + 1.2) if s > 0 else (-PY - 1.2, -PY + 0.01)
        g.append(box(-XW + 3.5, XW - 3.5, y0, y1, -1.0, 1.0))
        g.append(box(-XW - 1.2, -XW + 0.01, s * 21.3 - 0.7, s * 21.3 + 0.7, -ZC + 1.0, ZC - 1.0))
    return union(g)


# ---------------------------------------------------------------- foot and disc (top; bottom = flip)
def knuckle_section(clear=0.0):
    """xz points of the knuckle (boss and stem), stowed."""
    stem = [_back(PR, d) for d in (0.0, 6.0)]
    pts = circle_pts(PR, KR + clear)
    for c in stem:
        for dx, dz in ((1, -1), (-1, 1)):
            pts.append((c[0] + dx * (1.5 + clear) / math.sqrt(2), c[1] + dz * (1.5 + clear) / math.sqrt(2)))
    return pts


@functools.cache
def foot():
    f = box(-H, H, -FY, FY, ZF, H).intersect(outline())
    f = f.union(box(PR[0] - 7.0, H, -KNUCKLE_Y - 0.8, KNUCKLE_Y + 0.8, ZF, H))  # square hinge corner round the knuckle guide
    f = f.cut(_swept_hull(knuckle_section(0.2), -KNUCKLE_Y - 0.2, KNUCKLE_Y + 0.2))  # knuckle guide, 0.2 slide clearance
    f = f.cut(_swept_hull(circle_pts(PR, AXLE_D / 2 + 0.3), -FY - 1, FY + 1))  # the axle's path beyond the knuckle
    notch = circle_pts(GC, tip_r(N_SMALL) + 0.5) + circle_pts(GI, tip_r(N_IDLER) + 0.5) + circle_pts(PR, tip_r(N_AXLE) + 0.5)
    notch = notch + [(x + SLIDE[0], z + SLIDE[1]) for x, z in notch]
    f = f.cut(hull_xz(notch + [(x, H + 1) for x, _ in notch], NOTCH_Y, FY + 1))  # room for the sled's inner gears (layer 2), open to the top so the foot drops past them
    for c, r in ((GC, tip_r(N_SMALL) + 0.5), (GI, tip_r(N_IDLER) + 0.5)):  # as the foot opens, the sled's inner gears sweep through its +y corner
        v = (c[0] + SLIDE[0] - P[0], c[1] + SLIDE[1] - P[1])
        rots = [(P[0] + v[0] * math.cos(math.radians(t)) - v[1] * math.sin(math.radians(t)), P[1] + v[0] * math.sin(math.radians(t)) + v[1] * math.cos(math.radians(t))) for t in range(0, 181, 10)]
        f = f.cut(union([hull_xz(circle_pts(rots[k], r) + circle_pts(rots[k + 1], r), NOTCH_Y, FY + 1) for k in range(len(rots) - 1)]))
    f = f.cut(cyl_z(DISC_D / 2 + 0.3, DISC_Z0 - 0.15, H + 1))  # disc recess
    f = f.cut(cyl_z(BEARING[1] / 2 - 0.025, ZF - 1, BEARING_Z[1] + 0.01))  # bearing seat, 0.05 interference, pressed in from below
    r0, r1, t = BEARING_LIP
    f = f.union(ring_z(r0, r1, BEARING_Z[1], BEARING_Z[1] + t)).union(ring_z(BEARING[1] / 2 - 0.05, r1, DISC_Z0 - 0.2, BEARING_Z[1]))  # collar and lip over the outer race
    f = f.cut(ring_z(CONTACT_R[0] - 0.3, CONTACT_R[1] + 0.3, DISC_Z0 - 2.2, DISC_Z0))  # sprung contacts
    f = f.cut(cyl_z(1.7, ZF - 1, ZF + 3.6, *BOLT_XY))  # stow bolt hole
    f = f.cut(cyl_z(LOCK_PIN_D / 2 + 0.05, ZF - 1, DISC_Z0 + 0.1, 0.0, -LOCK_R))  # lock pin bore
    f = f.cut(box(-3.5, 3.5, -LOCK_R - 2.0, -LOCK_R + 2.0, ZF - 1, ZF + 2.0))  # its lever and return spring
    f = f.cut(box(-29.0, -3.0, -LOCK_R - 0.6, -LOCK_R + 0.6, ZF - 1, ZF + 0.9))  # the SMA wire's groove, crimp at the -x end
    f = f.cut(box(-2.0, 2.0, 8.0, 15.0, ZF - 1, ZF + 1.2))  # FFC from the foot board to the core
    return f


@functools.cache
def disc():
    d = cyl_z(DISC_D / 2, DISC_Z0, H)
    d = d.cut(cyl_z(HUB[1] + 0.05, DISC_Z0 - 0.01, DISC_Z0 + HUB[2]))  # the stainless hub's flange
    d = d.cut(ring_z(BEARING_LIP[0] - 0.3, BEARING_LIP[1] + 0.3, DISC_Z0 - 0.01, BEARING_Z[1] + BEARING_LIP[2] + 0.05))  # relief over the foot's lip
    for a in HUB_SCREW_ANGLES:  # M2 x 3 brass heat-set inserts for the hub's three screws
        d = d.cut(cyl_z(1.6, DISC_Z0 + HUB[2] - 0.01, DISC_Z0 + HUB[2] + 3.0, *polar(HUB_SCREW_R, a)))
    d = d.cut(ring_z(LOCK_RING[0] - 0.05, LOCK_RING[1] + 1, DISC_Z0 - 0.01, DISC_Z0 + LOCK_RING[2]))  # lock ring seat (bonded)
    for k in range(4):  # the engaged lock pin reaches 0.65 past the ring's inner edge
        d = d.cut(cyl_z(LOCK_PIN_D / 2 + 0.1, DISC_Z0 - 0.01, DISC_Z0 + LOCK_RING[2], *polar(LOCK_R, 90 * k - 90)))
    d = d.cut(ring_z(BAY_R[0], BAY_R[1], BAY_Z0 - 0.05, H - LIP_T))  # mechanism bay under the lip
    d = d.cut(slots_cut(H, -1)).cut(plate_seat()).cut(mag_pockets(H, -1))
    for a in PIN_ANGLES:  # pin head holes in the lip and their 12-degree arcs
        d = d.cut(arc_slot(PIN_R, a, a + 12, 2 * HEAD_R + 0.2, H - LIP_T - 0.01, H + 0.01))
    return d.cut(ring_z(CONTACT_R[0], CONTACT_R[1], DISC_Z0 - 0.01, DISC_Z0 + 0.6))  # ring PCB seat


HUB_SCREW_R, HUB_SCREW_ANGLES = 6.2, (60.0, 180.0, 300.0)


@functools.cache
def hub():
    """303 stainless hub (turned): pressed into the 6800's bore, flange screwed to the disc by three
    M2 x 4 A2 screws; the 5 mm shaft's spigot enters its bore and the 2.5 mm cross pin its slot."""
    h = cyl_z(HUB[0], BEARING_Z[0], DISC_Z0).union(cyl_z(HUB[1], DISC_Z0, DISC_Z0 + HUB[2]))
    h = h.cut(cyl_z(SHAFT_D / 2 + 0.05, BEARING_Z[0] - 1, BEARING_Z[0] + 3.0))
    h = h.cut(box(-CROSS_PIN[0] / 2 - 0.05, CROSS_PIN[0] / 2 + 0.05, -CROSS_PIN[1] / 2 - 0.3, CROSS_PIN[1] / 2 + 0.3, BEARING_Z[0] - 1, BEARING_Z[0] + 2.6))
    for a in HUB_SCREW_ANGLES:
        h = h.cut(cyl_z(1.1, DISC_Z0 - 1, DISC_Z0 + HUB[2] + 1, *polar(HUB_SCREW_R, a)))
    return h


@functools.cache
def lock_parts():
    """The stowed-and-open disc lock (top foot): steel ring under the disc rim with four notches, the
    spring pin in the foot (engaged), its bellcrank lever and the SMA wire (0.15 mm Flexinol, 32 mm)."""
    ring = ring_z(LOCK_RING[0], LOCK_RING[1], DISC_Z0, DISC_Z0 + LOCK_RING[2])
    for k in range(4):
        n = box(LOCK_RING[0] - 0.5, LOCK_R, -LOCK_PIN_D / 2 - 0.05, LOCK_PIN_D / 2 + 0.05, DISC_Z0 - 1, DISC_Z0 + 2).union(cyl_z(LOCK_PIN_D / 2 + 0.05, DISC_Z0 - 1, DISC_Z0 + 2, LOCK_R, 0.0))
        ring = ring.cut(n.rotate((0, 0, 0), (0, 0, 1), 90 * k - 90))
    pin = cyl_z(LOCK_PIN_D / 2, ZF + 0.8, DISC_Z0 + LOCK_RING[2] - 0.05, 0.0, -LOCK_R)
    lever = box(-3.0, 3.0, -LOCK_R - 1.5, -LOCK_R + 1.5, ZF + 0.0, ZF + 0.8)
    sma = cyl_x(0.3, -28.5, -3.0, -LOCK_R, ZF + 0.45)  # drawn 0.6 thick (wire 0.15 in a crimped sleeve)
    return {
        "lock_ring": (ring, "bought", "steel"),
        "lock_pin": (pin, "bought", "steel"),
        "lock_lever": (lever, "print", "PETG"),
        "lock_sma": (sma, "bought", "steel"),
    }


@functools.cache
def cam_ring():
    r = ring_z(CAM_R[0], CAM_R[1], BAY_Z0, BAY_Z0 + RAMP_BASE)
    for a in PIN_ANGLES:  # the ramp each slider climbs when the ring turns (drawn as its full-height end block)
        r = r.union(cyl_z(1.4 * K, BAY_Z0 + RAMP_BASE, BAY_Z0 + RAMP_BASE + RISE, *polar(PIN_R, a - 22)))
    return r


@functools.cache
def pins():
    """M2.5 x 8 pan-head screws (head flush with the face when retracted) and their thin nuts (the sliders)."""
    z0 = BAY_Z0 + RAMP_BASE
    parts = []
    for a in PIN_ANGLES:
        x, y = polar(PIN_R, a)
        nut = cq.Workplane("XY").polygon(6, 5.0 / math.cos(math.radians(30))).extrude(SLIDER_T).translate((x, y, z0))
        parts.append(nut.union(cyl_z(STEM_R, z0 + SLIDER_T, H - LIP_T, x, y)).union(cyl_z(HEAD_R, H - LIP_T, H, x, y)))
    return union(parts)


def bolt():
    return cyl_z(1.5, PLATE_Z[0] - 1.5, ZF + 3.0, *BOLT_XY)  # guided by its hole in the drive plate


def floor_tile():
    """One cell of floor: the top of a passive cube, 10K thick, with the cube's chamfer on its top
    edges so tiles laid edge to edge leave the V-groove a hinge knuckle sits in."""
    t = box(-H, H, -H, H, -10 * K, 0).edges(">Z").chamfer(C)
    return t.cut(slots_cut(0.0, -1)).cut(mag_pockets(0.0, -1))


# ---------------------------------------------------------------- disc drive parts (top disc)
@functools.cache
def disc_drive():
    """Top disc drive: motor (with its encoder), three stages on two fixed pins and the centre
    shaft, the shaft's E-clip under the bridge, the 1.2 mm stainless drive plate over the train."""
    mx, my = DISC_MOTOR_XY
    m0, m1, m2 = DT_M
    z_bot = MOTOR_FACE_Z - DISC_MOTOR["L"]
    motor = box(mx - N20_H / 2, mx + N20_H / 2, my - N20_W / 2, my + N20_W / 2, z_bot, MOTOR_FACE_Z)
    motor = motor.union(box(mx - N20_H / 2, mx + N20_H / 2, my - N20_W / 2, my + N20_W / 2, z_bot - DISC_MOTOR["enc_L"], z_bot))  # encoder, inside the outline
    motor = motor.union(cyl_z(1.5, MOTOR_FACE_Z, LC[1], mx, my))  # D shaft, cut to length
    c1 = cyl_z(tip_r(DT[1], m0), *LC, *C1_XY).union(cyl_z(tip_r(DT[2], m1), LB[0], LC[0], *C1_XY)).cut(cyl_z(1.5, 0, 30, *C1_XY))
    c2 = cyl_z(tip_r(DT[3], m1), *LB, *C2_XY).union(cyl_z(tip_r(DT[4], m2), LA[0], LB[0], *C2_XY)).cut(cyl_z(2.0, 0, 30, *C2_XY))
    cg = cyl_z(tip_r(DT[5], m2), *LA).cut(cyl_z(SHAFT_D / 2, 0, 30))
    cross = cyl_y(CROSS_PIN[0] / 2, -CROSS_PIN[1] / 2, CROSS_PIN[1] / 2, 0.0, BEARING_Z[0] + 1.3)
    shaft = cyl_z(SHAFT_D / 2, *SHAFT_Z).cut(cyl_y(CROSS_PIN[0] / 2, -SHAFT_D, SHAFT_D, 0.0, BEARING_Z[0] + 1.3))  # cross-drilled 2.5
    eclip = ring_z(SHAFT_D / 2 - 0.4, SHAFT_D / 2 + 1.6, BRIDGE_Z[0] - 0.7, BRIDGE_Z[0] - 0.1)  # DIN 6799 size 4 for the 5 mm shaft
    sx = [((a + b2) / 2, CRADLE[3] - 0.8) for a, b2 in STRAP_XS]
    plate = hull_xy(circle_pts((0, 0), 4.5) + circle_pts(C1_XY, 3.0) + circle_pts(C2_XY, 3.5) + circle_pts(DRIVE_POST, 2.0) + circle_pts(BOLT_XY, 2.6) + sum((circle_pts(c, 2.5) for c in sx), []), *PLATE_Z)
    for c, r in [((0, 0), SHAFT_D / 2 + 0.05), (C1_XY, 1.52), (C2_XY, 0.77), (DRIVE_POST, 1.1), (BOLT_XY, 1.6)] + [(c, 1.1) for c in sx]:
        plate = plate.cut(cyl_z(r, 0, 30, *c))
    return {
        "disc_motor": (motor, "bought", "motor"),
        "pinion": (cyl_z(tip_r(DT[0], m0), *LC, mx, my).cut(cyl_z(1.5, 0, 30, mx, my)), "bought", "steel"),
        "c1": (c1, "bought", "steel"),
        "c2": (c2, "bought", "steel"),
        "c1_pin": (cyl_z(1.5, BRIDGE_Z[0], PLATE_Z[1], *C1_XY), "bought", "steel"),
        "c2_pin": (cyl_z(2.0, BRIDGE_Z[0], LB[1], *C2_XY).union(cyl_z(0.75, LB[1], PLATE_Z[1], *C2_XY)), "bought", "steel"),  # 4 mm, turned down to a 1.5 mm spigot through stage 1
        "centre_gear": (cg, "bought", "steel"),
        "centre_shaft": (shaft, "bought", "steel"),
        "cross_pin": (cross, "bought", "steel"),
        "shaft_eclip": (eclip, "bought", "steel"),
        "drive_plate": (plate, "bought", "steel"),
    }


def main_board():
    """Main board (1.6 FR4) on the cover's inner face, parts on its inner side, notched round the
    three cover tabs; the USB-C receptacle sits in the cover's pocket."""
    x0, x1, y0, y1, z0, z1 = BOARD
    uy, uz = USB_YZ
    b = box(x0, x0 + 1.6, y0, y1, z0, z1).union(box(x0 + 1.6, x1, -13.0, 13.0, -14.5, 14.5)).union(box(-H + 0.6, x0, uy - 1.75, uy + 1.75, uz - 4.6, uz + 4.6))
    for yy, zz in COVER_SCREWS:
        s = 1 if yy > 0 else -1
        b = b.cut(box(x0 - 1, x1 + 1, min(s * 13.5, s * 20), max(s * 13.5, s * 20), zz - 3.0, zz + 3.0))
    for yy, zz in BOARD_SCREWS:
        b = b.cut(cyl_x(1.1, x0 - 1, x0 + 2, yy, zz))
    return b


SLED_MOVES_WITH_FOOT = ("knuckle", "axle", "g_axle", "enc_magnet_hinge")  # these turn with the foot when it opens (the axle is keyed to the knuckle)
FOOT_PARTS = ("foot", "disc", "cam_ring", "pins", "bearing", "lip_disc", "hub", "mag_disc", "lock_ring", "lock_pin", "lock_lever", "lock_sma")


def build(top_open=0.0, deployed=(), sled=None):
    """Every part placed. top_open opens the top foot by that many degrees about its (deployed) hinge;
    `deployed` names hinges ('top', 'bottom') slid out; `sled` = {'top': f, 'bottom': f} sets a stroke
    fraction directly (0 stowed, 1 out). Returns {name: (shape, kind, material)}."""
    parts = {}
    frac = {t: (1.0 if t in deployed else 0.0) for t in ("top", "bottom")}
    if top_open:
        frac["top"] = 1.0
    frac.update(sled or {})

    def add(name, shape, kind, material, opens=False):
        if opens and top_open:
            shape = shape.rotate((P[0], 0, P[1]), (P[0], 1, P[1]), top_open)
        parts[name] = (shape, kind, material)

    add("body", body(), "print", "PETG")
    for tag, f in (("top", lambda s: s), ("bottom", flip)):
        op = tag == "top"
        add(f"foot_{tag}", f(foot()), "print", "PETG", op)
        add(f"disc_{tag}", f(disc()), "print", "PETG", op)
        add(f"cam_ring_{tag}", f(cam_ring()), "print", "PETG", op)
        add(f"pins_{tag}", f(pins()), "bought", "steel", op)
        add(f"bearing_{tag}", f(ring_z(BEARING[0] / 2, BEARING[1] / 2, *BEARING_Z)), "bought", "steel", op)
        add(f"lip_disc_{tag}", f(lip_plate(True)), "bought", "steel", op)
        add(f"hub_{tag}", f(hub()), "bought", "steel", op)
        for name, (shape, kind, mat) in lock_parts().items():
            add(f"{name}_{tag}", f(shape), kind, mat, op)
        add(f"mag_disc_{tag}", f(magnets(H, -1)), "bought", "magnet", op)
        for name, (shape, kind, mat) in sled_parts().items():
            add(f"{name}_{tag}", f(slid(shape, frac[tag])), kind, mat, op and name in SLED_MOVES_WITH_FOOT)
        for name, (shape, kind, mat) in deploy_fixed().items():
            add(f"{name}_{tag}", f(shape), kind, mat)
        for name, (shape, kind, mat) in deploy_moving().items():
            add(f"{name}_{tag}", f(shape.translate((RETRACT * frac[tag], 0, 0))), kind, mat)
        for name, (shape, kind, mat) in disc_drive().items():
            add(f"{name}_{tag}", f(shape), kind, mat)
        add(f"bolt_{tag}", f(bolt()), "bought", "steel")
    for axis, sign in PASSIVE_FACES:
        nm = f"{'p' if sign > 0 else 'n'}{axis}"
        add(f"lip_{nm}", on_face(lip_plate(False), axis, sign), "bought", "steel")
        add(f"mag_{nm}", on_face(magnets(H, -1), axis, sign), "bought", "magnet")
    add("cell", box(CELL_X0, CELL_X0 + CELL[1], -CELL[2] / 2, CELL[2] / 2, -CELL[0] / 2, CELL[0] / 2), "bought", "cell")
    add("side_cover", cover(), "print", "PETG")
    add("strip_top", strip(), "print", "PETG")
    add("strip_bottom", flip(strip()), "print", "PETG")
    add("main_board", main_board(), "bought", "pcb")
    return parts
# ---------------------------------------------------------------- checks
def volume(shape):
    try:
        return shape.val().Volume()
    except Exception:
        return float("nan")


def bbox_overlap(a, b, pad=0.0):
    p, q = a.val().BoundingBox(), b.val().BoundingBox()
    return not (p.xmax + pad < q.xmin or q.xmax + pad < p.xmin or p.ymax + pad < q.ymin or q.ymax + pad < p.ymin or p.zmax + pad < q.zmin or q.zmax + pad < p.zmin)


# pairs that touch on purpose (fits): seats, shafts in bores, gears on shafts, meshes are drawn as tip
# circles so meshing gears overlap by design (tooth depth), the body holds the bearings' neighbours.
INTENDED = {
    frozenset(p)
    for t in ("top", "bottom")
    for p in (
        (f"g_motor_{t}", f"g_compound_{t}"), (f"g_compound_{t}", f"g_idler_{t}"), (f"g_idler_{t}", f"g_axle_{t}"), (f"g_axle_{t}", f"axle_{t}"),
        (f"g_motor_{t}", f"hinge_motor_{t}"), (f"eclips_{t}", f"axle_{t}"), (f"eclips_{t}", f"g_compound_{t}"), (f"eclips_{t}", f"g_idler_{t}"),
        (f"pin_{t}", f"carrier_{t}"), (f"disc_motor_{t}", f"pinion_{t}"), (f"pinion_{t}", f"c1_{t}"), (f"c1_{t}", f"c2_{t}"), (f"c2_{t}", f"centre_gear_{t}"),
        (f"hub_{t}", f"bearing_{t}"), (f"foot_{t}", f"bearing_{t}"), (f"body", f"c1_pin_{t}"), (f"body", f"c2_pin_{t}"), (f"centre_shaft_{t}", f"centre_gear_{t}"),
        (f"centre_shaft_{t}", f"shaft_eclip_{t}"),
    )
}


# ---------------------------------------------------------------- clearance checks (2026-11-17)
CELL_BOX = None


def _cell_box():
    global CELL_BOX
    if CELL_BOX is None:
        CELL_BOX = box(-H, H, -H, H, -H, H)
    return CELL_BOX


def _solid(shape):
    return shape.val() if hasattr(shape, "val") else shape


def _compound(shapes):
    return cq.Compound.makeCompound([_solid(s) for s in shapes])


def _common(a, b):
    try:
        return a.intersect(b).Volume()
    except Exception:
        return float("nan")


def _bb(shape):
    return _solid(shape).BoundingBox()


def _near(parts, test):
    return [sh for _, (sh, _, _) in parts.items() if test(_bb(sh))]


def _matrix(rot, trans):
    return cq.Matrix([[float(rot[i][0]), float(rot[i][1]), float(rot[i][2]), float(trans[i])] for i in range(3)])


def _moved(shape, rot, trans):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf

    t = gp_Trsf()
    r = rot
    t.SetValues(float(r[0][0]), float(r[0][1]), float(r[0][2]), float(trans[0]), float(r[1][0]), float(r[1][1]), float(r[1][2]), float(trans[1]), float(r[2][0]), float(r[2][1]), float(r[2][2]), float(trans[2]))
    return cq.Shape.cast(BRepBuilderAPI_Transform(_solid(shape).wrapped, t, True).Shape())


def _rotations():
    import itertools

    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = [[0, 0, 0] for _ in range(3)]
            for i in range(3):
                m[i][perm[i]] = signs[i]
            det = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
            if det == 1:
                out.append(m)
    return out


def _apply(rot, v):
    return tuple(sum(rot[i][j] * v[j] for j in range(3)) for i in range(3))


def check_stowed_proud(parts):
    """Stowed: nothing beyond either disc plane; anything outside the cell only beyond +x, within the
    +x neighbour's chamfer void, and within |y| <= EDGE_END."""
    rows, worst, hit = [], 0.0, 0.0
    neighbours = _compound([outline().translate((S * dx, S * dy, 0)) for dx, dy in ((1, 0), (0, 1), (0, -1), (-1, 0))])
    zmax = max(_bb(sh).zmax for sh, _, _ in parts.values())
    zmin = min(_bb(sh).zmin for sh, _, _ in parts.values())
    for name, (sh, _, _) in sorted(parts.items()):
        out = sh.cut(_cell_box())
        v = volume(out)
        if v > 0.01:
            bb = out.val().BoundingBox()
            into = _common(_solid(out), neighbours)
            hit = max(hit, into)
            rows.append(f"  outside the cell: {name} {v:.2f} mm3 (into a lattice neighbour {into:.3f} mm3), x {bb.xmin:.2f}..{bb.xmax:.2f}, y {bb.ymin:.2f}..{bb.ymax:.2f}, z {bb.zmin:.2f}..{bb.zmax:.2f}")
            worst = max(worst, abs(bb.ymin), abs(bb.ymax))
    ok = zmax <= H + 1e-3 and zmin >= -H - 1e-3 and worst <= EDGE_END + 1e-3 and hit <= 0.05
    return [f"stowed, nothing proud of a disc plane: z {zmin:.3f}..{zmax:.3f}; outside-cell parts end at |y| {worst:.2f} (limit {EDGE_END}) and reach {hit:.3f} mm3 into side neighbours: {'PASS' if ok else 'FAIL'}"] + rows, ok


def check_partner_sweep(parts, step=15.0):
    """A partner bonded disc to disc on the top disc (its own hinge facing ours) turns a full turn."""
    ours = _compound(_near(parts, lambda b: b.zmax > ZC))
    theirs = _near(parts, lambda b: b.zmin < -ZC)
    worst, rows = 0.0, []
    ang = 0.0
    while ang < 360.0 - 1e-6:
        v = _common(ours, _compound([_solid(sh).translate(cq.Vector(0, 0, S)).rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), ang) for sh in theirs]))
        worst = max(worst, v)
        if v > 0.05:
            rows.append(f"  partner at {ang:.1f} deg: {v:.2f} mm3")
        ang += step
    ok = worst <= 0.05
    return [f"partner turning on the top disc, 0..360 deg every {step:g} deg: worst overlap {worst:.3f} mm3: {'PASS' if ok else 'FAIL'}"] + rows, ok


def tiles(n=1):
    return [floor_tile().translate((i * S, j * S, 0)) for i in range(-n, n + 1) for j in range(-n, n + 1)]


def check_yaw_sweep(parts, step=7.5):
    """Standing on the bottom disc on V-groove tiles (top at z = -25), the cube yaws a full turn."""
    floor = _compound([t.translate((0, 0, -H)) for t in tiles(1)])
    ours = _near(parts, lambda b: b.zmin < -ZC)
    worst, ang = 0.0, 0.0
    while ang < 90.0 - 1e-6:  # the tile floor repeats every 90 degrees under a cube turning about a cell centre
        worst = max(worst, _common(_compound([_solid(sh).rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), ang) for sh in ours]), floor))
        ang += step
    ok = worst <= 0.05
    return [f"standing cube yawing on V-groove tiles, 0..90 deg every {step:g} deg (period 90): worst overlap {worst:.3f} mm3: {'PASS' if ok else 'FAIL'}"], ok


def check_vertex_neighbours():
    """Our top hinge deployed; every neighbour cell round each end vertex of that edge, every
    orientation in which one of the neighbour's hinge edges runs through the vertex at right angles
    to ours, with that hinge deployed too (the worst case)."""
    ours_parts = build(deployed={"top"})
    theirs = {"top": build(deployed={"top"}), "bottom": build(deployed={"bottom"})}
    rows, worst, n = [], 0.0, 0
    for vy in (H, -H):
        vertex = (H, vy, H)
        near = lambda b: max(b.xmin - vertex[0], vertex[0] - b.xmax, b.ymin - vertex[1], vertex[1] - b.ymax, b.zmin - vertex[2], vertex[2] - b.zmax) < 18.0
        ours = _compound(_near(ours_parts, near))
        for cx in (0, S):
            for cy in (0, S if vy > 0 else -S):
                for cz in (0, S):
                    if (cx, cy, cz) == (0, 0, 0):
                        continue
                    for rot in _rotations():
                        for which, zc in (("top", H), ("bottom", -H)):
                            p0 = _apply(rot, (H, 0, zc))
                            p0 = (p0[0] + cx, p0[1] + cy, p0[2] + cz)
                            d = _apply(rot, (0, 1, 0))
                            if d[1] != 0:
                                continue  # parallel to ours, not perpendicular
                            w = tuple(vertex[i] - p0[i] for i in range(3))
                            cross = (w[1] * d[2] - w[2] * d[1], w[2] * d[0] - w[0] * d[2], w[0] * d[1] - w[1] * d[0])
                            if any(abs(c_) > 1e-6 for c_ in cross):
                                continue  # its hinge line misses the vertex
                            moved = [_moved(sh, rot, (cx, cy, cz)) for sh, _, _ in theirs[which].values()]
                            moved = [m for m in moved if near(m.BoundingBox())]
                            if not moved:
                                continue
                            n += 1
                            v = _common(ours, _compound(moved))
                            worst = max(worst, v)
                            if v > 0.05:
                                rows.append(f"  vertex {vertex}, neighbour cell ({cx}, {cy}, {cz}), its {which} hinge: {v:.2f} mm3")
    ok = worst <= 0.05
    return [f"perpendicular hinge edges at a shared vertex, both deployed: {n} placements, worst overlap {worst:.3f} mm3: {'PASS' if ok else 'FAIL'}"] + rows, ok


def check_pivot_channel():
    """Deployed hinge against what a pivot meets: (a) a climb, the foot bonded on top of a support
    cube (a stowed cube, placed in the foot's frame, turned so that its own hinges are not on our
    edge: the planner rule); (b) a floor roll, the foot bonded on V-groove
    tiles. Our body stays put; the foot and what it is bonded to turn about the edge line."""
    rows, worst = [], 0.0
    near = lambda b: b.xmax > H - 18.0 and b.zmax > H - 18.0
    support = build()
    for ang in (90.0, 135.0, 180.0):
        ps = build(ang)
        ours = _compound(_near(ps, near))
        moved = []
        for sh, _, _ in support.values():
            m = _solid(sh).rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), 180).translate(cq.Vector(0, 0, S)).rotate(cq.Vector(P[0], 0, P[1]), cq.Vector(P[0], 1, P[1]), ang)
            if near(m.BoundingBox()):
                moved.append(m)
        v = _common(ours, _compound(moved))
        worst = max(worst, v)
        rows.append(f"  climb, foot at {ang:.0f} deg: overlap {v:.3f} mm3")
    for ang in (0.0, 45.0, 90.0):
        ps = build(ang, deployed={"top"})
        ours = _compound(_near(ps, near))
        floor = []
        for t in tiles(1):
            m = _solid(t).rotate(cq.Vector(0, 0, 0), cq.Vector(1, 0, 0), 180).translate(cq.Vector(0, 0, H))
            if ang:
                m = m.rotate(cq.Vector(P[0], 0, P[1]), cq.Vector(P[0], 1, P[1]), ang)
            floor.append(m)
        v = _common(ours, _compound(floor))
        worst = max(worst, v)
        rows.append(f"  floor roll (mirrored: tiles on the top foot), foot at {ang:.0f} deg: overlap {v:.3f} mm3")
    ok = worst <= 0.05
    return [f"deployed hinge in the pivot channel: worst overlap {worst:.3f} mm3: {'PASS' if ok else 'FAIL'}"] + rows, ok


# ---------------------------------------------------------------- hand assembly (2026-11-17)
# Each step moves its parts (one bench subassembly) along straight segments into place, in the top
# frame; the bottom half repeats them flipped (cube turned over). start = final - sum(moves).
DOWN = (0, 0, -45)
SIDE_STEPS = [
    ("OUT switch into its pocket in the +x wall", ["switch_out"], [DOWN, (2.1, 0, 0)]),
    ("-y sled plate (plate B)", ["plate_ny"], [DOWN, (0, -1.4, 0)]),
    ("+y sled subassembly: plate A, hinge motor (2 x M1.6 on the bench), gears, E-clips", ["plate_py", "hinge_motor", "g_motor", "g_compound", "g_idler", "g_axle", "eclips"], [DOWN, (0, 5.1, 0)]),
    ("deploy module: FIT0577, IN board and switch, carrier with its nut", ["deploy_motor", "deploy_in_board", "switch_in", "carrier", "nut"], [DOWN, (2.0, 0, 0)]),
    ("deploy straps (hooked under the -y ledges)", ["deploy_strap"], [(0, 0, -30), (0, -2.0, 0)]),
    ("carrier pin, through the access hole in the -y face", ["pin"], [(0, 15, 0)]),
    ("disc motor with its encoder and pressed pinion (bench), straight down through the strip opening", ["disc_motor", "pinion"], [(0, 0, -45)]),
    ("centre shaft with its gear (bench)", ["centre_shaft", "centre_gear"], [(0, 0, -45)]),
    ("shaft E-clip under the bridge, through the -x opening", ["shaft_eclip"], [(45, 0, 0)]),
    ("C2 pin, pressed into the bridge", ["c2_pin"], [(0, 0, -30)]),
    ("C2 cluster onto its pin", ["c2"], [(0, 0, -30)]),
    ("C1 pin, pressed into the bridge", ["c1_pin"], [(0, 0, -30)]),
    ("C1 cluster onto its pin", ["c1"], [(0, 0, -30)]),
    ("drive plate over the pins and the straps", ["drive_plate"], [(0, 0, -20)]),
    ("stow bolt down through its hole in the drive plate", ["bolt"], [(0, 0, -30)]),
    ("cross pin, pressed through the shaft from -y", ["cross_pin"], [(0, 0, -25), (0, 10, 0)]),
    ("+y wall strip over the drive", ["strip"], [(0, 0, -30)]),
    ("foot stack (bench: bearing, hub screwed to the disc, cam ring, pins, lock) with the knuckle", list(FOOT_PARTS) + ["knuckle"], [DOWN]),
    ("hinge encoder board on its spacers, through the chamfer", ["enc_board_hinge", "enc_spacers"], [(-10, 0, -10)]),
    ("hinge axle, along -y through the +y chamfer", ["axle"], [(0, -45, 0)]),
    ("hinge encoder magnet onto the axle end, through the chamfer", ["enc_magnet_hinge"], [(-10, 0, -10)]),
]
SLED_STEPS = 6  # the first six steps (sled and deploy drive) are done on both halves before either disc drive: the bottom deploy motor passes where the top disc motor stands
FIRST_STEPS = [("cell, through the -x opening", ["cell"], [(45, 0, 0)])]
LAST_STEPS = [("-x cover with the main board (bench), lip ring and magnets", ["side_cover", "main_board", "lip_nx", "mag_nx"], [(8, 0, 0)])]
SCREWS = [  # (label, head centre, driver direction (out of the screw), head radius, after this step) in the top frame
    *[(f"strap and drive plate M2 x 8 at x {(a + b) / 2:.1f}", ((a + b) / 2, CRADLE[3] - 0.8, PLATE_Z[1]), (0, 0, 1), 1.9, "drive plate over the pins and the straps") for a, b in STRAP_XS],
    ("drive plate M2 x 6 into the bridge post", (*DRIVE_POST, PLATE_Z[1]), (0, 0, 1), 1.9, "drive plate over the pins and the straps"),
    *[(f"strip M2 x 8 countersunk at x {x:g}", (x, H, STRIP_SCREW_Z), (0, 1, 0), 1.9, "+y wall strip over the drive") for x in STRIP_SCREWS],
]
COVER_SCREW_LIST = [(f"cover M2 x 6 countersunk at y {yy:g}, z {zz:g}", (-H, yy, zz), (-1, 0, 0), 1.9) for yy, zz in COVER_SCREWS]


def _flipv(v):
    return (v[0], -v[1], -v[2])


def _sweep_hits(moving, placed, moves, step=1.0, limit=0.05):
    """Move the shapes `moving` (final poses) from final - sum(moves) along the moves; return the
    first collision with a placed part as (segment, distance, moving name, placed name, mm3) or None."""
    total = [sum(m[i] for m in moves) for i in range(3)]
    pos = [-t for t in total]
    for k, mv in enumerate(moves):
        length = math.sqrt(sum(c * c for c in mv))
        n = max(1, int(math.ceil(length / step)))
        lo = [min(pos[i], pos[i] + mv[i]) for i in range(3)]
        hi = [max(pos[i], pos[i] + mv[i]) for i in range(3)]
        local = {}
        for mname, msh in moving.items():
            bb = _bb(msh)
            region = box(bb.xmin + lo[0] - 0.5, bb.xmax + hi[0] + 0.5, bb.ymin + lo[1] - 0.5, bb.ymax + hi[1] + 0.5, bb.zmin + lo[2] - 0.5, bb.zmax + hi[2] + 0.5)
            for pname, psh in placed.items():
                if frozenset((_base(mname), _base(pname))) in INTENDED_BASE or not bbox_overlap(region, psh):
                    continue
                cut = psh.intersect(region)
                if volume(cut) > 1e-6:
                    local[(mname, pname)] = cut
        for j in range(n + 1):
            d = [pos[i] + mv[i] * j / n for i in range(3)]
            for (mname, pname), cut in local.items():
                ms = moving[mname].translate(tuple(d))
                if not bbox_overlap(ms, cut):
                    continue
                v = volume(ms.intersect(cut))
                if v > limit:
                    return (k, length * j / n, mname, pname, v)
        pos = [pos[i] + mv[i] for i in range(3)]
    return None


def _base(name):
    for t in ("_top", "_bottom"):
        if name.endswith(t):
            return name[: -len(t)]
    return name


INTENDED_BASE = {frozenset(_base(n) for n in pair) for pair in INTENDED}


def check_assembly():
    """Every part goes in, in order, along its stated straight segments without collision."""
    parts = build()
    placed = {n: v[0] for n, v in parts.items() if n == "body" or (n.startswith(("lip_p", "mag_p", "lip_ny", "mag_ny")))}
    rows, ok = [], True
    seq = [(lab, names, moves, "") for lab, names, moves in FIRST_STEPS]
    for phase in (SIDE_STEPS[:SLED_STEPS], SIDE_STEPS[SLED_STEPS:]):  # both sleds first, then both drives
        for tag in ("top", "bottom"):
            for lab, names, moves in phase:
                mv = moves if tag == "top" else [_flipv(m) for m in moves]
                seq.append((lab, [f"{n}_{tag}" for n in names], mv, tag))
    seq += [(lab, names, moves, "") for lab, names, moves in LAST_STEPS]
    for i, (lab, names, moves, tag) in enumerate(seq, 1):
        moving = {n: parts[n][0] for n in names}
        hit = _sweep_hits(moving, placed, moves)
        path = " then ".join(f"({m[0]:g}, {m[1]:g}, {m[2]:g})" for m in moves)
        if hit:
            ok = False
            k, dist, mn, pn, v = hit
            rows.append(f"  {i:2d} {tag:6s} {lab}: FAIL, segment {k + 1} at {dist:.1f} mm: {mn} x {pn} {v:.2f} mm3")
        else:
            rows.append(f"  {i:2d} {tag:6s} {lab}: clear along {path}")
        placed.update(moving)
        for slab, head, axis, hr, after in SCREWS:
            if after == lab:
                h = head if tag == "top" else (head[0], -head[1], -head[2])
                a = axis if tag == "top" else _flipv(axis)
                rows.append(_driver_row(slab, h, a, hr, placed))
                ok = ok and "FAIL" not in rows[-1]
    for slab, head, axis, hr in COVER_SCREW_LIST:
        rows.append(_driver_row(slab, head, axis, hr, placed))
        ok = ok and "FAIL" not in rows[-1]
    missing = sorted(set(parts) - set(placed))
    if missing:
        ok = False
        rows.append(f"  parts never placed: {', '.join(missing)}")
    return [f"hand assembly, {len(seq)} steps, every screw's driver path (3 mm shaft, 40 mm, plus head clearance): {'PASS' if ok else 'FAIL'}"] + rows, ok


def _driver_row(label, head, axis, head_r, placed):
    """A 3 mm driver shaft 40 mm long and a head clearance (head + 0.3, 2 mm) out of the screw head."""
    ax = [c for c in axis]
    def cyl_along(r, d0, d1):
        c = cq.Workplane("XY").circle(r).extrude(d1 - d0).translate((0, 0, d0))
        rot = cq.Vector(0, 0, 1).cross(cq.Vector(*ax))
        if rot.Length > 1e-9:
            ang = math.degrees(math.acos(max(-1.0, min(1.0, ax[2]))))
            c = c.rotate((0, 0, 0), rot.toTuple(), ang)
        elif ax[2] < 0:
            c = c.rotate((0, 0, 0), (1, 0, 0), 180)
        return c.translate(head)
    tool = cyl_along(head_r + 0.3, 0.05, 2.0).union(cyl_along(1.5, 0.05, 42.0))
    worst = max([(volume(tool.intersect(sh)), n) for n, sh in placed.items() if bbox_overlap(tool, sh)] or [(0.0, "-")])
    return f"     driver for {label}: {'clear' if worst[0] <= 0.05 else f'FAIL, hits {worst[1]} {worst[0]:.2f} mm3'}"


def check_drivers():
    return ["driver paths are checked inside the assembly sequence (above)"], True


def check_skin(t=0.8):
    """The body keeps a t mm skin under every outer surface except where it is meant to open: the
    foot slots, the -x cover opening, the hinge edges' channels, the passive face pockets, the
    carrier pin's access holes."""
    inner = cq.Workplane("XY").box(S - 2 * t, S - 2 * t, S - 2 * t).edges().chamfer(C - t * (2 - math.sqrt(2)))
    shell = outline().cut(inner)
    allowed = [box(-H - 1, H + 1, -PYF, PYF, ZC, H + 1), box(-H - 1, H + 1, -PYF, PYF, -H - 1, -ZC),
               box(-H - 1, -XW + 0.01, -COVER[0] - 0.5, COVER[0] + 0.5, -ZC - 0.01, ZC + 0.01)]
    band = 2 * H - C - 2.5
    edge = hull_xz([(band - H - 1, H + 1), (H + 1, band - H - 1), (H + 1, H + 1)], -H - 1, H + 1)
    allowed += [edge, flip(edge)]
    window = box(11.0, 27.5, L1_Y[0] - 0.5, L1_Y[1] + 1.0, 20.0, H + 1)  # the deployed L1 gear passes 0.9 mm outside the y-z chamfer
    enc_corner = box(19.5, H + 1, HENC_BOARD_Y[0] - 0.8, -21.5, 19.5, H + 1)  # the hinge encoder board stands in the y-z chamfer at the end of the hinge edge
    allowed += [strip_opening(), flip(strip_opening()), window, flip(window), enc_corner, flip(enc_corner)]
    for axis, sign in PASSIVE_FACES:
        allowed.append(on_face(cyl_z(PLATE_R[1] + 0.3, H - POCKET_DEPTH - 0.5, H + 1), axis, sign))
    for f in (lambda s: s, flip):
        allowed.append(f(cyl_y(PIN_D / 2 + 0.25, -H - 1, -YB + 0.01, *PIN_XZ)))
    void = shell.cut(body())
    for a in allowed:
        void = void.cut(a)
    v = volume(void)
    rows = [f"body skin >= {t} mm outside the intended openings: {v:.2f} mm3 of breach: {'PASS' if v <= 0.05 else 'FAIL'}"]
    if v > 0.05:
        for sol in void.val().Solids():
            if sol.Volume() > 0.01:
                bb = sol.BoundingBox()
                rows.append(f"  {sol.Volume():.2f} mm3 at x {bb.xmin:.1f}..{bb.xmax:.1f} y {bb.ymin:.1f}..{bb.ymax:.1f} z {bb.zmin:.1f}..{bb.zmax:.1f}")
    return rows, v <= 0.05


def interference(parts, label, only=None, limit=0.05):
    names = sorted(parts)
    rows, checked = [], 0
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if only and not (a in only or b in only):
                continue
            if frozenset((a, b)) in INTENDED:
                continue
            sa, sb = parts[a][0], parts[b][0]
            if not bbox_overlap(sa, sb):
                continue
            checked += 1
            v = volume(sa.intersect(sb))
            if v > limit:
                rows.append((a, b, v))
    return [f"{label}: {checked} pairs with touching boxes intersected, {len(rows)} overlap > {limit} mm3"] + [f"  {a} x {b}: {v:.2f} mm3" for a, b, v in rows]


DENSITY = {"PETG": 1.27, "PLA": 1.24, "steel": 7.85, "brass": 8.5, "magnet": 7.5}
INFILL = {"body": 0.75, "foot_top": 0.8, "foot_bottom": 0.8}  # effective solid fraction (walls + infill), guess
BOUGHT_G = {"motor_hinge": 10.5, "motor_disc": 9.5, "motor_deploy": 10.0, "cell": 7.0, "pcb_main": 7.0, "pcb_small": 0.5}  # grams (guess; FIT0577 about 10 g, source: BOM.md)


def masses(parts):
    rows = []
    for name, (shape, kind, mat) in sorted(parts.items()):
        if mat == "motor":
            g = BOUGHT_G["motor_hinge" if "hinge" in name else "motor_deploy" if "deploy" in name else "motor_disc"]
        elif mat == "cell":
            g = BOUGHT_G["cell"]
        elif mat == "pcb":
            g = BOUGHT_G["pcb_main" if name == "main_board" else "pcb_small"]
        else:
            g = volume(shape) / 1000 * DENSITY[mat] * (INFILL.get(name, 1.0 if kind == "bought" else 0.9))
        rows.append((name, g))
    extra = [
        ("wiring, FFCs, connectors", 4.0),
        ("SMA wires, crimps, springs (6 wires)", 1.5),
        ("screws and heat-set inserts", 4.0),
        ("disc ring PCBs, sprung contacts, foot flex", 2.0),
        ("disc lock pins and levers", 0.6),
    ]
    return rows, extra


def inertia(parts, rows):
    """Mass-weighted inertia about the centre, each part taken as a uniform solid of its mass."""
    grams = dict(rows)
    ixx = iyy = izz = 0.0
    cx = cy = cz = mtot = 0.0
    for name, (shape, _, _) in parts.items():
        m = grams[name] / 1000
        s = shape.val()
        c = s.Center()
        v = s.Volume()
        if v <= 0:
            continue
        mtot += m
        cx, cy, cz = cx + m * c.x, cy + m * c.y, cz + m * c.z
        bb = s.BoundingBox()
        dx, dy, dz = bb.xlen / 1000, bb.ylen / 1000, bb.zlen / 1000
        px, py, pz = c.x / 1000, c.y / 1000, c.z / 1000
        ixx += m * ((dy**2 + dz**2) / 12 + py**2 + pz**2)
        iyy += m * ((dx**2 + dz**2) / 12 + px**2 + pz**2)
        izz += m * ((dx**2 + dy**2) / 12 + px**2 + py**2)
    return (cx / mtot, cy / mtot, cz / mtot), (ixx, iyy, izz)


# ---------------------------------------------------------------- export
PRINT_ORIENT = {  # rotation (axis, degrees) that puts the part on the bed, see ASSEMBLY.md print table
    "body": ((1, 0, 0), 90),  # stands on its -y face: chamfers and pockets print without support
    "foot_top": ((0, 1, 0), 180),  # outer face down
    "disc_top": ((0, 1, 0), 180),  # face down
    "cam_ring_top": ((0, 0, 1), 0),
    "floor_tile": ((0, 0, 1), 0),
    "side_cover": ((0, 1, 0), -90),  # outer face down
    "strip_top": ((1, 0, 0), 90),  # inner face down, insert holes vertical
}  # the rest print as modelled (flat side down)


def export(parts):
    pdir, bdir = HERE / "print", HERE / "bought"
    for d in (pdir, bdir):
        d.mkdir(exist_ok=True)
        for old in list(d.glob("*.step")) + list(d.glob("*.stl")) + list(d.glob("*.dxf")):  # parts of earlier designs
            old.unlink()
    printed = {k: v[0] for k, v in parts.items() if v[1] == "print" and not k.endswith("_bottom")}
    printed["floor_tile"] = floor_tile()
    for name, shape in printed.items():
        axis, deg = PRINT_ORIENT.get(name, ((0, 0, 1), 0))
        s = shape.rotate((0, 0, 0), axis, deg) if deg else shape
        bb = s.val().BoundingBox()
        s = s.translate((-(bb.xmin + bb.xmax) / 2, -(bb.ymin + bb.ymax) / 2, -bb.zmin))
        stem = name.replace("_top", "")
        cq.exporters.export(s, str(pdir / f"{stem}.step"))
        cq.exporters.export(s, str(pdir / f"{stem}.stl"), tolerance=0.02, angularTolerance=0.2)
    for active, stem in ((False, "lip_ring_passive"), (True, "lip_ring_disc")):  # laser-cutting outlines, 0.5 mm stainless
        flat = lip_plate(active).translate((0, 0, -(H - SKIN - PLATE_T / 2)))
        cq.exporters.export(flat.section(), str(bdir / f"{stem}.dxf"), exportType="DXF")
    for name, (shape, kind, _) in parts.items():
        if kind == "bought" and name.endswith("_top") or name in ("cell", "main_board"):
            cq.exporters.export(shape, str(bdir / f"{name.replace('_top', '')}.step"))
    colours = {"PETG": (0.85, 0.85, 0.82), "steel": (0.55, 0.57, 0.6), "brass": (0.8, 0.65, 0.25), "motor": (0.35, 0.4, 0.5), "cell": (0.3, 0.45, 0.75), "pcb": (0.15, 0.5, 0.25), "magnet": (0.6, 0.2, 0.2)}
    for fname, ps in (("cube50_assembly.step", parts), ("cube50_open90.step", build(90.0))):
        asm = cq.Assembly(name="cube50")
        for name, (shape, _, mat) in ps.items():
            asm.add(shape, name=name, color=cq.Color(*colours.get(mat, (0.7, 0.7, 0.7))))
        asm.save(str(HERE / fname))


def summary():
    rows = [
        f"S {S:g} mm, H {H:g}, chamfer {C:g} (channel radius {C / math.sqrt(2):.2f}); foot {TF:.2f} thick, core z +-{ZC:.2f}, x +-{XW:.2f}, y +-{PY:.2f}",
        f"sled: retract {RETRACT:.2f} in x and z ({STROKE:.2f} diagonal); stowed axle {PR}; motor box x {MX1 - N20_W:.2f}..{MX1:.2f}, z {MZ1 - N20_H:.2f}..{MZ1:.2f}, y {YA - HINGE_MOTOR_L:.2f}..{YA:.2f}",
        f"hinge train m{M}: motor {N_MOTOR}T at ({G2[0]:.2f}, {G2[1]:.2f}), compound {N_BIG}T/{N_SMALL}T at ({GC[0]:.2f}, {GC[1]:.2f}), idler {N_IDLER}T at ({GI[0]:.2f}, {GI[1]:.2f}), axle {N_AXLE}T; ratio {HINGE_RATIO:.3f}",
        f"deploy: FIT0577 body x {DEP_X0:.2f}..{DEP_X0 + FIT_L:.2f} at y {DEP_Y}, z {DEP_Z}; carrier x {CAR_X[0]:.2f}..{CAR_X[1]:.2f} (+{RETRACT:.2f}); screw used {CAR_X[1] + RETRACT - INB_X[0]:.2f} of {SCREW_L:g} mm",
        f"disc drive: motor ({DISC_MOTOR_XY[0]:.2f}, {DISC_MOTOR_XY[1]:.2f}) z {MOTOR_FACE_Z - DISC_MOTOR['L'] - DISC_MOTOR['enc_L']:.2f}..{MOTOR_FACE_Z:.2f}; C1 ({C1_XY[0]:.2f}, {C1_XY[1]:.2f}), C2 ({C2_XY[0]:.2f}, {C2_XY[1]:.2f}); {DT[0]}:{DT[1]} m{DT_M[0]} z {LC[0]}..{LC[1]}, {DT[2]}:{DT[3]} m{DT_M[1]} z {LB[0]}..{LB[1]}, {DT[4]}:{DT[5]} m{DT_M[2]} z {LA[0]}..{LA[1]}; ratio {DISC_RATIO:.3f}",
        f"face magnets: r {MAG_R}, at +-{MAG_DELTA} + 90k deg, {MAG_D} x {MAG_T} under {MAG_SKIN}; wall between pockets {2 * MAG_R * math.sin(math.radians(min(MAG_DELTA, 45 - MAG_DELTA))) - MAG_D - 0.1:.2f}; to the lip ring {PLATE_R[0] - 0.1 - MAG_R - MAG_D / 2 - 0.05:.2f}",
    ]
    return rows


def main():
    out = summary()
    print("\n".join(out), flush=True)
    parts = build()
    out += interference(parts, "stowed")
    print(out[-1] if len(out[-1]) < 200 else "", flush=True)
    for f in (0.5, 1.0):
        out += interference(build(sled={"top": f, "bottom": f}), f"sled travel {f:g} of the stroke (both sleds), feet stowed", only=SLED_NAMES)
    out += interference(build(sled={"top": 0.0, "bottom": 0.0}), "sled travel 0 (stowed)", only=SLED_NAMES)
    moving = [n for n in parts if n.endswith("_top") and (n.split("_top")[0] in FOOT_PARTS or n.split("_top")[0] in SLED_MOVES_WITH_FOOT)]
    for ang in (15.0, 45.0, 90.0, 135.0, 180.0):
        out += interference(build(ang), f"top foot open {ang:.0f} deg", only=moving)
    print("\n".join(out[7:]), flush=True)
    for name, fn in (("proud", lambda: check_stowed_proud(parts)), ("partner", lambda: check_partner_sweep(parts)), ("yaw", lambda: check_yaw_sweep(parts)), ("vertex", check_vertex_neighbours), ("channel", check_pivot_channel), ("skin", check_skin), ("assembly", check_assembly)):
        if "--fast" in sys.argv and name in ("vertex", "partner"):
            continue
        lines, ok = fn()
        out += lines
        print("\n".join(lines[:12]), flush=True)
    rows, extra = masses(parts)
    total = sum(g for _, g in rows) + sum(g for _, g in extra)
    out.append("mass (g):")
    out += [f"  {n}: {g:.1f}" for n, g in rows] + [f"  {n}: {g:.1f}" for n, g in extra]
    out.append(f"  TOTAL {total:.1f} g")
    com, inert = inertia(parts, rows)
    out.append(f"centre of mass (mm, parts only) ({com[0]:.1f}, {com[1]:.1f}, {com[2]:.1f}); inertia about the centre (kg m2) {inert[0]:.2e}, {inert[1]:.2e}, {inert[2]:.2e}")
    out += torque_budget(total / 1000)
    text = "\n".join(out)
    print("\n".join(out[-6:]))
    (HERE / "report.txt").write_text(text + "\n")
    if not QUICK:
        export(parts)
        print("exported to", HERE)


SLED_NAMES = [f"{n}_{t}" for t in ("top", "bottom") for n in ("axle", "knuckle", "plate_py", "plate_ny", "hinge_motor", "g_motor", "g_compound", "g_idler", "g_axle", "eclips", "enc_board_hinge", "enc_magnet_hinge", "enc_spacers", "carrier", "nut", "pin")]


def torque_budget(mass_kg, peel=0.08):
    grav = mass_kg * 9.81 * S / 1000 / math.sqrt(2)
    hinge = 0.245 * HINGE_RATIO * 0.95**3
    rpm = 31.0 / HINGE_RATIO
    return [
        f"pivot: gravity {grav:.4f} N m (m g S / sqrt 2, CAD mass {mass_kg * 1000:.0f} g) + magnet peel {peel:.3f} = {grav + peel:.4f}; hinge 0.245 x {HINGE_RATIO:.3f} x 0.95^3 = {hinge:.4f} N m; margin {hinge / (grav + peel):.2f} (gravity only {hinge / grav:.2f}); foot free speed {rpm:.1f} rpm ({rpm * 6:.0f} deg/s)",
        *disc_speed_torque(),
    ]


def disc_speed_torque(eta=0.95**3):
    """The disc drive's speed-torque line (Pololu figures for the motor, 0.95 per mesh a guess)."""
    m = DISC_MOTOR
    w_free = m["free_rpm"] * 2 * math.pi / 60 / DISC_RATIO
    t_stall = m["stall_Nm"] * DISC_RATIO * eta
    t_cap = m["rating_Nm"] * DISC_RATIO * eta

    def w(t):
        return w_free * (1 - t / t_stall)

    def amps(t):
        return m["free_A"] + (m["stall_A"] - m["free_A"]) * t / t_stall

    return [
        f"disc drive: free {w_free:.2f} rad/s; cap {t_cap:.2f} N m at the gearbox rating ({m['rating_Nm']} N m, chop {amps(t_cap):.2f} A) at {w(t_cap):.2f} rad/s; 2.2 N m at {w(2.2):.2f} rad/s, {amps(2.2):.2f} A; theoretical stall {t_stall:.2f} N m",
        f"disc teeth (Lewis, hardened steel) at the cap: stage 1 {lewis(m['rating_Nm'], DT[0], DT_M[0], LC[1] - LC[0]):.0f} MPa, stage 2 {lewis(m['rating_Nm'] * DT[1] / DT[0] * 0.95, DT[2], DT_M[1], LB[1] - LB[0]):.0f}, stage 3 {lewis(t_cap / (DT[5] / DT[4]) / 0.95, DT[4], DT_M[2], LA[1] - LA[0]):.0f}",
    ]


def lewis(torque, teeth, module, face):
    y = {10: 0.201, 12: 0.245, 13: 0.261, 14: 0.277, 15: 0.290, 16: 0.296}[teeth]
    return torque / (pitch_r(teeth, module) / 1000) / (face * module * y)


if __name__ == "__main__":
    main()
