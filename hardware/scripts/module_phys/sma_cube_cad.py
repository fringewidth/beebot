"""Printable CAD of the SMA cube (hardware/src/beebot/hardware/sma_cube.py) at the chosen point.

Writes STEP and STL per printed part, an assembly STEP with every part placed (printed parts plus
envelopes of bought parts: magnets, steel, Didel gears, rotor, battery), an overlap check of the
placed parts (static and swept for the moving edge bolts and corner carriers), a fit report, and the
unfolded flex PCB net (SVG and DXF), all into hardware/sma_cube/.

Body frame as sma_cube.py: twist discs on +x and -x, plain faces +-y and +-z. All lengths in mm.

    .venv/bin/python hardware/scripts/module_phys/sma_cube_cad.py            # L from chosen.json
    .venv/bin/python hardware/scripts/module_phys/sma_cube_cad.py --L 16     # override L (mm)
    .venv/bin/python hardware/scripts/module_phys/sma_cube_cad.py --no-export  # checks only

Design decisions that differ from the concept (each is reported in hardware/sma_cube/README.md):

1. Frame is a square tube (four plain faces) plus two end plates (x faces with the disc window and a
   bulkhead that carries the hub bearing); internals go in through the open tube ends.
1. Hook tabs of the edge bolts are separate press-in parts, one per bolt face (hooks_per_side 1), with
   the tab width cut to what fits between the bayonet groove and the corner sphere pockets.
1. Bayonet lips point inward (radially), the only way the groove ring clears the hook slots at all.
1. Leg hinge axes sit at the knuckle radius inside both faces (0.5 mm off the lattice edge line), so the
   knuckle stays inside the cube's envelope.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cadquery as cq
import numpy as np
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "hardware" / "sma_cube"
CHOSEN = ROOT / "hardware" / "runs" / "module_phys" / "sma_cube" / "chosen.json"
sys.path[:0] = [str(ROOT / part / "src") for part in ("v0", "v1", "v2", "hardware")]

# Printer: Elegoo Saturn 4 Ultra 16K, 14 x 19 um pixels; limits as given for hobby MSLA tough resin.
PIXEL_XY = 0.019
PIXEL_X_MIN = 0.014
MIN_WALL = 0.30
MIN_HOLE = 0.35
MIN_FEATURE = 0.12

FLEX = 0.11  # flex PCB thickness (2-layer, coverlay)
SKIN = 0.30  # resin skin
CHAMFER = 0.50
CLEAR = 0.05  # sliding clearance between printed parts

# Didel catalogue (geared_joint.DIDEL, V): outer diameters, mm; face width A 0.8
DIDEL_OD = {"G209": 2.2, "G220": 4.4, "G220_pinion": 2.2, "G240": 8.4, "GW205": 2.8}
DIDEL_FACE = 0.8


# ---------------------------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------------------------


@dataclass
class Geo:
    """Every dimension the CAD uses, mm, derived from chosen.json's design (SI, fractions of L)."""

    L: float
    d: dict

    def __post_init__(self):
        d, L = self.d, self.L
        self.h = L / 2
        self.Ro = self.h - FLEX
        self.Ri = self.Ro - SKIN
        # bonds
        self.d_c = d["corner_frac"] * L
        self.cover = d["cover"] * 1e3
        self.a_yz = self.h - self.cover - self.d_c / 2
        self.a_x = self.a_yz - d["setback_frac"] * L
        self.d_r = d["ring_frac"] * L
        self.ring_r = d["ring_r_frac"] * L
        self.r_d = d["disc_r_frac"] * L
        self.t_d = d["disc_t_frac"] * L
        self.tread = d["tread"] * 1e3
        self.r_g = 0.9 * self.r_d
        self.insert_d, self.insert_t = 0.8, 0.3
        # hooks (bayonet tab size; the bolt tabs are cut to fit, see hook_fit)
        self.hook_b = d["hook_w"] * L
        self.hook_t = d["hook_t"] * L
        self.hook_l = d["hook_l"] * L
        self.shank = 0.45  # bayonet shank radial thickness (A)
        self.shank_x = 0.60  # bolt tab shank thickness along x (A)
        self.tab_rise = FLEX + SKIN + 0.04  # tab shank length beyond the own face: partner flex, skin, clearance
        # legs
        self.leg_l = d["leg_l_frac"] * L
        self.leg_w = d["leg_w_frac"] * L
        self.leg_t = d["leg_t"] * 1e3
        self.r_k = 0.35
        self.pin_d = 0.25  # steel hinge pin (music wire)
        self.k_ax = self.h - self.r_k  # knuckle axis inset from both faces
        self.floor = 0.30  # recess floor under each leg
        # capstan
        self.pitch = d["leg_pitch"] * 1e3
        self.r_c = d["leg_cap_r_frac"] * L
        self.n_teeth = int(round(2 * math.pi * self.r_c / self.pitch))
        self.r_c = self.n_teeth * self.pitch / (2 * math.pi)
        self.tooth_h = 0.4 * self.pitch
        self.cap_face = 0.9
        self.drum_r = self.r_k * d["leg_ratio"]
        # bolts and carriers
        self.stroke = d["rel_stroke"] * 1e3  # bolt travel
        self.free = 0.6  # first part of the travel frees the hooks; the carriers move over the rest
        self.retract = self.stroke - self.free  # corner sphere retraction along the edge bisector
        self.bar = 0.06 * L
        self.car_half = self.d_c / 2 + 0.30  # carrier half width along x
        # joint (geared_joint.nominal_design: Didel G209 swing gear and idler, G220 on the worm arbor,
        # Didel worm GW205 into a G220 on the sun shaft, printed split-ring planetary into the disc)
        from beebot.hardware.geared_joint import nominal_design

        jd = nominal_design(max(L, 11.0) * 1e-3)
        self.joint = jd
        wf = jd.train.meshes[-1]
        worm = jd.train.meshes[-2]
        self.wf_m = wf.module * 1e3
        self.wf_sun, self.wf_planet, self.wf_n = wf.sun, wf.planet, wf.planets
        self.wf_r1, self.wf_r2 = wf.ring1, wf.ring2
        self.wf_face = wf.face * 1e3
        # ring wall 0.3 mm (the printer's minimum wall) instead of geared_joint's 0.5 mm: the rotor slab
        # under the plain face then needs the rotor only about 0.15 mm thinner (option B, 2026-11-16)
        self.ring_wall = 0.30
        self.wf_outer = wf.module * 1e3 * wf.ring2 / 2 + 1.25 * wf.module * 1e3 + self.ring_wall
        self.wf_outer_design = wf.outer_radius() * 1e3
        self.wf_name = wf.name
        self.worm_rp = worm.worm_r * 1e3  # GW205 pitch radius
        self.g220_rp = 0.2 * 20 / 2
        self.rotor_r = jd.motor.r_o * 1e3
        self.rotor_t_design = jd.motor.t_r * 1e3
        # x layers of the +x joint (outer to inner), 0.08 mm axial gaps
        self.disc_x0 = self.h - self.t_d
        self.ring_sx = self.h - self.cover - self.d_r / 2  # ring sphere centre x
        self.boss_x0 = self.ring_sx - self.d_r / 2 - 0.15  # underside of the ring-sphere bosses
        self.r2_x = (self.boss_x0 - 0.08 - self.wf_face, self.boss_x0 - 0.08)  # output ring (disc hub)
        self.r1_x = (self.r2_x[0] - 0.08 - self.wf_face, self.r2_x[0] - 0.08)  # fixed ring (end plate)
        self.hub_r = 0.9
        # storage (lipo TTWWLL envelope, thickness along x, width along y, length along z)
        st = d["storage"]
        if st.startswith("lipo"):  # lipoTTWWLL: thickness TT/10 mm, width WW, length LL
            self.batt = (int(st[4:6]) / 10, float(st[6:8]), float(st[8:10]))
        else:  # CeraCharge 1812 cells: 4.4 x 3.2 x 1.1 each, stacked
            ncell = int(st[-1])
            self.batt = (1.1 * ncell, 3.2, 4.4)
        # option B: the cell (thickness along x) stays in the middle slab but sits off-axis toward +z,
        # so the Didel worms can come inboard under it; the sun G220 sits just outside the cell
        self.worm_x = self.batt[0] / 2 + CLEAR + DIDEL_FACE / 2
        self.g220_x = (self.worm_x - DIDEL_FACE / 2, self.worm_x + DIDEL_FACE / 2)
        self.worm_z = -(self.g220_rp + self.worm_rp)
        self.batt_z0 = self.worm_z + DIDEL_OD["GW205"] / 2 + CLEAR
        self.y_gear = self.batt[1] / 2 + CLEAR  # selector gear layer (y) just outside the cell
        # rotor slab directly under the plain face: above the fixed ring, below the leg recess floor
        self.rotor_y0 = self.wf_outer + CLEAR
        self.rotor_y1 = self.h - self.leg_t - CLEAR - self.floor - CLEAR
        self.rotor_t = min(self.rotor_t_design, self.rotor_y1 - self.rotor_y0)
        self.cap_x = self.g220_x[1]  # x-legs' capstan ring layer starts outboard of the sun G220


def load_design(path=CHOSEN):
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------------------------


def xf(shape, R=None, t=(0, 0, 0)):
    """Apply the orthogonal 3x3 matrix R (rows; a reflection is applied as a mirror about x = 0
    followed by a rotation) and translate by t. Returns a new shape."""
    R = np.eye(3) if R is None else np.asarray(R, float)
    s = shape.wrapped
    if np.linalg.det(R) < 0:
        M = gp_Trsf()
        M.SetMirror(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0)))
        s = BRepBuilderAPI_Transform(s, M, True).Shape()
        R = R @ np.diag([-1.0, 1.0, 1.0])
    T = gp_Trsf()
    T.SetValues(R[0, 0], R[0, 1], R[0, 2], float(t[0]), R[1, 0], R[1, 1], R[1, 2], float(t[1]), R[2, 0], R[2, 1], R[2, 2], float(t[2]))
    return cq.Shape.cast(BRepBuilderAPI_Transform(s, T, True).Shape())


def box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    s = hi - lo
    return cq.Solid.makeBox(float(s[0]), float(s[1]), float(s[2]), cq.Vector(*lo))


def cyl(r, length, p, axis):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    return cq.Solid.makeCylinder(float(r), float(length), cq.Vector(*map(float, p)), cq.Vector(*a))


def ring(r_in, r_out, length, p, axis):
    return cyl(r_out, length, p, axis).cut(cyl(r_in, length, p, axis))


def sphere(r, c):
    return cq.Solid.makeSphere(float(r), cq.Vector(*map(float, c)), angleDegrees1=-90, angleDegrees2=90)


def fuse(shapes):
    shapes = [s for s in shapes if s is not None]
    out = shapes[0]
    if len(shapes) > 1:
        out = out.fuse(*shapes[1:]).clean()
    return out


def toothed(n, r_root, r_tip, top_frac, root_frac, face, square=False):
    """Toothed disc about +z from z=0 to face. square: rectangular teeth (ratchet); else trapezoid
    (straight spur approximation of an involute)."""
    pts = []
    pitch_ang = 2 * math.pi / n
    for k in range(n):
        a0 = k * pitch_ang
        if square:
            hw = pitch_ang * top_frac / 2
            for a, r in ((a0 - hw, r_root), (a0 - hw, r_tip), (a0 + hw, r_tip), (a0 + hw, r_root)):
                pts.append((r * math.cos(a), r * math.sin(a)))
        else:
            hb = pitch_ang * root_frac / 2
            ht = pitch_ang * top_frac / 2
            for a, r in ((a0 - hb, r_root), (a0 - ht, r_tip), (a0 + ht, r_tip), (a0 + hb, r_root)):
                pts.append((r * math.cos(a), r * math.sin(a)))
    return cq.Workplane("XY").polyline(pts).close().extrude(face).val()


def spur(m, z, face):
    rp = m * z / 2
    return toothed(z, rp - 1.25 * m, rp + m, 0.25, 0.62, face)


AXES = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}
C4 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)  # +90 deg about x: y -> z, z -> -y
NX = np.diag([-1.0, -1.0, 1.0])  # +x joint -> -x joint (180 deg about z)
# leg name -> rotation taking the XP leg (lies in +y, hinge edge +y/+z, extends to -z) to it
LEG_R = {
    "XP": np.eye(3),
    "XN": np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], float),  # lies in +z on the same +y/+z edge (pair)
    "DP": np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], float),  # lies in -y, hinge on the +x/-y edge
    "DN": np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]], float),
}
BOLTS = ["pp", "np", "nn", "pn"]  # C4^k of pp; sign of (y, z) of the edge
BOLT_EDGE = {"pp": (1, 1), "np": (-1, 1), "nn": (-1, -1), "pn": (1, -1)}
# Faces: pp's tab 'a' goes into +y, 'b' into +z; C4 carries both.
# Both tabs of pp sit under the x-leg pair (XP in +y, XN in +z): dropped. Until 2026-11-16 XN lay in -y
# on the nn edge and the dropped tabs were (pp, a) and (nn, a); same count, 6 tabs either way.
DROPPED_TABS = {("pp", "a"), ("pp", "b")}


def diag_frame(sy, sz):
    """Unit vectors (n, t) in the yz plane for the diagonal toward (sy, sz)."""
    n = np.array([0.0, sy, sz]) / math.sqrt(2)
    t = np.array([0.0, -sz, sy]) / math.sqrt(2)
    return n, t


# ---------------------------------------------------------------------------------------------
# Fits computed before any solid is built
# ---------------------------------------------------------------------------------------------


def groove_dims(g: Geo):
    """Bayonet groove on a plain face (inward lips): slit, channel and boss radii, channel depth."""
    s_in = g.r_g - g.shank / 2 - 0.03
    s_out = g.r_g + g.shank / 2 + 0.03
    ch_in = g.r_g - g.shank / 2 - g.hook_l - 0.08
    ch_out = s_out + 0.02
    depth = g.hook_t + 0.05
    return {"slit": (s_in, s_out), "channel": (ch_in, ch_out), "boss": (ch_in - MIN_WALL, ch_out + MIN_WALL), "depth": depth, "boss_depth": depth + MIN_WALL}


def hook_fit(g: Geo):
    """Largest bolt tab width (along the face, across x) for which one own slot (+u) and one receive
    slot (-u) fit in each edge region between the bayonet groove boss and the corner sphere pockets."""
    gd = groove_dims(g)
    rbo = gd["boss"][1]
    zb1 = g.Ri - 0.11
    x_out = g.a_x - (g.d_c / 2 + 0.05) - MIN_WALL
    slot_len = g.shank_x + g.stroke
    best = None
    b = g.hook_b
    while b >= 0.3:
        zb0 = zb1 - b - CLEAR
        x_in = (math.sqrt(rbo**2 - zb0**2) if zb0 < rbo else 0.0) + MIN_WALL
        if x_out - x_in >= slot_len:
            best = {"b": round(b, 3), "band": (zb0, zb1), "u": (x_in + x_out) / 2, "slot_len": slot_len, "x_in": x_in, "x_out": x_out}
            break
        b -= 0.01
    # with the design's 2 tabs a side: 4 slots per edge region
    need2 = 2 * (2 * slot_len + MIN_WALL)
    zb0 = zb1 - g.hook_b - CLEAR
    x_in_full = (math.sqrt(rbo**2 - zb0**2) if zb0 < rbo else 0.0) + MIN_WALL
    return {"fit": best, "design_b": g.hook_b, "two_per_side_need_mm": round(need2, 2), "region_mm": round(2 * (x_out - MIN_WALL), 2),
            "full_width_slot_room_mm": round(x_out - x_in_full, 2), "groove_boss_r": round(rbo, 3)}


# ---------------------------------------------------------------------------------------------
# Printed parts (each built in world coordinates at its reference placement)
# ---------------------------------------------------------------------------------------------


def chamfered_cube(g: Geo):
    """Resin outer envelope: cube of half side Ro with every edge chamfered."""
    return cq.Workplane().box(2 * g.Ro, 2 * g.Ro, 2 * g.Ro).edges().chamfer(CHAMFER).val()


def face_cuts_common(g: Geo, hf):
    """Cuts on every plain face (as solids in world coordinates): bayonet slit arcs and notches,
    steel insert holes, corner sphere pockets (swept along the release), hook slots."""
    gd = groove_dims(g)
    cuts, adds = [], []
    s_in, s_out = gd["slit"]
    ch_in, ch_out = gd["channel"]
    b_in, b_out = gd["boss"]
    alpha = (g.hook_b / 2 + CLEAR) / g.r_g
    for k in range(4):
        R = np.linalg.matrix_power(C4, k)
        # the +y face, then rotated; face coordinates (x, z), centre at (0, Ri.., 0)
        y0 = g.Ri
        boss = ring(b_in, b_out, gd["boss_depth"], (0, y0 - gd["boss_depth"], 0), (0, 1, 0))
        adds.append(xf(boss, R))
        chan = ring(ch_in, ch_out, gd["depth"] + 0.001, (0, y0 - gd["depth"], 0), (0, 1, 0))
        cuts.append(xf(chan, R))
        # slit arcs: entry on the face axes (0, 90, 180, 270 deg in the (x, z) plane), lock +45 deg
        for q in range(4):
            a0 = q * math.pi / 2 - alpha
            a1 = q * math.pi / 2 + math.pi / 4 + alpha
            pts_o = [(s_out * math.cos(a), s_out * math.sin(a)) for a in np.linspace(a0, a1, 24)]
            pts_i = [(s_in * math.cos(a), s_in * math.sin(a)) for a in np.linspace(a1, a0, 24)]
            arc = cq.Workplane("XZ").polyline(pts_o + pts_i).close().extrude(-1.0).val()  # XZ normal is -y; extrude(-1) goes +y
            cuts.append(xf(xf(arc, None, (0, y0 - 0.01, 0)), R))
            # entry notch: lip removed, the tab and its inward lip drop in
            pts_o = [(s_out * math.cos(a), s_out * math.sin(a)) for a in np.linspace(q * math.pi / 2 - alpha, q * math.pi / 2 + alpha, 8)]
            pts_i = [(ch_in * math.cos(a), ch_in * math.sin(a)) for a in np.linspace(q * math.pi / 2 + alpha, q * math.pi / 2 - alpha, 8)]
            notch = cq.Workplane("XZ").polyline(pts_o + pts_i).close().extrude(-1.0).val()
            cuts.append(xf(xf(notch, None, (0, y0 - 0.01, 0)), R))
        # steel insert holes on the face diagonals (through the skin; the flex covers them)
        for sx in (-1, 1):
            for sz in (-1, 1):
                p = (sx * g.ring_r / math.sqrt(2), y0 - 0.01, sz * g.ring_r / math.sqrt(2))
                cuts.append(xf(cyl(g.insert_d / 2 + 0.03, SKIN + 0.2, p, (0, 1, 0)), R))
    # corner sphere pockets, swept inward along the edge bisector by the release stroke
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                n, _ = diag_frame(sy, sz)
                c = np.array([sx * g.a_x, sy * g.a_yz, sz * g.a_yz])
                cuts.append(sphere(g.d_c / 2 + 0.03, c))
                cuts.append(cyl(g.d_c / 2 + 0.03, g.retract, c, -n))
                cuts.append(sphere(g.d_c / 2 + 0.03, c - g.retract * n))
    # hook slots, one own (+u) and one receive (-u) per edge region, except under the XP/XN pair (pp)
    hf_fit = hf["fit"]
    if hf_fit:
        zb0, zb1 = hf_fit["band"]
        u, sl = hf_fit["u"], hf_fit["slot_len"]
        for k, bolt in enumerate(BOLTS):
            R = np.linalg.matrix_power(C4, k)
            for tab in ("a", "b"):
                if (bolt, tab) in DROPPED_TABS:
                    continue
                for sgn in (1, -1):
                    if tab == "a":  # +y face of pp, band near z = +edge
                        s = box((sgn * u - sl / 2 - CLEAR, g.Ri - 0.01, zb0), (sgn * u + sl / 2 + CLEAR, g.h + 0.5, zb1 + CLEAR))
                    else:  # +z face of pp, band near y = +edge
                        s = box((sgn * u - sl / 2 - CLEAR, zb0, g.Ri - 0.01), (sgn * u + sl / 2 + CLEAR, zb1 + CLEAR, g.h + 0.5))
                    cuts.append(xf(s, R))
    return cuts, adds


def leg_recess(g: Geo, name):
    """Floor pad (add) and recess plus knuckle clearance (cut) for one leg, world coordinates."""
    R = LEG_R[name]
    w2 = g.leg_w / 2 + 0.10
    z_tip = g.h - g.leg_l - 0.10
    pad = box((-w2 - 0.3, g.h - g.leg_t - CLEAR - g.floor, z_tip - 0.3), (w2 + 0.3, g.Ri, g.k_ax))
    rec = box((-w2, g.h - g.leg_t - CLEAR, z_tip), (w2, g.h + 0.5, g.h + 0.5))
    kn = cyl(g.r_k + CLEAR, 2 * w2, (-w2, g.k_ax, g.k_ax), (1, 0, 0))
    # pin bore through the frame knuckle bosses
    pin = cyl(g.pin_d / 2 + 0.02, 2 * (w2 + 0.65), (-(w2 + 0.65), g.k_ax, g.k_ax), (1, 0, 0))
    bosses = [cyl(g.r_k, 0.5, (s * (w2 + 0.05) + (0 if s > 0 else -0.5), g.k_ax, g.k_ax), (1, 0, 0)) for s in (1, -1)]
    # disc-pin sockets beside the leg strip (on the axis across the leg)
    r_p = 0.23 * g.L
    sockets = [cyl(0.25, SKIN + g.floor + 0.2, (s * r_p, g.Ri - 0.25, 0), (0, 1, 0)) for s in (1, -1)]
    return [xf(pad, R)] + [xf(b, R) for b in bosses], [xf(rec, R), xf(kn, R), xf(pin, R)] + [xf(s, R) for s in sockets]


def carrier_rails(g: Geo):
    """Guide rails on the outer-x side of each corner carrier (frame), world coordinates. On the +x
    end (the side the bolts push toward) the rail runs from the carrier's retracted position up to the
    plane the bolt's ramp nose reaches at full stroke (x - n = kb + stroke + 0.1), so the bolt never
    enters it; the latched carrier's first ~0.5 mm of travel is guided by its sphere in the swept skin
    pocket, then by the rail."""
    out = []
    k, n_bot = wedge_k(g)
    kb = k - g.free
    x0 = g.a_x + g.car_half + CLEAR
    for sx in (-1, 1):
        if sx > 0:
            b = box((x0, g.Ri - 3.0, g.Ri - 3.0), (x0 + MIN_WALL, g.Ri + 0.01, g.Ri + 0.01))
            tri = b.intersect(halfspace_n(math.sqrt(2) * (n_bot - g.retract))).intersect(ramp_halfspace(kb + g.stroke + 0.1, 1))
        else:
            b = box((x0, g.Ri - 2.2, g.Ri - 2.2), (x0 + MIN_WALL, g.Ri + 0.01, g.Ri + 0.01))
            tri = b.intersect(halfspace_n(2 * g.Ri - 2.6))
        for sy in (-1, 1):
            for sz in (-1, 1):
                out.append(xf(tri, np.diag([float(sx), float(sy), float(sz)])))
    return out


def halfspace_n(s):
    """Big solid occupying y + z >= s."""
    q = 1 / math.sqrt(2)
    R = np.column_stack([[1, 0, 0], [0, q, q], [0, -q, q]])
    return xf(box((-20, 0, -20), (20, 40, 20)), R, (0, s / 2, s / 2))


def frame(g: Geo, hf):
    """Four plain faces (tube from x = -Ri to Ri), with every face feature; carrier rails."""
    outer = chamfered_cube(g).intersect(box((-g.Ri, -g.h, -g.h), (g.Ri, g.h, g.h)))
    tube = outer.cut(box((-g.Ri - 1, -g.Ri, -g.Ri), (g.Ri + 1, g.Ri, g.Ri)))
    cuts, adds = face_cuts_common(g, hf)
    for name in LEG_R:
        a, c = leg_recess(g, name)
        adds += a
        cuts += c
    adds += carrier_rails(g)
    body = fuse([tube] + adds)
    body = body.intersect(box((-g.Ri, -g.h - 1, -g.h - 1), (g.Ri, g.h + 1, g.h + 1)))
    body = body.cut(fuse(cuts))
    return body.clean()


def internal_ring(m, z, face, r_out):
    """Ring gear with z internal (straight, trapezoid) teeth about +z, z = 0..face."""
    rp = m * z / 2
    gaps = toothed(z, rp - m, rp + 1.25 * m, 0.25, 0.62, face + 0.02)
    gaps = xf(gaps, None, (0, 0, -0.01)).fuse(cq.Solid.makeCylinder(rp - m + 0.001, face + 0.02, cq.Vector(0, 0, -0.01)))
    return cq.Solid.makeCylinder(r_out, face).cut(gaps).clean()


ZX = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)  # local z -> world x


def end_plate(g: Geo, side=1):
    """End plate on +x (side 1) or -x (side -1): x face ring with the disc window and chamfered outer
    edges, the hinge bosses and recess of the D-leg on that end (DP on +x, DN on -x), and the fixed
    ring R1 of the split-ring planetary on three posts (diagonals; the worm diagonal is left free)."""
    plate = chamfered_cube(g).intersect(box((g.Ri, -g.h, -g.h), (g.h, g.h, g.h)))
    plate = plate.cut(cyl(g.r_d + 0.10, 2, (g.Ri - 0.5, 0, 0), (1, 0, 0)))
    r1 = xf(internal_ring(g.wf_m, g.wf_r1, g.wf_face, g.wf_outer), ZX, (g.r1_x[0], 0, 0))
    inner = [r1]
    r_post = g.r_d + 0.15 + 0.25  # posts run outside the disc rim to the end plate
    for sy, sz in ((1, 1), (-1, 1), (-1, -1)):
        n, _ = diag_frame(sy, sz)
        ang = math.atan2(n[1], n[2])  # rotate +z onto n about x
        Rr = np.array([[1, 0, 0], [0, math.cos(ang), math.sin(ang)], [0, -math.sin(ang), math.cos(ang)]])
        inner.append(xf(box((g.r1_x[0], -0.25, g.wf_outer - 0.2), (g.r1_x[1], 0.25, r_post + 0.25)), Rr))
        inner.append(xf(box((g.r1_x[0], -0.25, r_post - 0.25), (g.Ri + 0.01, 0.25, r_post + 0.25)), Rr))
    base = fuse([plate] + inner)
    if side < 0:
        base = xf(base, NX)
    ad, cuts = leg_recess(g, "DP" if side > 0 else "DN")
    lo, hi = (g.Ri, g.h + 0.01) if side > 0 else (-g.h - 0.01, -g.Ri)
    ad = [a.intersect(box((lo, -g.h - 1, -g.h - 1), (hi, g.h + 1, g.h + 1))) for a in ad]
    body = fuse([base] + [a for a in ad if a.Volume() > 1e-6]).cut(fuse(cuts))
    return body.clean()


def disc_hub(g: Geo):
    """+x disc and its hub, which is the output ring R2 of the split-ring planetary (internal teeth),
    4 bayonet tabs (inward lips) on the diagonals at 0.9 disc radius, 4 ring sphere pockets on
    bosses, two sprung pin bores."""
    x0 = g.disc_x0
    disc = cyl(g.r_d, g.t_d, (x0, 0, 0), (1, 0, 0))
    r2 = xf(internal_ring(g.wf_m, g.wf_r2, g.wf_face, g.wf_outer), ZX, (g.r2_x[0], 0, 0))
    r_root = g.wf_m * g.wf_r2 / 2 + 1.25 * g.wf_m
    cup = ring(r_root, g.wf_outer, x0 - g.r2_x[1] + 0.01, (g.r2_x[1] - 0.01, 0, 0), (1, 0, 0))
    parts = [disc, r2, cup]
    cuts = []
    for q in range(4):
        a = math.pi / 4 + q * math.pi / 2
        c, s = math.cos(a), math.sin(a)
        p = np.array([g.ring_sx, g.ring_r * c, g.ring_r * s])
        parts.append(cyl(g.d_r / 2 + 0.15, x0 - g.boss_x0 + 0.01, (g.boss_x0, p[1], p[2]), (1, 0, 0)))
        cuts.append(sphere(g.d_r / 2 + 0.02, p))
        cuts.append(cyl(g.d_r / 2 + 0.02, p[0] - g.boss_x0 + 0.01, (g.boss_x0 - 0.01, p[1], p[2]), (1, 0, 0)))
        rise = g.tab_rise
        sh = box((g.h - 0.01, g.r_g - g.shank / 2, -g.hook_b / 2), (g.h + rise + g.hook_t, g.r_g + g.shank / 2, g.hook_b / 2))
        lip = box((g.h + rise, g.r_g - g.shank / 2 - g.hook_l, -g.hook_b / 2), (g.h + rise + g.hook_t, g.r_g, g.hook_b / 2))
        Rq = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        parts.append(xf(fuse([sh, lip]), Rq))
    r_p = 0.23 * g.L
    for s in (1, -1):
        parts.append(cyl(0.45, x0 - g.boss_x0 + 0.01, (g.boss_x0, s * r_p, 0), (1, 0, 0)))
        cuts.append(cyl(0.03 * g.L / 2 + 0.02, g.h - g.boss_x0 + 0.1, (g.boss_x0 + 0.15, s * r_p, 0), (1, 0, 0)))
    body = fuse(parts).cut(fuse(cuts))
    return body.clean()


def planet(g: Geo):
    """Planet of the split-ring stage: one printed gear spanning both rings (in the design the R2
    half is profile-shifted; drawn unshifted), 0.25 mm steel axle bore. Local axis +z."""
    face = g.r2_x[1] - g.r1_x[0]
    return spur(g.wf_m, g.wf_planet, face).cut(cq.Solid.makeCylinder(0.14, face + 1, cq.Vector(0, 0, -0.5))).clean()


def sun_shaft(g: Geo):
    """Printed sun gear on its shaft (the Didel G220 is pressed on the shaft below). Local axis +z,
    z = 0 at the G220 layer bottom."""
    sun = xf(spur(g.wf_m, g.wf_sun, g.wf_face), None, (0, 0, g.r1_x[0] - g.g220_x[0]))
    shaft = cq.Solid.makeCylinder(0.30, g.r2_x[1] - g.g220_x[0], cq.Vector(0, 0, 0))
    return fuse([sun, shaft])


def leg_knuckle_spans(g: Geo):
    """x spans (XP frame) of a leg's own knuckle segments and of the gaps left for its partner's. The
    x-legs XP and XN share the +y/+z hinge pin; XN is XP turned by x -> -x, so own segments
    [-w2, -w2/2 - c] and [c, w2/2 - c] map onto the gaps and the knuckles interleave like a hinge."""
    w2, c = g.leg_w / 2, CLEAR
    own = [(-w2, -w2 / 2 - c), (c, w2 / 2 - c)]
    gaps = [(-w2 / 2 - c, c), (w2 / 2 - c, w2)]
    return own, gaps


def leg(g: Geo):
    """XP leg: plate in the +y face from the hinge (z = k_ax) to the tip, two knuckle segments (the
    partner leg's segments sit in the gaps, on the same pin) with pin bore and tendon groove, the
    plate notched round the partner's knuckles, steel shim pocket at the tip (outside face). One part
    for all four legs."""
    w2 = g.leg_w / 2
    z_tip = g.h - g.leg_l
    y0 = g.h - g.leg_t
    own, gaps = leg_knuckle_spans(g)
    plate = box((-w2, y0, z_tip), (w2, g.h, g.k_ax))
    plate = plate.cut(fuse([cyl(g.r_k + CLEAR, b - a, (a, g.k_ax, g.k_ax), (1, 0, 0)) for a, b in gaps]))
    kn = [cyl(g.r_k, b - a, (a, g.k_ax, g.k_ax), (1, 0, 0)) for a, b in own]
    body = fuse([plate] + kn)
    body = body.cut(cyl(g.pin_d / 2 + 0.02, g.leg_w + 0.2, (-w2 - 0.1, g.k_ax, g.k_ax), (1, 0, 0)))
    xg = (own[1][0] + own[1][1]) / 2
    groove = ring(g.r_k - 0.10, g.r_k + 0.1, 0.20, (xg - 0.10, g.k_ax, g.k_ax), (1, 0, 0))
    body = body.cut(groove)
    shim = box((-w2 + 0.5, g.h - 0.12, z_tip + 0.3), (w2 - 0.5, g.h + 0.1, z_tip + 1.3))
    body = body.cut(shim)
    return body.clean()


def leg_capstan(g: Geo):
    """Capstan ratchet: ring with n_teeth square teeth at the leg pitch, web, axle bore, tendon drum
    of the knuckle radius times leg_ratio. Local: axis +z, ring from z=0 to cap_face, drum below."""
    ring_ = toothed(g.n_teeth, g.r_c - g.tooth_h / 2 - 0.0, g.r_c + g.tooth_h / 2, 0.5, 0.5, g.cap_face, square=True)
    web_in = g.r_c - g.tooth_h / 2 - 0.35
    ring_ = ring_.cut(cq.Solid.makeCylinder(web_in, 5, cq.Vector(0, 0, -1)))
    web = cq.Solid.makeCylinder(web_in + 0.01, 0.35, cq.Vector(0, 0, 0)).cut(cq.Solid.makeCylinder(0.45, 1, cq.Vector(0, 0, -0.1)))
    hub = cq.Solid.makeCylinder(0.45, g.cap_face, cq.Vector(0, 0, 0))
    drum = cq.Solid.makeCylinder(g.drum_r, 0.6, cq.Vector(0, 0, -0.6))
    flange = cq.Solid.makeCylinder(g.drum_r + 0.12, 0.1, cq.Vector(0, 0, -0.7))
    body = fuse([ring_, web, hub, drum, flange]).cut(cq.Solid.makeCylinder(g.pin_d / 2 + 0.02, 5, cq.Vector(0, 0, -2)))
    return body.clean()


def pawl(g: Geo):
    """Generic pawl on a flexure: anchor pad, flexure beam, head with 3 teeth at the 0.2 mm pitch on
    its -Y side (tips at Y = 0), wire-crimp eye. Local: beam along +X, width along Z (0.8)."""
    w = 0.8
    th = g.tooth_h
    pad = box((-0.7, th + 0.15, 0), (0.0, th + 0.85, w))
    beam = box((0.0, th + 0.15, 0), (1.6, th + 0.15 + MIN_WALL, w))
    head = box((1.6, th, 0), (1.6 + 3 * g.pitch + 0.1, th + 0.55, w))
    teeth = [box((1.6 + 0.05 + k * g.pitch, 0.0, 0), (1.6 + 0.05 + k * g.pitch + g.pitch / 2, th + 0.01, w)) for k in range(3)]
    eye = cyl(0.475, w, (1.6 + 3 * g.pitch + 0.1 + 0.35, th + 0.55, 0), (0, 0, 1))
    body = fuse([pad, beam, head, eye] + teeth)
    body = body.cut(cyl(0.175, w + 0.2, (1.6 + 3 * g.pitch + 0.1 + 0.35, th + 0.55, -0.1), (0, 0, 1)))
    return body.clean()


def wedge_k(g: Geo):
    """Ramp plane between the bolt's nose and its carrier: sign * x - n = k (n along the bisector)."""
    x_end = g.a_x - g.car_half - CLEAR
    n_bot = math.sqrt(2) * g.a_yz - g.d_c / 2 - 0.30
    return x_end - n_bot, n_bot


def ramp_halfspace(k, sign):
    """Solid occupying sign * x - (y + z)/sqrt2 >= k."""
    q = 1 / math.sqrt(2)
    nu = np.array([sign * q, -0.5, -0.5])
    t = np.array([0, -q, q])
    R = np.column_stack([nu, t, np.cross(nu, t)])
    return xf(box((0, -20, -20), (40, 20, 20)), R, (sign * k, 0, 0))


NOSE_T = 0.15  # half thickness of the bolt's nose plates across the edge bisector plane
FIN_T = 0.23  # half thickness of the rack fin: 0.46, leaving a 0.30 core between the 0.08 teeth cut in both faces
RACK_X0 = {"drive": -4.4, "lock": -4.4}  # rack pawls' local origin x (pad at x0 - 0.7, eye at x0 + 2.65)


def fin_n0(g: Geo):
    """Lowest n (along the edge bisector) of the bolt's rack fin and its pawls: the pawls (0.8 wide
    along n, 1.27 across) must stay under both plain faces' bayonet bosses (z, y < boss top)."""
    boss_top = g.Ri - groove_dims(g)["boss_depth"]
    return math.sqrt(2) * (boss_top - 0.05) - (FIN_T + 0.01 + PAWL_RECT[3]) - 0.8


def bolt_bar(g: Geo, hf, with_a=False, knuckle_notch=False):
    """Edge bolt pp (edge +y/+z) at the latched position: bar of section 0.06 L in the corner, 45 deg
    ramp noses at both ends (thin sloped plates reaching inward along the edge bisector by the
    retraction), a rack fin on the bisector plane under the bar with 0.2 mm square teeth on both
    sides (drive pawl on +t, lock pawl on -t; the pawls sit under both faces' bayonet bosses), tab
    sockets, and reliefs for a partner's hook lips under the skin."""
    s = g.bar
    lo, hi = g.Ri - 0.02 - s, g.Ri - 0.02
    k, _ = wedge_k(g)
    kb = k - g.free  # the noses meet the carriers' ramps after the first g.free of travel (hooks free first)
    x_nose = g.a_x - g.d_c / 2 - 0.08  # noses stop short of the spheres
    body = box((-x_nose, lo, lo), (x_nose, hi, hi))
    q = 1 / math.sqrt(2)
    Rn = np.column_stack([[1, 0, 0], [0, q, q], [0, -q, q]])  # local Y -> n, Z -> t
    n_lo = math.sqrt(2) * lo - g.retract
    for sgn in (1, -1):
        xa, xb = sorted((sgn * (x_nose - 1.2 - g.retract - 1.0), sgn * x_nose))
        blk = xf(box((xa, n_lo, -NOSE_T), (xb, math.sqrt(2) * lo + 0.3, NOSE_T)), Rn)
        blk = blk.intersect(ramp_halfspace(kb - 0.8, sgn))
        body = body.fuse(blk)
    body = body.cut(fuse([ramp_halfspace(kb, 1), ramp_halfspace(kb, -1)]))
    # rack fin: covers both pawl heads over the whole travel
    n0 = fin_n0(g)
    heads = [x0 + 1.6 for x0 in RACK_X0.values()]
    fx0, fx1 = min(heads) - g.stroke - 0.15, max(heads) + 0.7 + 0.15
    body = body.fuse(xf(box((fx0, n0 - 0.05, -FIN_T), (fx1, math.sqrt(2) * lo + 0.3, FIN_T)), Rn))
    nt = int((fx1 - fx0) / g.pitch)
    teeth = []
    for side in (1, -1):
        t0, t1 = sorted((side * (FIN_T - g.tooth_h), side * (FIN_T + 0.01)))
        teeth += [xf(box((fx0 + j * g.pitch, n0, t0), (fx0 + j * g.pitch + g.pitch / 2, n0 + 0.8, t1)), Rn) for j in range(nt)]
    body = body.cut(fuse(teeth))
    # chamfer the edge-side corner to clear the leg knuckles (pp shares its edge with the XP/XN pair)
    body = body.cut(halfspace_n(2 * g.k_ax - (g.r_k + CLEAR) * math.sqrt(2)))
    # clear both faces' bayonet bosses (rings under the skin) over the whole travel (+x by stroke)
    gd = groove_dims(g)
    r_b = gd["boss"][1] + CLEAR
    y_b = g.Ri - gd["boss_depth"] - CLEAR
    for Rf in (np.eye(3), np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], float)):
        st = fuse([cyl(r_b, 2.0, (0, y_b, 0), (0, 1, 0)), cyl(r_b, 2.0, (-g.stroke, y_b, 0), (0, 1, 0)),
                   box((-g.stroke, y_b, -r_b), (0, y_b + 2.0, r_b))])
        body = body.cut(xf(st, Rf))
    if knuckle_notch:
        # bolt pp runs under the XP / XN leg recess floors (pads from y or z = pad_y0 to the skin) for
        # |x| <= the pad half width; notch the bar there over the whole travel
        pad_y0 = g.h - g.leg_t - CLEAR - g.floor
        X = g.leg_w / 2 + 0.10 + 0.3 + CLEAR
        cut_ = fuse([box((-X - g.stroke, pad_y0 - CLEAR, -20), (X, 20, 20)), box((-X - g.stroke, -20, pad_y0 - CLEAR), (X, 20, 20))])
        body = body.cut(cut_)
    fit = hf["fit"]
    if fit:
        zb0, zb1 = fit["band"]
        u, b, sl = fit["u"], fit["b"], fit["slot_len"]
        lip_depth = g.hook_t + 0.04
        c_b = -u - g.stroke / 2
        cuts = [box((c_b - g.shank_x / 2 - 0.01, zb0 - 0.01, hi - 0.3), (c_b + g.shank_x / 2 + 0.01, zb0 + b + 0.01, hi + 0.1)),
                box((u - sl / 2 - g.hook_l - g.stroke - 0.05, zb0 - 0.01, hi - lip_depth), (u + sl / 2 + g.hook_l + 0.05, hi + 0.1, hi + 0.1))]
        if with_a:
            c_a = u - g.stroke / 2
            cuts += [box((c_a - g.shank_x / 2 - 0.01, hi - 0.3, zb0 - 0.01), (c_a + g.shank_x / 2 + 0.01, hi + 0.1, zb0 + b + 0.01)),
                     box((-u - sl / 2 - g.hook_l - g.stroke - 0.05, hi - lip_depth, zb0 - 0.01), (-u + sl / 2 + g.hook_l + 0.05, hi + 0.1, hi + 0.1))]
        body = body.cut(fuse(cuts))
    return body.clean()


def rack_pawl_frames(g: Geo):
    """Placement (rotation, origin) of the drive and lock pawls of bolt pp, beside its rack fin."""
    q = 1 / math.sqrt(2)
    xh, nh, th = np.array([1.0, 0, 0]), np.array([0, q, q]), np.array([0, -q, q])
    n0 = fin_n0(g)
    Rd = np.column_stack([xh, th, np.cross(xh, th)])  # teeth face -t (the fin), width along -n
    od = RACK_X0["drive"] * xh + (FIN_T + 0.01) * th + (n0 + 0.8) * nh
    Rl = np.column_stack([xh, -th, np.cross(xh, -th)])  # teeth face +t, width along +n
    ol = RACK_X0["lock"] * xh - (FIN_T + 0.01) * th + n0 * nh
    return (Rd, od), (Rl, ol)


def hook_tab(g: Geo, hf, which="b"):
    """Bolt hook tab of pp at its latched position: shank pressed 0.3 into the bar socket, rising
    through the own skin and a partner's flex and skin, lip pointing -x (the latch direction).
    which 'b': into +z (band in y); 'a': into +y (band in z)."""
    fit = hf["fit"]
    zb0, b, u = fit["band"][0], fit["b"], fit["u"]
    hi = g.Ri - 0.02
    top = g.h + g.tab_rise
    c = -u - g.stroke / 2 if which == "b" else u - g.stroke / 2
    shank = box((c - g.shank_x / 2, zb0, hi - 0.3), (c + g.shank_x / 2, zb0 + b, top + g.hook_t))
    lip = box((c - g.shank_x / 2 - g.hook_l, zb0, top), (c - g.shank_x / 2 + 0.01, zb0 + b, top + g.hook_t))
    tab = fuse([shank, lip])
    if which == "a":
        tab = xf(tab, np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], float))  # swap y and z (a mirror; the tab is symmetric)
    return tab


def corner_carrier(g: Geo):
    """Carrier of the corner sphere at (+a_x, +a_yz, +a_yz), latched: a cup around the sphere in
    the corner triangle; its inner-x face is the 45 deg ramp the bolt's nose pushes (inward along the
    edge bisector). World coordinates for this corner."""
    c = np.array([g.a_x, g.a_yz, g.a_yz])
    hi = g.Ri - 0.02
    k, n_bot = wedge_k(g)
    blk = box((g.a_x - g.car_half, hi - 2.0, hi - 2.0), (g.a_x + g.car_half, hi, hi))
    blk = blk.intersect(halfspace_n(n_bot * math.sqrt(2)))
    blk = blk.intersect(ramp_halfspace(k + CLEAR * math.sqrt(2), 1))
    blk = blk.cut(sphere(g.d_c / 2 + 0.02, c))
    return blk.clean()


# ---------------------------------------------------------------------------------------------
# Bought parts (envelopes)
# ---------------------------------------------------------------------------------------------


PAWL_EYE = np.array([1.6 + 0.6 + 0.1 + 0.35, 0.08 + 0.55, 0.4])  # crimp-pin eye centre in the pawl's local frame
PAWL_RECT = (-0.7, 3.13, 0.0, 1.11)  # pawl outline in its local X, Y (pad to eye boss)
ZUP = np.eye(3)
XPOS = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)  # capstan local z -> +x, x -> +y, y -> +z
XNEG = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]], float)  # local z -> -x, x -> +y, y -> -z


def capstan_slots(g: Geo):
    """Option B capstan layout: leg -> (origin of the capstan's local frame, rotation local -> world,
    world keep-in box (lo, hi) for its pawls). Local frame: axis +z, ring z = 0..cap_face, tendon drum
    below (z < 0, toward the cell or the -z wire plane).
    1. XP, XN (the leg pair on the +y/+z edge): axis along x, beside the off-axis cell above the
       worms, ring layer just outboard of the sun G220 (-x and +x side).
    1. DP, DN: axis along z, in the band under the -z face, below the worms and the selector gear
       layers, either side of y = 0."""
    # x-pair: centre height so the ring clears the sun G220 pinion; keep-in under the corner fins
    # (y <= the fin's lowest point), above the sun G220 pinion, under the leg recess floor
    q = 1 / math.sqrt(2)
    y_fin = q * fin_n0(g) - 0.15  # a point with y, z <= y_fin is below the fin's n range
    zb = 4.9 * g.L / 18.6
    za = g.worm_z - DIDEL_OD["G220"] / 2 - 0.2 - g.cap_face  # ring top 0.2 under the arbor G220s
    ya_max = math.sqrt(2) * (fin_n0(g) - 0.15) + za  # pawls stay below the -z corners' rack pawls (n < fin_n0)
    ya = 0.5 * ya_max + 0.4
    hi_z = min(g.h - g.leg_t - CLEAR - g.floor - CLEAR, g.Ri - groove_dims(g)["boss_depth"] - 0.1 - 0.4)  # under the x-pair wire lanes
    xa = g.r1_x[0] - 0.1
    return {
        "XN": (np.array([g.cap_x, 0.0, zb]), XPOS, ((g.cap_x, -y_fin, 1.3), (g.cap_x + 0.9, y_fin, hi_z))),
        "XP": (np.array([-g.cap_x, 0.0, zb]), XNEG, ((-g.cap_x - 0.9, -y_fin, 1.3), (-g.cap_x, y_fin, hi_z))),
        "DN": (np.array([0.0, ya, za]), ZUP, ((-xa, 0.15, za), (xa, ya_max, za + 0.9))),
        "DP": (np.array([0.0, -ya, za]), ZUP, ((-xa, -ya_max, za), (xa, -0.15, za + 0.9))),
    }


CAPSTANS = {"XP": -1, "XN": 1, "DP": 1, "DN": -1}  # kept for the wire map: x side of the far end


def _pawl_corners(r_tip, phi, beam):
    rv = np.array([math.cos(phi), math.sin(phi)])
    tau = beam * np.array([-math.sin(phi), math.cos(phi)])
    x0, x1, y0, y1 = PAWL_RECT
    base = rv * r_tip - tau * 1.95
    return np.array([base + X * tau + Y * rv for X, Y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))])


def _sat_overlap(A, B, gap=0.06):
    for P in (A, B):
        for k in range(4):
            e = P[(k + 1) % 4] - P[k]
            nrm = np.array([-e[1], e[0]]) / np.linalg.norm(e)
            pa, pb = A @ nrm, B @ nrm
            if pa.max() + gap <= pb.min() or pb.max() + gap <= pa.min():
                return False
    return True


STRUCT = ("frame", "end_plate_px", "end_plate_nx")  # parts a wire anchor may be fixed to
RIBBON_R = 0.12  # leg wire ribbon (5 x 38 um side by side, 0.05 apart) envelope radius
ANCHOR_GAP = 0.10  # anchor crimp end to the wall
CRIMP_LEN = 0.60  # crimp tube length (not active wire)
LEVER_A = 0.50  # wire arm of the stroke lever, mm (pawl arm = k x LEVER_A)
K_GRID = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)  # crank ratios tried (pawl stroke / wire stroke)


class Scene:
    """Tessellated obstacles for straight-ray queries (Moller-Trumbore, numpy), with a per-part
    bounding-box prefilter. Used to find straight SMA wire corridors (no bends at all)."""

    def __init__(self):
        self.parts = []  # (name, lo, hi, V0, E1, E2)

    def add(self, name, shape, tol=0.03):
        vs, tris = shape.tessellate(tol, 0.3)
        if not tris:
            return
        V = np.array([[v.x, v.y, v.z] for v in vs])
        T = np.array(tris)
        a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
        self.parts.append((name, V.min(0) - 1e-6, V.max(0) + 1e-6, a, b - a, c - a))

    def _cast1(self, o, d, skip):
        best = (np.inf, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_d = np.where(np.abs(d) > 1e-12, 1.0 / np.where(np.abs(d) > 1e-12, d, 1.0), np.inf)
        for name, lo, hi, V0, E1, E2 in self.parts:
            if name in skip:
                continue
            with np.errstate(invalid="ignore"):
                t1, t2 = (lo - o) * inv_d, (hi - o) * inv_d
            t1 = np.where(np.isnan(t1), -np.inf, t1)
            t2 = np.where(np.isnan(t2), np.inf, t2)
            tmin, tmax = np.minimum(t1, t2).max(), np.maximum(t1, t2).min()
            if tmax < max(tmin, 0.0) or tmin > best[0]:
                continue
            P = np.cross(d, E2)
            det = (E1 * P).sum(1)
            ok = np.abs(det) > 1e-14
            inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
            Tv = o - V0
            u = (Tv * P).sum(1) * inv
            Q = np.cross(Tv, E1)
            v = (Q @ d) * inv
            t = (Q * E2).sum(1) * inv
            hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6)
            if hit.any():
                tt = t[hit].min()
                if tt < best[0]:
                    best = (float(tt), name)
        return best

    def cast(self, o, d, r=0.0, skip=()):
        """Distance along unit d from o to the first obstacle, for a cylinder of radius r (centre
        ray and six rays on its surface), and the obstacle's name."""
        o, d = np.asarray(o, float), np.asarray(d, float) / np.linalg.norm(d)
        offs = [np.zeros(3)]
        if r > 0:
            a = np.cross(d, [1.0, 0, 0])
            if np.linalg.norm(a) < 0.3:
                a = np.cross(d, [0, 1.0, 0])
            a /= np.linalg.norm(a)
            b = np.cross(d, a)
            offs += [r * (math.cos(k * math.pi / 3) * a + math.sin(k * math.pi / 3) * b) for k in range(6)]
        best = (np.inf, None)
        for of in offs:
            t, n = self._cast1(o + of, d, set(skip))
            if t < best[0]:
                best = (t, n)
        return best


def pawl_pose(g: Geo, origin, R, phi, beam):
    """World rotation and origin of a capstan pawl at angle phi with beam sense, its eye, its pull
    direction (+tau, the ring tangent) and the in-plane outward normal (+rv)."""
    r_tip = g.r_c + g.tooth_h / 2 + 0.01
    rv = np.array([math.cos(phi), math.sin(phi), 0.0])
    tau = beam * np.array([-math.sin(phi), math.cos(phi), 0.0])
    nz = np.cross(tau, rv)
    Rp = np.column_stack([tau, rv, nz])
    z0 = 0.05 if nz[2] > 0 else 0.85
    o_loc = rv * r_tip - tau * 1.95 + np.array([0, 0, z0])
    return R @ Rp, origin + R @ o_loc, origin + R @ (o_loc + Rp @ PAWL_EYE), R @ tau, R @ rv


def lever_k(stroke_need, active_mm):
    """Lever ratio pawl stroke / wire stroke for a straight wire at 3 % strain."""
    return stroke_need / (0.03 * active_mm) if active_mm > 0 else float("inf")


def straight_route(scene, start, d, r, skip, lmax=14.0):
    """Free straight run from start along d (to a structural wall); returns (free, hit, active)."""
    t, hit = scene.cast(start, d, r=r, skip=skip)
    t = min(t, lmax)
    active = t - ANCHOR_GAP - CRIMP_LEN if hit in STRUCT else 0.0
    return t, hit, max(active, 0.0)


def _seg_rect(p, d, ln, hw):
    """2D rectangle (corners) of a wire of half width hw from p along d for ln."""
    n = np.array([-d[1], d[0]])
    return np.array([p + n * hw, p + d * ln + n * hw, p + d * ln - n * hw, p - n * hw])


def pawl_angles(g: Geo, origin, R, box_lo, box_hi, scene=None, skip=()):
    """Search angles (15 deg steps) and beam directions for the three pawls so that they do not
    overlap each other (0.06 mm gap) and stay inside the keep-in box; best = largest box margin."""
    r_tip = g.r_c + g.tooth_h / 2 + 0.01
    opts = [(math.radians(a), bm) for a in range(0, 360, 10) for bm in (1, -1)]
    corners = {}
    margin = {}
    lo, hi = np.asarray(box_lo), np.asarray(box_hi)
    for o in opts:
        c2 = _pawl_corners(r_tip, *o)
        corners[o] = c2
        w = np.array([origin + R @ np.array([p[0], p[1], 0.45]) for p in c2])
        m = min((w - lo).min(), (hi - w).min()) if True else 0
        # only the two in-plane world axes matter (the third is the ring layer)
        ax = [i for i in range(3) if abs(R[i, 2]) < 0.5]
        m = min(min((w[:, i] - lo[i]).min(), (hi[i] - w[:, i]).min()) for i in ax)
        margin[o] = m
    good = [o for o in opts if margin[o] >= 0.05]
    if scene is not None:
        return _pawl_angles_rays(g, origin, R, good, corners, margin, scene, skip)
    best, best_m = None, -1e9
    for i, o1 in enumerate(good):
        for j, o2 in enumerate(good):
            if j <= i or abs(o1[0] - o2[0]) < 1e-6 or _sat_overlap(corners[o1], corners[o2]):
                continue
            for o3 in good[j + 1:]:
                if abs(o3[0] - o1[0]) < 1e-6 or abs(o3[0] - o2[0]) < 1e-6:
                    continue
                if _sat_overlap(corners[o1], corners[o3]) or _sat_overlap(corners[o2], corners[o3]):
                    continue
                m = min(margin[o1], margin[o2], margin[o3])
                if m > best_m:
                    best, best_m = (o1, o2, o3), m
    return best, best_m


def _pawl_angles_rays(g, origin, R, good, corners, margin, scene, skip):
    """(open, lock, close) with open and close of opposite beam senses (opposite drive directions),
    no overlaps, best box margin. The wires are routed afterwards through bellcranks."""
    best, best_m = None, -1e9
    for o_open in good:
        for o_close in good:
            if o_close[1] == o_open[1] or abs(o_close[0] - o_open[0]) < 1e-6 or _sat_overlap(corners[o_open], corners[o_close]):
                continue
            m0 = min(margin[o_open], margin[o_close])
            if m0 <= best_m:
                continue
            for o_lock in good:
                if abs(o_lock[0] - o_open[0]) < 1e-6 or abs(o_lock[0] - o_close[0]) < 1e-6:
                    continue
                if _sat_overlap(corners[o_lock], corners[o_open]) or _sat_overlap(corners[o_lock], corners[o_close]):
                    continue
                m = min(m0, margin[o_lock])
                if m > best_m:
                    best, best_m = (o_open, o_lock, o_close), m
    return best, best_m


def capstan_group(g: Geo, name, scene=None, skip=()):
    """Capstan of one leg and its three pawls (open, lock, close), tooth tips 0.01 mm off the ring."""
    origin, R, (lo, hi) = capstan_slots(g)[name]
    out = {f"capstan_{name}": xf(leg_capstan(g), R, origin)}
    best, m = pawl_angles(g, origin, R, lo, hi, scene, skip)
    if best is None:
        raise RuntimeError(f"no pawl arrangement fits for capstan {name} at L {g.L}")
    pw = pawl(g)
    r_tip = g.r_c + g.tooth_h / 2 + 0.01
    for j, (phi, beam) in enumerate(best):
        rv = np.array([math.cos(phi), math.sin(phi), 0.0])
        tau = beam * np.array([-math.sin(phi), math.cos(phi), 0.0])
        nz = np.cross(tau, rv)
        Rp = np.column_stack([tau, rv, nz])
        z0 = 0.05 if nz[2] > 0 else 0.85
        o_loc = rv * r_tip - tau * 1.95 + np.array([0, 0, z0])
        out[f"pawl_{name}{j}"] = xf(pw, R @ Rp, origin + R @ o_loc)
        out[f"_eye_{name}{j}"] = origin + R @ (o_loc + Rp @ PAWL_EYE)
        out[f"_dir_{name}{j}"] = R @ tau
        out[f"_nrm_{name}{j}"] = R @ rv
    out["_margin"] = m
    return out, origin


def bought_and_moving(g: Geo, hf):
    """Every placed part: name -> (shape, kind) with kind printed/bought, plus swept envelopes of the
    moving parts (bolts, carriers) over their release stroke."""
    parts = {}
    sweeps = {}
    # frame and joints
    parts["frame"] = (frame(g, hf), "printed")
    parts["end_plate_px"] = (end_plate(g, 1), "printed")
    parts["end_plate_nx"] = (end_plate(g, -1), "printed")
    dh = disc_hub(g)
    joint = {"disc_hub": (dh, "printed"), "sun_shaft": (xf(sun_shaft(g), ZX, (g.g220_x[0], 0, 0)), "printed")}
    a_sp = g.wf_m * (g.wf_sun + g.wf_planet) / 2
    pl = planet(g)
    for k in range(g.wf_n):
        ang = 2 * math.pi * k / g.wf_n
        joint[f"planet{k}"] = (xf(pl, ZX, (g.r1_x[0], a_sp * math.cos(ang), a_sp * math.sin(ang))), "printed")
    # Didel G220 (20/9) on the sun shaft, catalogue envelope: 20T then its 9T pinion below
    joint["didel_G220_sun"] = (fuse([cyl(DIDEL_OD["G220"] / 2, DIDEL_FACE, (g.g220_x[0], 0, 0), (1, 0, 0)),
                                     cyl(DIDEL_OD["G220_pinion"] / 2, DIDEL_FACE, (g.g220_x[1], 0, 0), (1, 0, 0))]), "bought")  # pinion faces outboard
    # Didel worm GW205 (axis y) tangent to that G220 on the -z side, under the off-axis cell; its arbor
    # runs out along +y to the selector gear layer just outside the cell, where its G220 (20T) is driven
    # by a G209 idler on the selector's swing arm from the G209 on the rotor pin (rotor axis = face centre)
    W = np.array([g.worm_x, 0.0, g.worm_z])
    y_g = g.y_gear
    joint["didel_GW205"] = (cyl(DIDEL_OD["GW205"] / 2, 2.5, W - np.array([0, 1.25, 0]), (0, 1, 0)), "bought")
    joint["worm_arbor"] = (cyl(0.30, y_g + DIDEL_FACE + 0.1 + 1.6, W - np.array([0, 1.6, 0]), (0, 1, 0)), "bought")
    joint["didel_G220_arbor"] = (fuse([cyl(DIDEL_OD["G220"] / 2, DIDEL_FACE, W + np.array([0, y_g, 0]), (0, 1, 0)),
                                       cyl(DIDEL_OD["G220_pinion"] / 2, DIDEL_FACE, W + np.array([0, y_g - DIDEL_FACE, 0]), (0, 1, 0))]), "bought")
    P = np.array([0.0, 0.0, 0.0])  # (x, -, z) of the rotor axis: centred on the +y face, inside the groove boss
    d = np.array([P[0] - W[0], P[2] - W[2]])
    dist = float(np.linalg.norm(d))
    a_, b_ = 2.9, 1.8
    xa = (a_**2 - b_**2 + dist**2) / (2 * dist)
    hh = math.sqrt(max(a_**2 - xa**2, 0.0))
    e = d / dist
    I2 = np.array([W[0], W[2]]) + xa * e - hh * np.array([-e[1], e[0]])
    I = np.array([I2[0], 0.0, I2[1]])
    joint["didel_G209_idler"] = (cyl(DIDEL_OD["G209"] / 2, DIDEL_FACE, I + np.array([0, y_g, 0]), (0, 1, 0)), "bought")
    joint["didel_G209_rotor"] = (cyl(DIDEL_OD["G209"] / 2, DIDEL_FACE, P + np.array([0, y_g, 0]), (0, 1, 0)), "bought")
    joint["rotor"] = (cyl(g.rotor_r, g.rotor_t, P + np.array([0, g.rotor_y0, 0]), (0, 1, 0)), "bought")
    joint["rotor_pin"] = (cyl(0.15, g.rotor_y0 - y_g, P + np.array([0, y_g, 0]), (0, 1, 0)), "bought")
    for k, (sh, kind) in joint.items():
        parts[f"{k}_px"] = (sh, kind)
        parts[f"{k}_nx"] = (xf(sh, NX), kind)
    joint_layout = {"r2_x": [round(v, 3) for v in g.r2_x], "r1_x": [round(v, 3) for v in g.r1_x], "g220_x": [round(v, 3) for v in g.g220_x],
                    "worm_centre": W.round(3).tolist(), "idler": I.round(3).tolist(), "rotor_axis": P.round(3).tolist(),
                    "rotor_slab_y": [round(g.rotor_y0, 3), round(g.rotor_y0 + g.rotor_t, 3)], "rotor_t": round(g.rotor_t, 3), "rotor_t_design": round(g.rotor_t_design, 3),
                    "wolfrom": g.wf_name, "wolfrom_outer_r": round(g.wf_outer, 3), "wolfrom_outer_r_design": round(g.wf_outer_design, 3),
                    "worm_x": round(g.worm_x, 3), "arbor_G220_vs_R1_x_margin": round(g.r1_x[0] - (g.worm_x + DIDEL_OD["G220"] / 2), 3),
                    "cell_z": [round(g.batt_z0, 3), round(g.batt_z0 + g.batt[2], 3)]}
    # ring spheres (in the discs), disc pins
    for side, R in (("px", np.eye(3)), ("nx", NX)):
        for q in range(4):
            a = math.pi / 4 + q * math.pi / 2
            parts[f"ring_sphere_{side}{q}"] = (xf(sphere(g.d_r / 2, (g.ring_sx, g.ring_r * math.cos(a), g.ring_r * math.sin(a))), R), "bought")
        for s in (1, -1):
            parts[f"disc_pin_{side}{'p' if s > 0 else 'n'}"] = (xf(cyl(0.03 * g.L / 2, 0.8, (g.h - 0.8 + 0.3, s * 0.23 * g.L, 0), (1, 0, 0)), R), "bought")
    # corner spheres and carriers; carriers swept over the release
    car = corner_carrier(g)
    nvec = {}
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                R = np.diag([float(sx), float(sy), float(sz)])
                if sx * sy * sz < 0:
                    pass  # reflections: the carrier is symmetric about its own planes, print a mirrored copy
                key = f"{'p' if sx > 0 else 'n'}{'p' if sy > 0 else 'n'}{'p' if sz > 0 else 'n'}"
                c = np.array([sx * g.a_x, sy * g.a_yz, sz * g.a_yz])
                parts[f"corner_sphere_{key}"] = (sphere(g.d_c / 2, c), "bought")
                cr = xf(car, R)
                parts[f"carrier_{key}"] = (cr, "printed")
                nv, _ = diag_frame(sy, sz)
                sweeps[f"carrier_{key}"] = [xf(cr, None, -s * nv) for s in np.linspace(0, g.retract, 5)]
    # steel inserts on the plain faces (in the skin holes)
    for k in range(4):
        R = np.linalg.matrix_power(C4, k)
        for sx in (-1, 1):
            for sz in (-1, 1):
                p = (sx * g.ring_r / math.sqrt(2), g.Ro - g.insert_t, sz * g.ring_r / math.sqrt(2))
                parts[f"insert_{k}{sx}{sz}"] = (xf(cyl(g.insert_d / 2, g.insert_t, p, (0, 1, 0)), R), "bought")
    # legs, pins, shims, capstans, pawls
    lg = leg(g)
    for name, R in LEG_R.items():
        parts[f"leg_{name}"] = (xf(lg, R), "printed")
        w2 = g.leg_w / 2 + 0.75
        if name != "XN":  # XP and XN share one hinge pin (interleaved knuckles): modelled once, as hinge_pin_XP
            parts[f"hinge_pin_{name}"] = (xf(cyl(g.pin_d / 2, 2 * w2, (-w2, g.k_ax, g.k_ax), (1, 0, 0)), R), "bought")
        z_tip = g.h - g.leg_l
        parts[f"tip_shim_{name}"] = (xf(box((-g.leg_w / 2 + 0.55, g.h - 0.11, z_tip + 0.35), (g.leg_w / 2 - 0.55, g.h - 0.01, z_tip + 1.25)), R), "bought")
    cap_layout = {}
    eyes = {}
    pw = pawl(g)
    # edge bolts with tabs and rack pawls; swept over the release stroke
    bars = {(False, False): bolt_bar(g, hf, with_a=False), (True, False): bolt_bar(g, hf, with_a=True)}
    tabs = {w: hook_tab(g, hf, w) for w in ("a", "b")} if hf["fit"] else {}
    lo = g.Ri - 0.02 - g.bar
    for k, bolt in enumerate(BOLTS):
        R = np.linalg.matrix_power(C4, k)
        with_a = (bolt, "a") not in DROPPED_TABS
        notch = bolt == "pp"  # the x-leg pair's shared hinge and recess floors lie on edge pp
        if (with_a, notch) not in bars:
            bars[(with_a, notch)] = bolt_bar(g, hf, with_a=with_a, knuckle_notch=notch)
        b = xf(bars[(with_a, notch)], R)
        parts[f"bolt_{bolt}"] = (b, "printed")
        pieces = [b]
        for w, tb in tabs.items():
            if (bolt, w) in DROPPED_TABS:
                continue
            ts = xf(tb, R)
            parts[f"hook_tab_{bolt}{w}"] = (ts, "printed")
            pieces.append(ts)
        # rack pawls beside the rack fin (drive on +t, lock on -t), under both faces' bayonet bosses
        (Rd, od), (Rl, ol) = rack_pawl_frames(g)
        drive = xf(pw, Rd, od)
        lock = xf(pw, Rl, ol)
        eyes[f"bolt_{bolt}_drive"] = R @ (od + Rd @ PAWL_EYE)
        eyes[f"bolt_{bolt}_lock"] = R @ (ol + Rl @ PAWL_EYE)
        parts[f"rack_pawl_{bolt}0"] = (xf(drive, R), "printed")
        parts[f"rack_pawl_{bolt}1"] = (xf(lock, R), "printed")
        sweeps[f"bolt_{bolt}"] = [fuse([xf(p, None, (sft, 0, 0)) for p in pieces]) for sft in np.linspace(0, g.stroke, 5)]
    # battery (lipo401012 envelope) in the middle slab
    bt, bw, bl = g.batt
    parts["battery"] = (box((-bt / 2, -bw / 2, g.batt_z0), (bt / 2, bw / 2, g.batt_z0 + bl)), "bought")
    # capstans last: their pawls are placed so the open and close wires get straight corridors
    for name in CAPSTANS:
        origin, R, _ = capstan_slots(g)[name]
        parts[f"capstan_{name}"] = (xf(leg_capstan(g), R, origin), "printed")
    scene = Scene()
    for n, (sh, _) in parts.items():
        scene.add(n, sh)
    for n, states in sweeps.items():
        for k, st in enumerate(states[1:]):
            scene.add(f"sweep_{n}_{k}", st)
    routes = []
    # bolt rack pawls: drive (tooth step 0.25 mm along +x) and lock (lift 0.15 mm via a bellcrank)
    q = 1 / math.sqrt(2)
    th = np.array([0, -q, q])
    for k, bolt in enumerate(BOLTS):
        R = np.linalg.matrix_power(C4, k)
        xw = R @ np.array([1.0, 0, 0])
        # drive: eye along +x (rack step); crank arm out along +t. lock: lifted along -t (off the fin)
        # by a crank arm along +x; both wires run along +x toward the end plate.
        routes.append(_route(scene, f"bolt-{bolt}", f"rack_pawl_{bolt}0", eyes[f"bolt_{bolt}_drive"], xw, R @ th, 0.25, 0.05, [xw]))
        routes.append(_route(scene, f"unlock-{bolt}", f"rack_pawl_{bolt}1", eyes[f"bolt_{bolt}_lock"], R @ (-th), xw, 0.15, 0.03,
                             [math.cos(math.radians(an)) * xw + math.sin(math.radians(an)) * (R @ th) for an in range(0, 360, 15)]))
    for r in routes:
        _scene_add_route(scene, r)
    groups = {}
    for name in CAPSTANS:  # all pawls first, so every wire route sees every pawl
        shapes, centre = capstan_group(g, name, scene, skip=())
        groups[name] = shapes
        cap_layout[name] = centre.round(3).tolist()
        for k, sh in shapes.items():
            if k == "_margin":
                cap_layout[name + "_pawl_box_margin"] = round(sh, 3)
            elif k.startswith("_eye_"):
                eyes[k[5:]] = sh
            elif k.startswith("_"):
                continue
            elif k.startswith("capstan_"):
                continue
            else:
                parts[k] = (sh, "printed")
                scene.add(k, sh)
    for name in CAPSTANS:
        shapes = groups[name]
        for j, role in ((0, "open"), (2, "close")):
            tw, rw = shapes[f"_dir_{name}{j}"], shapes[f"_nrm_{name}{j}"]
            dirs = [math.cos(math.radians(an)) * tw + math.sin(math.radians(an)) * rw for an in range(0, 360, 15) if abs(an - 180) > 35]
            r = _route(scene, f"{name}-{role}", f"pawl_{name}{j}", shapes[f"_eye_{name}{j}"], tw, rw, 0.25, RIBBON_R, dirs)
            routes.append(r)
            _scene_add_route(scene, r)
    return parts, sweeps, {"joint": joint_layout, "capstans": cap_layout, "eyes": {k: np.round(v, 3).tolist() for k, v in eyes.items()}, "routes": routes}


def _route(scene, rid, pawl_name, eye, tau, rv, need, r, dirs):
    """Straight SMA element (no pulleys, no bends) driving a pawl eye along tau through a bellcrank:
    pivot P = eye + rv k a, wire arm P -> W = P + (d x n) a (a = LEVER_A, n = tau x rv), so a pull
    along d turns the crank and moves the eye along +tau by k x the wire stroke. Of the candidate
    wire directions dirs the one with the longest active run is taken; the run ends at a wall
    (anchor post into it) or, if it ends at a moving or bought part, short of it on a bracket to the
    nearest wall. k = need / (3 % of the active length), at least 1 (a longer wire is cut to
    need / 3 %)."""
    eye, tau, rv = (np.asarray(v, float) for v in (eye, tau, rv))
    tau, rv = tau / np.linalg.norm(tau), rv / np.linalg.norm(rv)
    n = np.cross(tau, rv)
    skip = (pawl_name,)
    best = None
    for d in dirs:
        d = np.asarray(d, float) / np.linalg.norm(d)
        e2 = np.cross(d, n)
        for k in K_GRID:  # smallest crank ratio this direction supports
            P = eye + rv * k * LEVER_A
            W = P + e2 * LEVER_A
            t, hit = scene.cast(W, d, r=r, skip=skip)
            t = min(t, 20.0)
            a = t - ANCHOR_GAP - CRIMP_LEN - (0.0 if hit in STRUCT else 0.3) if hit is not None else 0.0  # None: leaks out of a window
            if a > 0.5 and lever_k(need, min(a, need / 0.03)) <= k + 1e-9:
                cand = {"d": d, "e2": e2, "t": t, "hit": hit, "a": a, "k": k}
                if best is None or (k, -min(a, need / 0.03)) < (best["k"], -min(best["a"], need / 0.03)):
                    best = cand
                break
    if best is None:
        return {"id": rid, "pawl": pawl_name, "eye": eye.tolist(), "tau": tau.tolist(), "rv": rv.tolist(), "dir": list(map(float, dirs[0])), "pivot": eye.tolist(),
                "start": eye.tolist(), "free_mm": 0.0, "hit": None, "active_mm": 0.0, "lever_k": float("inf"), "need_mm": need, "r": r, "bend_deg": 0.0}
    b = best
    used = min(b["a"], need / 0.03)
    k = b["k"]
    P = eye + rv * k * LEVER_A
    W = P + b["e2"] * LEVER_A
    # the anchor crimp (r 0.25) must fit where the wire ends: step the wire back until it does
    for _ in range(15):
        tc, hc = scene.cast(W + b["d"] * used, b["d"], r=0.25, skip=skip)
        if tc >= CRIMP_LEN + 0.05:
            break
        used -= 0.2
    if used <= 0.5 or lever_k(need, used) > max(K_GRID):
        used = 0.0
    elif lever_k(need, used) > k + 1e-9:
        k = min(x for x in K_GRID if x >= lever_k(need, used) - 1e-9)
        P = eye + rv * k * LEVER_A
        W = P + b["e2"] * LEVER_A
    out = {"id": rid, "pawl": pawl_name, "eye": eye.tolist(), "tau": tau.tolist(), "rv": rv.tolist(), "dir": b["d"].tolist(),
           "pivot": P.tolist(), "start": W.tolist(), "free_mm": round(float(b["t"]), 3), "hit": b["hit"], "active_mm": round(float(used), 3),
           "lever_k": round(float(k), 3), "need_mm": need, "r": r, "bend_deg": round(math.degrees(math.acos(np.clip(np.dot(b["d"], tau), -1, 1))), 1)}
    if b["hit"] not in STRUCT and used > 0:
        A = W + b["d"] * (used + CRIMP_LEN / 2)
        u = np.cross(b["d"], [1.0, 0, 0])
        if np.linalg.norm(u) < 0.3:
            u = np.cross(b["d"], [0, 1.0, 0])
        u /= np.linalg.norm(u)
        v = np.cross(b["d"], u)
        brk = None
        for ang in range(0, 360, 30):
            dv = math.cos(math.radians(ang)) * u + math.sin(math.radians(ang)) * v
            tb, hb = scene.cast(A + dv * 0.3, dv, r=0.15, skip=skip)
            if hb in STRUCT and (brk is None or tb < brk[0]):
                brk = (tb + 0.3, dv)
        if brk is None:
            out["active_mm"] = 0.0
        else:
            out["bracket"] = {"from": A.tolist(), "dir": brk[1].tolist(), "len": round(float(brk[0]), 3)}
    return out


def _scene_add_route(scene, r):
    if r["active_mm"] <= 0:
        return
    sh = route_solids(r)
    for n, s_ in sh.items():
        scene.add(n, s_)


def route_solids(r):
    """Wire, crimps, anchor post or bracket, return lead and its pin, bellcrank and pivot of one route."""
    W, d = np.array(r["start"]), np.array(r["dir"])
    eye, tau, rv, P = np.array(r["eye"]), np.array(r["tau"]), np.array(r["rv"]), np.array(r["pivot"])
    n = np.cross(tau, rv)
    a, t = r["active_mm"], r["free_mm"]
    rid = r["id"]
    out = {f"wire_{rid}": cyl(r["r"], a, W, d)}
    c0 = W + d * a
    out[f"crimp_{rid}_anchor"] = cyl(0.25, CRIMP_LEN, c0, d)
    if "bracket" in r:
        bk = r["bracket"]
        out[f"bracket_{rid}"] = cyl(0.20, bk["len"] + 0.3, np.array(bk["from"]), np.array(bk["dir"]))
    else:
        out[f"post_{rid}"] = cyl(0.10, t - a - CRIMP_LEN + 0.6, c0 + d * CRIMP_LEN, d)  # steel pin into the wall, to the flex pad
    out[f"crimp_{rid}_eye"] = cyl(0.20, 0.4, W, d)
    # return lead (50 um enamelled copper with a slack loop) from the crank pin beside the wire, to a
    # pin through the same wall
    off = np.cross(d, n) if np.linalg.norm(np.cross(d, n)) > 0.5 else n
    off = off / np.linalg.norm(off)
    l0 = W + n * 0.30
    ln = a + CRIMP_LEN + 0.3 if "bracket" in r else t - 0.05
    out[f"lead_{rid}"] = cyl(0.04, ln, l0, d)
    if "bracket" not in r:
        out[f"leadpin_{rid}"] = cyl(0.06, 0.65, l0 + d * ln, d)
    # bellcrank: arms P -> eye and P -> W (0.3 x 0.5 section), pivot pin along n
    for nm, q in (("lever_out", eye), ("lever_in", W)):
        v = q - P
        L_ = float(np.linalg.norm(v))
        if L_ > 1e-6:
            u = v / L_
            w_ = np.cross(n, u)
            out[f"{nm}_{rid}"] = xf(box((-0.15, -0.15, -0.25), (L_ + 0.15, 0.15, 0.25)), np.column_stack([u, w_, n]), P)
    out[f"pivot_{rid}"] = cyl(0.125, 0.8, P - n * 0.4, n)
    return out


def face_uv(p):
    """Nearest face of a point and its panel coordinates (+-y: u = x, v = z; +-z: u = x, v = y;
    +-x: u = y, v = z)."""
    x, y, z = map(float, p)
    a = int(np.argmax(np.abs([x, y, z])))
    sg = "+" if (x, y, z)[a] > 0 else "-"
    if a == 0:
        return sg + "x", y, z
    if a == 1:
        return sg + "y", x, z
    return sg + "z", x, y


ROUTE_PREFIX = ("wire_", "crimp_", "post_", "leadpin_", "lead_", "lever_out_", "lever_in_", "pivot_", "bracket_")
ROUTE_PAWL = {}  # route id -> the pawl it drives (filled by wire_map)


def route_id(name):
    for pre in ROUTE_PREFIX:
        if name.startswith(pre):
            rid = name[len(pre):]
            for suf in ("_anchor", "_eye"):
                if rid.endswith(suf):
                    rid = rid[: -len(suf)]
            return rid
    return None


def route_wall_points(r):
    """Where the anchor post (or bracket) and the return lead's pin meet the wall."""
    W, d = np.array(r["start"]), np.array(r["dir"])
    n = np.cross(np.array(r["tau"]), np.array(r["rv"]))
    if "bracket" in r:
        bk = r["bracket"]
        p = np.array(bk["from"]) + np.array(bk["dir"]) * bk["len"]
        return p, p + 0.0
    return W + d * r["free_mm"], W + n * 0.30 + d * (r["free_mm"] - 0.05)


def wire_map(g: Geo, routes, design):
    """Every SMA element from the straight routes (no bends: every wire is a straight run from a
    bellcrank at its pawl to a crimp on a post or bracket at a wall). Legs: one ribbon of
    leg_strands x leg_wire per pawl, split floor / full at the anchor crimp (two channels); bolts:
    rel_strands x rel_wire; unlock: four 25 um segments in series (one channel)."""
    wires = []
    floor_n = math.ceil(design.get("leg_floor_frac", 0.5) * design["leg_strands"] - 1e-9)
    for r in routes:
        ROUTE_PAWL[r["id"]] = r["pawl"]
        if r["active_mm"] <= 0:
            continue
        W, d = np.array(r["start"]), np.array(r["dir"])
        path = [W.tolist(), (W + d * r["active_mm"]).tolist()]
        pa, pl = route_wall_points(r)
        fa, ua, va = face_uv(pa)
        fl, ul, vl = face_uv(pl)
        base = {"route": r["id"], "path": path, "strand_mm": r["active_mm"], "lever_k": r["lever_k"], "bend_deg_at_crank": r["bend_deg"],
                "pins": []}
        if r["id"].startswith(("XP", "XN", "DP", "DN")):
            for grp, ns in (("floor", floor_n), ("full", design["leg_strands"] - floor_n)):
                cr = [(fa, ua, va, "anchor pad (" + grp + ")")] + ([(fl, ul, vl, "return lead pad")] if grp == "floor" else [])
                wires.append(dict(base, id=f"{r['id']}-{grp}", channel=f"{r['id']}-{grp}", actuator=f"leg {r['id'][:2]} capstan, {r['id'][3:]} pawl",
                                  d_um=round(design["leg_wire"] * 1e6), strands=ns, crimps=cr))
        elif r["id"].startswith("bolt-"):
            wires.append(dict(base, id=r["id"], channel=r["id"], actuator=f"edge bolt {r['id'][5:]}, rack drive pawl", d_um=round(design["rel_wire"] * 1e6),
                              strands=design["rel_strands"], crimps=[(fa, ua, va, "anchor pad"), (fl, ul, vl, "return lead pad")]))
        else:
            wires.append(dict(base, id=r["id"], channel="unlock", actuator=f"unlock of bolt {r['id'][7:]} lock pawl (series segment)", d_um=25,
                              strands=1, crimps=[(fa, ua, va, "anchor pad (series link)"), (fl, ul, vl, "lead pad (series link)")]))
    return wires


def wire_solids(routes):
    out = {}
    for r in routes:
        if r["active_mm"] > 0:
            out.update(route_solids(r))
    return out


# ---------------------------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------------------------


def _bb_overlap(A, B, tol=0.0):
    return not (A.xmax < B.xmin - tol or B.xmax < A.xmin - tol or A.ymax < B.ymin - tol or B.ymax < A.ymin - tol or A.zmax < B.zmin - tol or B.zmax < A.zmin - tol)


def _attached(a, b, parts):
    """Pairs that touch by design and are not checked: a wire and the pawl it pulls, wires of the
    same bundle, components in their skin pockets (frame, end plates)."""
    ra, rb = route_id(a), route_id(b)
    if ra is not None and ra == rb:
        return True  # one SMA element: wire, crimps, crank, pivot, lead, post or bracket
    if {a, b} & {"board"} and (a.startswith(("bcomp_", "tail_")) or b.startswith(("bcomp_", "tail_"))):
        return True  # soldered on the board / tail bonded to its edge
    if {a, b} & {"battery"} and (a.startswith("batt_tab") or b.startswith("batt_tab")):
        return True  # tab welded to the cell terminal
    for x, y, rx in ((a, b, ra), (b, a, rb)):
        if rx is None:
            continue
        if y == ROUTE_PAWL.get(rx):
            return True  # crank on its pawl's eye pin
        if y in STRUCT and x.startswith(("post_", "leadpin_", "bracket_")):
            return True  # anchor post / lead pin pressed into the wall, bracket printed onto it
    return False


def overlaps(parts: dict, sweeps: dict, g: Geo, thresh=0.01):
    names = list(parts)
    bbs = {n: parts[n][0].BoundingBox() for n in names}
    pairs = [("sun_shaft", "didel_G220_sun"), ("didel_G220_sun", "didel_GW205"), ("worm_arbor", "didel_GW205"), ("worm_arbor", "didel_G220_arbor"),
             ("didel_G220_arbor", "didel_G209_idler"), ("didel_G209_idler", "didel_G209_rotor"), ("rotor_pin", "didel_G209_rotor"), ("rotor_pin", "rotor")]
    pairs += [(f"planet{k}", o) for k in range(3) for o in ("sun_shaft", "disc_hub")]
    mesh = {frozenset((f"{a}_{side}", f"{b}_{side}")) for side in ("px", "nx") for a, b in pairs}
    mesh |= {frozenset((f"planet{k}_{side}", f"end_plate_{side}")) for k in range(3) for side in ("px", "nx")}
    static, meshing, flexc = [], [], []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if not _bb_overlap(bbs[a], bbs[b]) or _attached(a, b, parts):
                continue
            try:
                v = parts[a][0].copy().intersect(parts[b][0].copy()).Volume()
            except Exception as e:  # noqa: BLE001
                v = float("nan")
            if not (v <= thresh):
                (meshing if frozenset((a, b)) in mesh else static).append((a, b, round(v, 4)))
    swept = []
    for m, states in sweeps.items():
        if m.startswith("bolt_"):
            bolt = m.split("_")[1]
            skip = {m} | {n for n in names if n.startswith("hook_tab_" + bolt) or (n.startswith("carrier_") and n[-2:] == bolt)}
        else:
            key = m.split("_")[1]
            skip = {m, "corner_sphere_" + key, "bolt_" + key[1:]} | {n for n in names if n.startswith("hook_tab_" + key[1:])}
        for b in names:
            if b in skip:
                continue
            worst = 0.0
            for sh in states[1:]:
                if not _bb_overlap(sh.BoundingBox(), bbs[b]):
                    continue
                try:
                    worst = max(worst, sh.copy().intersect(parts[b][0].copy()).Volume())
                except Exception:  # noqa: BLE001
                    worst = float("nan")
            st = next((x[2] for x in static if {x[0], x[1]} == {m, b}), 0.0)
            if not (worst <= st + thresh):
                swept.append((m, b, round(worst, 4)))
    # outside the cube envelope (beyond the flex outer surface); the tabs and tread stand proud by design
    env = box((-g.h, -g.h, -g.h), (g.h, g.h, g.h))
    outside = []
    for n in names:
        bb = bbs[n]
        if max(abs(bb.xmin), abs(bb.xmax), abs(bb.ymin), abs(bb.ymax), abs(bb.zmin), abs(bb.zmax)) <= g.h + 1e-6:
            continue
        v = parts[n][0].copy().cut(env).Volume()
        if v > thresh:
            outside.append((n, round(v, 4)))
    return static, swept, outside, meshing, flexc


def feature_table(g: Geo, hf, table=None):
    """Printer-limit table: (feature, size mm, pixels at 19 um, at risk)."""
    fit = hf["fit"] or {}
    gd = groove_dims(g)
    rows = [
        ("frame skin wall", SKIN, "wall"),
        ("leg recess floor", g.floor, "wall"),
        ("corner sphere cover (resin over sphere, plain faces)", g.cover, "wall"),
        ("ring sphere cover (disc face over sphere)", g.cover, "wall"),
        ("bayonet groove boss wall", MIN_WALL, "wall"),
        ("bayonet slit width (radial)", gd["slit"][1] - gd["slit"][0], "hole"),
        ("bayonet tab lip thickness (axial)", g.hook_t, "feature"),
        ("bayonet tab lip overhang (radial, inward)", g.hook_l, "feature"),
        ("bayonet tab width (tangential)", g.hook_b, "feature"),
        ("bolt hook tab width (fitted)", fit.get("b", float("nan")), "feature"),
        ("bolt hook tab shank (along x)", g.shank_x, "feature"),
        ("bolt hook slot width", fit.get("b", 0) + CLEAR, "hole"),
        ("bolt bar section", g.bar, "feature"),
        ("bolt bar left under a partner lip (relief)", g.bar - (g.hook_t + 0.04), "wall"),
        ("rack / capstan tooth pitch", g.pitch, "feature"),
        ("rack / capstan tooth width (half pitch)", g.pitch / 2, "feature"),
        ("rack / capstan tooth height (0.4 pitch)", g.tooth_h, "feature"),
        ("capstan web (ring to hub)", 0.35, "wall"),
        ("capstan axle bore (0.25 pin)", g.pin_d + 0.04, "hole"),
        ("tendon drum wall around axle bore", g.drum_r - (g.pin_d / 2 + 0.02), "wall"),
        ("leg plate thickness", g.leg_t, "wall"),
        ("leg knuckle wall around pin bore", g.r_k - (g.pin_d / 2 + 0.02), "wall"),
        ("leg knuckle wall at tendon groove", g.r_k - 0.10 - (g.pin_d / 2 + 0.02), "wall"),
        ("hinge pin bore (0.25 pin)", g.pin_d + 0.04, "hole"),
        ("tip shim pocket depth", 0.12, "feature"),
        ("pawl flexure beam", MIN_WALL, "wall"),
        ("pawl tooth width", g.pitch / 2, "feature"),
        ("pawl tooth height", g.tooth_h, "feature"),
        ("pawl crimp-pin eye bore", 0.35, "hole"),
        ("pawl eye wall", 0.475 - 0.175, "wall"),
        ("split-ring tooth height (2.25 m, module %.3f)" % g.wf_m, 2.25 * g.wf_m, "feature"),
        ("split-ring tooth top land (0.25 pitch)", 0.25 * math.pi * g.wf_m, "feature"),
        ("split-ring tooth root width (0.62 pitch)", 0.62 * math.pi * g.wf_m, "feature"),
        ("split-ring ring wall (outside the tooth roots)", 0.5, "wall"),
        ("planet axle bore (0.25 pin)", 0.28, "hole"),
        ("sun shaft diameter", 0.60, "feature"),
        ("disc pin bore", 0.03 * g.L + 0.04, "hole"),
        ("steel insert hole", g.insert_d + 0.06, "hole"),
        ("disc-pin socket", 0.5, "hole"),
        ("end plate ring posts", 0.5, "wall"),
        ("carrier guide rail", MIN_WALL, "wall"),
    ]
    out = []
    if table:
        pk = table.get("packing")
        out.append({"feature": "packing (0.8 x inner volume / parts volume; hand assembly needs >= 1.5)", "size_mm": pk, "px_19um": None, "limit_mm": 1.5, "kind": "assembly", "at_risk": bool(pk is None or pk < 1.7)})
        lw = g.d["leg_wire"] * 1e6
        out.append({"feature": "leg wire %.0f um: 3 %% stroke of a %.1f mm run is %.2f mm; a thick wire needs a longer run (or a lever) for 1.25 teeth plus pawl lift" % (lw, 0.8 * (g.L - 0.8), 0.024 * (g.L - 0.8)),
                    "size_mm": round(0.024 * (g.L - 0.8), 3), "px_19um": None, "limit_mm": round(1.25 * g.pitch, 3), "kind": "actuator", "at_risk": bool(lw >= 50)})
    for name, size, kind in rows:
        px = size / PIXEL_XY
        lim = {"wall": MIN_WALL, "hole": MIN_HOLE, "feature": MIN_FEATURE}[kind]
        risk = not (size >= lim - 1e-9) or px < 8
        out.append({"feature": name, "size_mm": round(size, 3), "px_19um": round(px, 1), "limit_mm": lim, "kind": kind, "at_risk": bool(risk)})
    return out


def fit_report(g: Geo, hf, layout):
    """Plain fit numbers, mm (negative margin = does not fit by that much)."""
    r = {}
    gd = groove_dims(g)
    # joint (nominal_design, option B): x stack from the cube axis out to the disc. The cell (x half
    # thickness) is followed by the sun / arbor G220 layer at worm_x; the arbor G220 (OD) must clear
    # the fixed ring R1, which starts at r1_x[0]; the worm sits under the cell (z), not beside it.
    r["joint_x_stack"] = {"cell_half": round(g.batt[0] / 2, 3), "g220_layer_x": [round(v, 3) for v in g.g220_x],
                          "r1_x": [round(v, 3) for v in g.r1_x], "r2_x": [round(v, 3) for v in g.r2_x], "disc_x0": round(g.disc_x0, 3)}
    r["joint_x_depth_margin"] = round(g.r1_x[0] - (g.worm_x + DIDEL_OD["G220"] / 2), 3)  # arbor G220 rim vs R1 inner face (x)
    r["worm_vs_cell_z_margin"] = round(g.batt_z0 - (g.worm_z + DIDEL_OD["GW205"] / 2), 3)
    r["wolfrom_outer_r"] = round(g.wf_outer, 3)
    r["wolfrom_outer_r_design"] = round(g.wf_outer_design, 3)
    r["ring_wall"] = g.ring_wall
    r["wolfrom_vs_ring_sphere_boss_r"] = round(g.ring_r + g.d_r / 2 + 0.15, 3)
    r["rotor_diameter"] = round(2 * g.rotor_r, 3)
    r["rotor_t"] = round(g.rotor_t, 3)
    r["rotor_t_design"] = round(g.rotor_t_design, 3)
    r["rotor_slab_y"] = [round(g.rotor_y0, 3), round(g.rotor_y0 + g.rotor_t, 3)]
    r["rotor_slab_room"] = round(g.rotor_y1 - g.rotor_y0, 3)
    r["capstan_pawl_box_margins"] = {k[:2]: v for k, v in layout["capstans"].items() if k.endswith("_margin")}
    r["bolt_travel"] = g.stroke
    r["sphere_retraction"] = round(g.retract, 3)
    # bolt hooks
    r["hook_tab_width_fitted"] = (hf["fit"] or {}).get("b")
    r["hook_tab_width_design"] = round(g.hook_b, 3)
    r["hook_two_per_side_slot_length_need"] = hf["two_per_side_need_mm"]
    r["hook_region_length"] = hf["region_mm"]
    r["hook_full_width_slot_room"] = hf["full_width_slot_room_mm"]
    # legs vs steel inserts and bayonet notches
    ins_in = g.ring_r / math.sqrt(2) - g.insert_d / 2
    r["leg_strip_half_width"] = round(g.leg_w / 2 + 0.10, 3)
    r["leg_vs_insert_margin"] = round(ins_in - (g.leg_w / 2 + 0.10), 3)
    r["leg_strip_covers_groove_deg_each_end"] = round(2 * math.degrees(math.asin(min(1.0, (g.leg_w / 2 + 0.1) / g.r_g))), 1)
    r["bayonet_throw_window_deg"] = round(90 - r["leg_strip_covers_groove_deg_each_end"], 1)
    # knuckle on the lattice edge line
    r["knuckle_axis_offset_from_edge_line"] = round(math.sqrt(2) * g.r_k, 3)
    r["knuckle_protrusion_if_on_edge_line"] = round(g.r_k, 3)
    # tabs proud of the faces
    r["disc_tab_proud_of_face"] = round(g.tab_rise + g.hook_t, 3)
    r["bolt_tab_proud_of_face"] = round(g.tab_rise + g.hook_t, 3)
    r["tread_proud"] = g.tread
    r["battery_mm"] = list(g.batt)
    return r


# ---------------------------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------------------------

EXPORT = {
    "frame": "frame",
    "end_plate": "end_plate_px",
    "end_plate_nx": "end_plate_nx",
    "disc_hub": "disc_hub_px",
    "planet": "planet0_px",
    "sun_shaft": "sun_shaft_px",
    "leg": "leg_XP",
    "leg_capstan": "capstan_XP",
    "pawl": None,
    "edge_bolt": "bolt_pp",
    "hook_tab": "hook_tab_ppb",
    "corner_carrier": "carrier_ppp",
}


def export(parts, g: Geo, pawl_shape, out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for fname, key in EXPORT.items():
        if key is not None and key not in parts:
            continue  # e.g. no hook tab fits at this L
        s = pawl_shape if key is None else parts[key][0]
        cq.exporters.export(s, str(out / f"{fname}.step"))
        cq.exporters.export(s, str(out / f"{fname}.stl"), tolerance=0.005, angularTolerance=0.1)
        written += [f"{fname}.step", f"{fname}.stl"]
    assy = cq.Assembly(name="sma_cube")
    colours = {"printed": cq.Color(0.75, 0.75, 0.8), "bought": cq.Color(0.8, 0.5, 0.2)}
    for n, (s, kind) in parts.items():
        assy.add(s, name=n, color=colours[kind])
    assy.save(str(out / "assembly.step"))
    written.append("assembly.step")
    return written


# ---------------------------------------------------------------------------------------------
# Flex net
# ---------------------------------------------------------------------------------------------


FACE_AX = {"+x": (0, 1, 2), "-x": (0, 1, 2), "+y": (1, 0, 2), "-y": (1, 0, 2), "+z": (2, 0, 1), "-z": (2, 0, 1)}  # normal, u, v axes
HT_LEVELS = (0.35, 0.75, 1.10)  # component heights (pocket depth below the flex) the masks are built for


def interior_blocked(g: Geo, shapes, res=0.05, clear=0.05):
    """Per face and height level, a (v, u) raster of where something inside the cube (any part,
    swept envelope, wire, crimp, lever, bracket; not the skin itself) lies within that height of
    the outer skin surface. Each obstacle is cut to a slab under the skin, tessellated, and its
    triangles rasterised with their shallowest depth (conservative); dilated by clear."""
    from PIL import Image, ImageDraw
    h = g.h
    grid = np.arange(-h + res / 2, h, res)
    N = len(grid)
    deep = max(HT_LEVELS) + clear
    out = {}
    for f, (ax, au, av) in FACE_AX.items():
        sg = 1.0 if f[0] == "+" else -1.0
        lo = [-h - 1] * 3
        hi = [h + 1] * 3
        lo[ax], hi[ax] = sorted((sg * (g.Ro - deep), sg * (g.Ri - 0.001)))
        slab = box(tuple(lo), tuple(hi))
        sbb = slab.BoundingBox()
        imgs = {ht: Image.new("1", (N, N), 0) for ht in HT_LEVELS}
        drs = {ht: ImageDraw.Draw(im) for ht, im in imgs.items()}
        for name, sh in shapes.items():
            if not _bb_overlap(sh.BoundingBox(), sbb):
                continue
            try:
                pc = sh.copy().intersect(slab)
                if pc.Volume() < 1e-7:
                    continue
                vs, tris = pc.tessellate(0.02, 0.3)
            except Exception:  # noqa: BLE001
                continue
            V = np.array([[v.x, v.y, v.z] for v in vs])
            if len(V) == 0:
                continue
            depth = g.Ro - sg * V[:, ax]
            U = (V[:, au] + h) / res - 0.5
            Vv = (V[:, av] + h) / res - 0.5
            for t in tris:
                dmin = depth[list(t)].min()
                poly = [(U[i], Vv[i]) for i in t]
                for ht in HT_LEVELS:
                    if dmin < ht + clear:
                        drs[ht].polygon(poly, fill=1, outline=1)
        for ht, im in imgs.items():
            m = np.array(im, bool)
            k = int(math.ceil(clear / res))
            mm = m.copy()
            for di in range(-k, k + 1):
                for dj in range(-k, k + 1):
                    mm |= np.roll(np.roll(m, di, 0), dj, 1)
            out[(f, ht)] = mm
    return out


def flex_net(g: Geo, hf, design_table, wires_list, out=OUT, blocked=None, tab_pads=None):
    """Unfolded flex PCB net: six L x L panels in a cross; windows, pads, footprints, coils, fold
    lines, stiffener outlines, power rails; SVG and DXF. Returns a placement report."""
    import ezdxf

    Lm = g.L
    h = g.h
    # panel origins (lower-left) in net coordinates; cross: column +y / +z / -y / -z, +x and -x beside +z
    P = {"+y": (Lm, 3 * Lm), "+z": (Lm, 2 * Lm), "-y": (Lm, Lm), "-z": (Lm, 0.0), "-x": (0.0, 2 * Lm), "+x": (2 * Lm, 2 * Lm)}
    # panel local coords (u, v) in [-h, h], mapped to the face's (e1, e2) directions; panel centre c
    items = []  # (layer, kind, data, label)

    def C(face, u, v):
        x0, y0 = P[face]
        return (x0 + h + u, y0 + h + v)

    # outlines and fold lines
    for f in P:
        x0, y0 = P[f]
        items.append(("OUTLINE", "rect", (x0, y0, Lm, Lm), None))
    folds = [((Lm, 3 * Lm), (2 * Lm, 3 * Lm)), ((Lm, 2 * Lm), (2 * Lm, 2 * Lm)), ((Lm, Lm), (2 * Lm, Lm)), ((Lm, 2 * Lm), (Lm, 3 * Lm)), ((2 * Lm, 2 * Lm), (2 * Lm, 3 * Lm))]
    for a, b in folds:
        items.append(("FOLD", "line", (a, b), None))
    # each plain face: (face, leg strip axis 'u' or 'v' (the strip's long direction), strip sign of hinge end)
    # panel u = face e1, v = face e2 with e1 = x for +-y faces drawn with x horizontal
    # strips: +y XP along z (v, hinge at +z), +z XN along y (v, hinge at +y), -y DP along x (u, hinge at
    # +x), -z DN along u (hinge at -x)
    strip = {"+y": "v", "-y": "u", "+z": "v", "-z": "u"}
    hinge_end = {"+y": 1, "-y": 1, "+z": 1, "-z": -1}
    leg_on = {"+y": "XP", "+z": "XN", "-y": "DP", "-z": "DN"}
    gd = groove_dims(g)
    free_masks = {}
    res = 0.05
    grid = np.arange(-h + res / 2, h, res)
    U, V = np.meshgrid(grid, grid, indexing="xy")
    for f in ("+y", "-y", "+z", "-z"):
        mask = np.ones_like(U, bool)
        # chamfer and edge band (0.5 mm chamfer -> flex bends there; keep 0.6 from edges)
        mask &= (np.abs(U) < h - 0.6) & (np.abs(V) < h - 0.6)
        # leg window
        w2 = g.leg_w / 2 + 0.15
        if strip[f] == "v":
            lo, hi = sorted((hinge_end[f] * h, hinge_end[f] * (h - g.leg_l - 0.15)))
            items.append(("WINDOW", "rect", (C(f, -w2, lo)[0], C(f, -w2, lo)[1], 2 * w2, hi - lo), f"leg window ({leg_on[f]})"))
            mask &= ~((np.abs(U) < w2 + 0.1) & (V > lo - 0.1) & (V < hi + 0.1))
        else:
            lo, hi = sorted((hinge_end[f] * h, hinge_end[f] * (h - g.leg_l - 0.15)))
            items.append(("WINDOW", "rect", (C(f, lo, -w2)[0], C(f, lo, -w2)[1], hi - lo, 2 * w2), f"leg window ({leg_on[f]})"))
            mask &= ~((np.abs(V) < w2 + 0.1) & (U > lo - 0.1) & (U < hi + 0.1))
        # bayonet slit arcs (through the flex) and notches
        s_in, s_out = gd["slit"]
        ch_in = gd["channel"][0]
        alpha = (g.hook_b / 2 + CLEAR) / g.r_g
        R_ = np.hypot(U, V)
        A_ = np.mod(np.arctan2(V, U), 2 * math.pi)
        for q in range(4):
            a0 = q * math.pi / 2 - alpha
            a1 = q * math.pi / 2 + math.pi / 4 + alpha
            pts = [C(f, s_out * math.cos(a), s_out * math.sin(a)) for a in np.linspace(a0, a1, 24)] + [C(f, s_in * math.cos(a), s_in * math.sin(a)) for a in np.linspace(a1, a0, 24)]
            items.append(("WINDOW", "poly", pts, None))
            pts = [C(f, s_out * math.cos(a), s_out * math.sin(a)) for a in np.linspace(q * math.pi / 2 - alpha, q * math.pi / 2 + alpha, 6)] + [C(f, ch_in * math.cos(a), ch_in * math.sin(a)) for a in np.linspace(q * math.pi / 2 + alpha, q * math.pi / 2 - alpha, 6)]
            items.append(("WINDOW", "poly", pts, None))
        mask &= ~((R_ > ch_in - 0.1) & (R_ < s_out + 0.1))
        # steel insert spots (flex covers them, keep-out)
        for su in (-1, 1):
            for sv in (-1, 1):
                pu, pv = su * g.ring_r / math.sqrt(2), sv * g.ring_r / math.sqrt(2)
                items.append(("KEEPOUT", "circle", (C(f, pu, pv), g.insert_d / 2), None))
                mask &= np.hypot(U - pu, V - pv) > g.insert_d / 2 + 0.1
        # corner spheres: window 0.4 mm (retains the sphere) and a contact ring pad
        for su in (-1, 1):
            for sv in (-1, 1):
                if strip[f] == "v":
                    pu, pv = su * g.a_x, sv * g.a_yz
                else:
                    pu, pv = su * g.a_x, sv * g.a_yz
                # on +-y faces, x is u; on +-z faces x is u too; the other coordinate is the face's e2
                items.append(("PAD", "ring", (C(f, pu, pv), 0.2, 0.55), "corner contact"))
                mask &= np.hypot(U - pu, V - pv) > 0.75
        # hook slots
        fit = hf["fit"]
        if fit:
            zb0, zb1 = fit["band"]
            u_, sl = fit["u"], fit["slot_len"]
            dropped_edge = {"+y": 1, "-y": -1}.get(f)
            for sv in (-1, 1):
                if dropped_edge is not None and sv == dropped_edge:
                    continue
                for su in (-1, 1):
                    x0_, x1_ = su * u_ - sl / 2 - CLEAR, su * u_ + sl / 2 + CLEAR
                    y0_, y1_ = (zb0, zb1 + CLEAR) if sv > 0 else (-zb1 - CLEAR, -zb0)
                    a = C(f, x0_, y0_)
                    items.append(("WINDOW", "rect", (a[0], a[1], x1_ - x0_, y1_ - y0_), None))
                    mask &= ~((U > x0_ - 0.1) & (U < x1_ + 0.1) & (V > y0_ - 0.1) & (V < y1_ + 0.1))
        # disc-pin sockets beside the strip
        r_p = 0.23 * g.L
        for s in (-1, 1):
            pu, pv = (s * r_p, 0.0) if strip[f] == "v" else (0.0, s * r_p)
            items.append(("WINDOW", "circle", (C(f, pu, pv), 0.25), None))
            mask &= np.hypot(U - pu, V - pv) > 0.45
        free_masks[f] = mask
    for f in ("+x", "-x"):
        mask = (np.abs(U) < h - 0.6) & (np.abs(V) < h - 0.6)
        items.append(("WINDOW", "circle", (C(f, 0, 0), g.r_d + 0.10), "disc window"))
        mask &= np.hypot(U, V) > g.r_d + 0.25
        # D-leg hinge knuckles cut the +x / -x panel edge next to +z / -z
        free_masks[f] = mask

    # component footprints (w, h mm) and how many
    wires = design_table["wires"]
    comps = [  # outer flex: the FETs next to the SMA anchors (the board carries MCU, IMU, charger, LDO, shift registers, tantalums)
        ("FET DMG1012T SOT-523", 1.6, 1.6, wires["wires"] + 6, 0.70),
    ]
    placed, unplaced = [], []

    def take(f, w, hh, target=None, ht=0.0):
        m = free_masks[f]
        if blocked is not None and ht > SKIN + 1e-6:
            lev = min(x for x in HT_LEVELS if x >= ht - 1e-9)
            m = m & ~blocked[(f, lev)]
            band = h - (ht + 0.15)  # a part standing in by ht must stay ht + 0.15 from the edge (neighbour face's parts)
            m = m & (np.abs(U) < band) & (np.abs(V) < band)
        nw, nh = int(math.ceil((w + 0.2) / res)), int(math.ceil((hh + 0.2) / res))
        # summed-area table for a free rectangle search
        S = np.pad(m.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        H, Wd = m.shape
        if nh > H or nw > Wd:
            return None
        tot = S[nh:, nw:] - S[:-nh, nw:] - S[nh:, :-nw] + S[:-nh, :-nw]
        ok = np.argwhere(tot == nw * nh)
        if len(ok) == 0:
            return None
        # prefer positions near the target (u, v), else near the panel centre
        if target is None:
            cen = np.array([H / 2 - nh / 2, Wd / 2 - nw / 2])
        else:
            cen = np.array([(target[1] + h) / res - nh / 2, (target[0] + h) / res - nw / 2])
        i, j = ok[np.argmin(((ok - cen) ** 2).sum(1))]
        free_masks[f][i:i + nh, j:j + nw] = False
        u0 = grid[j] - res / 2 + 0.1
        v0 = grid[i] - res / 2 + 0.1
        return u0, v0

    tab_report = []
    for (f, u, v, lab) in (tab_pads or []):
        ok = take(f, 1.2, 1.4, (u, v))
        if ok is None:
            unplaced.append(lab)
            continue
        a = C(f, ok[0], ok[1])
        items.append(("PAD", "rect", (a[0], a[1], 1.2, 1.4), lab))
        tab_report.append({"pad": lab, "face": f, "u": round(ok[0] + 0.6, 2), "v": round(ok[1] + 0.7, 2), "tab_u": round(u, 2), "tab_v": round(v, 2)})
    order = ["-z", "-y", "+z", "+y", "+x", "-x"]
    crimp = []
    n_cr = wires["crimps"]
    for w in wires_list:
        for k, (f, u, v, what) in enumerate(w["crimps"]):
            pos = take(f, 0.6, 0.8, (u, v))
            if pos is None and f[1] == "x":
                pos = None
            if pos is None:
                unplaced.append(f"crimp pad {w['id']}")
                continue
            a = C(f, pos[0], pos[1])
            items.append(("PAD", "rect", (a[0], a[1], 0.6, 0.8), w["id"] if k == 0 else None))
            crimp.append({"wire": w["id"], "face": f, "u": round(pos[0] + 0.3, 2), "v": round(pos[1] + 0.4, 2), "target_u": round(u, 2), "target_v": round(v, 2),
                          "offset_mm": round(math.hypot(pos[0] + 0.3 - u, pos[1] + 0.4 - v), 2), "what": what})
    comp3d = []
    for name, w, hh, cnt, ht in comps:
        for k in range(cnt):
            done = False
            for f in order:
                for (ww, hv) in ((w, hh), (hh, w)):
                    pos = take(f, ww, hv, None, ht)
                    if pos:
                        a = C(f, pos[0], pos[1])
                        items.append(("COMP", "rect", (a[0], a[1], ww, hv), name.split(" ")[0] if k == 0 else None))
                        placed.append((name, f, round(pos[0], 2), round(pos[1], 2)))
                        comp3d.append((name, f, pos[0], pos[1], ww, hv, ht))
                        done = True
                        break
                if done:
                    break
            if not done:
                unplaced.append(name)
    # IR LED + photodiode (0402: 1.0 x 0.5) as near each face centre as the face allows
    ir = []
    for f in P:
        for nm in ("IR LED 0402", "photodiode 0402"):
            pos = take(f, 1.0, 0.5, None, 0.35) if f in free_masks else None
            if pos:
                a = C(f, pos[0], pos[1])
                items.append(("COMP", "rect", (a[0], a[1], 1.0, 0.5), nm.split(" ")[0] if nm.startswith("IR") else "PD"))
                ir.append((nm, f, round(pos[0], 2), round(pos[1], 2), round(math.hypot(pos[0] + 0.5, pos[1] + 0.25), 2)))
                comp3d.append((nm, f, pos[0], pos[1], 1.0, 0.5, 0.35))
            else:
                unplaced.append(f"{nm} ({f})")
    # Schottky 0201 at the 8 corners: 2 per corner on the two panels' corner zones (16)
    sch = 0
    for f in ("+y", "-y", "+z", "-z"):
        for su in (-1, 1):
            for sv in (-1, 1):
                pu, pv = su * (g.a_x - 1.0), sv * (g.a_yz - 0.2)
                pos = take(f, 0.6, 0.3, (pu, pv), 0.30)
                if pos is None:
                    unplaced.append(f"Schottky 0201 ({f})")
                    continue
                a = C(f, pos[0], pos[1])
                items.append(("COMP", "rect", (a[0], a[1], 0.6, 0.3), "D" if sch == 0 else None))
                comp3d.append(("Schottky 0201", f, pos[0], pos[1], 0.6, 0.3, 0.30))
                sch += 1
    # SMA crimp pads: two per wire, 0.6 x 0.8, in rows near where the wires end (edges of the plain faces)
    # stator coils: 3 sector coils, 2 layers, on flaps folding in to the rotor (the rotor axis is the
    # worm's diagonal, see README); flap off the -x / +x panel's outer edge
    coils = []
    for f, side in (("+x", 1), ("-x", -1)):
        x0, y0 = P[f]
        cx = x0 + h
        cy = y0 + Lm + 4.7
        r_o = g.rotor_r
        items.append(("OUTLINE", "rect", (cx - 4.5, cy - 4.5, 9.0, 9.0), f"stator flap {f}"))
        items.append(("FOLD", "line", ((cx - 4.5, y0 + Lm), (cx + 4.5, y0 + Lm)), None))
        for j in range(3):
            a0, a1 = 2 * math.pi * j / 3 + 0.12, 2 * math.pi * (j + 1) / 3 - 0.12
            pts = [(cx + 0.3 * r_o * math.cos(a), cy + 0.3 * r_o * math.sin(a)) for a in np.linspace(a0, a1, 10)] + [(cx + r_o * math.cos(a), cy + r_o * math.sin(a)) for a in np.linspace(a1, a0, 20)]
            items.append(("COIL", "poly", pts, f"coil {j + 1}" if j == 0 else None))
        coils.append(f)
        items.append(("STIFF", "circle", ((cx, cy), r_o + 0.3), None))
        # bridge between the panel and the flap
        bx = x0 + (Lm if side > 0 else -0.2)
    # battery tab pads: where the cell's two nickel tabs come through the skin (placed before the rest)
    for f in ("+x", "-x"):
        x0, y0 = P[f]
        items.append(("STIFF", "rect", (x0 + 0.6, y0 + 0.6, Lm - 1.2, Lm - 1.2), "stiffener (end plate ring)" if f == "+x" else None))
    # power rails: VBAT ring around the four plain panels near the edges, VSW to crimp pads
    rails = []
    for f in ("+y", "+z", "-y", "-z"):
        x0, y0 = P[f]
        rails.append(((x0 + 0.35, y0 + 0.35), (x0 + Lm - 0.35, y0 + 0.35)))
        rails.append(((x0 + 0.35, y0 + Lm - 0.35), (x0 + Lm - 0.35, y0 + Lm - 0.35)))
    for f in ("+y", "+z", "-y", "-z"):
        x0, y0 = P[f]
        rails.append(((x0 + 0.35, y0 + 0.35), (x0 + 0.35, y0 + Lm - 0.35)))
        rails.append(((x0 + Lm - 0.35, y0 + 0.35), (x0 + Lm - 0.35, y0 + Lm - 0.35)))
    for a, b in rails:
        items.append(("RAIL", "line", (a, b), None))

    # write SVG and DXF
    W_, H_ = 3 * Lm + 10, 4 * Lm + 2
    colours = {"OUTLINE": "#000", "FOLD": "#888", "WINDOW": "#c00", "KEEPOUT": "#c80", "PAD": "#b8860b", "COMP": "#06c", "COIL": "#a0a", "STIFF": "#0a0", "RAIL": "#e60"}
    dash = {"FOLD": ' stroke-dasharray="0.6,0.4"', "STIFF": ' stroke-dasharray="0.3,0.3"'}
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="-5 -1 {W_} {H_}" width="{W_ * 20}" height="{H_ * 20}">',
           '<rect x="-5" y="-1" width="100%" height="100%" fill="#fff"/>',
           f'<g transform="translate(0,{4 * Lm}) scale(1,-1)" fill="none" stroke-width="0.04">']
    texts = []
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    for lay in colours:
        doc.layers.add(lay)
    for lay, kind, data, label in items:
        col = colours[lay]
        st = f'stroke="{col}"{dash.get(lay, "")}'
        if kind == "rect":
            x, y, w, hh = data
            svg.append(f'<rect x="{x:.3f}" y="{y:.3f}" width="{w:.3f}" height="{hh:.3f}" {st}/>')
            msp.add_lwpolyline([(x, y), (x + w, y), (x + w, y + hh), (x, y + hh)], close=True, dxfattribs={"layer": lay})
            anchor = (x + w / 2, y + hh / 2)
        elif kind == "line":
            (a, b) = data
            svg.append(f'<line x1="{a[0]:.3f}" y1="{a[1]:.3f}" x2="{b[0]:.3f}" y2="{b[1]:.3f}" {st}/>')
            msp.add_line(a, b, dxfattribs={"layer": lay})
            anchor = a
        elif kind == "poly":
            pts = " ".join(f"{p[0]:.3f},{p[1]:.3f}" for p in data)
            svg.append(f'<polygon points="{pts}" {st}/>')
            msp.add_lwpolyline(data, close=True, dxfattribs={"layer": lay})
            anchor = data[0]
        elif kind == "circle":
            (c, r) = data
            svg.append(f'<circle cx="{c[0]:.3f}" cy="{c[1]:.3f}" r="{r:.3f}" {st}/>')
            msp.add_circle(c, r, dxfattribs={"layer": lay})
            anchor = c
        elif kind == "ring":
            (c, r0, r1) = data
            svg.append(f'<circle cx="{c[0]:.3f}" cy="{c[1]:.3f}" r="{r1:.3f}" {st}/><circle cx="{c[0]:.3f}" cy="{c[1]:.3f}" r="{r0:.3f}" stroke="#c00"/>')
            msp.add_circle(c, r1, dxfattribs={"layer": lay})
            msp.add_circle(c, r0, dxfattribs={"layer": "WINDOW"})
            anchor = c
        if label:
            texts.append((anchor, label, col))
            msp.add_text(label, dxfattribs={"layer": lay, "height": 0.25}).set_placement(anchor)
    svg.append("</g>")
    for (x, y), label, col in texts:
        svg.append(f'<text x="{x:.2f}" y="{4 * Lm - y:.2f}" font-size="0.32" font-family="sans-serif" fill="{col}">{label}</text>')
    for f, (x0, y0) in P.items():
        svg.append(f'<text x="{x0 + 0.3:.2f}" y="{4 * Lm - y0 - Lm + 0.7:.2f}" font-size="0.6" font-family="sans-serif" fill="#000">{f} face</text>')
        msp.add_text(f"{f} face", dxfattribs={"layer": "OUTLINE", "height": 0.5}).set_placement((x0 + 0.3, y0 + Lm - 0.8))
    legend = ["black: panel outline (14 mm, 0.11 mm 2-layer flex)", "grey dashed: fold lines", "red: windows (legs, bayonet slits and notches, hook slots, sockets, disc)",
              "orange: insert keep-outs", "gold: pads (corner contacts, SMA crimps, battery)", "blue: footprints", "violet: stator sector coils (2 layers)", "green dashed: stiffeners", "orange-red: VBAT rails (routed); other nets unrouted"]
    for k, line in enumerate(legend):
        svg.append(f'<text x="{2 * Lm + 2:.2f}" y="{4 * Lm - Lm * 1.0 + 0.6 * k - 6:.2f}" font-size="0.42" font-family="sans-serif" fill="#000">{line}</text>')
    svg.append("</svg>")
    out.mkdir(parents=True, exist_ok=True)
    (out / "flex_net.svg").write_text("\n".join(svg))
    doc.saveas(str(out / "flex_net.dxf"))
    cnt = {}
    for name, f, *_ in placed:
        cnt.setdefault(name, []).append(f)
    largest = {}
    for f, m in free_masks.items():
        largest[f] = {"flat": round(float(m.sum()) * res * res, 1)}
        if blocked is not None:
            for ht in HT_LEVELS:
                largest[f][str(ht)] = round(float((m & ~blocked[(f, ht)]).sum()) * res * res, 1)
    return {"placed": {k: len(v) for k, v in cnt.items()}, "placed_faces": {k: sorted(set(v)) for k, v in cnt.items()}, "unplaced": unplaced, "ir": ir,
            "crimp_pads": crimp, "crimp_needed": n_cr, "battery_tab_pads": tab_report, "components_3d": comp3d, "schottky": sch, "coils": coils, "free_area_left_mm2": largest}


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------


def component_solids(g: Geo, comp3d):
    """Envelopes of the flex components inside the cube: each sits on the flex's inner side in a
    pocket through the 0.3 mm skin and stands into the cavity by (height - skin)."""
    out = {}
    for k, (name, f, u0, v0, w, hh, ht) in enumerate(comp3d):
        if ht <= SKIN + 1e-6:
            continue  # sits within its skin pocket
        if f in ("+y", "-y", "+z", "-z"):
            s_ = 1 if f[0] == "+" else -1
            n0, n1 = sorted((s_ * (g.Ro - ht), s_ * g.Ri))  # the part in the skin pocket is not an overlap
            if f[1] == "y":
                b = box((u0, n0, v0), (u0 + w, n1, v0 + hh))
            else:
                b = box((u0, v0, n0), (u0 + w, v0 + hh, n1))
        else:
            s_ = 1 if f[0] == "+" else -1
            n0, n1 = sorted((s_ * (g.Ro - ht), s_ * g.Ri))
            b = box((n0, u0, v0), (n1, u0 + w, v0 + hh))
        out[f"comp_{k}_{name.split(' ')[0]}"] = b
    return out


BOARD_T = 0.20  # rigid board (FR4 0.2 mm, 2 layers)
TAIL_W, TAIL_T = 2.0, 0.12  # flex tails from the board to the outer flex (40 lines at 0.2 mm pitch on 2 layers over two tails)
BOARD_COMPS = [  # name, w, h, count, height (mm)
    ("MCU CH32V003F4U6 QFN-20 3x3", 3.0, 3.0, 1, 0.75),
    ("IMU LIS2DW12 LGA-12 2x2", 2.0, 2.0, 1, 0.70),
    ("charger TP4054 SOT-23-5", 3.0, 3.0, 1, 1.10),
    ("shift register 74HC595BQ DHVQFN-16 2.5x3.5", 2.5, 3.5, 4, 0.85),
    ("tantalum 1206", 3.2, 1.6, 6, 1.10),
    ("LDO XC6206 SOT-23", 3.0, 3.0, 1, 1.10),
]


def _slab_raster(shapes, ax, lo, hi, h, res, clear):
    """Raster (rows = second in-plane axis, cols = first) of everything between lo and hi along ax."""
    from PIL import Image, ImageDraw
    au, av = [i for i in range(3) if i != ax]
    grid = np.arange(-h + res / 2, h, res)
    N = len(grid)
    blo, bhi = [-h - 1] * 3, [h + 1] * 3
    blo[ax], bhi[ax] = lo, hi
    slab = box(tuple(blo), tuple(bhi))
    sbb = slab.BoundingBox()
    im = Image.new("1", (N, N), 0)
    dr = ImageDraw.Draw(im)
    for name, sh in shapes.items():
        if not _bb_overlap(sh.BoundingBox(), sbb):
            continue
        try:
            pc = sh.copy().intersect(slab)
            if pc.Volume() < 1e-7:
                continue
            vs, tris = pc.tessellate(0.02, 0.3)
        except Exception:  # noqa: BLE001
            continue
        V = np.array([[v.x, v.y, v.z] for v in vs])
        U = (V[:, au] + h) / res - 0.5
        W = (V[:, av] + h) / res - 0.5
        for t in tris:
            dr.polygon([(U[i], W[i]) for i in t], fill=1, outline=1)
    m = np.array(im, bool)
    k = int(math.ceil(clear / res))
    mm = m.copy()
    for di in range(-k, k + 1):
        for dj in range(-k, k + 1):
            mm |= np.roll(np.roll(m, di, 0), dj, 1)
    return mm, grid


def inner_board(g: Geo, shapes, res=0.05):
    """Rigid board in the layer between the -z capstans (DP, DN ring tops) and the joint's gear
    layer under the off-axis cell: outline = the free region of that layer containing the cube axis;
    the board's parts on its upper side, each where nothing stands within its height; two flex tails
    from the board's edge down through the -z skin to the outer flex. Returns solids, a placement
    report and the tail pads."""
    from scipy import ndimage
    za = capstan_slots(g)["DN"][0][2]
    z0 = za + g.cap_face + CLEAR
    z1 = z0 + BOARD_T
    h = g.h
    free_b, grid = _slab_raster(shapes, 2, z0, z1, h, res, CLEAR)
    free_b = ~free_b & (np.abs(grid)[None, :] < g.Ri - 0.4) & (np.abs(grid)[:, None] < g.Ri - 0.4)
    lab, _ = ndimage.label(free_b)
    c = len(grid) // 2
    lab0 = lab[c, c]
    board_m = (lab == lab0) if lab0 > 0 else np.zeros_like(free_b)
    board_m = ndimage.binary_opening(board_m, iterations=4)  # no slivers under 0.4 mm
    lab, _ = ndimage.label(board_m)
    board_m = lab == lab[c, c] if lab[c, c] > 0 else board_m
    # board solid from 0.25 mm cells
    cell = 5
    boxes = []
    for i0 in range(0, len(grid) - cell, cell):
        row = board_m[i0:i0 + cell].all(0)
        cols = row.reshape(-1)[: (len(grid) // cell) * cell].reshape(-1, cell).all(1)
        j = 0
        while j < len(cols):
            if cols[j]:
                j1 = j
                while j1 + 1 < len(cols) and cols[j1 + 1]:
                    j1 += 1
                x0, x1 = grid[j * cell] - res / 2, grid[(j1 + 1) * cell - 1] + res / 2
                y0, y1 = grid[i0] - res / 2, grid[i0 + cell - 1] + res / 2
                boxes.append(box((x0, y0, z0), (x1, y1, z1)))
                j = j1 + 1
            else:
                j += 1
    solids = {"board": fuse(boxes).clean()} if boxes else {}
    # the board as built (cells), for the placement mask
    built = np.zeros_like(board_m)
    for b in boxes:
        bb = b.BoundingBox()
        jj = (grid > bb.xmin) & (grid < bb.xmax)
        ii = (grid > bb.ymin) & (grid < bb.ymax)
        built[np.ix_(ii, jj)] = True
    occ = built.copy()
    rasters = {}
    placed, unplaced = [], []
    for name, w, hh, cnt, ht in BOARD_COMPS:
        if ht not in rasters:
            m_, _ = _slab_raster(shapes, 2, z1, z1 + ht + CLEAR, h, res, CLEAR)
            rasters[ht] = ~m_
        for k in range(cnt):
            done = False
            for ww, hv in ((w, hh), (hh, w)):
                m = occ & rasters[ht]
                nw, nh = int(math.ceil((ww + 0.2) / res)), int(math.ceil((hv + 0.2) / res))
                S = np.pad(m.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
                tot = S[nh:, nw:] - S[:-nh, nw:] - S[nh:, :-nw] + S[:-nh, :-nw]
                ok = np.argwhere(tot == nw * nh)
                if len(ok) == 0:
                    continue
                cen = np.array([c - nh / 2, c - nw / 2])
                i, j = ok[np.argmin(((ok - cen) ** 2).sum(1))]
                occ[i:i + nh, j:j + nw] = False
                x0, y0 = grid[j] - res / 2 + 0.1, grid[i] - res / 2 + 0.1
                solids[f"bcomp_{len(placed)}_{name.split(' ')[0]}"] = box((x0, y0, z1), (x0 + ww, y0 + hv, z1 + ht))
                placed.append({"part": name, "x": round(x0, 2), "y": round(y0, 2), "w": ww, "h": hv, "ht": ht})
                done = True
                break
            if not done:
                unplaced.append(name)
    # tails: vertical strips (width along x, thickness along y) from the board down through the -z skin
    tails, pads = [], []
    tried = []
    for yt in (4.1, -4.1, 4.4, -4.4, 3.8, -3.8, 4.7, -4.7, 5.0, -5.0, 2.0, -2.0, 0.0):
        for xc in (3.3, -3.3, 3.6, -3.6, 3.0, -3.0, 2.6, -2.6, 0.0):
            if len(tails) == 2:
                break
            if any(abs(xc - t[0]) < TAIL_W + 0.3 and abs(yt - t[1]) < 0.5 for t in tails):
                continue
            jj = (grid > xc - TAIL_W / 2) & (grid < xc + TAIL_W / 2)
            ii = np.abs(grid - yt) < 0.3
            if not built[np.ix_(ii, jj)].all():
                continue
            strip = box((xc - TAIL_W / 2, yt - TAIL_T / 2, -g.Ri), (xc + TAIL_W / 2, yt + TAIL_T / 2, z0))
            grown = box((xc - TAIL_W / 2 - 0.05, yt - TAIL_T / 2 - 0.05, -g.Ri), (xc + TAIL_W / 2 + 0.05, yt + TAIL_T / 2 + 0.05, z0 - 0.001))
            gb = grown.BoundingBox()
            clash = None
            for n_, sh in shapes.items():
                if n_ in solids or not _bb_overlap(gb, sh.BoundingBox()):
                    continue
                if grown.intersect(sh.copy()).Volume() > 1e-6:
                    clash = n_
                    break
            tried.append((xc, yt, clash))
            if clash is None:
                tails.append((xc, yt))
                solids[f"tail_{len(tails)}"] = strip
                pads.append(("-z", xc, yt, f"board tail {len(tails)}"))
    rep = {"z": [round(z0, 3), round(z1, 3)], "area_mm2": round(float(built.sum()) * res * res, 1),
           "bbox": [round(float(grid[built.any(0)].min()), 2), round(float(grid[built.any(0)].max()), 2), round(float(grid[built.any(1)].min()), 2), round(float(grid[built.any(1)].max()), 2)] if built.any() else None,
           "placed": placed, "unplaced": unplaced, "tails": tails, "tail_tries": tried[:20], "head_room": {str(k): round(float(v.sum()) * res * res, 1) for k, v in rasters.items()}}
    return solids, rep, pads


def battery_tabs(g: Geo, parts):
    """The cell's two nickel tabs (1.0 x 0.1 mm) leave its top (3 x 10 face, z = cell top), run
    sideways along x just above it, then up through the +z skin to the flex beside the XN leg
    strip. First pair of candidate positions that touch nothing (0.05 mm gap) is taken."""
    bt, bw, bl = g.batt
    zt = g.batt_z0 + bl
    w, t = 1.0, 0.10
    out = {}
    cands = []
    for xe in (4.0, 4.5, 5.0, 5.5, 6.0, -4.0, -4.5, -5.0, -5.5, -6.0):
        for yc in (0.0, 1.2, -1.2, 2.4, -2.4, 3.6, -3.6):
            sx = 1 if xe > 0 else -1
            hor = box((min(0.0, xe), yc - w / 2, zt + 0.05), (max(0.0, xe) + sx * t / 2 * 0, yc + w / 2, zt + 0.05 + t))
            ver = box((xe - t / 2, yc - w / 2, zt + 0.05), (xe + t / 2, yc + w / 2, g.Ri))
            sh = hor.fuse(ver).clean()
            grown = box((min(0.0, xe) - 0.05, yc - w / 2 - 0.05, zt + 0.0), (max(0.0, xe) + 0.05, yc + w / 2 + 0.05, zt + 0.2)).fuse(
                box((xe - t / 2 - 0.05, yc - w / 2 - 0.05, zt), (xe + t / 2 + 0.05, yc + w / 2 + 0.05, g.Ri)))
            bb = grown.BoundingBox()
            clash = None
            for n, (s_, k) in parts.items():
                if n == "battery" or k == "component" or not _bb_overlap(bb, s_.BoundingBox()):
                    continue
                if grown.intersect(s_.copy()).Volume() > 1e-6:
                    clash = n
                    break
            if clash is None:
                cands.append((xe, yc, sh))
    chosen = []
    for c in cands:
        if all(abs(c[1] - o[1]) >= 1.4 or abs(c[0] - o[0]) >= 1.4 for o in chosen) and (not chosen or c[0] == chosen[0][0]):
            chosen.append(c)
        if len(chosen) == 2:
            break
    pads = []
    for lab, (xe, yc, sh) in zip(("BAT+", "BAT-"), chosen):
        out[f"batt_tab_{'p' if lab == 'BAT+' else 'n'}"] = sh
        pads.append(("+z", xe, yc, lab))
    return out, pads, [(c[0], c[1]) for c in cands]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, default=None, help="cube side, mm (default: chosen.json)")
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--design", default=str(CHOSEN), help="design json (default: chosen.json)")
    ap.add_argument("--out", default=str(OUT), help="output directory")
    ap.add_argument("--set", action="append", default=[], help="design override key=value (SI / fractions, as chosen.json)")
    a = ap.parse_args(argv)
    ch = load_design(a.design)
    out = Path(a.out)
    d = ch["design"]
    overrides = {}
    for kv in a.set:
        k_, v_ = kv.split("=", 1)
        overrides[k_] = float(v_)
        d[k_] = float(v_)
    L = a.L if a.L else d["L"] * 1e3
    g = Geo(L, d)
    hf = hook_fit(g)
    parts, sweeps, layout = bought_and_moving(g, hf)
    for n, (s, _) in parts.items():
        if not s.isValid():
            print("invalid solid:", n)
    wires = wire_map(g, layout["routes"], d)
    channels = sorted({w["channel"] for w in wires})
    if len(channels) != ch["table"]["wires"]["wires"]:
        print("SMA channel count differs from the design table:", len(channels), ch["table"]["wires"]["wires"])
    printed_parts = dict(parts)
    for n, sh in wire_solids(layout["routes"]).items():
        kind = "printed" if n.startswith(("lever_", "bracket_")) else "wire"
        parts[n] = (sh, kind)
        if n.startswith(("lever_", "bracket_")):
            printed_parts[n] = (sh, "printed")
    tabs, tab_pads, tab_cands = battery_tabs(g, parts)
    for n, sh in tabs.items():
        parts[n] = (sh, "bought")
    shapes = {n: s_ for n, (s_, k) in parts.items()}
    for n, states in sweeps.items():
        for k_, st in enumerate(states[1:]):
            shapes[f"sweep_{n}_{k_}"] = st
    board, board_rep, tail_pads = inner_board(g, shapes)
    for n, sh in board.items():
        kind = "component" if n.startswith("bcomp_") else ("printed" if n == "board" else "bought")
        parts[n] = (sh, kind)
        shapes[n] = sh
    blocked = interior_blocked(g, shapes)
    flex = flex_net(g, hf, ch["table"], wires, out, blocked=blocked, tab_pads=tab_pads + tail_pads)
    for n, sh in component_solids(g, flex["components_3d"]).items():
        parts[n] = (sh, "component")
    static, swept, outside, meshing, flexc = overlaps(parts, sweeps, g)
    report = {
        "design_file": str(Path(a.design).resolve().relative_to(ROOT)) if Path(a.design).resolve().is_relative_to(ROOT) else a.design,
        "L_mm": L,
        "hook_fit": hf,
        "layout": layout,
        "fit": fit_report(g, hf, layout),
        "overlaps_static": static,
        "overlaps_swept": swept,
        "intended_contacts": meshing,
        "battery_tab_candidates": tab_cands,
        "board": board_rep,
        "design_overrides": overrides,
        "outside_envelope": outside,
        "features": feature_table(g, hf, ch["table"]),
        "wires": wires,
        "flex": {k: v for k, v in flex.items() if k != "components_3d"},
        "volumes_mm3": {n: round(s.Volume(), 3) for n, (s, _) in parts.items()},
        "kinds": {n: k for n, (_, k) in parts.items()},
    }
    if not a.no_export:
        report["written"] = export(printed_parts, g, pawl(g), out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "check.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: report[k] for k in ("fit", "hook_fit")}, indent=1, default=str))
    print(f"static overlaps > 0.01 mm3: {len(static)}, total {sum(x[2] for x in static):.2f} mm3")
    print(f"swept overlaps: {len(swept)}")
    print("outside envelope:", outside)
    print(json.dumps(report["flex"], indent=1, default=str)[:3000])
    return report


if __name__ == "__main__":
    main()
