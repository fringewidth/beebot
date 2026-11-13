"""One complete 50 mm cube with a turning face: every part, interference checks, mass and cost.

Builds on cube_turning_face.py (the first body and disc; its sizes for the recess, the 6803 bearing
and the detent balls are kept). The cube spans x, y = -25..25, z = 0..50; the turning face is +Z.

Run:  .venv/bin/python hardware/rotating_cube/cube_full.py
Writes cube_full_<part>.step and .stl for every part, cube_full_assembly.step, prints the
interference table, the mass and inertia (copied into src/beebot/hardware/cube_turning.py) and the
cost table. render_exploded.py renders the assembled and exploded views from the same parts.

Sizes marked BOUGHT are the usual catalogue sizes of off-the-shelf parts, written from memory and
not checked against a datasheet. Everything marked GUESS is a choice of this file.

Decisions (argued in runs/handoff/rotating-cube-poc.md):

1. The turning-face gearmotor sits in the body, on the disc's axis, and drives the disc's hub by
   its D shaft. The motor's wires never cross the bearing and its reaction goes into the body.
1. The disc's magnets are electropermanent and need a current pulse only when they switch, which
   happens at rest. Two spring pins in the body ride on two copper tracks under the disc (a face
   slip ring); nothing else crosses the bearing.
1. Magnets: 48 identical electropermanent units. 24 lie in the edges, two an edge, each serving
   the two faces that meet there (as cube_swarm's edge magnets do); 4 more on every face on a
   12 mm circle, which on the turning face ride on the disc. A face pattern has to map onto itself
   under quarter turns and under mirroring, or two faces do not mate; three units of three faces do
   not fit in a corner, which is why the outer ones are in the edges.
1. The disc stands 0.3 mm proud of the face so that a bonded neighbour rides on it and not on the
   ring of the body around it.
1. Edge hinge: an N20 gearmotor and, on each of the 12 edges, a 3 mm axle with two latch hooks.
   The train from the motor to an axle and the way a hook takes hold of the neighbour's axle are
   NOT designed: HARDWARE.md says so of the hinge cube too, and nothing found here fits. The parts
   are stand-ins that reserve the volume and the mass.
"""

import itertools
import math
import sys
from pathlib import Path

import cadquery as cq

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import cube_turning_face as first  # noqa: E402

SIDE = first.SIDE
H = SIDE / 2
TOP = SIDE
CHAMFER = 1.5  # all 12 edges (cube_hinge's result rests on chamfered edges) GUESS
WALL = 2.0  # GUESS
PROUD = 0.3  # the disc above the top face GUESS
CLEAR = first.CLEAR

# Turning face
RECESS_D = first.RECESS_D  # 36
DISC_D = first.DISC_D  # 35.6
DISC_T = 9.5  # plate, 0.3 of it proud GUESS
DISC_Z0 = TOP + PROUD - DISC_T  # 40.8: underside of the plate
FLOOR_Z = DISC_Z0 - CLEAR  # 40.6: recess floor
BEARING_ID, BEARING_OD, BEARING_W = first.BEARING_ID, first.BEARING_OD, first.BEARING_W  # 6803: 17 x 26 x 5 BOUGHT
SEAT_Z = FLOOR_Z - BEARING_W  # 35.6
BLOCK_Z = SEAT_Z - 2.0  # 33.6: underside of the solid block that carries the bearing
BORE_D = 20.0  # through hole under the bearing
HUB_D, HUB_L = BEARING_ID, BEARING_W + CLEAR
RETAINER_D, RETAINER_T = 19.6, 1.0  # washer screwed under the hub: holds the disc in the bearing GUESS
BALL_D, BALL_PROTRUDE, DIMPLE_R = first.BALL_D, first.BALL_PROTRUDE, first.DIMPLE_R
BALL_Z = 43.1
BALL_R = DISC_D / 2 - BALL_D / 2 + BALL_PROTRUDE  # 16.8: ball centre when seated
SPRING_D, SPRING_L = 2.8, 5.0  # GUESS: 1 N at this length
TRACK_ID, TRACK_OD, TRACK_T = 27.0, 35.0, 0.6  # ring board under the disc, two copper tracks GUESS
PIN_D, PIN_L, PIN_R = 1.5, 5.6, (14.5, 16.5)  # spring pins, one per track GUESS (0.9-1.5 mm pogo pins are BOUGHT sizes)

# N20 gearmotor: gearbox 12 x 10 x 9, motor 12 x 10 x 15, D shaft 3 x 10. BOUGHT
N20_W, N20_H, N20_GEARBOX, N20_MOTOR, N20_SHAFT_D, N20_SHAFT_L = 12.0, 10.0, 9.0, 15.0, 3.0, 10.0
TURN_MOTOR_Z = 7.0  # its tail; gearbox ends at 31, shaft at 41 inside the hub
EDGE_MOTOR_X, EDGE_MOTOR_Z = -12.5, 13.0  # centre of the edge-hinge motor's section; its axis is along y GUESS

# Electropermanent unit: an AlNiCo 5 rod and an N42 rod, 4 dia x 4, side by side between two steel
# pole plates 1.5 thick, in a coil. Envelope 10 x 7 x 6 (edge units 10 x 6 x 6). GUESS, sized from
# Knaian's Robot Pebbles unit (3.2 mm rods, 4.4 N) by rod area: 7 N a unit.
EPM_L, EPM_W, EPM_D = 10.0, 7.0, 6.0
INNER_R = 12.0  # the inner four's circle; cube_turning.Face.radius = 0.24 L
EDGE_UNIT_AT = 12.5  # along the edge from its middle; cube.magnet_sites puts the edge magnets at 0.25 L
EDGE_UNIT_IN = 18.5  # the unit's section runs from here to 24.5 along both face normals

# Edge hinge stand-ins
AXLE_D, AXLE_L, AXLE_IN = 3.0, 12.0, 3.5  # steel pin in the middle of every edge, this far inside both faces GUESS
HOOK = (2.0, 3.0, 6.0)  # latch hook: thickness along the axle, width, length from the axle GUESS
HOOK_AT = 4.5  # along the axle from its middle

BATTERY = (5.0, 20.0, 25.0)  # 1S LiPo 502025, about 200 mAh BOUGHT
BATTERY_AT = (10.5, 0.0, 19.5)
BOARD = (21.0, 4.6, 24.0)  # 1.6 mm board with 3 mm of parts on it GUESS
BOARD_AT = (5.5, 15.3, 20.0)

DISC_ANGLES = (0.0, 22.5, 45.0, 67.5, 90.0)
HOOK_ANGLES = (0.0, 45.0, 90.0)

# Masses in grams. measured: none. "volume" rows are CAD volume x density x fill.
RHO_PRINT = 1.27e-3  # PETG, g/mm3
FILL_BODY, FILL_DISC = 0.7, 0.8  # GUESS: walls print solid, the top block at about 40 percent
MASS = {
    "bearing": 7.0,  # 6803ZZ BOUGHT (a solid steel ring of its size would be 11.9)
    "turn_motor": 9.5,  # N20 BOUGHT
    "edge_motor": 9.5,
    "epm_unit": 2.0,  # GUESS: rods 0.74, poles 0.85, copper 0.4
    "ball": 0.11,
    "spring": 0.1,
    "battery": 5.0,  # BOUGHT
    "board": 7.0,  # GUESS: board 1.6, controller, 18 pulse channels, two capacitors
    "axle": 0.66,  # steel, 3 x 12
    "hook": 0.25,
    "retainer": 0.5,
    "track": 0.6,
    "pin": 0.1,
    "duct": 1.0,  # the wires inside it
}
NOT_DRAWN = {"wiring, connectors": 4.0, "screws, motor and battery mounts": 6.0, "edge drive train (not designed)": 10.0}  # GUESS

COST = [  # USD for one cube in small numbers. All GUESS, from memory of hobby prices.
    ("2 N20 gearmotors with encoders", 16.0),
    ("6803 bearing", 2.0),
    ("48 electropermanent units (96 rods, 96 pole plates, winding)", 60.0),
    ("18 pulse channels (H bridges), 2 capacitors", 14.0),
    ("controller with radio, IMU, charger", 8.0),
    ("1S LiPo 502025", 4.0),
    ("printed body and disc", 3.0),
    ("ring board, 2 spring pins", 3.0),
    ("balls, springs, 12 axles, 24 hooks, screws", 6.0),
    ("edge drive train (not designed)", 10.0),
]


def box(size, at):
    return cq.Workplane("XY").box(*size).translate(at)


def cyl(d, h, x=0.0, y=0.0, z=0.0):
    return first.cyl(d, h, x, y, z)


def turned(shape, degrees, axis=(0, 0, 1), through=(0, 0, 0)):
    end = tuple(p + a for p, a in zip(through, axis))
    return shape.rotate(through, end, degrees) if degrees else shape


def union(shapes):
    out = shapes[0]
    for s in shapes[1:]:
        out = out.union(s)
    return out


def outline():
    return cq.Workplane("XY").box(SIDE, SIDE, SIDE, centered=(True, True, False)).edges().chamfer(CHAMFER)


FACES = [((1, 0, 0), (0, 1, 0), (0, 0, 1)), ((-1, 0, 0), (0, 0, 1), (0, 1, 0)), ((0, 1, 0), (0, 0, 1), (1, 0, 0)), ((0, -1, 0), (1, 0, 0), (0, 0, 1)), ((0, 0, 1), (1, 0, 0), (0, 1, 0)), ((0, 0, -1), (0, 1, 0), (1, 0, 0))]  # normal, u, v
CENTRE = (0.0, 0.0, H)


def on_face(shape, face, proud=0.0):
    """A shape drawn on the +Z face about the origin (z = 0 the face, -z inward) moved to `face`."""
    n, u, v = face
    m = cq.Matrix([[u[0], v[0], n[0], CENTRE[0] + n[0] * (H + proud)], [u[1], v[1], n[1], CENTRE[1] + n[1] * (H + proud)], [u[2], v[2], n[2], CENTRE[2] + n[2] * (H + proud)]])
    return cq.Workplane(obj=shape.val().transformGeometry(m))


def inner_units(face, proud=0.0):
    """The four units of one face on the inner circle, long side along the circle."""
    units = []
    for k in range(4):
        angle = 45 + 90 * k
        unit = box((EPM_W, EPM_L, EPM_D), (INNER_R, 0, -EPM_D / 2)).rotate((0, 0, 0), (0, 0, 1), angle)
        units.append(on_face(unit, face, proud))
    return units


def edges():
    """The 12 edges as (direction axis 0-2, sign a, sign b) with a, b on the other two axes."""
    return [(axis, a, b) for axis in range(3) for a in (-1, 1) for b in (-1, 1)]


def along_edge(edge, s, inward):
    """The point `s` along an edge from its middle and `inward` inside both faces."""
    axis, a, b = edge
    u, v = [k for k in range(3) if k != axis]
    p = [0.0, 0.0, H]
    p[axis] += s
    p[u] += a * (H - inward)
    p[v] += b * (H - inward)
    return tuple(p)


def edge_units():
    """Two units an edge: 10 along it, 6 x 6 across, flush with both faces, clipped by the chamfer."""
    hull = outline()
    section = H - EDGE_UNIT_IN
    units = []
    for edge in edges():
        for s in (-EDGE_UNIT_AT, EDGE_UNIT_AT):
            size = [section] * 3
            size[edge[0]] = EPM_L
            units.append(box(size, along_edge(edge, s, section / 2)).intersect(hull))
    return units


def axles():
    out = []
    for edge in edges():
        direction = [0, 0, 0]
        direction[edge[0]] = 1
        pin = cq.Workplane("XY").circle(AXLE_D / 2).extrude(AXLE_L).translate((0, 0, -AXLE_L / 2))
        if edge[0] == 0:
            pin = pin.rotate((0, 0, 0), (0, 1, 0), 90)
        elif edge[0] == 1:
            pin = pin.rotate((0, 0, 0), (1, 0, 0), 90)
        out.append(pin.translate(along_edge(edge, 0.0, AXLE_IN)))
    return out


def hooks(angle=0.0):
    """Two hooks an axle, each a bar with a bore for the axle. At 0 they lie inside the cube: on the
    four edges of the turning face they point straight down beside the recess, elsewhere from the
    axle toward the cube's centre line. `angle` swings them about the axle; on the turning face's
    edges the swing is outward (at 90 the tip is outside the side face)."""
    pins = axles()
    out = []
    for index, edge in enumerate(edges()):
        axis, a, b = edge
        u, v = [k for k in range(3) if k != axis]
        spin = [0, 0, 0]
        spin[axis] = 1
        handed = -1 if axis == 1 else 1  # a turn about +y takes z to x
        top = axis != 2 and b > 0
        rest = (-90 if top else math.degrees(math.atan2(-b, -a))) * handed
        swing = angle * (a * handed if top else 1)
        for s in (-HOOK_AT, HOOK_AT):
            size, offset = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
            size[axis], size[u], size[v] = HOOK[0], HOOK[2], HOOK[1]
            offset[u] = HOOK[2] / 2 - HOOK[1] / 2
            pivot = along_edge(edge, s, AXLE_IN)
            hook = box(size, tuple(p + o for p, o in zip(pivot, offset))).cut(pins[index])
            out.append(turned(hook, rest + swing, tuple(spin), pivot))
    return out


def n20(axis):
    """Gearmotor with its tail at the origin: along +z with its 12 mm side along x ("z"), or along
    +y with its 12 mm side along z ("y")."""
    length = N20_MOTOR + N20_GEARBOX
    motor = box((N20_W, N20_H, length), (0, 0, length / 2)).intersect(cyl(N20_W + 1.0, length)).union(cyl(N20_SHAFT_D, N20_SHAFT_L, z=length))
    if axis == "y":
        motor = motor.rotate((0, 0, 0), (0, 0, 1), 90).rotate((0, 0, 0), (1, 0, 0), -90)
    return motor


def build(disc_angle=0.0, hook_angle=0.0):
    """Every part as {name: (workplane, group)}; group "disc" turns with the disc."""
    fixed_faces = [f for f in FACES if f[0] != (0, 0, 1)]
    top = FACES[4]
    detent = abs((disc_angle + 45) % 90 - 45) < 1e-6  # the balls sit in their dimples
    ball_r = BALL_R if detent else BALL_R - BALL_PROTRUDE  # else the springs are pushed in by the wall

    inner_fixed = [u for f in fixed_faces for u in inner_units(f)]
    inner_disc = inner_units(top, PROUD)
    edge_magnets = edge_units()
    pins = axles()
    latch = hooks(hook_angle)
    swept = [h for a in HOOK_ANGLES for h in hooks(a)]

    turn_motor = n20("z").translate((0, 0, TURN_MOTOR_Z))
    edge_motor = n20("y").translate((EDGE_MOTOR_X, -(N20_MOTOR + N20_GEARBOX) / 2, EDGE_MOTOR_Z))
    battery = box(BATTERY, BATTERY_AT)
    board = box(BOARD, BOARD_AT)
    bearing = cyl(BEARING_OD, BEARING_W, z=SEAT_Z).cut(cyl(BEARING_ID, BEARING_W, z=SEAT_Z))
    retainer = cyl(RETAINER_D, RETAINER_T, z=SEAT_Z - RETAINER_T).cut(cyl(N20_SHAFT_D + 0.4, RETAINER_T, z=SEAT_Z - RETAINER_T))
    track = cyl(TRACK_OD, TRACK_T, z=DISC_Z0).cut(cyl(TRACK_ID, TRACK_T, z=DISC_Z0))
    spring_pins = [cyl(PIN_D, PIN_L, 0, r, DISC_Z0 - PIN_L) for r in PIN_R]
    duct = box((4.0, 3.0, DISC_Z0 - PIN_L - 32.0), (0, 15.5, (DISC_Z0 - PIN_L + 32.0) / 2))

    balls, springs = [], []
    for k in range(4):
        balls.append(cq.Workplane("XY").sphere(BALL_D / 2).translate((ball_r, 0, BALL_Z)).rotate((0, 0, 0), (0, 0, 1), 90 * k))
        spring = cq.Workplane("YZ").circle(SPRING_D / 2).extrude(SPRING_L).translate((ball_r - BALL_D / 2 - SPRING_L, 0, BALL_Z))
        springs.append(spring.rotate((0, 0, 0), (0, 0, 1), 90 * k))

    # body: chamfered shell, the block that carries the bearing, windows and bores for what sits in it
    body = outline()
    body = body.cut(box((SIDE - 2 * WALL, SIDE - 2 * WALL, BLOCK_Z - WALL), (0, 0, (BLOCK_Z + WALL) / 2)))
    body = body.cut(cyl(RECESS_D, TOP - FLOOR_Z + 1, z=FLOOR_Z)).cut(cyl(BEARING_OD, BEARING_W, z=SEAT_Z)).cut(cyl(BORE_D, SEAT_Z - BLOCK_Z + 1, z=BLOCK_Z - 1))
    for k in range(4):
        body = body.cut(cq.Workplane("XY").sphere(DIMPLE_R).translate((BALL_R, 0, BALL_Z)).rotate((0, 0, 0), (0, 0, 1), 90 * k))
    for part in inner_fixed + edge_magnets + pins + swept + spring_pins + [duct]:
        body = body.cut(part)
    for edge in edges():  # a slot round each hook so that it can swing out through the corner
        for s in (-HOOK_AT, HOOK_AT):
            spin = [0, 0, 0]
            spin[edge[0]] = 1
            slot = cq.Workplane("XY").circle(HOOK[2] + 0.4).extrude(HOOK[0] + 0.6).translate((0, 0, -(HOOK[0] + 0.6) / 2))
            if edge[0] == 0:
                slot = slot.rotate((0, 0, 0), (0, 1, 0), 90)
            elif edge[0] == 1:
                slot = slot.rotate((0, 0, 0), (1, 0, 0), 90)
            body = body.cut(slot.translate(along_edge(edge, s, AXLE_IN)))

    # disc: plate, hub with a bore for the D shaft, pockets
    disc = cyl(DISC_D, DISC_T, z=DISC_Z0).union(cyl(HUB_D, HUB_L, z=DISC_Z0 - HUB_L))
    disc = disc.cut(cyl(N20_SHAFT_D + 0.1, DISC_Z0 - HUB_L + 6.0, z=0)).cut(track)
    for part in inner_disc:
        disc = disc.cut(part)
    for k in range(4):
        pocket = cq.Workplane("YZ").circle((BALL_D + 0.2) / 2).extrude(BALL_D + SPRING_L + 1.0).translate((DISC_D / 2 - BALL_D - SPRING_L, 0, BALL_Z))
        disc = disc.cut(pocket.rotate((0, 0, 0), (0, 0, 1), 90 * k))

    parts = {
        "body": (body, "body"),
        "disc": (disc, "disc"),
        "bearing": (bearing, "body"),
        "retainer": (retainer, "disc"),
        "turn_motor": (turn_motor, "body"),
        "edge_motor": (edge_motor, "body"),
        "battery": (battery, "body"),
        "board": (board, "body"),
        "track": (track, "disc"),
        "pins": (union(spring_pins), "body"),
        "duct": (duct, "body"),
        "balls": (union(balls), "disc"),
        "springs": (union(springs), "disc"),
        "epm_edge": (union(edge_magnets), "body"),
        "epm_inner": (union(inner_fixed), "body"),
        "epm_disc": (union(inner_disc), "disc"),
        "axles": (union(pins), "body"),
        "hooks": (union(latch), "body"),
    }
    if disc_angle:
        parts = {name: (turned(shape, disc_angle) if group == "disc" else shape, group) for name, (shape, group) in parts.items()}
    return parts


COUNT = {"balls": 4, "springs": 4, "epm_edge": 24, "epm_inner": 20, "epm_disc": 4, "axles": 12, "hooks": 24, "pins": 2}
COLOURS = {
    "body": (0.72, 0.72, 0.75),
    "disc": (0.91, 0.53, 0.17),
    "bearing": (0.24, 0.43, 0.85),
    "retainer": (0.35, 0.35, 0.4),
    "turn_motor": (0.23, 0.64, 0.34),
    "edge_motor": (0.10, 0.45, 0.25),
    "battery": (0.85, 0.80, 0.25),
    "board": (0.10, 0.35, 0.20),
    "track": (0.80, 0.55, 0.20),
    "pins": (0.85, 0.75, 0.30),
    "duct": (0.80, 0.20, 0.20),
    "balls": (0.85, 0.85, 0.90),
    "springs": (0.55, 0.55, 0.60),
    "epm_edge": (0.60, 0.25, 0.65),
    "epm_inner": (0.75, 0.30, 0.55),
    "epm_disc": (0.90, 0.35, 0.45),
    "axles": (0.30, 0.30, 0.33),
    "hooks": (0.20, 0.60, 0.80),
}
LABELS = {
    "body": "body (printed, 2 mm walls)",
    "disc": "turning disc",
    "bearing": "6803 bearing",
    "retainer": "retainer washer",
    "turn_motor": "turning-face gearmotor (N20)",
    "edge_motor": "edge-hinge gearmotor (N20)",
    "battery": "1S LiPo 502025",
    "board": "controller board",
    "track": "ring board, two tracks",
    "pins": "two spring pins",
    "duct": "wire duct",
    "balls": "4 detent balls",
    "springs": "4 detent springs",
    "epm_edge": "24 edge magnets (switchable)",
    "epm_inner": "20 face magnets (switchable)",
    "epm_disc": "4 disc magnets (switchable)",
    "axles": "12 edge axles",
    "hooks": "24 latch hooks",
}


def volume(shape):
    return shape.val().Volume()


def overlap(a, b):
    box_a, box_b = a.val().BoundingBox(), b.val().BoundingBox()
    apart = box_a.xmin > box_b.xmax or box_b.xmin > box_a.xmax or box_a.ymin > box_b.ymax or box_b.ymin > box_a.ymax or box_a.zmin > box_b.zmax or box_b.zmin > box_a.zmax
    if apart:
        return 0.0
    try:
        return a.intersect(b).val().Volume()
    except Exception:
        return 0.0


def interference():
    """Volume shared by every pair of parts, at every disc angle and hook angle. Returns the rows
    above 0.01 mm3 and the number of pair checks."""
    rows, checks = [], 0
    still = build()
    names = list(still)
    for a, b in itertools.combinations(names, 2):
        checks += 1
        v = overlap(still[a][0], still[b][0])
        if v > 0.01:
            rows.append((a, b, "disc 0, hooks 0", round(v, 2)))
    for angle in DISC_ANGLES[1:]:
        parts = build(disc_angle=angle)
        for a, b in itertools.combinations(names, 2):
            if (parts[a][1] == "disc") != (parts[b][1] == "disc"):
                checks += 1
                v = overlap(parts[a][0], parts[b][0])
                if v > 0.01:
                    rows.append((a, b, f"disc {angle}", round(v, 2)))
    for angle in HOOK_ANGLES[1:]:
        latch = union(hooks(angle))
        for b in names:
            if b != "hooks":
                checks += 1
                v = overlap(latch, still[b][0])
                if v > 0.01:
                    rows.append(("hooks", b, f"hooks {angle}", round(v, 2)))
    return rows, checks


def masses(parts):
    """{name: grams} for the drawn parts."""
    out = {}
    for name, (shape, _) in parts.items():
        if name == "body":
            out[name] = volume(shape) * RHO_PRINT * FILL_BODY
        elif name == "disc":
            out[name] = volume(shape) * RHO_PRINT * FILL_DISC
        elif name.startswith("epm"):
            out[name] = COUNT[name] * MASS["epm_unit"]
        else:
            key = {"balls": "ball", "springs": "spring", "axles": "axle", "hooks": "hook", "pins": "pin"}.get(name, name)
            out[name] = COUNT.get(name, 1) * MASS[key]
    return out


def inertia(parts, grams, names):
    """Mass (kg), centre of mass (mm) and the inertia tensor about it (kg m2) of the named parts,
    each taken as uniform over its own volume."""
    total, moment = 0.0, cq.Vector(0, 0, 0)
    rows = []
    for name in names:
        solid = parts[name][0].val()
        m = grams[name] / 1000
        c = cq.Shape.centerOfMass(solid)
        i = cq.Shape.matrixOfInertia(solid)  # about the part's centre of mass, for unit density
        scale = m / solid.Volume()
        rows.append((m, c, [[i[r][k] * scale * 1e-6 for k in range(3)] for r in range(3)]))
        total += m
        moment = moment + c * m
    com = moment / total
    tensor = [[0.0] * 3 for _ in range(3)]
    for m, c, i in rows:
        d = [(c.x - com.x) / 1000, (c.y - com.y) / 1000, (c.z - com.z) / 1000]
        dd = sum(v * v for v in d)
        for r in range(3):
            for k in range(3):
                tensor[r][k] += i[r][k] + m * ((dd if r == k else 0.0) - d[r] * d[k])
    return total, com, tensor


def export(parts):
    assembly = cq.Assembly()
    for name, (shape, _) in parts.items():
        cq.exporters.export(shape, str(HERE / f"cube_full_{name}.step"))
        cq.exporters.export(shape, str(HERE / f"cube_full_{name}.stl"), tolerance=0.05, angularTolerance=0.3)
        assembly.add(shape, name=name, color=cq.Color(*COLOURS[name]))
    assembly.save(str(HERE / "cube_full_assembly.step"))


def main():
    parts = build()
    export(parts)
    print(f"{len(parts)} parts exported to {HERE}")

    rows, checks = interference()
    print(f"\ninterference: {checks} pair checks at disc angles {DISC_ANGLES} and hook angles {HOOK_ANGLES}; pairs sharing more than 0.01 mm3:")
    for row in rows:
        print("  ", row)
    if not rows:
        print("   none")

    grams = masses(parts)
    print("\nmass (g):")
    for name in parts:
        print(f"  {LABELS[name]:34s} {grams[name]:6.1f}")
    for name, g in NOT_DRAWN.items():
        print(f"  {name + ' (not drawn)':34s} {g:6.1f}")
    drawn = sum(grams.values())
    total = drawn + sum(NOT_DRAWN.values())
    print(f"  {'total':34s} {total:6.1f}   (drawn parts {drawn:.1f})")
    magnets = grams["epm_edge"] + grams["epm_inner"] + grams["epm_disc"]
    print(f"  of which switchable magnets {magnets:.0f} g ({magnets / total:.0%}); what the turning face adds (disc, bearing, retainer, motor, balls, springs, track, pins): "
          f"{sum(grams[k] for k in ('disc', 'bearing', 'retainer', 'turn_motor', 'balls', 'springs', 'track', 'pins')):.1f} g")

    m, com, tensor = inertia(parts, grams, list(parts))
    spread = sum(NOT_DRAWN.values()) / 1000  # the parts not drawn, taken as spread evenly through the cube
    side = SIDE / 1000
    about = [tensor[k][k] + spread * side**2 / 6 for k in range(3)]
    whole = m + spread
    print("\nmass and inertia (for cube_turning.py):")
    print(f"  MASS = {whole:.3f} kg; centre of mass of the drawn parts ({com.x:.1f}, {com.y:.1f}, {com.z:.1f}) mm (the cube's centre is (0, 0, 25))")
    print(f"  inertia about the centre of mass, x y z: {about[0]:.3e} {about[1]:.3e} {about[2]:.3e} kg m2; INERTIA = mean / (MASS side^2) = {sum(about) / 3 / (whole * side**2):.3f} (a uniform cube: 0.167)")
    disc_names = [name for name, (_, group) in parts.items() if group == "disc"]
    dm, dcom, dtensor = inertia(parts, grams, disc_names)
    print(f"  DISC_MASS = {dm:.4f} kg ({', '.join(disc_names)}); DISC_INERTIA about its axis = {dtensor[2][2]:.2e} kg m2; its centre of mass z = {dcom.z:.1f} mm")

    print("\ncost (USD, one cube, all guesses):")
    for name, usd in COST:
        print(f"  {name:62s} {usd:6.1f}")
    print(f"  {'total':62s} {sum(usd for _, usd in COST):6.1f}")

    print("\nrenders: .venv/bin/python hardware/rotating_cube/render_exploded.py")


if __name__ == "__main__":
    main()
