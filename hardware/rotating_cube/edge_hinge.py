"""Edge-hinge cube pair: a 50 mm mover on a 50 mm support, turning about their shared edge.

Recreated on 2026-11-14 from the exported STEP/STL files and the constants left in the bytecode of
the first version, whose source was lost; the parts and their sizes are the same.

HARDWARE.md ("The hinge cube in MuJoCo") says the edge latch and the torque source are not
designed. This file picks one layout to show the parts of one pivot between two cubes, marked
GUESS where it is a choice. It is an illustration of the motion, not a mechanism: the motor is not
connected to the axle and the latches are plain discs. cube_full.py has the complete cube.

1. Support: a 50 mm cube on the floor. Its top-right edge has a cylindrical groove (radius 5 mm).
1. Mover: a 50 mm cube on top of the support. Its bottom-right edge has the matching groove.
1. Axle: an 8 mm pin in both grooves, along the pivot edge, fixed to the support. GUESS
1. Motor: a 12 x 15 mm gearmotor stand-in inside the mover, next to the axle. GUESS
1. Latch: two 16 mm discs, one at each end of the axle, stand-ins for a latch. GUESS

The pivot edge sits at x = 0, z = 50. The mover starts over x in [-50, 0], z in [50, 100] and the
support over x in [-50, 0], z in [0, 50]. Turning the mover about the pivot tips it onto the edge
(90 degrees) and lands it on the support's right-hand face (180 degrees).

Usage: .venv/bin/python hardware/rotating_cube/edge_hinge.py
"""
from pathlib import Path

import cadquery as cq

SIDE = 50.0
GROOVE_R = 5.0
AXLE_R = 4.0
AXLE_L = SIDE
MOTOR_D = 12.0
MOTOR_L = 15.0
MOTOR_POCKET_D = 12.4
LATCH_D = 16.0
LATCH_T = 4.0
PIVOT_X = 0.0
PIVOT_Z = SIDE
SUPPORT_CENTRE = (-25.0, 0.0, 25.0)
MOVER_CENTRE = (-25.0, 0.0, 75.0)
MOTOR_AT = (-12.0, 62.0, 2.0)  # x, z, and where it starts along y


def box_at(centre):
    return cq.Workplane("XY").box(SIDE, SIDE, SIDE).translate(centre)


def cyl_y(radius, length, x, z, y0):
    """A cylinder along y from y0 to y0 + length, at (x, z)."""
    return cq.Workplane("XZ", origin=(0, y0, 0)).center(x, z).circle(radius).extrude(-length)


def build_support():
    return box_at(SUPPORT_CENTRE).cut(cyl_y(GROOVE_R, SIDE, PIVOT_X, PIVOT_Z, -SIDE / 2))


def build_mover():
    mover = box_at(MOVER_CENTRE).cut(cyl_y(GROOVE_R, SIDE, PIVOT_X, PIVOT_Z, -SIDE / 2))
    return mover.cut(cyl_y(MOTOR_POCKET_D / 2, MOTOR_L, MOTOR_AT[0], MOTOR_AT[1], MOTOR_AT[2]))


def build_axle():
    return cyl_y(AXLE_R, AXLE_L, PIVOT_X, PIVOT_Z, -AXLE_L / 2)


def build_motor():
    return cyl_y(MOTOR_D / 2, MOTOR_L, MOTOR_AT[0], MOTOR_AT[1], MOTOR_AT[2])


def build_latches():
    near = cyl_y(LATCH_D / 2, LATCH_T, PIVOT_X, PIVOT_Z, -AXLE_L / 2 - LATCH_T)
    return near.union(cyl_y(LATCH_D / 2, LATCH_T, PIVOT_X, PIVOT_Z, AXLE_L / 2))


def export(parts, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, shape in parts.items():
        cq.exporters.export(shape, str(out_dir / f"edge_hinge_{name}.step"))
        cq.exporters.export(shape, str(out_dir / f"edge_hinge_{name}.stl"))


def main():
    parts = {"support": build_support(), "mover": build_mover(), "axle": build_axle(), "motor": build_motor(), "latches": build_latches()}
    export(parts, Path(__file__).parent)
    for name, shape in parts.items():
        box = shape.val().BoundingBox()
        print(f"{name}: volume {shape.val().Volume():.0f} mm3, x {box.xmin:.0f}..{box.xmax:.0f}, y {box.ymin:.0f}..{box.ymax:.0f}, z {box.zmin:.0f}..{box.zmax:.0f}")
    print(f"mover and axle overlap: {parts['mover'].intersect(parts['axle']).val().Volume():.3f} mm3 (should be ~0)")


if __name__ == "__main__":
    main()
