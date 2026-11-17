"""Figures for ASSEMBLY.md and DESIGN.md, rendered offscreen (VTK) from cube50_cad.build().

Run from the repo root:  .venv/bin/python hardware/cube50/cad/render.py
Writes hardware/cube50/figures/: assembled.png (outside, and the core with the top foot and the
front wall cut away), exploded.png (every part pulled out, labelled), foot_open.png (the top foot at
0, 90 and 180 degrees), pivot.png (a floor roll, the body turning about the bonded foot),
steps.png (the four assembly stages of ASSEMBLY.md).
"""

import math
import sys
from pathlib import Path

import numpy as np
import vtk
from PIL import Image, ImageDraw, ImageFont
from vtk.util.numpy_support import vtk_to_numpy

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import cube50_cad as cad  # noqa: E402

FIG = HERE.parent / "figures"
COLOUR = {"PETG": (0.86, 0.86, 0.82), "steel": (0.55, 0.57, 0.62), "brass": (0.82, 0.66, 0.26), "motor": (0.36, 0.42, 0.55),
          "cell": (0.30, 0.46, 0.78), "pcb": (0.16, 0.52, 0.27), "magnet": (0.62, 0.22, 0.22)}
FOOT_COLOUR = (0.93, 0.62, 0.30)
DISC_COLOUR = (0.45, 0.62, 0.85)


def mesh(shape):
    vertices, triangles = shape.val().tessellate(0.1, 0.3)
    points = vtk.vtkPoints()
    for p in vertices:
        points.InsertNextPoint(p.x, p.y, p.z)
    cells = vtk.vtkCellArray()
    for t in triangles:
        cells.InsertNextCell(3, list(t))
    data = vtk.vtkPolyData()
    data.SetPoints(points)
    data.SetPolys(cells)
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(data)
    normals.SetFeatureAngle(40)
    normals.SplittingOn()
    normals.Update()
    return normals.GetOutput()


def colour_of(name, material):
    if name.startswith("foot"):
        return FOOT_COLOUR
    if name.startswith("disc_") and "motor" not in name and "idler" not in name:
        return DISC_COLOUR
    return COLOUR.get(material, (0.7, 0.7, 0.7))


def renderer_for(items, size, clip=None):
    """items: [(polydata, colour, offset)]. clip: (origin, normal) cuts every piece."""
    ren = vtk.vtkRenderer()
    ren.SetBackground(1, 1, 1)
    for data, colour, offset in items:
        mapper = vtk.vtkPolyDataMapper()
        if clip is not None:
            plane = vtk.vtkPlane()
            plane.SetOrigin(*clip[0])
            plane.SetNormal(*clip[1])
            cutter = vtk.vtkClipPolyData()
            cutter.SetInputData(data)
            cutter.SetClipFunction(plane)
            mapper.SetInputConnection(cutter.GetOutputPort())
        else:
            mapper.SetInputData(data)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.SetPosition(*offset)
        actor.GetProperty().SetColor(*colour)
        actor.GetProperty().SetSpecular(0.15)
        ren.AddActor(actor)
    win = vtk.vtkRenderWindow()
    win.SetOffScreenRendering(1)
    win.SetSize(*size)
    win.AddRenderer(ren)
    return win, ren


def shot(win, ren, azimuth, elevation, distance, focus=(0, 0, 0)):
    cam = ren.GetActiveCamera()
    a, e = math.radians(azimuth), math.radians(elevation)
    cam.SetFocalPoint(*focus)
    cam.SetPosition(focus[0] + distance * math.cos(e) * math.cos(a), focus[1] + distance * math.cos(e) * math.sin(a), focus[2] + distance * math.sin(e))
    cam.SetViewUp(0, 0, 1)
    cam.SetViewAngle(30)
    ren.ResetCameraClippingRange()
    win.Render()
    grab = vtk.vtkWindowToImageFilter()
    grab.SetInput(win)
    grab.Update()
    img = grab.GetOutput()
    w, h, _ = img.GetDimensions()
    return Image.fromarray(vtk_to_numpy(img.GetPointData().GetScalars()).reshape(h, w, -1)[::-1, :, :3].copy())


def to_screen(ren, point, height):
    c = vtk.vtkCoordinate()
    c.SetCoordinateSystemToWorld()
    c.SetValue(*point)
    x, y = c.GetComputedDisplayValue(ren)
    return x, height - y


def font(size):
    for f in ("/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def caption(img, text, size=22):
    d = ImageDraw.Draw(img)
    d.text((14, 10), text, fill=(20, 20, 20), font=font(size))
    return img


def items_of(parts, skip=()):
    return [(mesh(s), colour_of(n, m), (0, 0, 0)) for n, (s, _, m) in parts.items() if n not in skip]


def assembled():
    parts = cad.build()
    w1, r1 = renderer_for(items_of(parts), (800, 800))
    a = caption(shot(w1, r1, 35, 25, 300), f"Cube{cad.S:g}, feet stowed (top foot orange, disc blue)")
    cut = [n for n in parts if n.endswith("_top") and any(n.startswith(k) for k in cad.FOOT_PARTS)]
    w2, r2 = renderer_for(items_of(parts, skip=cut), (800, 800), clip=((0, -2, 0), (0, 1, 0)))
    b = caption(shot(w2, r2, -60, 35, 230), "Top foot removed, front half cut at y = -2 mm")
    out = Image.new("RGB", (1600, 800), "white")
    out.paste(a, (0, 0))
    out.paste(b, (800, 0))
    out.save(FIG / "assembled.png")


def _mirror(d):
    """Top-side offsets turned into the bottom side's (180 degrees about x)."""
    return {k.replace("_top", "_bottom"): (v[0], -v[1], -v[2]) for k, v in d.items() if k.endswith("_top")}


EXPLODE_TOP = {
    "foot_top": (0, 0, 75), "disc_top": (0, 0, 115), "cam_ring_top": (0, 0, 135), "pins_top": (0, 0, 150), "bearing_top": (0, 0, 92), "hub_top": (0, 0, 100), "lip_disc_top": (0, 0, 128),
    "lock_ring_top": (0, -30, 108), "lock_pin_top": (0, -35, 85), "lock_lever_top": (0, -40, 78), "lock_sma_top": (-10, -45, 72), "mag_disc_top": (0, 0, 140),
    "axle_top": (65, 0, 45), "knuckle_top": (65, 0, 60), "plate_py_top": (45, 45, 35), "plate_ny_top": (45, -45, 35), "hinge_motor_top": (40, 30, 25), "g_motor_top": (45, 55, 35), "g_compound_top": (45, 60, 40),
    "g_idler_top": (50, 50, 45), "g_axle_top": (55, 40, 45), "eclips_top": (45, 65, 40), "enc_board_hinge_top": (40, -70, 45), "enc_magnet_hinge_top": (40, -62, 45), "enc_spacers_top": (40, -55, 40),
    "deploy_motor_top": (0, -55, 30), "deploy_in_board_top": (10, -55, 30), "switch_in_top": (12, -55, 30), "switch_out_top": (55, -55, 30), "carrier_top": (30, -55, 30), "nut_top": (30, -55, 38), "pin_top": (30, -75, 30), "deploy_strap_top": (0, -55, 45),
    "disc_motor_top": (-45, 40, 10), "pinion_top": (-45, 40, 50), "c1_top": (-25, 35, 55), "c2_top": (-35, 10, 40), "c1_pin_top": (-25, 35, 30), "c2_pin_top": (-35, 10, 25), "centre_gear_top": (0, 0, 40), "centre_shaft_top": (0, 0, 28), "shaft_eclip_top": (0, 0, 15), "drive_plate_top": (-10, 10, 62), "bolt_top": (-45, -20, 40), "cross_pin_top": (0, -25, 28), "strip_top": (-10, 50, 75),
}
EXPLODE = {**EXPLODE_TOP, **_mirror(EXPLODE_TOP), "lip_px": (40, 0, 0), "lip_py": (0, 40, 0), "lip_ny": (0, -40, 0), "mag_px": (36, 0, 0), "mag_py": (0, 36, 0), "mag_ny": (0, -36, 0),
           "side_cover": (-70, 0, 0), "lip_nx": (-80, 0, 0), "mag_nx": (-84, 0, 0), "main_board": (-50, 0, 0), "cell": (75, -75, 0), "body": (0, 0, 0)}
LABELS = ["foot_top", "disc_top", "hub_top", "bearing_top", "lock_ring_top", "lock_pin_top", "plate_py_top", "plate_ny_top", "hinge_motor_top", "axle_top", "knuckle_top", "carrier_top", "deploy_motor_top",
          "disc_motor_top", "c1_top", "c2_top", "centre_gear_top", "drive_plate_top", "enc_board_hinge_top", "side_cover", "main_board", "cell", "bolt_top", "body"]


def exploded():
    parts = cad.build()
    items = [(mesh(s), colour_of(n, m), EXPLODE.get(n, (0, 0, 0))) for n, (s, _, m) in parts.items()]
    win, ren = renderer_for(items, (1400, 1400))
    img = shot(win, ren, 30, 18, 520)
    d = ImageDraw.Draw(img)
    f = font(17)
    for n in LABELS:
        bb = parts[n][0].val().BoundingBox()
        c = np.array([(bb.xmin + bb.xmax) / 2, (bb.ymin + bb.ymax) / 2, (bb.zmin + bb.zmax) / 2]) + np.array(EXPLODE.get(n, (0, 0, 0)))
        x, y = to_screen(ren, c, 1400)
        label = n.replace("_top", "").replace("_", " ")
        d.line([(x, y), (x + 28, y - 22)], fill=(60, 60, 60), width=1)
        d.text((x + 30, y - 34), label, fill=(10, 10, 10), font=f)
    caption(img, f"Cube{cad.S:g} exploded (top-side parts labelled; the bottom side repeats them turned 180 degrees about x)")
    img.save(FIG / "exploded.png")


def foot_open():
    tiles = []
    for ang in (0.0, 90.0, 180.0):
        parts = cad.build(ang)
        win, ren = renderer_for(items_of(parts), (600, 600))
        tiles.append(caption(shot(win, ren, -55, 20, 330, focus=(15, 0, 15)), f"top foot open {ang:.0f} degrees"))
    out = Image.new("RGB", (1800, 600), "white")
    for k, t in enumerate(tiles):
        out.paste(t, (600 * k, 0))
    out.save(FIG / "foot_open.png")


def pivot():
    """A floor roll: the bottom foot stays bonded to the floor, the body turns 90 degrees about the
    hinge (drawn at 0, 45 and 90 degrees, lighter = earlier). A climb is the same motion through
    180 degrees with the foot bonded to a neighbour's side face."""
    parts = cad.build(deployed={"bottom"})
    moving_foot = tuple(f"{n}_bottom" for n in cad.FOOT_PARTS + cad.SLED_MOVES_WITH_FOOT)
    items = []
    for k, ang in enumerate((0, 45, 90)):
        shade = 0.35 + 0.65 * (k + 1) / 3
        for n, (s, _, m) in parts.items():
            if n in moving_foot:
                if k < 2:
                    continue
                shp = s
            else:
                shp = s.rotate((cad.H, 0, -cad.H), (cad.H, 1, -cad.H), -ang)
            c = np.array(colour_of(n, m)) * shade + (1 - shade)
            items.append((mesh(shp), tuple(c), (0, 0, 0)))
    win, ren = renderer_for(items, (900, 600))
    floor = vtk.vtkCubeSource()
    floor.SetBounds(-45, 165, -45, 45, -cad.H - 1, -cad.H)
    fm = vtk.vtkPolyDataMapper()
    fm.SetInputConnection(floor.GetOutputPort())
    fa = vtk.vtkActor()
    fa.SetMapper(fm)
    fa.GetProperty().SetColor(0.75, 0.78, 0.8)
    ren.AddActor(fa)
    img = caption(shot(win, ren, -90, 12, 420, focus=(40, 0, 0)), "Floor roll: bottom foot bonded to the floor, body turns 90 degrees about the hinge", 20)
    img.save(FIG / "pivot.png")


STEPS = [  # the order of ASSEMBLY.md section 2 (checked step by step in cube50_cad.check_assembly)
    ("1. Cell in through the -x opening; OUT switch, sled plates, hinge motor", ("body", "lip_p", "lip_ny", "mag_p", "mag_ny", "cell", "switch_out", "plate_", "hinge_motor", "g_", "eclips")),
    ("2. Deploy module, straps, carrier pin", ("deploy_", "switch_in", "carrier", "nut", "pin_top", "pin_bottom")),
    ("3. Disc motor, shaft, C2, C1, drive plate, bolt, cross pin, y-wall strip", ("disc_motor", "pinion", "centre_", "shaft_eclip", "c1", "c2", "drive_plate", "bolt", "cross_pin", "strip")),
    ("4. Feet (bench stacks), axles, hinge encoders, -x cover", ("foot", "disc_", "bearing", "hub", "lock_", "cam_ring", "pins", "lip_disc", "mag_disc", "axle", "knuckle", "enc_", "side_cover", "main_board", "lip_nx", "mag_nx")),
]


def steps():
    """Assembly steps: each panel adds the parts of one step to the previous ones."""
    parts = cad.build()
    tiles, shown = [], set()
    for title, prefixes in STEPS:
        for n in parts:
            if any(n.startswith(p) for p in prefixes):
                shown.add(n)
        items = [(mesh(s), colour_of(n, m), (0, 0, 0)) for n, (s, _, m) in parts.items() if n in shown]
        win, ren = renderer_for(items, (700, 600), clip=((0, -2, 0), (0, 1, 0)) if title.startswith(("2", "3")) else None)
        tiles.append(caption(shot(win, ren, -55, 28, 250), title, 17))
    out = Image.new("RGB", (1400, 1200), "white")
    for k, t in enumerate(tiles):
        out.paste(t, (700 * (k % 2), 600 * (k // 2)))
    out.save(FIG / "steps.png")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    assembled()
    exploded()
    foot_open()
    pivot()
    steps()
    print("figures in", FIG)
