"""Parametric CAD of one 50 mm cube module with a turning face (CadQuery).

Axis of the turning face is +Z; the cube spans z = 0..50, x,y = -25..25.
Run:  .venv/bin/python hardware/rotating_cube/cube_turning_face.py
Writes STEP and STL for body and disc, prints volumes, masses, interference,
and renders runs/handoff/media/rotating-cube-cad.png.

GUESS marks values that are placeholders, not measured or sourced.
"""
import math
from pathlib import Path

import cadquery as cq

# ---- parameters (mm) ----
SIDE = 50.0
CLEAR = 0.2                    # radial and axial clearance disc to body
RECESS_D = 36.0                # recess in the top face (GUESS)
RECESS_DEPTH = 14.4            # disc 14.0 + 0.2 under + 0.2 above
BEARING_ID = 17.0              # 6803 style: 17 x 26 x 5 (standard size)
BEARING_OD = 26.0
BEARING_W = 5.0
SHOULDER_D = 20.0              # through hole under the bearing seat (GUESS)
SHOULDER_Z = 20.0              # bottom of the cavity under the bearing
DISC_D = RECESS_D - 2 * CLEAR
DISC_T = 14.0
HUB_D = BEARING_ID
HUB_L = BEARING_W + CLEAR
MOTOR_D = 12.0                 # N20-like gearmotor envelope (GUESS)
MOTOR_L = 15.0
MOTOR_POCKET_D = MOTOR_D + 0.4
MOTOR_POCKET_DEPTH = 16.0
SHAFT_D = 3.0
SHAFT_L = 8.0
MAG_D = 6.0                    # disc magnet 6 x 3 (GUESS)
MAG_DEPTH = 3.0
MAG_FACE_OFFSET = 15.0         # magnet grid on the five fixed faces
RING_MAG_R = 24.0              # ring magnets at 45 deg
DISC_MAG_R = 13.0              # disc magnets at 45 deg
DETENT_COUNT = 4               # 90 degrees
BALL_D = 3.0                   # steel ball (GUESS)
BALL_POCKET_D = BALL_D + 0.2
BALL_POCKET_DEPTH = 5.0        # radial, from the rim
BALL_PROTRUDE = 0.5
DIMPLE_R = 1.6                 # dimple sphere radius in the recess wall (GUESS)

RHO_PLA = 1.24e-3              # g/mm3
RHO_STEEL = 7.8e-3
RHO_MAGNET = 7.5e-3            # NdFeB
RHO_MOTOR = 6.0e-3             # effective, N20 gearmotor ~10 g (GUESS)
INFILL = 0.30                  # effective solid fraction of printed shells+infill (GUESS)

TOP = SIDE
FLOOR_Z = TOP - RECESS_DEPTH          # recess floor
SEAT_Z = FLOOR_Z - BEARING_W          # bearing seat bottom
DISC_Z0 = FLOOR_Z + CLEAR             # disc plate underside


def cyl(d, h, x=0.0, y=0.0, z=0.0):
    return cq.Workplane("XY").workplane(offset=z).center(x, y).circle(d / 2).extrude(h)


def build_body():
    b = cq.Workplane("XY").box(SIDE, SIDE, SIDE, centered=(True, True, False))
    b = b.cut(cyl(RECESS_D, RECESS_DEPTH, z=FLOOR_Z))
    b = b.cut(cyl(BEARING_OD, BEARING_W, z=SEAT_Z))
    b = b.cut(cyl(SHOULDER_D, SEAT_Z - SHOULDER_Z, z=SHOULDER_Z))
    # magnet pockets on the five fixed faces
    o = MAG_FACE_OFFSET
    pts = [(o, o), (-o, o), (o, -o), (-o, -o)]
    h = SIDE / 2
    for x, y in pts:  # bottom face (z=0)
        b = b.cut(cyl(MAG_D, MAG_DEPTH, x, y, 0))
    for x, z in pts:  # +/-Y faces
        zc = SIDE / 2 + z
        b = b.cut(cq.Workplane("XZ").workplane(offset=-h).center(x, zc).circle(MAG_D / 2).extrude(MAG_DEPTH))   # y=+h inward
        b = b.cut(cq.Workplane("XZ").workplane(offset=h).center(x, zc).circle(MAG_D / 2).extrude(-MAG_DEPTH))   # y=-h inward
    for y, z in pts:  # +/-X faces
        zc = SIDE / 2 + z
        b = b.cut(cq.Workplane("YZ").workplane(offset=h - MAG_DEPTH).center(y, zc).circle(MAG_D / 2).extrude(MAG_DEPTH))
        b = b.cut(cq.Workplane("YZ").workplane(offset=-h).center(y, zc).circle(MAG_D / 2).extrude(MAG_DEPTH))
    # ring magnets on the top face around the recess
    for k in range(4):
        a = math.radians(45 + 90 * k)
        b = b.cut(cyl(MAG_D, MAG_DEPTH, RING_MAG_R * math.cos(a), RING_MAG_R * math.sin(a), TOP - MAG_DEPTH))
    # detent dimples in the recess wall, centred on the ball axis
    zc = DISC_Z0 + DISC_T / 2
    ball_c = DISC_D / 2 - BALL_D / 2 + BALL_PROTRUDE
    for k in range(DETENT_COUNT):
        a = math.radians(90 * k)
        s = cq.Workplane("XY").sphere(DIMPLE_R).translate((ball_c * math.cos(a), ball_c * math.sin(a), zc))
        # sphere concentric with the ball when the ball is seated
        b = b.cut(s)
    return b


def build_disc():
    d = cyl(DISC_D, DISC_T, z=DISC_Z0)
    d = d.union(cyl(HUB_D, HUB_L, z=DISC_Z0 - HUB_L))
    d = d.cut(cyl(MOTOR_POCKET_D, MOTOR_POCKET_DEPTH, z=DISC_Z0 - HUB_L))
    for k in range(4):
        a = math.radians(45 + 90 * k)
        d = d.cut(cyl(MAG_D, MAG_DEPTH, DISC_MAG_R * math.cos(a), DISC_MAG_R * math.sin(a), DISC_Z0 + DISC_T - MAG_DEPTH))
    zc = DISC_Z0 + DISC_T / 2
    for k in range(DETENT_COUNT):
        a = 90 * k
        p = (cq.Workplane("YZ").workplane(offset=DISC_D / 2 - BALL_POCKET_DEPTH).center(0, zc)
             .circle(BALL_POCKET_D / 2).extrude(BALL_POCKET_DEPTH + 1)).rotate((0, 0, 0), (0, 0, 1), a)
        d = d.cut(p)
    return d


def build_bearing():
    return cyl(BEARING_OD, BEARING_W, z=SEAT_Z).cut(cyl(BEARING_ID, BEARING_W, z=SEAT_Z))


def build_motor():
    m = cyl(MOTOR_D, MOTOR_L, z=DISC_Z0 - HUB_L + 0.4)
    return m.union(cyl(SHAFT_D, SHAFT_L, z=DISC_Z0 - HUB_L + 0.4 - SHAFT_L))


def build_balls(angle=0.0):
    zc = DISC_Z0 + DISC_T / 2
    ball_c = DISC_D / 2 - BALL_D / 2 + BALL_PROTRUDE
    out = None
    for k in range(DETENT_COUNT):
        a = math.radians(90 * k)
        s = cq.Workplane("XY").sphere(BALL_D / 2).translate((ball_c * math.cos(a), ball_c * math.sin(a), zc))
        out = s if out is None else out.union(s)
    return out.rotate((0, 0, 0), (0, 0, 1), angle)


def vol(w):
    return w.val().Volume()


def inter(a, b):
    try:
        return vol(a.intersect(b))
    except Exception:
        return 0.0


def render(parts, path):
    """Offscreen VTK render: left a section (front half removed), right the outside."""
    import numpy as np
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    from PIL import Image

    def actor(shape, color):
        v, t = shape.val().tessellate(0.1, 0.25)
        pts = vtk.vtkPoints()
        for p in v:
            pts.InsertNextPoint(p.x, p.y, p.z)
        cells = vtk.vtkCellArray()
        for tri in t:
            cells.InsertNextCell(3, list(tri))
        pd = vtk.vtkPolyData()
        pd.SetPoints(pts)
        pd.SetPolys(cells)
        nrm = vtk.vtkPolyDataNormals()
        nrm.SetInputData(pd)
        nrm.SetFeatureAngle(40)
        nrm.SplittingOn()
        m = vtk.vtkPolyDataMapper()
        m.SetInputConnection(nrm.GetOutputPort())
        a = vtk.vtkActor()
        a.SetMapper(m)
        a.GetProperty().SetColor(*color)
        return a

    palette = {"body": (0.65, 0.65, 0.68), "disc": (0.91, 0.53, 0.17), "bearing": (0.24, 0.43, 0.85), "motor": (0.23, 0.64, 0.34)}
    imgs = []
    for section, pos in ((True, (0, -170, 110)), (False, (110, -130, 120))):
        ren = vtk.vtkRenderer()
        ren.SetBackground(1, 1, 1)
        for name, shape, _ in parts:
            s = shape.cut(cq.Workplane("XY").box(200, 100, 200, centered=(True, True, True)).translate((0, -50, 25))) if section else shape
            ren.AddActor(actor(s, palette[name]))
        cam = ren.GetActiveCamera()
        cam.SetFocalPoint(0, 0, 25)
        cam.SetPosition(*pos)
        cam.SetViewUp(0, 0, 1)
        ren.ResetCameraClippingRange()
        win = vtk.vtkRenderWindow()
        win.SetOffScreenRendering(1)
        win.SetSize(700, 700)
        win.AddRenderer(ren)
        win.Render()
        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(win)
        w2i.Update()
        im = w2i.GetOutput()
        arr = np.frombuffer(bytes(vtk.util.numpy_support.vtk_to_numpy(im.GetPointData().GetScalars())), dtype=np.uint8)
        arr = arr.reshape(700, 700, -1)[::-1, :, :3]
        imgs.append(arr)
    Image.fromarray(np.concatenate(imgs, axis=1)).save(path)


def main():
    here = Path(__file__).parent
    media = here.parents[1] / "runs" / "handoff" / "media"
    media.mkdir(parents=True, exist_ok=True)
    body, disc, bearing, motor, balls = build_body(), build_disc(), build_bearing(), build_motor(), build_balls()
    for name, shape in (("body", body), ("disc", disc)):
        cq.exporters.export(shape, str(here / f"cube_turning_face_{name}.step"))
        cq.exporters.export(shape, str(here / f"cube_turning_face_{name}.stl"))

    print("radial clearance recess-disc:", (RECESS_D - DISC_D) / 2, "mm; axial under/over:", CLEAR, "/", TOP - (DISC_Z0 + DISC_T), "mm")
    print("body ∩ disc volume (mm3):", round(inter(body, disc), 6))
    for ang in (0, 22.5, 45, 67.5, 90, 135, 180, 270):
        print(f"  disc rotated {ang:6.1f} deg: body ∩ disc =", round(inter(body, disc.rotate((0, 0, 0), (0, 0, 1), ang)), 6))
    print("body ∩ bearing:", round(inter(body, bearing), 6), " disc ∩ bearing:", round(inter(disc, bearing), 6))
    print("motor ∩ body:", round(inter(motor, body), 6), " motor ∩ disc:", round(inter(motor, disc), 6))
    for ang in (0, 45):
        print(f"  balls at {ang} deg ∩ body (spring compression, expected at 45):", round(inter(body, build_balls(ang)), 4))
    print("  balls ∩ disc (pocket):", round(inter(disc, balls), 4))

    mag_v = math.pi * (MAG_D / 2) ** 2 * MAG_DEPTH
    n_mag = 5 * 4 + 4 + 4
    ball_v = 4 * math.pi / 6 * BALL_D ** 3
    rows = [
        ("body (PLA, solid)", vol(body) * RHO_PLA),
        ("disc (PLA, solid)", vol(disc) * RHO_PLA),
        ("bearing (steel)", vol(bearing) * RHO_STEEL),
        ("motor (N20-like)", vol(motor) * RHO_MOTOR),
        (f"{n_mag} magnets", n_mag * mag_v * RHO_MAGNET),
        ("4 balls (steel)", ball_v * RHO_STEEL),
    ]
    for n, m in rows:
        print(f"{n:22s} {m:7.1f} g")
    solid_total = sum(m for _, m in rows)
    printed = (vol(body) + vol(disc)) * RHO_PLA
    infill_total = solid_total - printed + printed * INFILL
    print(f"volumes: body {vol(body):.0f} mm3, disc {vol(disc):.0f} mm3")
    print(f"total, printed parts solid: {solid_total:.1f} g; with printed parts at {INFILL:.0%} effective fill: {infill_total:.1f} g")
    render([("body", body, "#9a9a9a"), ("disc", disc, "#e8872b"), ("bearing", bearing, "#3d6fd8"), ("motor", motor, "#3aa356")],
           str(media / "rotating-cube-cad.png"))


if __name__ == "__main__":
    main()
