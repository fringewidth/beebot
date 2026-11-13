"""Assembled and exploded views of the complete turning-face cube (offscreen VTK), from cube_full.py.

Usage: .venv/bin/python hardware/rotating_cube/render_exploded.py
Writes runs/handoff/media/rotating-cube-exploded.png (every part pulled out of the body and
labelled) and, through assembled(), runs/handoff/media/rotating-cube-cad.png (left: the outside;
right: the front half of the body removed). pieces(), scene() and shot() are also what
scripts/rotating_cube_video.py animates.
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
import cube_full as full  # noqa: E402

MEDIA = HERE.parents[1] / "runs" / "handoff" / "media"
CENTRE = np.array([0.0, 0.0, 25.0])


def mesh(shape):
    """vtkPolyData with normals of a CadQuery shape."""
    vertices, triangles = shape.val().tessellate(0.1, 0.25)
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


def pieces():
    """The cube as [(label key, vtkPolyData, colour, centre (3,), blow-up offset (3,), turns with the disc)].
    Magnets, axles and hooks are one piece per face or edge so that each can move out its own way."""
    parts = full.build()
    out = []

    def add(key, shape, offset, disc=False):
        box = shape.val().BoundingBox()
        centre = np.array([(box.xmin + box.xmax) / 2, (box.ymin + box.ymax) / 2, (box.zmin + box.zmax) / 2])
        out.append((key, mesh(shape), full.COLOURS[key], centre, np.array(offset, dtype=float), disc))

    add("body", parts["body"][0], (0, 0, 0))
    add("epm_disc", parts["epm_disc"][0], (0, 0, 92), True)
    add("disc", parts["disc"][0], (0, 0, 66), True)
    add("balls", parts["balls"][0], (0, 0, 50), True)
    add("springs", parts["springs"][0], (0, 0, 50), True)
    add("track", parts["track"][0], (0, 0, 40), True)
    add("bearing", parts["bearing"][0], (0, 0, 26))
    add("pins", parts["pins"][0], (0, 0, 14))
    add("retainer", parts["retainer"][0], (0, 0, -64), True)
    add("turn_motor", parts["turn_motor"][0], (0, 0, -75))
    add("edge_motor", parts["edge_motor"][0], (-58, -40, 0))
    add("battery", parts["battery"][0], (62, -30, 0))
    add("board", parts["board"][0], (20, 62, 0))
    add("duct", parts["duct"][0], (-8, 58, 0))
    for face in full.FACES:
        if face[0] != (0, 0, 1):
            n = np.array(face[0], dtype=float)
            add("epm_inner", full.union(full.inner_units(face)), n * (38 if face[0][2] == 0 else 30))
    units, pins, latch = full.edge_units(), full.axles(), full.hooks()
    for k, edge in enumerate(full.edges()):
        out_dir = np.array(full.along_edge(edge, 0.0, 0.0)) - CENTRE
        out_dir /= np.linalg.norm(out_dir)
        add("epm_edge", full.union(units[2 * k : 2 * k + 2]), out_dir * 22)
        add("axles", pins[k], out_dir * 40)
        add("hooks", full.union(latch[2 * k : 2 * k + 2]), out_dir * 40)
    return out


def scene(items, size, section=False):
    """A renderer with one actor per piece. section: the front half of the body (y < 0) is cut
    away and the loose pieces in front of the cut are hidden. Returns (window, renderer, actors)."""
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1, 1, 1)
    actors = []
    for key, data, colour, centre, _, _ in items:
        mapper = vtk.vtkPolyDataMapper()
        if section and key == "body":
            plane = vtk.vtkPlane()
            plane.SetOrigin(0, 0, 0)
            plane.SetNormal(0, 1, 0)
            clip = vtk.vtkClipPolyData()
            clip.SetInputData(data)
            clip.SetClipFunction(plane)
            mapper.SetInputConnection(clip.GetOutputPort())
        else:
            mapper.SetInputData(data)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*colour)
        actor.GetProperty().SetSpecular(0.15)
        actor.SetVisibility(not (section and key != "body" and centre[1] < -6))
        renderer.AddActor(actor)
        actors.append(actor)
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetSize(*size)
    window.AddRenderer(renderer)
    return window, renderer, actors


def shot(window, renderer, azimuth, elevation, distance, focus=(0, 0, 25)):
    """Render from a camera on a sphere about `focus`; returns the image (h, w, 3) uint8."""
    camera = renderer.GetActiveCamera()
    a, e = math.radians(azimuth), math.radians(elevation)
    camera.SetFocalPoint(*focus)
    camera.SetPosition(focus[0] + distance * math.cos(e) * math.cos(a), focus[1] + distance * math.cos(e) * math.sin(a), focus[2] + distance * math.sin(e))
    camera.SetViewUp(0, 0, 1)
    camera.SetViewAngle(30)
    renderer.ResetCameraClippingRange()
    window.Render()
    grab = vtk.vtkWindowToImageFilter()
    grab.SetInput(window)
    grab.Update()
    image = grab.GetOutput()
    w, h, _ = image.GetDimensions()
    return vtk_to_numpy(image.GetPointData().GetScalars()).reshape(h, w, -1)[::-1, :, :3].copy()


def to_screen(renderer, point, height):
    coordinate = vtk.vtkCoordinate()
    coordinate.SetCoordinateSystemToWorld()
    coordinate.SetValue(*point)
    x, y = coordinate.GetComputedDisplayValue(renderer)
    return x, height - y


def font(size):
    for path in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def labelled(image, renderer, items, amount=1.0, text_size=20, foot=0):
    """Draw one label per kind of piece in the margins, with a line to one piece of that kind.
    foot: pixels at the bottom to keep clear."""
    picture = Image.fromarray(image)
    draw = ImageDraw.Draw(picture)
    face = font(text_size)
    height, width = image.shape[:2]
    anchors = {}
    for key, _, _, centre, offset, _ in items:
        x, y = to_screen(renderer, centre + amount * offset, height)
        anchors.setdefault(key, []).append((x, y))
    spots = {}
    for key, points in anchors.items():  # the piece furthest from the middle of the picture
        spots[key] = max(points, key=lambda p: abs(p[0] - width / 2))
    left = sorted([k for k in spots if spots[k][0] < width / 2], key=lambda k: spots[k][1])
    right = sorted([k for k in spots if spots[k][0] >= width / 2], key=lambda k: spots[k][1])
    for side, keys in (("left", left), ("right", right)):
        step = (height - 80 - foot) / max(len(keys), 1)
        for row, key in enumerate(keys):
            y = 40 + step * (row + 0.5)
            text = full.LABELS[key]
            w = draw.textlength(text, font=face)
            x = 24 if side == "left" else width - 24 - w
            draw.text((x, y - text_size / 2), text, fill=(20, 20, 20), font=face)
            draw.line([(x + w + 6 if side == "left" else x - 6, y), spots[key]], fill=(90, 90, 90), width=1)
    return np.array(picture)


def place(actors, items, amount, disc_angle=0.0):
    """Move every piece `amount` of the way out, the disc's pieces turned by disc_angle."""
    for actor, (_, _, _, _, offset, disc) in zip(actors, items):
        transform = vtk.vtkTransform()
        transform.Translate(*(amount * offset))
        if disc and disc_angle:
            transform.RotateZ(disc_angle)
        actor.SetUserTransform(transform)


def assembled(items=None):
    items = items or pieces()
    size = (900, 900)
    images = []
    for section in (False, True):
        window, renderer, actors = scene(items, size, section)
        images.append(shot(window, renderer, -55 if not section else -90, 28 if not section else 12, 210))
    MEDIA.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(images, axis=1)).save(MEDIA / "rotating-cube-cad.png")
    print("wrote", MEDIA / "rotating-cube-cad.png")


def exploded(items=None):
    items = items or pieces()
    size = (1800, 1300)
    window, renderer, actors = scene(items, size)
    place(actors, items, 1.0)
    image = shot(window, renderer, -62, 20, 560, focus=(0, 0, 32))
    image = labelled(image, renderer, items, 1.0, text_size=26)
    Image.fromarray(image).save(MEDIA / "rotating-cube-exploded.png")
    print("wrote", MEDIA / "rotating-cube-exploded.png")


if __name__ == "__main__":
    everything = pieces()
    assembled(everything)
    exploded(everything)
