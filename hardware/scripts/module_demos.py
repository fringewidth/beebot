"""Demos of the layout B module (hardware/src/beebot/hardware/module_b.py): every robot is the same 50 mm
module, every motion comes from the disc drives under gravity, contacts and friction.

One clip per demo in runs/module_demos/<demo>.mp4, one JSON line per demo in
runs/module_demos/results.jsonl, a still per demo in runs/module_demos/stills/, then a captioned
reel hardware/handoff/media/module-demos.mp4 and a contact sheet hardware/handoff/media/module-demos-sheet.png.
Memo: hardware/handoff/module-demos.md.

    .venv/bin/python hardware/scripts/module_demos.py all          # every demo (one process each), reel, sheet
    .venv/bin/python hardware/scripts/module_demos.py lone          # one demo
    .venv/bin/python hardware/scripts/module_demos.py reel sheet    # rejoin
"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from beebot.hardware.module_b import DISC_RADIUS, DISC_THICK, HALF, PITCH, PROUD, Rig, lattice

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "module_demos"
MEDIA = ROOT / "hardware" / "handoff" / "media"
SIZE = (1280, 720)
FPS = 30
TRACK = 2 * (HALF + PROUD - DISC_THICK / 2)  # m between the rims' mid-planes


def font(size):
    for path in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


class Reel:
    """Frames to an MP4 through ffmpeg, the present caption burned into a band at the foot and a
    title in the top left corner."""

    def __init__(self, path, title=""):
        path.parent.mkdir(parents=True, exist_ok=True)
        command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SIZE[0]}x{SIZE[1]}", "-r", str(FPS), "-i", "-"]
        self.pipe = subprocess.Popen(command + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(path)], stdin=subprocess.PIPE)
        self.caption = ""
        self.title = title
        self.face = font(25)
        self.big = font(30)
        self.frames = 0
        self.last = None

    def picture(self, image):
        picture = Image.fromarray(image).convert("RGB")
        draw = ImageDraw.Draw(picture, "RGBA")
        if self.title:
            draw.rectangle([0, 0, 30 + draw.textlength(self.title, font=self.big) + 30, 56], fill=(0, 0, 0, 150))
            draw.text((30, 12), self.title, fill=(255, 255, 255, 255), font=self.big)
        if self.caption:
            lines = self.caption.split("\n")
            top = SIZE[1] - 22 - 32 * len(lines)
            draw.rectangle([0, top - 10, SIZE[0], SIZE[1]], fill=(0, 0, 0, 170))
            for k, line in enumerate(lines):
                draw.text((30, top + 32 * k), line, fill=(255, 255, 255, 255), font=self.face)
        return picture

    def add(self, image, repeat=1):
        picture = self.picture(image)
        self.last = picture
        self.pipe.stdin.write(picture.tobytes() * repeat)
        self.frames += repeat

    def close(self):
        self.pipe.stdin.close()
        self.pipe.wait()


class Camera:
    """A MuJoCo renderer feeding a Reel at FPS of sim time (times `slow`). `follow`: a body the
    camera's look-at point drifts after."""

    def __init__(self, rig, reel, lookat, distance, azimuth, elevation, follow=None, slow=1.0):
        import mujoco

        self.mujoco = mujoco
        self.renderer = mujoco.Renderer(rig.model, SIZE[1], SIZE[0])
        self.camera = mujoco.MjvCamera()
        self.camera.lookat = lookat
        self.camera.distance = distance
        self.camera.azimuth, self.camera.elevation = azimuth, elevation
        self.offset = np.array(lookat) - (rig.position(follow) if follow else 0)
        self.follow = follow
        self.reel = reel
        self.every = 1 / (FPS * slow)
        self.due = rig.data.time
        self.still_at = None
        self.still = None

    def image(self, rig):
        if self.follow:
            target = rig.position(self.follow) + self.offset
            self.camera.lookat = 0.9 * np.array(self.camera.lookat) + 0.1 * target
        self.renderer.update_scene(rig.data, camera=self.camera)
        return self.renderer.render()

    def __call__(self, rig):
        if rig.data.time >= self.due - 1e-9:
            self.due += self.every
            self.reel.add(self.image(rig))
            if self.still_at is not None and rig.data.time >= self.still_at and self.still is None:
                self.still = self.reel.last

    def pause(self, rig, seconds):
        self.reel.add(self.image(rig), int(seconds * FPS))


def wheel_turn(angle):
    """Disc angle (rad) for a lone wheel pair to yaw `angle` (rad) on the spot."""
    return angle * TRACK / (2 * DISC_RADIUS)


def box(name, pos, size, mass, rgba="0.80 0.30 0.25 1", friction=0.8, free=True):
    joint = "<freejoint/>" if free else ""
    p = " ".join(f"{v:.6g}" for v in pos)
    s = " ".join(f"{v:.6g}" for v in size)
    return f'<body name="{name}" pos="{p}">{joint}<geom name="{name}" type="box" size="{s}" mass="{mass}" rgba="{rgba}" friction="{friction} 0.005 0.0001"/></body>\n'


def mark(pos, size=(0.01, 0.01), rgba="0.9 0.2 0.2 1"):
    return f'<geom type="box" pos="{pos[0]:.6g} {pos[1]:.6g} 0.0002" size="{size[0]:.6g} {size[1]:.6g} 0.0002" rgba="{rgba}" contype="0" conaffinity="0"/>\n'


def record(name, ok, criterion, modules, summary, notes, measured):
    """One JSON line per demo in results.jsonl (replacing an earlier line for the same demo)."""
    row = {"demo": name, "ok": bool(ok), "criterion": criterion, "modules": modules}
    row.update(summary)
    row["measured"] = measured
    row["notes"] = notes
    path = OUT / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    rows = [r for r in rows if r["demo"] != name] + [row]
    order = list(DEMOS)
    rows.sort(key=lambda r: order.index(r["demo"]) if r["demo"] in order else 99)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps(row))
    return row


def finish(name, reel, camera, caption_file):
    reel.close()
    stills = OUT / "stills"
    stills.mkdir(parents=True, exist_ok=True)
    (camera.still or reel.last).save(stills / f"{name}.png")
    (OUT / f"{name}.caption.txt").write_text(caption_file)


# ---------------------------------------------------------------- demos


def lone(args):
    """One module alone: two discs are its wheels, a body edge its skid."""
    rig = Rig(lattice([{"at": (0, 0, 0), "axis": "+x"}]), scene=mark((0, -0.33), (0.03, 0.002)))
    reel = Reel(OUT / "lone.mp4", "1. One module alone: a two-wheeled robot")
    cam = Camera(rig, reel, [0.0, -0.18, 0.02], 0.65, 60, -22, slow=1.0)
    cam.still_at = 2.2
    reel.caption = "One module stands on the rims of its two discs (52 mm, 1 mm proud of the body).\nBoth discs turn the same way: it rolls at 0.15 m/s (55 rpm)."
    rig.run(0.5, cam)
    start = rig.position("m0b")
    low = [1.0]

    def watch(r):
        cam(r)
        corners = np.array([[a, b, c] for a in (-HALF, HALF) for b in (-HALF, HALF) for c in (-HALF, HALF)])
        z = r.data.xpos[r.model.body("m0b").id][2] + corners @ r.data.xmat[r.model.body("m0b").id].reshape(3, 3)[2]
        low[0] = min(low[0], float(z.min()))

    rig.command(speeds={"0p": 5.8, "0n": 5.8}, ramp=0.4)
    rig.run(2.0, watch)
    rig.command(speeds={"0p": 0.0, "0n": 0.0}, ramp=0.4)
    rig.run(0.8, cam)
    leg1 = rig.position("m0b") - start
    yaw0 = rig.heading("m0b")
    reel.caption = "The discs turn opposite ways: it turns on the spot, a quarter turn."
    turn = wheel_turn(math.pi / 2)
    rig.command(targets={"0p": rig.angle("0p") - turn, "0n": rig.angle("0n") + turn}, seconds=1.2)
    rig.run(1.6, watch)
    yaw = math.degrees(rig.heading("m0b") - yaw0)
    yaw = (yaw + 180) % 360 - 180
    mid = rig.position("m0b")
    reel.caption = "And rolls on in the new direction."
    rig.command(speeds={"0p": 5.8, "0n": 5.8}, ramp=0.4)
    rig.run(1.5, watch)
    rig.command(speeds={"0p": 0.0, "0n": 0.0}, ramp=0.4)
    rig.run(0.8, watch)
    leg2 = rig.position("m0b") - mid
    plated = Rig(lattice([{"at": (0, 0, 0), "axis": "+x", "z0": DISC_RADIUS + 0.0005}]), plates=True)  # spec 3.5's fix: side plates 1.2 mm proud
    plated.run(0.3)
    p0 = plated.position("m0b")
    plated.command(speeds={"0p": 5.8, "0n": 5.8}, ramp=0.4)
    plated.run(2.0)
    plated_roll = float(np.linalg.norm((plated.position("m0b") - p0)[:2]))
    distance1 = float(np.linalg.norm(leg1[:2]))
    distance2 = float(np.linalg.norm(leg2[:2]))
    ok = distance1 >= 0.25 and abs(abs(yaw) - 90) <= 10 and distance2 >= 0.2
    measured = {"leg1_m": round(distance1, 3), "turn_deg": round(yaw, 1), "leg2_m": round(distance2, 3), "body_min_clearance_mm": round(low[0] * 1000, 2), "with_plates_rolled_m": round(plated_roll, 3), "with_plates_peak_nm": round(float(plated.peak.max()), 3)}
    reel.caption = f"Rolled {distance1 * 100:.0f} cm, turned {abs(yaw):.0f} degrees, rolled {distance2 * 100:.0f} cm (measured)."
    cam.pause(rig, 1.5)
    finish("lone", reel, cam, "One module alone is a two-wheeled robot: it rolls on its two discs and turns on the spot.")
    criterion = "leg 1 >= 25 cm, on-the-spot turn 90 +/- 10 degrees, leg 2 >= 20 cm"
    notes = ("Flat floor, no rolling resistance, rim friction 1.0. The body tilts onto an edge (the skid) when it accelerates (clearance in measured; below 0 means it touched). "
             "With spec 3.5's candidate fix modelled (side plates 1.2 mm proud, the pitch that keeps rims clear of neighbours), the module stands on its bottom plate and rolls the distance in with_plates_rolled_m.")
    return record("lone", ok, criterion, 1, rig.summary(), notes, measured)


def drive(rig, wheels, speed, until, watch, limit=10.0, ramp=0.4):
    """Speed mode on `wheels` ({motor: sign}) until until(rig) is true or `limit` s pass, then stop."""
    rig.command(speeds={w: sign * speed for w, sign in wheels.items()}, ramp=ramp)
    t = 0.0
    while not until(rig) and t < limit:
        rig.run(0.01, watch)
        t += 0.01
    rig.command(speeds={w: 0.0 for w in wheels}, ramp=ramp)
    rig.run(ramp + 0.2, watch)
    return t


def forward_of(rig, body):
    """Unit vector a wheel module rolls along for negative disc speeds (its axis turned +90 degrees)."""
    x = rig.data.xmat[rig.model.body(body).id].reshape(3, 3)[:, 0]
    return np.array([-x[1], x[0]]) / max(np.hypot(x[0], x[1]), 1e-9)


def cart(args):
    """Four modules: two wheel modules in a row, two deck modules on top, a 300 g load (task T2)."""
    cells = [
        {"at": (0, 0, 0), "axis": "+x"},
        {"at": (0, 1, 0), "axis": "+x", "parent": 0},
        {"at": (0, 0, 1), "axis": "+z", "parent": 0},
        {"at": (0, 1, 1), "axis": "+z", "parent": 1},
    ]
    top = DISC_RADIUS + PITCH + HALF + PROUD
    goal = np.array([-0.25, 0.35 + PITCH / 2])
    scene = box("load", (0, PITCH / 2, top + 0.02 + 0.0005), (0.045, 0.06, 0.02), 0.3) + mark((0, 0.35 + PITCH / 2), (0.05, 0.002)) + mark(goal, (0.015, 0.015))
    rig = Rig(lattice(cells), scene=scene)
    reel = Reel(OUT / "cart.mp4", "2. Cart: four modules carry 300 g")
    cam = Camera(rig, reel, [-0.08, 0.25, 0.03], 0.95, 135, -28)
    cam.still_at = 3.0
    reel.caption = "Two modules in a row are the wheels (four driven discs); two on top, axes upright, are the deck.\nIt carries a 300 g box 35 cm, turns a quarter on the spot and stops on the red mark."
    rig.run(0.5, cam)
    p0 = rig.position("load")
    wheels = {"0p": -1, "0n": -1, "1p": -1, "1n": -1}
    centre = lambda r: (r.position("m0b")[:2] + r.position("m1b")[:2]) / 2
    drive(rig, wheels, 4.0, lambda r: centre(r)[1] >= 0.35 + PITCH / 2 - 0.025, cam)
    h0 = rig.heading("m0b")
    reel.caption = "Turning on the spot: the left discs back, the right discs forward (skid steer)."
    drive(rig, {"0p": -1, "1p": -1, "0n": 1, "1n": 1}, 2.0, lambda r: abs(r.heading("m0b") - h0) >= math.radians(86), cam, ramp=0.25)
    turn = math.degrees(rig.heading("m0b") - h0)
    reel.caption = "Then on to the mark."
    drive(rig, wheels, 4.0, lambda r: (goal - centre(r)) @ forward_of(r, "m0b") <= 0.022, cam)
    miss = float(np.linalg.norm(centre(rig) - goal))
    carried = rig.position("load") - p0
    on = rig.position("load")[2] > top
    ok = on and miss <= 0.03 and abs(turn - 90) <= 10
    reel.caption = f"Stopped {miss * 100:.1f} cm from the mark after a {turn:.0f} degree turn; the box stayed on (measured)."
    cam.pause(rig, 1.5)
    finish("cart", reel, cam, "Cart: two wheel modules and two deck modules carry a 300 g box, turn on the spot and stop at a mark.")
    measured = {"miss_cm": round(miss * 100, 2), "turn_deg": round(turn, 1), "load_still_on": bool(on), "load_moved_cm": round(float(np.linalg.norm(carried[:2])) * 100, 1)}
    notes = ("Stops by feedback on the tracked position and the IMU yaw. A first try carried the box straight on the wheel modules' tops: the top rims (1 mm proud) "
             "turned against the box and conveyed it off the back, so the deck is two modules with upright axes. Deck neighbours' rims overlap the wheel modules (contacts excluded).")
    return record("cart", ok, "box still on the deck, turn 90 +/- 10 degrees, stop within 3 cm of the mark", 4, rig.summary(), notes, measured)


def conveyor(args):
    """A row of five modules whose top rims carry a box; two modules under the ends are legs (task T7)."""
    cells = [{"at": (0, 0, 1), "axis": "+y", "z0": DISC_RADIUS + 0.0}]
    for k in range(1, 5):
        cells.append({"at": (k, 0, 1), "axis": "+y", "parent": k - 1})
    cells.append({"at": (0, 0, 0), "axis": "+z", "parent": 0})
    cells.append({"at": (4, 0, 0), "axis": "+z", "parent": 4})
    cells[0]["z0"] = HALF + PROUD
    rollers = [f"{k}{s}" for k in range(5) for s in "pn"]
    top = HALF + PROUD + PITCH + DISC_RADIUS
    scene = box("parcel", (0.25 * PITCH, 0, top + 0.015 + 0.0005), (0.06, 0.035, 0.015), 0.1, rgba="0.75 0.62 0.40 1")
    rig = Rig(lattice(cells), scene=scene)
    reel = Reel(OUT / "conveyor.mp4", "3. Conveyor: five modules as rollers")
    cam = Camera(rig, reel, [0.1, 0, 0.06], 0.65, 120, -25)
    cam.still_at = 2.5
    reel.caption = "Five modules in a row, axes across the belt line, stand on two more as legs.\nThe top rims of all ten discs turn together and carry a 100 g parcel along."
    rig.run(0.5, cam)
    x0 = rig.position("parcel")[0]
    rig.command(speeds={r: 2.5 for r in rollers}, ramp=0.4)
    t = 0.0
    while rig.position("parcel")[0] < 3.6 * PITCH and t < 8:
        rig.run(0.01, cam)
        t += 0.01
    far = rig.position("parcel")[0]
    reel.caption = "Reversed: the same rollers send it back (sorting either way)."
    rig.command(speeds={r: -2.5 for r in rollers}, ramp=0.4)
    rig.run(2.0, cam)
    rig.command(speeds={r: 0.0 for r in rollers}, ramp=0.3)
    rig.run(0.6, cam)
    back = rig.position("parcel")[0]
    legs = rig.position("m0b")[:2]
    moved = far - x0
    ok = moved >= 0.15 and (far - back) >= 0.05 and rig.position("parcel")[2] > top
    reel.caption = f"Carried {moved * 100:.0f} cm out and {(far - back) * 100:.0f} cm back (measured); the conveyor itself stayed put."
    cam.pause(rig, 1.5)
    finish("conveyor", reel, cam, "Conveyor: a row of modules carries a parcel on the rims of their discs, then reverses.")
    measured = {"carried_cm": round(moved * 100, 1), "returned_cm": round((far - back) * 100, 1), "frame_drift_mm": round(float(np.linalg.norm(legs - rig.position("m0b")[:2])) * 1000, 2)}
    notes = "Rims at 2.5 rad/s, about 6.5 cm/s at the rim. The parcel spans both rims of each module (70 mm wide). Leg modules stand on their lower disc faces, held."
    return record("conveyor", ok, "parcel carried >= 15 cm along the row and >= 5 cm back, stays on", 7, rig.summary(), notes, measured)


def crane(args):
    """Slewing, luffing crane with a level platform: base slew, luff, level-keeping wrist (5 modules)."""
    cells = [
        {"at": (0, 0, 0), "axis": "+z", "mount": "n", "z0": HALF + PROUD},
        {"at": (0, 0, 1), "axis": "+y", "parent": 0},  # luff joint: its +y disc
        {"at": (0, 1, 1), "axis": "+z", "parent": 1},  # offset on the luff disc
        {"at": (1, 1, 1), "axis": "+y", "parent": 2},  # wrist: its +y disc keeps the platform level
        {"at": (1, 2, 1), "axis": "+z", "parent": 3},  # platform; the load sits on its top disc
    ]
    plate = HALF + PROUD + PITCH + HALF + PROUD
    scene = box("load", (PITCH, 2 * PITCH, plate + 0.02 + 0.0005), (0.02, 0.02, 0.02), 0.1, rgba="0.30 0.35 0.75 1")
    rig = Rig(lattice(cells), scene=scene)
    reel = Reel(OUT / "crane.mp4", "4. Crane: slew, luff and keep the load level")
    cam = Camera(rig, reel, [0.02, 0.04, 0.09], 0.62, 150, -18)
    cam.still_at = 4.5
    reel.caption = "Five modules: the bottom one is bolted down and slews the rest; the next luffs the jib;\nthe wrist turns the other way so the platform stays level. Load 100 g, resting on the platform by friction."
    rig.command(targets={"1p": math.radians(20), "3p": -math.radians(20)}, seconds=0.8)
    rig.run(1.2, cam)
    z0 = rig.position("load")[2]
    lift = math.radians(40)
    reel.caption = "Luffing up 60 degrees, the wrist keeping the platform level."
    rig.command(targets={"1p": -lift, "3p": lift}, seconds=2.5)
    rig.run(2.8, cam)
    high = rig.position("load")[2]
    reel.caption = "Slewing a quarter turn."
    a0 = math.atan2(*rig.position("load")[[1, 0]])
    rig.command(targets={"0n": math.pi / 2}, seconds=2.5)
    rig.run(2.8, cam)
    a1 = math.atan2(*rig.position("load")[[1, 0]])
    reel.caption = "Luffing down and setting the load on its new spot."
    rig.command(targets={"1p": math.radians(20), "3p": -math.radians(20)}, seconds=2.5)
    rig.run(2.9, cam)
    raised = high - z0
    slewed = math.degrees((a1 - a0 + math.pi) % (2 * math.pi) - math.pi)
    on = abs(rig.position("load")[2] - z0) < 0.005 and np.linalg.norm(rig.position("load")[:2] - rig.position("m4b")[:2]) < 0.02
    ok = raised >= 0.04 and abs(abs(slewed) - 90) <= 10 and on
    reel.caption = f"Raised {raised * 100:.1f} cm, slewed {abs(slewed):.0f} degrees, load still on the platform (measured)."
    cam.pause(rig, 1.5)
    finish("crane", reel, cam, "Crane: five modules slew, luff and keep a platform level while moving a 100 g load.")
    measured = {"raised_cm": round(raised * 100, 2), "slewed_deg": round(slewed, 1), "load_on": bool(on)}
    notes = "Bottom module bolted to the floor by its lower disc (mount). The load is not fixed to the platform: friction 0.8 against the disc face."
    return record("crane", ok, "load raised >= 4 cm, slewed 90 +/- 10 degrees, still on the platform at the end", 5, rig.summary(), notes, measured)


def bridge(args):
    """A five-module beam across a three-module gap holding 200 g (task T8), and a three-module shelf."""
    cells = [{"at": (k, 0, 0), "axis": "+y", **({"parent": k - 1} if k else {"z0": 0.06 + DISC_RADIUS + 0.0005})} for k in range(5)]
    base = len(cells)
    cells += [
        {"at": (2, 4, 0), "axis": "+z", "z0": HALF + PROUD},
        {"at": (2, 4, 1), "axis": "+z", "parent": base},
        {"at": (2, 4, 2), "axis": "+z", "parent": base + 1},
        {"at": (2, 3, 2), "axis": "+z", "parent": base + 2},
        {"at": (2, 2, 2), "axis": "+z", "parent": base + 3},
        {"at": (2, 1, 2), "axis": "+z", "parent": base + 4},
        {"at": (2, 5, 0), "axis": "+z", "parent": base},
        {"at": (2, 6, 0), "axis": "+z", "parent": base + 6},
        {"at": (2, 3, 0), "axis": "+z", "parent": base},
    ]
    cells[base]["at"] = (2, 4, 0)
    # The bridge's ends rest on two blocks, 3 modules (150 mm) apart.
    gap = 3 * PITCH
    scene = box("abut_l", (HALF - 0.04, 0, 0.03), (0.04, 0.05, 0.03), 1.0, rgba="0.55 0.55 0.58 1", free=False)
    scene += box("abut_r", (4 * PITCH - HALF + 0.04, 0, 0.03), (0.04, 0.05, 0.03), 1.0, rgba="0.55 0.55 0.58 1", free=False)
    deck = 0.06 + DISC_RADIUS + 0.0005 + DISC_RADIUS
    scene += box("weight", (2 * PITCH, 0, deck + 0.015 + 0.002), (0.02, 0.02, 0.015), 0.2, rgba="0.30 0.30 0.32 1")
    shelf_top = HALF + PROUD + 2 * PITCH + HALF + PROUD
    scene += box("weight2", (2 * PITCH, 1 * PITCH, shelf_top + 0.015 + 0.002), (0.02, 0.02, 0.015), 0.2, rgba="0.30 0.30 0.32 1")
    del gap
    rig = Rig(lattice(cells), scene=scene)
    reel = Reel(OUT / "bridge.mp4", "5. Structures: a bridge and a shelf")
    cam = Camera(rig, reel, [0.1, 0.12, 0.08], 0.75, 205, -20)
    cam.still_at = 2.0
    reel.caption = "Front: five modules span a 150 mm gap between two blocks; a 200 g weight is set on mid-span.\nBack: a column of three on a three-module foot holds a shelf of three, 200 g set on its tip."
    rig.run(0.6, cam)
    rig.run(2.5, cam)
    sag = rig.position("m0b")[2] - rig.position("m2b")[2]
    tip = rig.position(f"m{base + 5}b")[2] - rig.position(f"m{base + 2}b")[2]
    both = rig.position("weight")[2] > deck and rig.position("weight2")[2] > shelf_top
    summary = rig.summary()
    ok = both and summary["bond_moment_nm"] <= 0.8
    reel.caption = f"Both hold (rigid bonds). Largest bond bending moment {summary['bond_moment_nm']:.2f} N m against the 0.8 N m of spec 5.5 (measured)."
    cam.pause(rig, 2.0)
    finish("bridge", reel, cam, "Structures: five modules bridge a 150 mm gap under 200 g; six modules make a shelf holding 200 g.")
    measured = {"bridge_sag_mm": round(sag * 1000, 3), "shelf_drop_mm": round(tip * 1000, 3), "weights_held": bool(both)}
    notes = "Bonds are rigid, so the structures cannot fail here; what counts is the bond moment (measured, 50 ms average) against spec 5.5's 0.8 N m. Discs held by their servos."
    return record("bridge", ok, "both weights held and the largest bond bending moment <= 0.8 N m (spec 5.5)", len(cells), summary, notes, measured)


def screwdriver(args):
    """A bolted frame of three modules; the fourth hangs from the head's lower disc and turns a
    knob by its square body in the knob's square socket (property 5.12: a screw that fits a face)."""
    cells = [
        {"at": (0, 0, 1), "axis": "+z", "mount": "b", "z0": HALF + PROUD + 0.025},  # bolted to a post
        {"at": (1, 0, 1), "axis": "+z", "parent": 0},  # head: axis upright, lower disc is the drive
        {"at": (1, 0, 0), "axis": "+x", "parent": 1},  # the bit: a whole module turned by the head
    ]
    resist = 0.3
    bit_z = HALF + PROUD + 0.025
    floor = bit_z - HALF - 0.004  # socket floor 3 mm under the bit's rims
    wall = HALF + 0.0015
    socket = f'<body name="knob" pos="{PITCH} 0 0"><joint name="knob" type="hinge" axis="0 0 1" damping="0.01" armature="0.001"/>'
    socket += f'<geom type="cylinder" size="0.02 {(floor - 0.001) / 2:.5f}" pos="0 0 {(floor + 0.001) / 2:.5f}" rgba="0.60 0.25 0.25 1" mass="0.1"/>'
    for sy in (1, -1):  # two lugs either side of the bit: a slotted head
        socket += f'<geom type="box" size="0.018 0.003 0.008" pos="0 {sy * (wall + 0.003):.5f} {floor + 0.008:.5f}" rgba="0.60 0.25 0.25 1" mass="0.01" friction="0.5 0.005 0.0001"/>'
    socket += f'<geom type="box" size="0.012 0.002 0.0005" pos="0 0 {floor + 0.0005:.5f}" rgba="1 1 1 1" contype="0" conaffinity="0" mass="0"/>'
    socket += "</body>\n"
    post = HALF + PROUD + 0.025 + PITCH - HALF
    scene = socket + box("post", (-0.012, 0, post / 2), (0.01, 0.02, post / 2), 1, rgba="0.55 0.55 0.58 1", free=False)
    rig = Rig(lattice(cells), scene=scene)
    reel = Reel(OUT / "screwdriver.mp4", "6. Screwdriver: continuous turning against a stiff knob")
    cam = Camera(rig, reel, [0.04, 0, 0.06], 0.45, 140, -30)
    cam.still_at = 4.0
    reel.caption = f"Two modules on a bench post; the second is the head, and its lower disc turns the third, whose square\nbody sits between the two lugs of a knob's slotted head. The knob takes {resist} N m to turn (a stiff valve)."
    rig.run(0.5, cam)
    joint = rig.model.joint("knob")
    k0 = rig.data.qpos[joint.qposadr[0]]
    dof = joint.dofadr[0]

    def watch(r):  # the valve's dry friction, smoothed over 0.05 rad/s (MuJoCo's frictionloss went unstable against the bit)
        cam(r)
        r.data.qfrc_applied[dof] = -resist * math.tanh(r.data.qvel[dof] / 0.05)

    rig.command(speeds={"1n": 2.5}, ramp=0.5)
    rig.run(8.5, watch)
    rig.command(speeds={"1n": 0.0}, ramp=0.3)
    rig.run(0.5, watch)
    turns = (rig.data.qpos[joint.qposadr[0]] - k0) / (2 * math.pi)
    ok = abs(turns) >= 3
    reel.caption = f"{abs(turns):.1f} turns of the knob (measured); the drive pays {resist} N m plus drag, against 0.4 N m continuous."
    cam.pause(rig, 1.5)
    finish("screwdriver", reel, cam, f"Screwdriver: a module turns a knob that takes {resist} N m, three turns and more.")
    measured = {"knob_turns": round(float(turns), 2), "knob_resistance_nm": resist}
    notes = ("The bit is a whole module: its square body drives the knob's two lugs like a blade in a slotted screw (positive engagement, 1.5 mm clearance a side). The first module is bolted to a bench post (a fixture); "
             "a frame standing free on the floor would have to resist 0.3 N m of reaction by friction. The lugs sweep 37 mm from the knob axis, so no module may sit beside the bit. "
             "The knob's resistance is a smoothed Coulomb torque applied each step (MuJoCo's frictionloss went unstable against the bit). The bond moment here is not "
             "trustworthy: the rigid square bit drives the lugs on one corner at a time with up to about 44 N of mostly radial contact force, which a compliant real bit would not see.")
    return record("screwdriver", ok, "knob turned >= 3 turns against 0.3 N m", 3, rig.summary(), notes, measured)


def crank(args):
    """Crank and slider (a piston): a bolted module turns a crank; a rod of three modules drives a
    fourth that slides in a channel. The crank pin and piston pin are discs in zero-torque mode."""
    cells = [
        {"at": (0, 0, 2), "axis": "+y", "mount": "b", "z0": HALF},  # frame, bolted to a stand; its +y disc is the crank
        {"at": (0, 1, 2), "axis": "+z", "parent": 0},  # crank hub, on the axis
        {"at": (1, 1, 2), "axis": "+y", "parent": 1},  # crank arm; its +y disc is the crank pin
        {"at": (1, 2, 2), "axis": "+z", "parent": 2},  # rod, three modules
        {"at": (2, 2, 2), "axis": "+z", "parent": 3},
        {"at": (3, 2, 2), "axis": "+y", "parent": 4},  # rod end; its +y disc is the piston pin
        {"at": (3, 3, 2), "axis": "+z", "parent": 5},  # slider (the piston)
    ]
    zc = HALF + 2 * PITCH
    stand = HALF + 2 * PITCH - HALF
    scene = box("stand", (0, 0, stand / 2), (0.015, 0.015, stand / 2), 1, rgba="0.55 0.55 0.58 1", free=False)
    for sz in (1, -1):
        scene += box(f"rail{sz}", (2.5 * PITCH, 3 * PITCH, zc + sz * (HALF + 0.0004 + 0.003)), (2.4 * PITCH, HALF, 0.003), 1, rgba="0.55 0.55 0.58 0.9", free=False)
    for px in (0.0, 5.0):
        scene += box(f"rail_post{px:.0f}", (px * PITCH, 3 * PITCH, zc / 2), (0.004, 0.004, zc / 2 + HALF), 1, rgba="0.55 0.55 0.58 1", free=False)
    rig = Rig(lattice(cells), scene=scene)
    for name in ("2p", "5p"):
        rig.command(free=[name])
    reel = Reel(OUT / "crank.mp4", "7. Crank and slider: a piston from seven modules")
    cam = Camera(rig, reel, [0.09, 0.08, 0.12], 0.75, 205, -18)
    cam.still_at = 3.0
    reel.caption = "The frame module (bolted to a stand) turns the crank; three modules make the connecting rod; the last one\nslides between two rails as the piston. The crank pin and piston pin are discs left free (motor at zero torque)."
    rig.run(0.5, cam)
    xs = []
    angle0 = rig.angle("0p")
    rig.command(speeds={"0p": 2.5}, ramp=0.5)

    def watch(r):
        cam(r)
        xs.append(r.position("m6b")[0])

    rig.run(8.5, watch)
    rig.command(speeds={"0p": 0.0}, ramp=0.4)
    rig.run(0.8, watch)
    turns = (rig.angle("0p") - angle0) / (2 * math.pi)
    stroke = max(xs) - min(xs)
    ok = turns >= 3 and stroke >= 0.09
    reel.caption = f"{turns:.1f} crank turns, piston stroke {stroke * 1000:.0f} mm (two crank radii = {2 * PITCH * 1000:.0f} mm; measured)."
    cam.pause(rig, 1.5)
    finish("crank", reel, cam, "Crank and slider: one module turns a crank, a three-module rod drives a fourth module back and forth between rails.")
    measured = {"crank_turns": round(turns, 2), "stroke_mm": round(stroke * 1000, 1)}
    notes = ("The closed loop closes through contact: the piston module is held square by two rails (fixtures), no weld. The two passive pins are motors at zero torque, "
             "which a worm drive (spec 4.4) could not do: with a worm they would have to be driven in step with the crank.")
    return record("crank", ok, "crank >= 3 turns and piston stroke >= 90 mm", len(cells), rig.summary(), notes, measured)


def door(args):
    """One module pushes a door open against its closer spring (task T3)."""
    width, height, closer = 0.16, 0.12, 0.04
    hinge = (0.02, 0.12)
    scene = f'<body name="door" pos="{hinge[0]} {hinge[1]} 0.002"><joint name="door" type="hinge" axis="0 0 1" stiffness="{closer}" damping="0.01" range="-5 120" limited="true"/>'
    scene += f'<geom type="box" size="{width / 2} 0.002 {height / 2}" pos="{width / 2} 0.002 {height / 2}" mass="0.15" rgba="0.45 0.35 0.25 1" friction="0.4 0.005 0.0001"/>'
    scene += f'<geom type="cylinder" size="0.004 {height / 2}" pos="0 0 {height / 2}" rgba="0.3 0.3 0.3 1" contype="0" conaffinity="0"/></body>\n'
    scene += box("jamb", (hinge[0] - 0.012, hinge[1] + 0.01, 0.07), (0.008, 0.008, 0.07), 1, rgba="0.6 0.6 0.6 1", free=False)
    scene += box("stop", (hinge[0] + width + 0.012, hinge[1] + 0.012, 0.07), (0.008, 0.008, 0.07), 1, rgba="0.6 0.6 0.6 1", free=False)
    rig = Rig(lattice([{"at": (0, 0, 0), "axis": "+x"}]), scene=scene)
    adr = rig.model.jnt_qposadr[rig.model.body("m0b").jntadr[0]]
    rig.data.qpos[adr:adr + 2] = [hinge[0] + 0.11, -0.05]
    reel = Reel(OUT / "door.mp4", "8. Door: one module pushes a door open")
    cam = Camera(rig, reel, [0.10, 0.10, 0.04], 0.70, 235, -30)
    cam.still_at = 3.5
    reel.caption = f"A 150 g door on a closer spring ({closer} N m per radian). One module drives at it and keeps driving:\nthe door swings open and the module passes through."
    rig.run(0.5, cam)
    joint = rig.model.joint("door")
    angles = []

    def watch(r):
        cam(r)
        angles.append(math.degrees(r.data.qpos[joint.qposadr[0]]))

    drive(rig, {"0p": -1, "0n": -1}, 4.0, lambda r: r.position("m0b")[1] > hinge[1] + 0.12, watch, limit=8.0)
    widest = max(abs(a) for a in angles)
    through = rig.position("m0b")[1] > hinge[1] + 0.05
    ok = widest >= 60 and through
    reel.caption = f"Opened to {widest:.0f} degrees and through (measured); the closer shuts it behind."
    rig.run(1.5, watch)
    finish("door", reel, cam, "Door: one module pushes a sprung door open and drives through.")
    measured = {"door_max_deg": round(widest, 1), "module_through": bool(through)}
    notes = "Push by traction of the two rims only (module weight 1.18 N, rim friction 1.0); the door is a scene object on a sprung hinge."
    return record("door", ok, "door opened >= 60 degrees and the module drove through", 1, rig.summary(), notes, measured)


def walker(args):
    """Quadruped with four rotating legs (RHex style, one twist joint per leg): 13 modules."""
    cells = [
        {"at": (0, 0, 0), "axis": "+y", "z0": PITCH + DISC_RADIUS + 0.0005},  # 0 front body: its discs drive the front legs
        {"at": (-1, 0, 0), "axis": "+y", "parent": 0},  # 1 spacer
        {"at": (-2, 0, 0), "axis": "+y", "parent": 1},  # 2 spacer
        {"at": (-3, 0, 0), "axis": "+y", "parent": 2},  # 3 back body: its discs drive the back legs
        {"at": (0, 1, 0), "axis": "+x", "parent": 0},  # 4 front left hub
        {"at": (0, 1, -1), "axis": "+y", "parent": 4},  # 5 front left leg
        {"at": (0, -1, 0), "axis": "+x", "parent": 0},  # 6 front right hub
        {"at": (0, -1, -1), "axis": "+y", "parent": 6},  # 7 front right leg
        {"at": (-3, 1, 0), "axis": "+x", "parent": 3},  # 8 back left hub
        {"at": (-3, 2, 0), "axis": "+x", "parent": 8},  # 9 back left hub, outer
        {"at": (-3, 2, -1), "axis": "+y", "parent": 9},  # 10 back left leg
        {"at": (-3, -1, 0), "axis": "+x", "parent": 3},  # 11 back right hub
        {"at": (-3, -2, 0), "axis": "+x", "parent": 11},  # 12 back right hub, outer
        {"at": (-3, -2, -1), "axis": "+y", "parent": 12},  # 13 back right leg
    ]
    rig = Rig(lattice(cells))
    legs = {"0p": 0.25, "0n": 0.75, "3p": 0.0, "3n": 0.5}  # wave gait: one leg swings at a time
    period, duty, sweep = 4.0, 0.75, math.radians(20)
    reel = Reel(OUT / "walker.mp4", "9. Walker: four rotating legs, 14 modules")
    cam = Camera(rig, reel, [-0.03, 0.0, 0.05], 0.80, 140, -20, follow="m1b")
    cam.still_at = 6.0
    reel.caption = "Body of four modules; each leg is two or three modules on a body disc and turns all the way round (RHex style).\nWave gait: each leg pushes slowly through 40 degrees, then swings over the top; three feet are always down."
    rig.run(0.5, cam)

    # Buehler clock with smoothed corners: slow stance through 2 x sweep, fast swing over the top,
    # the rate changing over 5 % of a cycle at each corner (a step in rate saturates the drive).
    grid = np.linspace(0.0, 1.0, 4001)
    edge = 0.05

    def bump(u):
        rise = np.clip((u - (duty - edge / 2)) / edge, 0, 1)
        fall = np.clip((u - (1 - edge / 2)) / edge, 0, 1)
        rise, fall = rise * rise * (3 - 2 * rise), fall * fall * (3 - 2 * fall)
        early = np.clip((u + edge / 2) / edge, 0, 1)  # the fall wraps round to the start of the cycle
        early = 1 - early * early * (3 - 2 * early)
        return np.clip(rise - fall + np.where(u < edge / 2, early, 0.0), 0, 1)

    slow = 2 * sweep / duty
    weight = bump(grid)
    fast = (2 * math.pi - slow) / np.trapezoid(weight, grid)
    rate = slow + fast * weight
    angle = np.concatenate([[0.0], np.cumsum((rate[1:] + rate[:-1]) / 2 * np.diff(grid))])
    angle -= np.interp(duty / 2, grid, angle)

    def clock(t, phase):
        u = t / period + phase
        k, u = math.floor(u), u - math.floor(u)
        return 2 * math.pi * k + float(np.interp(u, grid, angle)), float(np.interp(u, grid, rate)) / period

    t0 = rig.data.time + 1.0
    start = {leg: clock(0.0, phase)[0] for leg, phase in legs.items()}
    reel.caption = "Legs to their starting phases."
    rig.command(targets=start, seconds=0.9)
    rig.run(1.0, cam)
    x0 = rig.position("m1b")[0]
    for leg, phase in legs.items():
        rig.follow(leg, lambda t, phase=phase: clock(t - t0, phase))
    low = [1.0]

    def watch(r):
        cam(r)
        low[0] = min(low[0], float(min(r.position(f"m{k}b")[2] for k in range(4))))

    reel.caption = "Walking: four gait cycles."
    rig.run(4 * period, watch)
    for leg in legs:
        rig.follow(leg, None)
    rig.run(0.5, watch)
    walked = rig.position("m1b")[0] - x0
    ok = walked >= 0.15 and low[0] > HALF + 0.002
    reel.caption = f"Walked {walked * 100:.0f} cm in four cycles ({4 * period:.1f} s); body never down (measured)." if ok else f"Walked {walked * 100:.0f} cm in {4 * period:.1f} s (measured)."
    cam.pause(rig, 1.5)
    finish("walker", reel, cam, "Walker: a 14-module quadruped whose four legs turn all the way round, walking in a wave gait.")
    measured = {"walked_cm": round(walked * 100, 1), "body_lowest_mm": round(low[0] * 1000, 1)}
    notes = ("Back legs pass 0.4 N m at touchdown: the heavier back (outer hubs) sags and the swinging foot meets the floor before the stance angle. "
             "Feet are the lower rims of the leg modules (curved, rubber). The back legs sit one module further out than the front legs so that "
             "a leg never meets the other side's hub or leg (a leg sweeps 82 mm from its axis); with a three-module body the front legs clipped the back hubs by 2.5 mm and stalled at 0.8 N m.")
    return record("walker", ok, "body moves >= 15 cm forward in four cycles, body module centres stay > 27 mm up", len(cells), rig.summary(), notes, measured)


def arm(args):
    """Pick and place (task T1) with a swing-arm: base yaw, a pendulum lift, a jaw of a moving and a
    fixed finger. 11 modules."""
    z0 = HALF + 0.006
    cells = [
        {"at": (0, 0, 0), "axis": "+z", "mount": "n", "z0": z0},  # 0 base: bolted by its lower disc; the body yaws on it
        {"at": (0, 0, 1), "axis": "+x", "parent": 0},  # 1 tower
        {"at": (0, 0, 2), "axis": "+x", "parent": 1},  # 2 tower
        {"at": (0, 1, 2), "axis": "+x", "parent": 2},  # 3 lift: its +x disc swings everything below
        {"at": (1, 1, 2), "axis": "+y", "parent": 3},  # 4 offset on the lift disc
        {"at": (1, 1, 1), "axis": "+y", "parent": 4},  # 5 wrist: its +y disc swings the finger
        {"at": (1, 2, 1), "axis": "+x", "parent": 5},  # 6 finger offset
        {"at": (1, 2, 0), "axis": "+x", "parent": 6},  # 7 moving finger; its +x disc face is the pad
        {"at": (2, 1, 1), "axis": "+x", "parent": 5},  # 8 fixed finger, three modules round
        {"at": (2, 1, 0), "axis": "+x", "parent": 8},  # 9
        {"at": (2, 2, 0), "axis": "+x", "parent": 9},  # 10 fixed finger; its -x disc face is the pad
    ]
    w = 0.014
    pad = 2 * PITCH - HALF  # the fixed pad's face, x at yaw 0
    bx = pad - 0.003 - w / 2
    pick = np.array([-2 * PITCH, bx])  # the box's start: the jaw's spot after a quarter turn of yaw
    scene = box("block", (pick[0], pick[1], 0.010 + 0.0002), (0.015, w / 2, 0.010), 0.04, rgba="0.85 0.25 0.20 1", friction=0.9)  # 14 mm across the jaw
    scene += mark((bx, 2 * PITCH), (0.012, 0.02), rgba="0.2 0.6 0.9 1")
    rig = Rig(lattice(cells), scene=scene)
    reel = Reel(OUT / "arm.mp4", "11. Arm: pick and place with a two-finger jaw")
    cam = Camera(rig, reel, [-0.03, 0.06, 0.05], 0.65, 340, -22)
    cam.still_at = 13.5
    reel.caption = "Eleven modules: a bolted base that turns (yaw), a tower, a lift joint that swings the hand like a pendulum,\nand a jaw: one finger swung by a disc, the other fixed. A 14 x 30 x 20 mm, 40 g block (red) is to go to the blue mark."
    rig.run(0.6, cam)
    lift, open_, yaw, squeeze = math.radians(24), math.radians(20), math.pi / 2, 0.3
    log = {}

    def go(targets, seconds, caption=None, extra=0.3, caps=None):
        if caption:
            reel.caption = caption
        rig.command(targets=targets, seconds=seconds, caps=caps)
        rig.run(seconds + extra, cam)

    go({"3p": lift}, 1.5, "Lift (the pendulum joint swings the hand up) and open the jaw.")
    go({"5p": open_}, 1.0)
    go({"0n": -yaw}, 3.0, "Turn a quarter to the block.")  # the base hangs on its bolted disc: body angle = -joint angle
    go({"3p": 0.0}, 1.5, "Lower around the block.")
    go({"5p": -0.3}, 1.2, f"Close with the finger's current limit at {squeeze} N m: it pinches the block against the fixed finger (friction only).", caps={"5p": squeeze})
    log["grip_z"] = rig.position("block")[2]
    go({"3p": lift}, 1.5, "Lift.")
    log["lifted"] = rig.position("block")[2] - log["grip_z"]
    go({"0n": 0.0}, 3.0, "Turn back a quarter, carrying the block.")
    go({"3p": 0.03}, 1.5, "Lower to just above the table and let go.")  # set down fully while gripping, the block wedges in the jaw and opening needs 0.8 N m
    go({"5p": open_}, 1.0, caps={"5p": 0.8})
    go({"3p": lift}, 1.2, "Lift clear.")
    end = rig.position("block")
    moved = float(np.linalg.norm(end[:2] - pick))
    miss = float(np.linalg.norm(end[:2] - np.array([bx, 2 * PITCH])))
    upright = rig.data.xmat[rig.model.body("block").id].reshape(3, 3)[2, 2] > 0.95  # the block turned a quarter with the arm
    ok = log["lifted"] >= 0.01 and moved >= 0.15 and miss <= 0.02 and upright and end[2] < 0.03
    reel.caption = f"Lifted {log['lifted'] * 100:.1f} cm, carried {moved * 100:.0f} cm, set down {miss * 1000:.0f} mm from the mark, upright (measured)." if ok else f"Lifted {log['lifted'] * 100:.1f} cm, moved {moved * 100:.0f} cm, {miss * 1000:.0f} mm from the mark (measured)."
    cam.pause(rig, 2.0)
    finish("arm", reel, cam, "Arm: an 11-module swing arm picks a block by friction between two fingers and places it 18 cm away.")
    measured = {"lifted_cm": round(log["lifted"] * 100, 2), "moved_cm": round(moved * 100, 1), "miss_mm": round(miss * 1000, 1), "upright": bool(upright)}
    notes = ("A swing arm, not a 4-DOF arm: lifting is a pendulum swing of the hand, so the lift is small (about 3 cm) and the reach is fixed by the yaw circle. "
             "Twist joints make a jaw expensive: a finger that swings needs an offset module, and the fixed finger needs three modules to reach the moving finger's plane. "
             "Block is 14 mm wide: the jaw closes in an arc, so a wider block is met at a steeper angle (about 37 degrees at 30 mm) and would squeeze out.")
    return record("arm", ok, "block lifted >= 1 cm, moved >= 15 cm, set down within 2 cm of the mark and upright", len(cells), rig.summary(), notes, measured)


def inchworm(args):
    """A two-joint crawler of seven modules (inchworm-like): two feet, each a module standing on its
    held discs under a post, joined by a three-module link with a bend joint at each end."""
    cells = [
        {"at": (0, 0, 0), "axis": "+y"},  # 0 back foot
        {"at": (0, 0, 1), "axis": "+y", "parent": 0},  # 1 back post: its +y disc is joint 1
        {"at": (0, 1, 1), "axis": "+x", "parent": 1},  # 2 link: offset on joint 1
        {"at": (1, 1, 1), "axis": "+z", "parent": 2},  # 3 link
        {"at": (2, 1, 1), "axis": "+y", "parent": 3},  # 4 link end: its -y disc is joint 2
        {"at": (2, 0, 1), "axis": "+x", "parent": 4},  # 5 front post
        {"at": (2, 0, 0), "axis": "+y", "parent": 5},  # 6 front foot
    ]
    rig = Rig(lattice(cells))
    reel = Reel(OUT / "inchworm.mp4", "10. Crawler: two bend joints, seven modules")
    cam = Camera(rig, reel, [0.12, 0.03, 0.04], 0.75, 125, -18, follow="m3b")
    cam.still_at = 9.0
    reel.caption = "Two feet (each a module on its held discs, under a post) joined by a three-module link; each end of the link\nis a bend joint made from a disc and an offset module. No wheel turns: every disc on the floor is held."
    rig.run(0.5, cam)
    start = (rig.position("m0b") + rig.position("m6b")) / 2
    amp, step, cycles = 0.6, 0.8, 8
    order = [("1p", -amp), ("4n", amp), ("1p", 0.0), ("4n", 0.0)]  # a loop in joint space; reversed, it crawls back
    reel.caption = "Gait: back joint, front joint, back joint home, front joint home. The load on each foot shifts as the link tilts,\nso the lightly loaded foot slips and rocks forward while the other one grips: about 2.4 cm a cycle."
    low = [1.0]

    def watch(r):
        cam(r)
        low[0] = min(low[0], float(r.position("m3b")[2]))

    for _ in range(cycles):
        for joint, angle in order:
            rig.command(targets={joint: angle}, seconds=step)
            rig.run(step + 0.2, watch)
    end = (rig.position("m0b") + rig.position("m6b")) / 2
    crawled = float(end[0] - start[0])
    drift = float(end[1] - start[1])
    upright = all(abs(rig.position(f"m{k}b")[2] - DISC_RADIUS) < 0.004 for k in (0, 6))
    ok = crawled >= 0.15 and upright
    reel.caption = f"Crawled {crawled * 100:.0f} cm in {cycles} cycles ({cycles * 4 * (step + 0.2):.0f} s), {abs(drift) * 100:.1f} cm sideways drift, both feet still upright (measured)."
    cam.pause(rig, 1.5)
    finish("inchworm", reel, cam, "Crawler: a seven-module inchworm-like chain with two bend joints crawls forward; no wheel turns.")
    measured = {"crawled_cm": round(crawled * 100, 1), "drift_cm": round(drift * 100, 1), "feet_upright": bool(upright), "seconds": cycles * 4 * (step + 0.2)}
    notes = ("Not a snake: a lone snake gait with twist joints needs an offset module per bend, and an offset module cannot turn next to the floor (its corners reach 35 mm, "
             "the axis of a module on its rims is 26 mm up), so both joints sit one module up on posts. The feet stand on held rims, so they rock on the rims' round edge "
             "rather than slide: the gait relies on the foot shape. The same joint loop run backwards crawls backwards (checked: -25 cm in 8 cycles).")
    return record("inchworm", ok, "crawls >= 15 cm forward in 8 cycles with both feet upright", len(cells), rig.summary(), notes, measured)


DEMOS = {"lone": lone, "cart": cart, "conveyor": conveyor, "crane": crane, "bridge": bridge, "screwdriver": screwdriver, "crank": crank, "door": door, "walker": walker, "inchworm": inchworm, "arm": arm}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("what", nargs="+")
    args = parser.parse_args()
    for what in args.what:
        if what == "all":
            for name in DEMOS:
                subprocess.run(["nice", "-n", "10", sys.executable, __file__, name], check=False)
            subprocess.run([sys.executable, __file__, "reel", "sheet"], check=True)
        elif what == "reel":
            reel_join()
        elif what == "sheet":
            sheet()
        else:
            DEMOS[what](args)


def reel_join():
    names = [n for n in DEMOS if (OUT / f"{n}.mp4").exists()]
    listing = OUT / "list.txt"
    listing.write_text("".join(f"file '{OUT / n}.mp4'\n" for n in names))
    MEDIA.mkdir(parents=True, exist_ok=True)
    out = MEDIA / "module-demos.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(out)], check=True)
    print("wrote", out, names)


def sheet():
    names = [n for n in DEMOS if (OUT / "stills" / f"{n}.png").exists()]
    rows = {json.loads(line)["demo"]: json.loads(line) for line in (OUT / "results.jsonl").read_text().splitlines() if line.strip()}
    cols = 3
    w, h = 640, 360
    band = 44
    grid = Image.new("RGB", (cols * w, math.ceil(len(names) / cols) * (h + band)), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    face = font(20)
    for k, name in enumerate(names):
        image = Image.open(OUT / "stills" / f"{name}.png").convert("RGB").resize((w, h))
        x, y = (k % cols) * w, (k // cols) * (h + band)
        grid.paste(image, (x, y))
        row = rows.get(name, {})
        verdict = "works" if row.get("ok") else "FAILED"
        peak = max(row.get("peak_torque_nm", {}).values(), default=0.0)
        shown = f"peak {peak:.2f} N m" if peak else "peak < 0.05 N m"
        draw.text((x + 12, y + h + 10), f"{name}: {verdict}, {row.get('modules', '?')} modules, {shown}", fill=(20, 20, 20), font=face)
    out = MEDIA / "module-demos-sheet.png"
    grid.save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
