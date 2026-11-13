"""The full video of the turning-face cube: runs/handoff/media/rotating-cube-full.mp4.

Four scenes, each rendered by its own process (VTK and MuJoCo each want the GL context) and joined
with ffmpeg:

1. cad: the CAD parts (hardware/rotating_cube/cube_full.py) turning, blown up with labels, put
   back, then a section with the disc turning. Animated, no physics.
1. move: one cube crossing the floor and climbing over a neighbour by edge pivots. MuJoCo physics
   (cube_turning.TurningSwarm), the cubes drawn with the CAD's STL meshes.
1. bond: a cube pivots onto another's turning face, is turned two full turns with a stop at every
   quarter turn and one full turn without stopping, and pivots off again. MuJoCo physics.
1. task: the crank task of runs/handoff/rotating-cube-poc.md, one seed that succeeds and one run
   that fails. MuJoCo physics.

    .venv/bin/python scripts/rotating_cube_video.py all
    .venv/bin/python scripts/rotating_cube_video.py task --best 0 --fail 3 --fail_face '{"torque": 0.2}'
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

ROOT = Path(__file__).resolve().parents[1]
CAD = ROOT / "hardware" / "rotating_cube"
MEDIA = ROOT / "runs" / "handoff" / "media"
PARTS = ROOT / "runs" / "cube_turning" / "video"
SIZE = (1280, 720)
FPS = 30


def font(size):
    for path in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


class Reel:
    """Frames to an MP4 through ffmpeg, with the present caption burned into a band at the foot."""

    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SIZE[0]}x{SIZE[1]}", "-r", str(FPS), "-i", "-"]
        self.pipe = subprocess.Popen(command + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(path)], stdin=subprocess.PIPE)
        self.caption = ""
        self.face = font(25)
        self.frames = 0

    def add(self, image, repeat=1):
        picture = Image.fromarray(image).convert("RGB")
        if self.caption:
            draw = ImageDraw.Draw(picture, "RGBA")
            lines = self.caption.split("\n")
            top = SIZE[1] - 22 - 32 * len(lines)
            draw.rectangle([0, top - 10, SIZE[0], SIZE[1]], fill=(0, 0, 0, 170))
            for k, line in enumerate(lines):
                draw.text((30, top + 32 * k), line, fill=(255, 255, 255, 255), font=self.face)
        self.pipe.stdin.write(picture.tobytes() * repeat)
        self.frames += repeat

    def close(self):
        self.pipe.stdin.close()
        self.pipe.wait()
        print(f"{self.frames / FPS:.1f} s")


def ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def cad(args):
    sys.path.insert(0, str(CAD))
    import render_exploded as view

    items = view.pieces()
    reel = Reel(PARTS / "1-cad.mp4")
    window, renderer, actors = view.scene(items, SIZE)
    reel.caption = "One 50 mm cube from the CAD: 207 g, two N20 gearmotors, 48 switchable magnets.\nThe orange disc is the turning face."
    for k in range(4 * FPS):
        view.place(actors, items, 0.0)
        reel.add(view.shot(window, renderer, -150 + 100 * k / (4 * FPS), 26, 250))
    reel.caption = "Blow-up: everything in the cube."
    for k in range(3 * FPS):
        u = ease(k / (3 * FPS - 1))
        view.place(actors, items, u)
        reel.add(view.shot(window, renderer, -50 - 12 * u, 26 - 6 * u, 250 + 330 * u, focus=(0, 0, 25 + 7 * u)))
    image = view.shot(window, renderer, -62, 20, 580, focus=(0, 0, 32))
    image = view.labelled(image, renderer, items, 1.0, text_size=19, foot=100)
    reel.caption = "Body, disc on a 6803 bearing, turning-face gearmotor, detent balls and springs, edge-hinge gearmotor,\n12 edge axles with latch hooks (stand-ins: that mechanism is not designed), magnets, battery, board."
    reel.add(image, 7 * FPS)
    reel.caption = "Reassembly."
    for k in range(3 * FPS):
        u = 1 - ease(k / (3 * FPS - 1))
        view.place(actors, items, u)
        reel.add(view.shot(window, renderer, -50 - 12 * u, 26 - 6 * u, 250 + 330 * u, focus=(0, 0, 25 + 7 * u)))
    window, renderer, actors = view.scene(items, SIZE, section=True)
    reel.caption = "Section: the gearmotor sits in the body and turns the disc by its hub.\nFour balls click into dimples at every quarter turn; two spring pins feed the disc's magnets."
    for k in range(5 * FPS):
        u = k / (5 * FPS - 1)
        quarters = min(int(u * 2.2), 1)
        within = min(max((u * 2.2 - quarters) / 0.75, 0.0), 1.0)
        view.place(actors, items, 0.0, 90.0 * (quarters + ease(within)) if quarters < 2 else 180.0)
        reel.add(view.shot(window, renderer, -90, 14, 190))
    reel.close()


class Camera:
    """A MuJoCo renderer feeding a Reel; `slow` times slower than real time."""

    def __init__(self, swarm, reel, lookat, distance, azimuth, elevation, slow=1.0):
        import mujoco

        self.renderer = mujoco.Renderer(swarm.model, SIZE[1], SIZE[0])
        self.camera = mujoco.MjvCamera()
        self.camera.lookat = [v * swarm.cube.side for v in lookat]
        self.camera.distance = distance * swarm.cube.side
        self.camera.azimuth, self.camera.elevation = azimuth, elevation
        self.reel = reel
        self.every = 1 / (FPS * slow)
        self.due = swarm.data.time

    def frame(self, swarm):
        if swarm.data.time >= self.due:
            self.due += self.every
            self.renderer.update_scene(swarm.data, camera=self.camera)
            self.reel.add(self.renderer.render())

    def still(self, swarm, seconds):
        self.renderer.update_scene(swarm.data, camera=self.camera)
        self.reel.add(self.renderer.render(), int(seconds * FPS))


def move(args):
    from beebot.hardware.cube_turning import TurningSwarm

    swarm = TurningSwarm([(0, 0, 0), (0, 0, 2)], seed=args.seed, meshes=str(CAD))
    reel = Reel(PARTS / "2-move.mp4")
    camera = Camera(swarm, reel, (2.0, 0.5, 0.8), 7.5, 105, -16, slow=1.3)
    reel.caption = "Moving, as the hinge cube does: latch an edge, magnets off, the edge motor (0.3 N m) turns the cube about it.\nMuJoCo physics; the cubes are drawn with the CAD parts."
    camera.still(swarm, 1.0)
    rows = []
    for direction in [(0, 0, 1), (1, 0, 1), (-1, 0, 1), (0, 0, 1)]:
        result = swarm.move(0, direction, watch=camera.frame)
        rows.append((result.primitive, result.detail))
        camera.still(swarm, 0.3)
    camera.still(swarm, 0.8)
    reel.close()
    print(rows)


def bond(args):
    from beebot.hardware.cube_turning import TurningSwarm

    swarm = TurningSwarm([(0, 0, 0), (0, 0, 1)], turning=[(2, 1), (0, 1)], seed=args.seed, meshes=str(CAD))
    reel = Reel(PARTS / "3-bond.mp4")
    camera = Camera(swarm, reel, (0.4, 0.5, 1.1), 7.5, 60, -22)
    reel.caption = "Bonding: the orange cube pivots onto the red cube's turning face; its magnets switch on when it arrives."
    camera.still(swarm, 1.0)
    rows = [swarm.move(1, (1, 0, -1), watch=camera.frame).detail]
    camera.still(swarm, 0.6)
    for k in range(8):
        reel.caption = f"The turning face turns its neighbour: quarter turn {k + 1} of 8. At each stop the detent balls drop in\nand the edge magnets pull the pair square (the click), then the motor lets go."
        row = swarm.turn(0, 1, carried=[1], watch=camera.frame)
        rows.append((row["detail"], row["error"]))
        camera.still(swarm, 0.15)
    reel.caption = "One full turn without stopping."
    row = swarm.turn(0, 4, carried=[1], watch=camera.frame)
    rows.append((row["detail"], row["error"]))
    camera.still(swarm, 0.5)
    reel.caption = "Releasing: magnets off, and the cube pivots away about an edge."
    rows.append(swarm.move(1, (-1, 0, 1), watch=camera.frame).detail)
    camera.still(swarm, 0.8)
    reel.close()
    print(rows)


def task(args):
    from beebot.hardware.cube_turning import Face, crank

    reel = Reel(PARTS / "4-task.mp4")
    holder = {}

    def built(swarm):
        holder["camera"] = Camera(swarm, reel, (0.6, 0.0, 2.0), 10.0, 40, -12, slow=args.slow)
        holder["camera"].still(swarm, 0.8)
        holder["swarm"] = swarm

    def watch(swarm):
        holder["camera"].frame(swarm)

    reel.caption = f"The task (seed {args.best}): the top cube pivots onto the arm, the yellow cube's turning face carries arm and load\ntwice round, stopping and re-locking at every quarter turn, and the load pivots back."
    best = crank(args.best, meshes=str(CAD), built=built, watch=watch)
    reel.caption = f"Seed {args.best}: {best['relocks']} of {best['relock_attempts']} quarter turns re-locked, {best['sim_seconds']:.1f} s, nothing dropped." if best["success"] else f"Seed {args.best} failed: {best['why']}"
    holder["camera"].still(holder["swarm"], 1.5)
    reel.caption = args.fail_caption
    fail = crank(args.fail, meshes=str(CAD), built=built, watch=watch, face=Face(**json.loads(args.fail_face)), **json.loads(args.fail_options))
    reel.caption = args.fail_caption + f"\nResult: {fail['why']} at step {fail['failed_at']}; {fail['relocks']} of {fail['quarters_asked']} quarter turns." if not fail["success"] else "This run did not fail."
    holder["camera"].still(holder["swarm"], 2.5)
    reel.close()
    print({k: v for k, v in best.items() if k != "steps"})
    print({k: v for k, v in fail.items() if k != "steps"})


def join(args):
    names = ["1-cad.mp4", "2-move.mp4", "3-bond.mp4", "4-task.mp4"]
    missing = [n for n in names if not (PARTS / n).exists()]
    listing = PARTS / "list.txt"
    listing.write_text("".join(f"file '{PARTS / n}'\n" for n in names if n not in missing))
    out = MEDIA / "rotating-cube-full.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(out)], check=True)
    print("wrote", out, "missing:", missing)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene", choices=["cad", "move", "bond", "task", "join", "all"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--best", type=int, default=0)
    parser.add_argument("--fail", type=int, default=0)
    parser.add_argument("--fail_face", default='{"torque": 0.2}')
    parser.add_argument("--fail_options", default="{}")
    parser.add_argument("--fail_caption", default="A failure: the same task with the face motor held to an N20 gearbox's rating, 0.2 N m.")
    parser.add_argument("--slow", type=float, default=1.0)
    args = parser.parse_args()
    if args.scene == "all":
        for scene in ("cad", "move", "bond", "task", "join"):
            subprocess.run([sys.executable, __file__, scene] + sys.argv[2:], check=True)
    else:
        {"cad": cad, "move": move, "bond": bond, "task": task, "join": join}[args.scene](args)


if __name__ == "__main__":
    main()
