"""Captioned MP4 and contact sheet of the chosen SMA cube (hardware/runs/module_phys/sma_cube/chosen.json):
lone moves, dock, climb, roll and descent on a structure, split, the 2 x 2 x 2 assembly, the arm,
walker and the new boom, and failures (a climb onto a lone support, which tips it).

    .venv/bin/python hardware/scripts/module_phys/sma_cube_media.py all        # every clip, then join and sheet
    .venv/bin/python hardware/scripts/module_phys/sma_cube_media.py climb_row  # one clip
    .venv/bin/python hardware/scripts/module_phys/sma_cube_media.py join

Writes hardware/runs/module_phys/sma_cube/media/<clip>.mp4 and stills, hardware/handoff/media/sma-cube.mp4 and
hardware/handoff/media/sma-cube-sheet.png. Each clip is one nominal-parameter run (seed shown); the
Monte Carlo rates are in outcomes.json. The assembly clip's run is also written to
hardware/runs/module_phys/sma_cube/assembly.jsonl.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "hardware" / "scripts"))
sys.path.insert(0, str(ROOT / "hardware" / "scripts" / "module_phys"))

import sma_cube_machines as scm  # noqa: E402  (patches small_joint_machines for the latch cube)
import sma_cube_sim as sim  # noqa: E402
from module_demos import Reel, font  # noqa: E402
from beebot.hardware.sma_cube import Params  # noqa: E402

OUTD = sim.OUT / "media"
MEDIA = ROOT / "hardware" / "handoff" / "media"
SIZE = (1280, 720)


class WorldCam:
    """Renders sma_cube.World frames into a Reel, one per `every` s of sim time; rebuilds its renderer
    when the move builds a new World."""

    def __init__(self, reel, lookat, distance, azimuth=135, elevation=-25, every=0.25, still_frac=0.6):
        self.reel = reel
        self.cam = mujoco.MjvCamera()
        self.cam.lookat = np.array(lookat, float)
        self.cam.distance, self.cam.azimuth, self.cam.elevation = distance, azimuth, elevation
        self.every = every
        self.r = None
        self.model = None
        self.due = 0.0
        self.frames = []
        self.world = None

    def __call__(self, w):
        if w.model is not self.model:
            self.model = w.model
            self.r = mujoco.Renderer(w.model, SIZE[1], SIZE[0])
            self.due = w.data.time
        self.world = w
        if w.data.time >= self.due - 1e-12:
            self.due += self.every
            self.r.update_scene(w.data, camera=self.cam)
            img = self.r.render()
            self.reel.add(img)
            self.frames.append(self.reel.last)

    def hold(self, seconds, caption=None):
        if caption is not None:
            self.reel.caption = caption
        if self.world is not None:
            self.r.update_scene(self.world.data, camera=self.cam)
            self.reel.add(self.r.render(), int(seconds * 30))


def verdict_text(r):
    if r is None:
        return "no move offered"
    bits = [("reached" if r.get("ok") else f"FAILED: {r.get('class')}")]
    if r.get("harmful"):
        bits.append("harmful (a neighbour moved)")
    if r.get("energy_J") is not None:
        bits.append(f"{r['energy_J']:.2f} J")
    if r.get("sim_s") is not None:
        bits.append(f"{r['sim_s']:.0f} s")
    return ", ".join(bits)


def swarm_clip(name, fn, title, caption, lookat_cells, distance_L=9.0, azimuth=135, elevation=-25, every=0.25, seed=1):
    d = sim.chosen()
    L = d.L
    OUTD.mkdir(parents=True, exist_ok=True)
    reel = Reel(OUTD / f"{name}.mp4", title)
    reel.caption = caption
    look = np.array(lookat_cells, float) * L + np.array([0, 0, 0.5 * L])
    cam = WorldCam(reel, look, distance_L * L, azimuth, elevation, every)
    r = fn(d, Params(seed=seed), record=cam)
    cam.hold(3.0, caption + "\n" + f"Outcome (nominal parameters, seed {seed}): " + verdict_text(r))
    reel.close()
    if cam.frames:
        cam.frames[int(0.6 * (len(cam.frames) - 1))].save(OUTD / f"{name}.png")
    (OUTD / f"{name}.json").write_text(json.dumps(r, default=float))
    return r


def assembly_clip():
    d = sim.chosen()
    rows = []
    reel = Reel(OUTD / "assembly.mp4", f"Eight {d.L * 1e3:.1f} mm cubes build a 2 x 2 x 2")
    reel.caption = "Each cube is planned by BFS over the grid moves and driven in MuJoCo; after every move the planner\nreads where the cube really is. Columns at x = 0 and x = 1; each column passes through a 3-cube step."
    cam = WorldCam(reel, np.array([0.5, -1.0, 0.5]) * d.L, 16 * d.L, 120, -28, every=2.0)
    r = sim.assembly(d, Params(seed=1), record=cam, log=lambda s: rows.append(s))
    cam.hold(4.0, f"{r['moves']} moves, {r['failed_moves']} did not land as planned, {r['sim_s']:.0f} s of sim time: " + ("2 x 2 x 2 built" if r["built"] else "NOT built") + (", nothing knocked" if not r["harmful"] else ", a neighbour was knocked"))
    reel.close()
    if cam.frames:
        cam.frames[-1].save(OUTD / "assembly.png")
    with (sim.OUT / "assembly.jsonl").open("a") as f:
        f.write(json.dumps({k: v for k, v in r.items()}, default=int) + "\n")
    (OUTD / "assembly.json").write_text(json.dumps(r, default=int))
    return r


def machine_clip(name, cam_args, title):
    sjm = scm.sjm
    L = scm.D.L
    s = L / sjm.REF
    reel = Reel(OUTD / f"{name}.mp4", title)
    lookat, dist, az, el, follow, slow = cam_args
    cam = sjm.Cam(reel, [v * s for v in lookat], dist * s, az, el, follow, slow)
    row = sjm.MACHINES[name](L, {}, cam)
    reel.close()
    img = reel.last
    if img is not None:
        img.save(OUTD / f"{name}.png")
    (OUTD / f"{name}.json").write_text(json.dumps(row, default=float))
    return row


CLIPS = {
    "tip_book": lambda: swarm_clip("tip_book", sim.MOVES["tip_book"], "Lone cube: BOOK tip", "The leg in the bottom face opens over its hinge edge; the cube rolls one cell, quasi-static\n(the leg is in compression past the balance point).", (0, 0, 0)),
    "tip_kick": lambda: swarm_clip("tip_kick", sim.MOVES["tip_kick"], "Lone cube: KICK tip", "A leg hinged at the bottom edge swings down and lifts the cube over its far edge;\nthe controller stops when the IMU shows the balance point passed; the last 45 deg are under gravity.", (0, 0, 0)),
    "yaw": lambda: swarm_clip("yaw", sim.MOVES["yaw"], "Lone cube: YAW on the disc", "Disc face down: the geared joint turns the body a quarter on the disc's silicone tread,\ncorrected by the IMU.", (0, 0, 0)),
    "self_right": lambda: swarm_clip("self_right", sim.MOVES["self_right"], "Lone cube: self-righting", "From a random orientation the cube plans (BFS) and tips back to its home orientation.", (0, 0, 0), distance_L=12),
    "dock": lambda: swarm_clip("dock", sim.MOVES["dock"], "DOCK onto a latched pair", "The last tip of an approach with a position and heading error; the corner spheres pull it in\nand the edge bolts latch (hooks on the x-parallel edges).", (0, 1.5, 0), distance_L=11),
    "climb_row": lambda: swarm_clip("climb_row", sim.MOVES["climb_row"], "CLIMB onto a braced row", "The bolts free the hooks, the far edge retracts its spheres; the leg presses the support's side and\nturns the cube about the shared top edge; at the top the bolts relatch.", (0, 1, 0), distance_L=11),
    "roll_top": lambda: swarm_clip("roll_top", sim.MOVES["roll_top"], "ROLL on top of a row", "Unlatch the far edge, book over the hinge edge, relatch on the next cube.", (0, 0, 0.5), distance_L=11),
    "book_descend": lambda: swarm_clip("book_descend", sim.MOVES["book_descend"], "DESCEND off a row", "Half turn down the side; the leg stays in compression on the row's top the whole way.", (0, 1, 0.3), distance_L=11),
    "split_pair": lambda: swarm_clip("split_pair", sim.MOVES["split_pair"], "SPLIT from a latched neighbour", "Both bolts on the face free the hooks and retract the four corner spheres, then the cube tips away.", (0, 0.5, 0), distance_L=11),
    "climb_lone": lambda: swarm_clip("climb_lone", sim.MOVES["climb_lone"], "Failure: CLIMB onto a lone support", "A lone support tips over: the combined centre of mass leaves its base. The planner never offers\nthis move (a support must be braced by a row or a cube under it).", (0, 1, 0), distance_L=11),
    "assembly": assembly_clip,
    "arm": lambda: machine_clip("arm", ([-0.03, 0.06, 0.05], 0.65, 340, -22, None, 1.0 / scm.sjm.TIME), "Arm: pick and place, latched bonds, geared joints"),
    "walker": lambda: machine_clip("walker", ([-0.03, 0.0, 0.05], 0.80, 140, -20, "c1b", 1.0 / 8.0), "Walker: four rotating legs, latched bonds"),
    "boom": lambda: machine_clip("boom", ([0.03, 0.02, 0.04], 0.75, 300, -25, None, 1.0 / 4.0), "New machine: five-cube latched boom"),
}
ORDER = ["tip_book", "tip_kick", "yaw", "self_right", "dock", "climb_row", "roll_top", "book_descend", "split_pair", "climb_lone", "assembly", "arm", "walker", "boom"]


def join():
    names = [n for n in ORDER if (OUTD / f"{n}.mp4").exists()]
    listing = OUTD / "list.txt"
    listing.write_text("".join(f"file '{OUTD / n}.mp4'\n" for n in names))
    MEDIA.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(MEDIA / "sma-cube.mp4")], check=True)
    cols, w, h, band = 4, 480, 270, 54
    grid = Image.new("RGB", (cols * w, math.ceil(len(names) / cols) * (h + band)), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    face = font(17)
    for k, n in enumerate(names):
        if not (OUTD / f"{n}.png").exists():
            continue
        x, y = (k % cols) * w, (k // cols) * (h + band)
        grid.paste(Image.open(OUTD / f"{n}.png").convert("RGB").resize((w, h)), (x, y))
        r = json.loads((OUTD / f"{n}.json").read_text()) if (OUTD / f"{n}.json").exists() else {}
        ok = r.get("ok") if "ok" in r else r.get("built")
        line2 = "ok_1.5x " + str(r.get("ok_1.5x")) if "ok_1.5x" in r else verdict_text(r)
        draw.text((x + 8, y + h + 4), f"{n}: {'works' if ok else 'FAILS'}", fill=(20, 20, 20), font=face)
        draw.text((x + 8, y + h + 26), line2[:52], fill=(60, 60, 60), font=face)
    grid.save(MEDIA / "sma-cube-sheet.png")
    print("wrote", MEDIA / "sma-cube.mp4", MEDIA / "sma-cube-sheet.png", names)


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what == "join":
        join()
        return
    names = ORDER if what == "all" else what.split(",")
    for n in names:
        t = time.time()
        r = CLIPS[n]()
        print(n, json.dumps({k: r.get(k) for k in ("ok", "ok_1.5x", "built", "class", "moves", "failed_moves")} if isinstance(r, dict) else r), round(time.time() - t), flush=True)
    if what == "all":
        join()


if __name__ == "__main__":
    main()
