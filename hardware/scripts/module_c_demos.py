"""Demos of the layout C cube (hardware/src/beebot/hardware/module_c.py): 50 mm cubes that drive in pairs on a
sprung wheel, pivot singly about an edge and bond with switchable connectors during the run. Every
motion comes from the wheel and disc motors and the pivot torque under gravity, contacts and
friction; poses are read through a noisy camera; a bond forms only inside the capture range.

Each demo runs over 10 seeds (placement and sensor noise differ); seed 0 is filmed. One clip per
demo in hardware/runs/module_pairs/<demo>.mp4, one JSON line per demo in hardware/runs/module_pairs/results.jsonl,
a still per demo in hardware/runs/module_pairs/stills/, then the captioned reel
hardware/handoff/media/module-pairs.mp4 and the contact sheet hardware/handoff/media/module-pairs-sheet.png.
Memo: hardware/handoff/module-pairs.md.

    .venv/bin/python hardware/scripts/module_c_demos.py all       # every demo (one process each, nice), packing, reel, sheet
    .venv/bin/python hardware/scripts/module_c_demos.py pairing   # one demo
    .venv/bin/python hardware/scripts/module_c_demos.py reel sheet
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from beebot.hardware.module_c import (
    MASS,
    PIVOT_TORQUE,
    SIDE,
    WHEEL_FACE,
    FACES,
    Failure,
    Placement,
    Planner,
    World,
    horizontal,
    nearest_lattice,
    packing,
    signed_angle,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "hardware" / "runs" / "module_pairs"
MEDIA = ROOT / "hardware" / "handoff" / "media"
SIZE = (1280, 720)
FPS = 30
SEEDS = 10
ONLY = None  # --seeds: run just these seeds, film none, record nothing (debugging)
GRAVITY_PIVOT = MASS * 9.81 * SIDE / math.sqrt(2)  # N m: the largest gravity torque about an edge


def font(size):
    for path in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


class Reel:
    """Frames to an MP4 through ffmpeg, the caption burned into a band at the foot, a title top left."""

    def __init__(self, path, title=""):
        path.parent.mkdir(parents=True, exist_ok=True)
        command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SIZE[0]}x{SIZE[1]}", "-r", str(FPS), "-i", "-"]
        self.pipe = subprocess.Popen(command + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(path)], stdin=subprocess.PIPE)
        self.caption = ""
        self.title = title
        self.face = font(25)
        self.big = font(30)
        self.last = None

    def picture(self, image):
        picture = Image.fromarray(image).convert("RGB")
        draw = ImageDraw.Draw(picture, "RGBA")
        if self.title:
            draw.rectangle([0, 0, 60 + draw.textlength(self.title, font=self.big), 56], fill=(0, 0, 0, 150))
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

    def close(self):
        self.pipe.stdin.close()
        self.pipe.wait()


class Camera:
    """Renders a World into a Reel: one frame per 1 / (FPS * slow) s of sim time (slow < 1 speeds
    the clip up). The look-at point drifts after the mean of `follow` cubes."""

    def __init__(self, world, reel, lookat, distance, azimuth, elevation, follow=None, slow=1.0):
        self.renderer = mujoco.Renderer(world.model, SIZE[1], SIZE[0])
        self.camera = mujoco.MjvCamera()
        self.camera.lookat = np.array(lookat, float)
        self.camera.distance = distance
        self.camera.azimuth, self.camera.elevation = azimuth, elevation
        self.follow = follow
        self.reel = reel
        self.every = 1 / (FPS * slow)
        self.due = world.data.time
        self.still_at = None
        self.still = None
        self.world = world

    def image(self, world):
        if self.follow:
            target = np.mean([world.pos(i) for i in self.follow], axis=0)
            target[2] = max(target[2], 0.03)
            self.camera.lookat = 0.95 * np.array(self.camera.lookat) + 0.05 * target
        self.renderer.update_scene(world.data, camera=self.camera)
        return self.renderer.render()

    def __call__(self, world):
        if world.data.time >= self.due - 1e-9:
            self.due += self.every
            self.reel.add(self.image(world))
            if self.still_at is not None and world.data.time >= self.still_at and self.still is None:
                self.still = self.reel.last

    def pause(self, seconds):
        self.reel.add(self.image(self.world), int(seconds * FPS))


def mark(pos, size=(0.01, 0.01), rgba="0.9 0.2 0.2 1"):
    return f'<geom type="box" pos="{pos[0]:.6g} {pos[1]:.6g} 0.0002" size="{size[0]:.6g} {size[1]:.6g} 0.0002" rgba="{rgba}" contype="0" conaffinity="0"/>\n'


def finish(name, reel, camera):
    reel.close()
    stills = OUT / "stills"
    stills.mkdir(parents=True, exist_ok=True)
    (camera.still or reel.last).save(stills / f"{name}.png")


def record(name, results, criterion, notes, extra=None):
    """One JSON line per demo in results.jsonl (replacing an earlier line for the same demo)."""
    ok = [r["ok"] for r in results]
    if ONLY is not None:
        print("seeds", ONLY, "ok", ok, [r.get("why") for r in results])
        for r in results:
            for e in r.get("last_events", []):
                print("   ", r["seed"], e)
        return None
    rate = sum(ok) / len(ok)
    row = {"demo": name, "ok": rate >= 0.8, "success_rate": round(rate, 2), "seeds": len(results), "criterion": criterion}
    keys = sorted({k for r in results for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "seed"})
    for k in keys:
        values = [r[k] for r in results if k in r and r[k] is not None]
        if values:
            row[k] = {"mean": round(float(np.mean(values)), 4), "min": round(float(np.min(values)), 4), "max": round(float(np.max(values)), 4)}
    row["failed_docks"] = int(sum(r.get("failed_docks_n", 0) for r in results))
    row["dock_attempts"] = int(sum(r.get("dock_attempts_n", 0) for r in results))
    row["harmful"] = int(sum(len(r.get("harm", [])) for r in results))
    row["failures"] = [{"seed": r["seed"], "why": r.get("why", "")} for r in results if not r["ok"]]
    row.update(extra or {})
    row["per_seed"] = results
    row["notes"] = notes
    path = OUT / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    rows = [r for r in rows if r["demo"] != name] + [row]
    order = list(DEMOS)
    rows.sort(key=lambda r: order.index(r["demo"]) if r["demo"] in order else 99)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps({k: v for k, v in row.items() if k != "per_seed"}))
    return row


def pivot_stats(world):
    if not world.pivot_peaks:
        return {}
    moving = max(p[0] for p in world.pivot_peaks)
    return {
        "pivots": world.pivot_count,
        "pivot_peak_moving_nm": round(moving, 4),
        "pivot_peak_nm": round(max(p[1] for p in world.pivot_peaks), 4),
        "pivot_saturated_s": round(sum(p[2] for p in world.pivot_peaks), 3),
        "pivot_margin_x": round(PIVOT_TORQUE / GRAVITY_PIVOT, 2),
    }


def common(world, ok, why="", **numbers):
    row = {"ok": bool(ok), "why": why, "sim_s": round(world.data.time, 2), "failed_docks_n": world.failed_docks, "dock_attempts_n": world.dock_attempts,
           "cube_metres_driven": round(world.driven, 3), "wheel_peak_nm": round(world.peak_wheel, 4), "bond_peak_force_n": round(world.peak_bond_force, 2),
           "bond_peak_torque_nm": round(world.peak_bond_torque, 3), "harm": world.harm}
    row.update(pivot_stats(world))
    row.update(numbers)
    if not ok:
        row["last_events"] = world.events[-6:]
    return row


def guard(world, cubes, label, moved=0.01):
    """Harm check after a step: a cube tipped off its faces, or one of `cubes` (bystanders) shifted."""
    before = {i: world.pos(i) for i in cubes}

    def check():
        world.check_harm(range(world.n), label)
        for i, p in before.items():
            d = float(np.linalg.norm((world.pos(i) - p)[:2]))
            if d > moved:
                world.harm.append({"t": round(world.data.time, 2), "cube": i, "what": f"knocked {d * 1000:.0f} mm", "during": label})
    return check


def pair_placements(rng, centre, yaw):
    """Two cubes bonded side by side (axles in a line), with placement error."""
    n = np.array([-math.sin(yaw), math.cos(yaw)])
    out = []
    for k in range(2):
        p = np.asarray(centre, float) + n * (k - 0.5) * (SIDE + 0.0005) + rng.uniform(-0.0015, 0.0015, 2) * 0.5
        out.append(Placement(tuple(p), yaw=yaw + math.radians(rng.uniform(-1.5, 1.5))))
    return out


def loose_two(rng, centre, spread=0.09, apart=0.14, downs=(WHEEL_FACE, WHEEL_FACE)):
    while True:
        a = np.asarray(centre) + rng.uniform(-spread, spread, 2)
        b = np.asarray(centre) + rng.uniform(-spread, spread, 2)
        if np.linalg.norm(a - b) > apart:
            break
    return [Placement(tuple(a), down=downs[0], yaw=rng.uniform(-math.pi, math.pi)), Placement(tuple(b), down=downs[1], yaw=rng.uniform(-math.pi, math.pi))]


def run_seeds(name, one, film):
    results = []
    for seed in ONLY if ONLY is not None else range(SEEDS):
        t = time.time()
        try:
            row = one(seed, film and seed == 0 and ONLY is None)
        except Failure as e:
            row = {"ok": False, "why": f"failure: {e}"}
        except RuntimeError as e:
            row = {"ok": False, "why": f"sim error: {e}"}
        row["seed"] = seed
        row["wall_s"] = round(time.time() - t, 1)
        print(name, seed, row["ok"], row.get("why", ""), flush=True)
        results.append(row)
    return results


# ---------------------------------------------------------------------- 1. lone cube


def lone(args):
    def one(seed, filmed):
        rng = np.random.default_rng(seed)
        w = World([Placement((0.0, 0.0), yaw=rng.uniform(-math.pi, math.pi))], seed=seed)
        P = Planner(w)
        cam = None
        if filmed:
            reel = Reel(OUT / "lone.mp4", "1. One cube: drives straight, pivots to turn")
            d0 = w.rolling(0)
            cam = Camera(w, reel, [d0[0] * 0.15, d0[1] * 0.15, 0.02], 0.75, 135, -25, follow=[0], slow=0.7)
            cam.still_at = 1.5
            w.watch = cam
            reel.caption = "A lone cube rests on its sprung wheel (20 mm, 1.5 mm proud) and one bottom edge.\nIt can only drive straight: 30 cm at 0.15 m/s."
        w.run(0.3)
        p0, r0 = P.look(0, 5)
        d = w.rolling(0, r0)
        start = w.pos(0)
        t0 = w.data.time
        P.drive_to(0, p0 + d * 0.30)
        t_drive = w.data.time - t0
        moved = w.pos(0) - start
        drift = float(np.cross(d, moved)[2])
        along = float(moved @ d)
        yaw0 = math.atan2(d[1], d[0])
        if filmed:
            w.watch.reel.caption = "To change heading it needs three pivots about floor edges (each a quarter turn, 150 deg/s):\nroll sideways, roll forward, roll back. One pivot alone turns the wheel face off the floor."
        check = guard(w, [], "heading turn")
        t1 = w.data.time
        outs = P.turn_heading(0, np.cross([0, 0, 1.0], d))
        check()
        t_turn = w.data.time - t1
        d2 = w.rolling(0)
        turned = math.degrees(signed_angle(d, d2))
        turned_abs = abs(turned)
        if filmed:
            w.watch.reel.caption = f"Heading turned {turned_abs:.1f} degrees in {t_turn:.1f} s; the wheel is down again. Then 20 cm in the new direction."
        p2, r2 = P.look(0, 5)
        P.drive_to(0, p2 + w.rolling(0, r2) * 0.20)
        ok = all(o in ("arrived", "near") for o in outs) and abs(turned_abs - 90) <= 5 and abs(drift) <= 0.005 and w.wheel_down(0) and not w.harm
        if filmed:
            w.watch.reel.caption = f"Straight run: 30 cm at {0.30 / t_drive:.2f} m/s average, {abs(drift) * 1000:.1f} mm sideways drift; turn {turned_abs:.1f} degrees by 3 pivots (measured)."
            cam.pause(1.5)
            finish("lone", w.watch.reel, cam)
        return common(w, ok, "" if ok else f"pivots {outs}, turned {turned_abs:.1f}, drift {drift * 1000:.1f} mm",
                      drive_speed_m_s=round(along / t_drive, 3), straight_drift_mm=round(abs(drift) * 1000, 2), heading_turn_deg=round(turned_abs, 2),
                      heading_turn_s=round(t_turn, 2), s_per_pivot=round(t_turn / 3, 2))

    results = run_seeds("lone", one, True)
    return record("lone", results, "30 cm straight with <= 5 mm sideways drift; heading turned 90 +/- 5 degrees by floor pivots, wheel down after; no harm",
                  "A lone cube drives straight only (one wheel, the body edge is a skid). A heading change takes three floor pivots (roll sideways, forward, back) and moves the cube one cell; "
                  "one pivot is a quarter turn about a floor edge and leaves the wheel face on a side. Floor pivots have no latch: the edge holds by friction (floor 0.5).")


# ---------------------------------------------------------------------- 2. pairing


def pairing(args):
    def one(seed, filmed):
        rng = np.random.default_rng(100 + seed)
        w = World(loose_two(rng, (0.0, 0.0)), seed=seed)
        P = Planner(w)
        cam = None
        if filmed:
            reel = Reel(OUT / "pairing.mp4", "2. Pairing: two loose cubes dock side by side")
            cam = Camera(w, reel, [0.0, 0.0, 0.02], 0.85, 120, -35, follow=[0, 1], slow=0.5)
            cam.still_at = 14.0
            w.watch = cam
            reel.caption = "Two loose cubes, wheels down, at random headings. Neither can steer, so:\none drives to where the other's line crosses its own; the other drives into its side face and pushes (squares its yaw)."
        w.run(0.3)
        t0 = w.data.time
        ok = P.pair_up(0, 1)
        t = w.data.time - t0
        w.run(0.5)
        check = guard(w, [], "pairing")
        check()
        bonded = frozenset((0, 1)) in w.bonds
        axles = abs(float(w.rot(0)[:, 1] @ w.rot(1)[:, 1]))
        ok = ok and bonded and axles > 0.995 and not w.harm
        if filmed:
            cam.reel.caption = f"Bonded side by side, axles in a line: a differential pair, in {t:.0f} s with {w.pivot_count} pivots and {w.failed_docks} failed dock(s) (measured)."
            cam.pause(1.5)
            finish("pairing", cam.reel, cam)
        return common(w, ok, "" if ok else "not bonded", pairing_s=round(t, 2))

    results = run_seeds("pairing", one, True)
    return record("pairing", results, "the two cubes end bonded side by side with their wheel axles in a line (a differential pair); no harm",
                  "Sequence (Planner.pair_up): if the two rolling lines are within 35 degrees of parallel, b turns 90 degrees first (three floor pivots); a drives to where b's line crosses its own; "
                  "b drives into a's side face and pushes, which squares its yaw (a cube cannot steer); a lines its centre up with b's; b latches the shared vertical edge and turns a quarter "
                  "(1.5 degrees past, so it slides in with its line pointing slightly outward) into the diagonal cell; both lean the way b will arrive; b slides in beside a; bond. "
                  "The quarter turn on the floor drags both cubes' wheels and edges: a turns too (about 43 degrees in a probe) and the pivot runs at its 0.1 N m clip for about 0.25 s.")


# ---------------------------------------------------------------------- 3. pair travel vs pivots


def travel(args):
    def pair_one(seed, filmed):
        rng = np.random.default_rng(200 + seed)
        yaw = rng.uniform(-0.2, 0.2) + math.pi / 2  # rolling along about +y
        w = World(pair_placements(rng, (0.0, 0.0), yaw), seed=seed, bonds=[(0, 1, None)], scene=mark((0.5, 0.5), (0.012, 0.012)) + mark((0.0, 0.5), (0.006, 0.006), "0.3 0.3 0.3 1"))
        P = Planner(w)
        cam = None
        if filmed:
            reel = Reel(OUT / "travel.mp4", "3. Pair travel: 1 m with a turn, stop at a mark")
            cam = Camera(w, reel, [0.2, 0.25, 0.02], 1.25, 225, -40, follow=[0, 1], slow=0.6)
            cam.still_at = 4.0
            w.watch = cam
            reel.caption = "A bonded pair drives as a differential robot: 0.5 m, a quarter turn on the spot, 0.5 m to the red mark.\nClosed loop on the camera pose (1 mm, 1 degree noise per frame)."
        w.run(0.3)
        t0 = w.data.time
        P.goto([0, 1], np.array([0.0, 0.5, 0.0]))
        P.goto([0, 1], np.array([0.5, 0.5, 0.0]))
        t = w.data.time - t0
        c = (w.pos(0) + w.pos(1)) / 2
        miss = float(np.linalg.norm(c[:2] - [0.5, 0.5]))
        path = 0.5 + 0.5
        ok = miss <= 0.01 and frozenset((0, 1)) in w.bonds and not w.harm
        if filmed:
            cam.reel.caption = f"1 m in {t:.1f} s ({path / t:.3f} m/s including the turn), stopped {miss * 1000:.1f} mm from the mark (measured)."
            cam.pause(1.5)
            finish("travel", cam.reel, cam)
        return common(w, ok, "" if ok else f"miss {miss * 1000:.1f} mm", speed_m_s=round(path / t, 4), time_s=round(t, 2), stop_error_mm=round(miss * 1000, 2))

    def pivot_one(seed, filmed):
        rng = np.random.default_rng(300 + seed)
        w = World([Placement((0.0, 0.0), down="+y", yaw=math.radians(rng.uniform(-1.5, 1.5)))], seed=seed)
        P = Planner(w)
        cam = None
        if filmed:
            reel = Reel(OUT / "travel_pivots.mp4", "3b. The same metre by single-cube pivots")
            cam = Camera(w, reel, [0.25, 0.25, 0.02], 1.25, 225, -40, follow=[0], slow=1.0)
            cam.still_at = 6.0
            w.watch = cam
            reel.caption = "One cube rolls the same path by floor pivots: 10 quarter turns along y, 10 along x (a turn costs nothing).\nNo latch on a flat floor: each edge holds by friction."
        w.run(0.3)
        t0 = w.data.time
        outs = []
        for k in range(20):
            outs.append(P.roll(0, [0, 1, 0] if k < 10 else [1, 0, 0]))
        t = w.data.time - t0
        p = w.pos(0)
        goal = np.array([0.5, 0.5])
        miss = float(np.linalg.norm(p[:2] - goal))
        ok = all(o in ("arrived", "near") for o in outs) and not w.harm
        if filmed:
            cam.reel.caption = f"1 m in {t:.1f} s ({1.0 / t:.3f} m/s); ended {miss * 1000:.0f} mm from the lattice target (measured, open loop)."
            cam.pause(1.5)
            finish("travel_pivots", cam.reel, cam)
        return common(w, ok, "" if ok else f"pivots {outs}", speed_m_s=round(1.0 / t, 4), time_s=round(t, 2), stop_error_mm=round(miss * 1000, 2))

    pair_results = run_seeds("travel", pair_one, True)
    pivot_results = run_seeds("travel_pivots", pivot_one, True)
    ps = [r["speed_m_s"] for r in pair_results if "speed_m_s" in r]
    vs = [r["speed_m_s"] for r in pivot_results if "speed_m_s" in r]
    comparison = {"pair_speed_m_s_mean": round(float(np.mean(ps)), 4), "pivot_speed_m_s_mean": round(float(np.mean(vs)), 4), "pair_over_pivots": round(float(np.mean(ps) / np.mean(vs)), 2),
                  "pivot_runs": {"success_rate": round(sum(r["ok"] for r in pivot_results) / len(pivot_results), 2),
                                 "stop_error_mm_mean": round(float(np.mean([r["stop_error_mm"] for r in pivot_results if "stop_error_mm" in r])), 2),
                                 "time_s_mean": round(float(np.mean([r["time_s"] for r in pivot_results if "time_s" in r])), 2),
                                 "pivot_peak_moving_nm_max": round(max(r.get("pivot_peak_moving_nm", 0) for r in pivot_results), 4),
                                 "per_seed": pivot_results}}
    return record("travel", pair_results, "pair stops within 10 mm of the mark after 1 m with a quarter turn, still bonded; no harm",
                  "Speed includes accelerating, slowing to the mark and the turn on the spot. The single-cube comparison rolls 20 quarter turns (10 along y, 10 along x) on a flat floor with no latch, "
                  "open loop; its stop error is to the ideal lattice point.", comparison)


# ---------------------------------------------------------------------- 4. split


def split(args):
    def one(seed, filmed):
        rng = np.random.default_rng(400 + seed)
        yaw = rng.uniform(-math.pi, math.pi)
        w = World(pair_placements(rng, (0.0, 0.0), yaw), seed=seed, bonds=[(0, 1, None)])
        P = Planner(w)
        cam = None
        if filmed:
            reel = Reel(OUT / "split.mp4", "4. Split: the pair unbonds, each cube goes its own way")
            cam = Camera(w, reel, [0.0, 0.0, 0.02], 0.9, 135, -35, follow=[0, 1], slow=0.6)
            cam.still_at = 3.0
            w.watch = cam
            reel.caption = "The pair drives 20 cm and stops; the connectors switch off."
        w.run(0.3)
        c, f, _ = P.group_pose([0, 1], 3)
        P.follow([0, 1], c, c + f * 0.2, speed=0.12)
        w.unbond(0, 1)
        w.run(0.3)
        if filmed:
            cam.reel.caption = "One cube drives on 20 cm; the other turns 90 degrees by three pivots and drives 20 cm its own way."
        check = guard(w, [], "split")
        p0 = w.pos(0)
        P.drive_straight(0, 0.20)
        outs = P.turn_heading(1, -w.rolling(0))
        P.drive_straight(1, 0.20)
        check()
        apart = float(np.linalg.norm((w.pos(0) - w.pos(1))[:2]))
        ok = frozenset((0, 1)) not in w.bonds and apart >= 0.15 and all(o in ("arrived", "near") for o in outs) and w.wheel_down(0) and w.wheel_down(1) and not w.harm
        if filmed:
            cam.reel.caption = f"Apart by {apart * 100:.0f} cm, both on their wheels (measured)."
            cam.pause(1.5)
            finish("split", cam.reel, cam)
        del p0
        return common(w, ok, "" if ok else f"apart {apart:.3f}, pivots {outs}", separation_m=round(apart, 3))

    results = run_seeds("split", one, True)
    return record("split", results, "after unbonding the cubes end >= 15 cm apart, both wheel down, all pivots arrived; no harm",
                  "Unbonding is switching the weld off; nothing holds the cubes but friction, so nothing springs apart.")


# ---------------------------------------------------------------------- 5. reorient then pair


def reorient(args):
    def one(seed, filmed):
        rng = np.random.default_rng(500 + seed)
        down = [f for f in FACES if f != WHEEL_FACE][seed % 5]
        w = World(loose_two(rng, (0.0, 0.0), spread=0.08, apart=0.15, downs=(WHEEL_FACE, down)), seed=seed)
        P = Planner(w)
        cam = None
        if filmed:
            reel = Reel(OUT / "reorient.mp4", "5. Reorient: wheel face not down, then pair")
            cam = Camera(w, reel, [0.0, 0.0, 0.02], 0.85, 120, -35, follow=[0, 1], slow=0.5)
            cam.still_at = 2.5
            w.watch = cam
            reel.caption = f"The right cube lies on its {down} face, wheel not down. It pivots about floor edges until the wheel is down\n(one pivot if the wheel faces sideways, two if it faces up), then the two pair up."
        w.run(0.3)
        away = horizontal(w.pos(1) - w.pos(0))
        used = P.wheel_down(1, away)
        w.run(0.3)
        ok = P.pair_up(0, 1)
        check = guard(w, [], "reorient")
        check()
        ok = ok and frozenset((0, 1)) in w.bonds and used <= 2 and not w.harm
        if filmed:
            cam.reel.caption = f"{used} pivot(s) to put the wheel down, then paired; {w.failed_docks} failed dock(s) (measured)."
            cam.pause(1.5)
            finish("reorient", cam.reel, cam)
        return common(w, ok, "" if ok else "not paired", reorient_pivots=used, start_face_down=down)

    results = run_seeds("reorient", one, True)
    return record("reorient", results, "a cube lying on a non-wheel face is wheel down after at most 2 pivots and then pairs; no harm",
                  "Seeds cycle through the five non-wheel faces (+x disc down, +y, -y, +z, -z), two seeds each.")


# ---------------------------------------------------------------------- 6. climb


def climb(args):
    def one(seed, filmed):
        rng = np.random.default_rng(600 + seed)
        yaw = rng.uniform(-math.pi, math.pi)
        n = np.array([-math.sin(yaw), math.cos(yaw)])
        cells = [np.array([0.0, 0.0]) + n * k * (SIDE + 0.0005) + rng.uniform(-0.00075, 0.00075, 2) for k in range(3)]
        pls = [Placement(tuple(c), yaw=yaw + math.radians(rng.uniform(-1.5, 1.5))) for c in cells]
        w = World(pls, seed=seed, bonds=[(0, 1, None), (1, 2, None)])
        P = Planner(w)
        cam = None
        if filmed:
            reel = Reel(OUT / "climb.mp4", "6. Climb: a cube pivots up onto its neighbour and back")
            cam = Camera(w, reel, [0.0, 0.0, 0.04], 0.6, math.degrees(yaw) + 180 - 60, -20, follow=[0, 1, 2], slow=1.5)
            cam.still_at = 1.2
            w.watch = cam
            reel.caption = "Three cubes bonded in a row (wheels down). The end cube latches the top edge it shares with its neighbour,\nunbonds, and turns half a turn about it (convex up), then bonds on top (shown at 2/3 speed)."
            cam.pause(1.5)
        w.run(0.3)
        check = guard(w, [], "climb")
        up = P.climb(2, 1)
        on_top = w.pos(2)[2] - w.pos(1)[2]
        bonded_up = frozenset((1, 2)) in w.bonds
        if filmed:
            cam.reel.caption = f"Up: {up}, bonded on top: {bonded_up}. Now back down to the floor cell it came from."
            cam.pause(1.5)
        w.run(0.3)
        down = P.climb_down(2, 1, horizontal(w.pos(1) - w.pos(0)))
        check()
        back = frozenset((1, 2)) in w.bonds and abs(w.pos(2)[2] - w.pos(1)[2]) < 0.004
        rel = w.rot(1).T @ (w.pos(2) - w.pos(1))
        ok = up in ("arrived", "near") and bonded_up and abs(on_top - SIDE) < 0.004 and down in ("arrived", "near") and back and not w.harm
        if filmed:
            cam.reel.caption = f"Down: {down}, bonded beside it again: {back}. Peak pivot torque {max(p[1] for p in w.pivot_peaks):.3f} N m against a 0.1 N m limit (measured)."
            cam.pause(1.5)
            finish("climb", cam.reel, cam)
        del rel
        return common(w, ok, "" if ok else f"up {up} bonded {bonded_up}, down {down} back {back}", height_on_top_mm=round(on_top * 1000, 2))

    results = run_seeds("climb", one, True)
    return record("climb", results, "the end cube of a bonded row of three climbs onto its neighbour (bonded, 50 +/- 4 mm above it) and back down to the floor (bonded beside it); no harm",
                  "The climb is a half turn about the top edge the two cubes share, latched to the neighbour (cube_hinge's latch: two soft point constraints, engage within 3 mm). "
                  "The row stands on three wheels in a line and leans 3.4 degrees on one edge; the hinge axis follows the support's real edge.")


# ---------------------------------------------------------------------- 7. assembly


ZONES = [(0.0, -0.42), (0.40, 0.25), (-0.40, 0.25)]


def assembly(args):
    def one(seed, filmed):
        rng = np.random.default_rng(700 + seed)
        pls = []
        for z in ZONES:
            downs = tuple(FACES[int(k)] for k in rng.integers(0, 6, 2))
            pls += loose_two(rng, z, spread=0.07, apart=0.14, downs=downs)
        site = np.array([0.0, 0.0, 0.0])
        w = World(pls, seed=seed, scene=mark((0.0, 0.0), (0.004, 0.004), "0.3 0.3 0.3 1"))
        P = Planner(w)
        cam = None

        def say(text):
            if filmed:
                cam.reel.caption = text
        if filmed:
            reel = Reel(OUT / "assembly.mp4", "7. Assembly: six loose cubes build a two-layer staircase")
            cam = Camera(w, reel, [0.0, 0.0, 0.03], 1.35, 150, -45, slow=0.3)
            w.watch = cam
            say("Six loose cubes in three places, lying on random faces at random headings. No help: every move is a wheel,\na pivot or a connector, poses come from a noisy camera, a bond forms only inside the capture range.")
        w.run(0.5)
        t0 = w.data.time
        stage = {}
        pairs = [(0, 1), (2, 3), (4, 5)]
        for a, b in pairs:
            say("Each cube that is not on its wheel pivots onto it (at most two pivots); then the two pair up.")
            for i, j in ((a, b), (b, a)):
                P.wheel_down(i, horizontal(w.pos(i) - w.pos(j)))
            others = [k for k in range(6) if k not in (a, b)]
            check = guard(w, others, f"pairing {a}-{b}", moved=0.01)
            if not P.pair_up(a, b):
                return common(w, False, f"pairing {a}-{b} failed", stage=stage)
            check()
        stage["paired_s"] = round(w.data.time - t0, 1)
        if filmed:
            cam.follow = None
            cam.camera.lookat = np.array([0.05, 0.0, 0.03])
            cam.camera.distance = 0.75
        say("The first pair drives to the site and stands with its axles along the row it will start.")
        check = guard(w, [2, 3, 4, 5], "pair 1 to site")
        P.goto([0, 1], site, final=[0, 1, 0])
        check()
        p0, p1 = P.look(0, 3)[0], P.look(1, 3)[0]
        a_row = horizontal(p1 - p0) if (p1 - p0)[0] > 0 else horizontal(p0 - p1)
        say("The other pairs drive over and slide in at the end of the row, axles in line\n(the faces they bond on are normal to the axles, so the cubes' 3.4 degree lean stays in the face plane).")
        for g in ((2, 3), (4, 5)):
            row = sorted([k for k in range(6) if k in w.group_of(0)], key=lambda k: float(P.look(k)[0] @ a_row))
            end = row[-1]
            pe = P.look(end, 3)[0]
            idle = [k for k in range(6) if k not in g and k not in row]
            check = guard(w, idle, f"pair {g} docks")
            if not P.slide_dock(list(g), [pe + a_row * SIDE, pe + 2 * a_row * SIDE], end):
                return common(w, False, f"slide dock of {g} failed", stage=stage)
            check()
        stage["row_s"] = round(w.data.time - t0, 1)
        row = sorted(range(6), key=lambda k: float(P.look(k)[0] @ a_row))
        say("Second layer: the end cube climbs onto its neighbour (half turn about the shared top edge),\nrolls two cells along the top, and the next cube climbs after it.")
        steps = [("climb", row[5], row[4]), ("traverse", row[5], row[3]), ("traverse", row[5], row[2]), ("climb", row[4], row[3])]
        for kind, mover, other in steps:
            out = P.climb(mover, other) if kind == "climb" else P.traverse(mover, other)
            w.check_harm(range(6), f"{kind} {mover}")
            if out not in ("arrived", "near"):
                return common(w, False, f"{kind} {mover} onto {other}: {out}", stage=stage)
        w.run(1.0)
        t = w.data.time - t0
        # check the result against the plan, in the frame of row[0]
        o, r = w.pos(row[0]), w.rot(row[0])
        x = a_row
        expected = {row[0]: (0, 0), row[1]: (1, 0), row[2]: (2, 0), row[3]: (3, 0), row[5]: (2, 1), row[4]: (3, 1)}
        errors = []
        for k, (i, layer) in expected.items():
            p = w.pos(k) - o
            errors.append(math.hypot(float(p @ x) - i * SIDE, float(p[2]) - layer * SIDE))
        wanted = [(row[0], row[1]), (row[1], row[2]), (row[2], row[3]), (row[2], row[5]), (row[3], row[4]), (row[4], row[5])]
        bonds = all(frozenset(b) in w.bonds for b in wanted)
        aligned = max(nearest_lattice(r.T @ w.rot(k))[1] for k in range(6))
        ok = bonds and max(errors) < 0.004 and aligned < 3.0 and not w.harm
        if filmed:
            say(f"Built in {t:.0f} s (one cube or pair moving at a time): two layers, all {len(wanted)} bonds closed.\n{w.pivot_count} pivots, {w.driven:.1f} cube-metres on wheels, {w.failed_docks} failed dock(s), {len(w.harm)} harmful event(s) (measured).")
            cam.camera.distance = 0.55
            cam.pause(2.5)
            finish("assembly", cam.reel, cam)
        return common(w, ok, "" if ok else f"bonds {bonds}, cell error {max(errors) * 1000:.1f} mm, misaligned {aligned:.1f} deg",
                      assembly_s=round(t, 1), cell_error_mm_max=round(max(errors) * 1000, 2), misalignment_deg_max=round(aligned, 2), stage=stage)

    results = run_seeds("assembly", one, True)
    return record("assembly", results, "six loose cubes end as a two-layer staircase (floor row of four, two on top over the last two), every planned bond closed, every cube within 4 mm and 3 degrees of its cell; no harm",
                  "Scripted planner for this goal: reorient, pair (three pairs, one after another), pair 1 to the site, pairs 2 and 3 slide in at the end of the row (axle direction), "
                  "then climb, roll, roll, climb. Cubes start in three areas 0.4-0.45 m from the site (two per area, random face down and heading) so lone cubes, which cannot steer, "
                  "do not drive through each other; that placement is a planner convenience. A 2x2x2 block was not attempted as the final plan: it needs pairs to dock front to back, "
                  "which fails the capture range (see the memo).")


# ---------------------------------------------------------------------- front-to-back pair docking (the failure that shaped the assembly)


def front_dock(args):
    def one(seed, filmed):
        rng = np.random.default_rng(800 + seed)
        y2 = rng.uniform(-math.pi, math.pi)
        w = World(pair_placements(rng, (0.0, 0.0), rng.uniform(-0.3, 0.3)) + pair_placements(rng, (0.32, 0.25), y2), seed=seed, bonds=[(0, 1, None), (2, 3, None)])
        P = Planner(w)
        cam = None
        if filmed:
            reel = Reel(OUT / "front_dock.mp4", "Failure: two pairs docking front to back")
            cam = Camera(w, reel, [0.12, 0.08, 0.03], 0.8, 160, -30, slow=0.5)
            cam.still_at = 13.0
            w.watch = cam
            reel.caption = "A pair docks head on into another pair's front faces. Each pair leans 3.4 degrees on an edge (two wheels in a line)."
        w.run(0.3)
        P.goto([0, 1], np.array([0.1, 0.1, 0.0]), final=[1, 0, 0])
        c, f, _ = P.group_pose([0, 1], 3)
        x1 = horizontal(f if f[0] > 0 else -f)
        ok = P.dock([2, 3], [0, 1], approach=-x1)
        docks = [e for e in w.events if "dock" in e]
        last = docks[-1] if docks else {}
        if filmed:
            cam.reel.caption = (f"Leaning the same way, the faces are {max(last.get('lateral_mm', [0])):.1f} mm out of line (2 x 1.5 mm proud wheel = 3.0 mm);\n"
                                f"leaning opposite ways they are 7 degrees apart. Capture is +/- 3 mm, +/- 5 degrees: docked {ok} (measured).")
            cam.pause(2.0)
            finish("front_dock", cam.reel, cam)
        return common(w, ok, "" if ok else f"last attempt lateral {last.get('lateral_mm')} mm, angle {last.get('angle_deg')} deg, gap {last.get('gap_mm')} mm",
                      lateral_mm_last=max(last.get("lateral_mm", [0])), angle_deg_last=max(last.get("angle_deg", [0])))

    results = run_seeds("front_dock", one, True)
    return record("front_dock", results, "a pair docks head on into another pair's front faces (both bonds inside the capture range) within 3 attempts",
                  "Kept as a failure. A cube on its wheel leans 3.4 degrees (1.5 mm over a 25 mm half side). Faces normal to the lean direction meet 2 x 1.5 = 3.0 mm out of line when both lean "
                  "the same way, or 7 degrees apart when they lean opposite ways; the spec's capture range is +/- 3 mm and +/- 5 degrees. Pressing harder makes them lean apart. "
                  "Faces normal to the axle (side by side, end to end) are not affected, which is why pairing and the assembly use only those.")


# ---------------------------------------------------------------------- packing


def pack(args):
    result = packing()
    path = OUT / "packing.json"
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result))
    return result


DEMOS = {"lone": lone, "pairing": pairing, "travel": travel, "split": split, "reorient": reorient, "climb": climb, "assembly": assembly, "front_dock": front_dock}
CLIPS = ["lone", "pairing", "travel", "travel_pivots", "split", "reorient", "climb", "front_dock", "assembly"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("what", nargs="+")
    parser.add_argument("--seeds", default=None, help="comma list: run only these seeds, no film, no record")
    args = parser.parse_args()
    global ONLY
    ONLY = [int(x) for x in args.seeds.split(",")] if args.seeds else None
    for what in args.what:
        if what == "all":
            for name in DEMOS:
                subprocess.run(["nice", "-n", "10", sys.executable, __file__, name], check=False)
            subprocess.run([sys.executable, __file__, "packing", "reel", "sheet"], check=True)
        elif what == "packing":
            pack(args)
        elif what == "reel":
            reel_join()
        elif what == "sheet":
            sheet()
        else:
            DEMOS[what](args)


def reel_join():
    names = [n for n in CLIPS if (OUT / f"{n}.mp4").exists()]
    listing = OUT / "list.txt"
    listing.write_text("".join(f"file '{OUT / n}.mp4'\n" for n in names))
    MEDIA.mkdir(parents=True, exist_ok=True)
    out = MEDIA / "module-pairs.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(out)], check=True)
    print("wrote", out, names)


def sheet():
    names = [n for n in CLIPS if (OUT / "stills" / f"{n}.png").exists()]
    rows = {json.loads(line)["demo"]: json.loads(line) for line in (OUT / "results.jsonl").read_text().splitlines() if line.strip()}
    if "travel" in rows:
        rows["travel_pivots"] = {"success_rate": rows["travel"]["pivot_runs"]["success_rate"], "ok": rows["travel"]["pivot_runs"]["success_rate"] >= 0.8}
    cols = 3
    w, h, band = 640, 360, 44
    grid = Image.new("RGB", (cols * w, math.ceil(len(names) / cols) * (h + band)), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    face = font(20)
    for k, name in enumerate(names):
        image = Image.open(OUT / "stills" / f"{name}.png").convert("RGB").resize((w, h))
        x, y = (k % cols) * w, (k // cols) * (h + band)
        grid.paste(image, (x, y))
        row = rows.get(name, {})
        verdict = "works" if row.get("ok") else "FAILS"
        draw.text((x + 12, y + h + 10), f"{name}: {verdict}, success {row.get('success_rate', '?')} over 10 seeds", fill=(20, 20, 20), font=face)
    out = MEDIA / "module-pairs-sheet.png"
    grid.save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
