"""Monte Carlo, calibration, optimisation, outcome model and media for the reaction-wheel cube
(hardware/src/beebot/hardware/rw_cube.py). CPU only, at most 6 worker processes.

    .venv/bin/python hardware/scripts/module_phys/rw_sim.py calibrate --moves roll_floor,climb_row --n 24
    .venv/bin/python hardware/scripts/module_phys/rw_sim.py mc --moves all --n 200
    .venv/bin/python hardware/scripts/module_phys/rw_sim.py opt --pop 24 --gens 8
    .venv/bin/python hardware/scripts/module_phys/rw_sim.py outcomes
    .venv/bin/python hardware/scripts/module_phys/rw_sim.py render

Writes hardware/runs/module_phys/rw_sim/: results.jsonl (one line per move and design), trials/<design>/<move>.jsonl
(one line per trial), calibrate.jsonl, pareto.json, outcomes.json; media to hardware/handoff/media/.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import asdict, replace
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / part / "src") for part in ("v0", "v1", "v2", "hardware")]

from beebot.hardware import rw_cube as rc  # noqa: E402
from beebot.hardware.mmsim.opt import wilson_interval  # noqa: E402

OUT = ROOT / "hardware" / "runs" / "module_phys" / "rw_sim"
MEDIA = ROOT / "hardware" / "handoff" / "media"
WORKERS = 6
MOVES = ["roll_floor", "roll_top", "climb_row", "climb_lone", "descend_row", "descend_lone", "split_pair", "split_row", "dock",
         "yaw", "bond", "fall_pair"]
SUCCESS = {"yaw": {"within_3deg"}, "bond": {"closed"}, "fall_pair": {"bond_held"}, "fall_lone": {"intact"}}
HARM = {"dislodged"}


def success_set(move):
    return SUCCESS.get(move, {"reached"})


def design_from(dd: dict) -> rc.Design:
    return rc.Design(**dd)


def one_trial(args):
    move, dd, seed, opts = args
    d = design_from(dd)
    t0 = time.time()
    try:
        if move == "yaw":
            tr = rc.run_yaw(d, seed, retries=opts.get("retries", 2))
        elif move == "bond":
            tr = rc.run_bond(d, seed)
        elif move in ("fall_pair", "fall_lone"):
            tr = rc.run_fall(d, seed, pair=(move == "fall_pair"))
        else:
            k = opts.get("k")
            if k is None and opts.get("kscale") is not None:
                kind = rc.MOVE_KIND[move]
                k = {"roll": d.k_roll, "top": d.k_top, "climb": d.k_climb, "descend": d.k_descend, "split": d.k_split}[kind] * opts["kscale"]
            tr = rc.run_move(d, move, seed, retries=opts.get("retries", 2), escalate=opts.get("escalate", 1.08), k_override=k)
        row = {"move": move, "seed": seed, "cls": tr.cls, "cls_first": tr.cls_first, "attempts": tr.attempts,
               "metrics": _clean(tr.metrics), "secs": round(time.time() - t0, 2)}
    except Exception as e:  # a crashed trial is recorded, never dropped
        row = {"move": move, "seed": seed, "cls": "sim_error", "cls_first": "sim_error", "attempts": 0,
               "metrics": {"error": repr(e)[:200]}, "secs": round(time.time() - t0, 2)}
    return row


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(round(float(x), 6))
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def run_pool(jobs, workers=WORKERS, chunks=1):
    ctx = get_context("spawn")
    with ctx.Pool(min(workers, WORKERS)) as pool:
        return pool.map(one_trial, jobs, chunksize=chunks)


def summarise(move, rows):
    n = len(rows)
    ok = success_set(move)
    k1 = sum(r["cls_first"] in ok for r in rows)
    kr = sum(r["cls"] in ok for r in rows)
    harm_first = sum(r["cls_first"] in HARM for r in rows)
    harm = sum(r["cls"] in HARM for r in rows)
    def iv(k):
        lo, hi = wilson_interval(k, n)
        return {"k": k, "n": n, "rate": k / n if n else 0.0, "lo": lo, "hi": hi}
    return {"single_try": iv(k1), "with_retry": iv(kr), "harmful_single": iv(harm_first), "harmful_with_retry": iv(harm),
            "classes_first": dict(Counter(r["cls_first"] for r in rows)), "classes_final": dict(Counter(r["cls"] for r in rows)),
            "mean_attempts": float(np.mean([r["attempts"] for r in rows])) if rows else 0.0}


def append_jsonl(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def worst_so_far(design_key, policy=None):
    best = {}
    p = OUT / "results.jsonl"
    if not p.exists():
        return None
    for line in p.read_text().splitlines():
        r = json.loads(line)
        if r.get("design_key") == design_key and r.get("kind") == "mc" and r.get("policy") == policy:
            best[r["move"]] = r["with_retry"]["rate"]
    return min(best.values()) if best else None


def cmd_mc(args):
    dd = json.loads(args.design) if args.design else {}
    d = design_from(dd)
    moves = MOVES if args.moves == "all" else args.moves.split(",")
    policy = args.policy
    for move in moves:
        t0 = time.time()
        seeds = range(args.seed0, args.seed0 + args.n)
        jobs = [(move, asdict(d), s, {"retries": args.retries, "escalate": args.escalate, "kscale": args.kscale}) for s in seeds]
        rows = run_pool(jobs)
        tdir = OUT / "trials" / f"{d.key()}__{policy}"
        tdir.mkdir(parents=True, exist_ok=True)
        with open(tdir / f"{move}.jsonl", "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        all_rows = [json.loads(l) for l in (tdir / f"{move}.jsonl").read_text().splitlines()]
        summ = summarise(move, all_rows)
        rec = {"kind": "mc", "move": move, "design_key": d.key(), "policy": policy, "design": asdict(d), "retries": args.retries,
               "escalate": args.escalate, "kscale": args.kscale, **summ, "secs": round(time.time() - t0, 1),
               "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        prev = worst_so_far(d.key(), policy)
        rates = [summ["with_retry"]["rate"]] + ([prev] if prev is not None else [])
        rec["worst_move_success"] = min(rates)
        append_jsonl(OUT / "results.jsonl", rec)
        s1, sr, h = summ["single_try"], summ["with_retry"], summ["harmful_with_retry"]
        print(f"{policy:8s} {move:13s} n={sr['n']:5d} first {s1['rate']:.3f} [{s1['lo']:.3f},{s1['hi']:.3f}]  final {sr['rate']:.3f} [{sr['lo']:.3f},{sr['hi']:.3f}]"
              f"  harm {h['rate']:.3f}  first {summ['classes_first']}  final {summ['classes_final']}  ({time.time() - t0:.0f}s)", flush=True)


def cmd_calibrate(args):
    """Single-try success over a grid of command multiplier k and catch duration, n MC trials each."""
    dd = json.loads(args.design) if args.design else {}
    moves = args.moves.split(",")
    ks = [float(x) for x in args.ks.split(",")]
    tcs = [float(x) for x in args.tcs.split(",")]
    jobs = []
    for move in moves:
        for tc in tcs:
            d = design_from({**dd, "catch_T": tc})
            for k in ks:
                for s in range(args.n):
                    jobs.append((move, asdict(d), 10_000 + s, {"retries": 0, "k": k}))
    t0 = time.time()
    rows = run_pool(jobs, chunks=2)
    i = 0
    for move in moves:
        for tc in tcs:
            line = []
            for k in ks:
                rr = rows[i: i + args.n]
                i += args.n
                summ = summarise(move, rr)
                rec = {"kind": "calibrate", "move": move, "catch_T": tc, "k": k, "design": dd, **summ}
                append_jsonl(OUT / "calibrate.jsonl", rec)
                c = summ["classes_first"]
                line.append(f"k{k}:{summ['single_try']['rate']:.2f}/h{summ['harmful_single']['rate']:.2f}")
            print(f"{move:13s} Tc {tc * 1e3:6.2f} ms  " + "  ".join(line), flush=True)
    print(f"{len(rows)} trials in {time.time() - t0:.0f}s")


# ----------------------------------------------------------------------------------------------
# Outcome model
# ----------------------------------------------------------------------------------------------

LATTICE = {  # detailed class -> what the voxel grid can show
    "reached": "reached", "stayed": "stayed", "fell_back": "stayed", "held": "stayed",
    "dislodged": "harm", "double_roll": "off", "strayed": "off", "off_lattice": "off", "slid_off": "off", "fell_off": "off",
    "overshot": "off", "reached_unbonded": "off", "sim_error": "off", "over_speed_limit": "stayed",
    "within_3deg": "reached", "not_within_3deg": "stayed", "tipped": "off", "closed": "reached", "gap_left": "stayed",
    "closed_tilted": "off", "bond_held": "reached", "bond_broke": "off", "intact": "reached", "no_rotor_along_axis": "off",
}

CLASS_TEXT = {
    "reached": "on the target cell, flat, turned as commanded (a quarter turn, or a half turn for climbs and descents), bonded where the move ends on a structure; every neighbour within 0.1 L and 10 degrees of where it was",
    "stayed": "back on the start cell in the start orientation (too little impulse); retryable",
    "fell_back": "climb: swung up and fell back onto the start cell; retryable",
    "held": "split: the bond held, the mover is back beside its neighbour; retryable",
    "dislodged": "harmful: a neighbour moved more than 0.1 L or turned more than 10 degrees (cube.DISTURBED)",
    "double_roll": "rolled two cells",
    "strayed": "flat and on the lattice but in another cell or orientation than the target or start",
    "off_lattice": "not on a lattice site: more than 0.25 L from a cell centre (slid or hopped), or not flat",
    "slid_off": "roll on top: ended on the floor beside the structure",
    "fell_off": "climb: ended on the floor somewhere else",
    "overshot": "climb: on the top layer but past the target",
    "reached_unbonded": "on the target cell but not flush against the cube below",
    "within_3deg": "yaw: heading within 3 degrees of the command, still flat",
    "not_within_3deg": "yaw: heading error above 3 degrees after the tries",
    "tipped": "yaw: the cube tipped",
    "closed": "bond: flush, aligned face bond", "gap_left": "bond: faces not closed", "closed_tilted": "bond: closed but tilted",
    "no_rotor_along_axis": "the mover is turned more than 8 degrees off the grid before the (re)try (it slid or twisted while settling or after an earlier try), so no flywheel lies along the roll axis",
    "bond_held": "fall: the pair is still bonded after the fall", "bond_broke": "fall: the bond opened", "sim_error": "the simulation diverged (counted as a failure)",
}


def _probs(counter, n):
    return {k: v / n for k, v in sorted(counter.items(), key=lambda kv: -kv[1])}


def compile_outcomes():
    designs = {}
    tdir = OUT / "trials"
    for ddir in sorted(tdir.iterdir()) if tdir.exists() else []:
        prims = {}
        dd = None
        for f in sorted(ddir.glob("*.jsonl")):
            rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
            if not rows:
                continue
            move = f.stem
            n = len(rows)
            first = Counter(r["cls_first"] for r in rows)
            final = Counter(r["cls"] for r in rows)
            ok = success_set(move)
            def iv(k):
                lo, hi = wilson_interval(k, n)
                return {"k": k, "n": n, "rate": k / n, "lo": lo, "hi": hi}
            lat1, latr = Counter(), Counter()
            for c, v in first.items():
                lat1[LATTICE.get(c, "off")] += v
            for c, v in final.items():
                latr[LATTICE.get(c, "off")] += v
            harm = [{"seed": r["seed"], "neighbour_moved_mm": r["metrics"].get("neighbour_moved_mm"),
                     "neighbour_rot_deg": r["metrics"].get("neighbour_rot_deg"), "mover_cell": r["metrics"].get("cell")}
                    for r in rows if r["cls"] in HARM][:50]
            by_mu = {}
            for lo_, hi_ in ((0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6)):
                sel = [r for r in rows if lo_ <= (r["metrics"].get("var") or {}).get("mu_floor", -1) < hi_]
                if sel:
                    by_mu[f"{lo_:.1f}-{hi_:.1f}"] = {"n": len(sel), "single_try": sum(r["cls_first"] in ok for r in sel) / len(sel),
                                                     "with_retry": sum(r["cls"] in ok for r in sel) / len(sel)}
            by_face = {}
            for r in rows:
                fd = r["metrics"].get("face_down")
                if fd:
                    e = by_face.setdefault(fd, [0, 0, 0])
                    e[0] += 1
                    e[1] += r["cls_first"] in ok
                    e[2] += r["cls"] in ok
            prims[move] = {
                "n": n,
                "single_try": {"success": iv(sum(first[c] for c in ok)), "probs": _probs(first, n), "counts": dict(first),
                               "lattice": _probs(lat1, n)},
                "with_retry": {"success": iv(sum(final[c] for c in ok)), "probs": _probs(final, n), "counts": dict(final),
                               "lattice": _probs(latr, n), "mean_attempts": float(np.mean([r["attempts"] for r in rows]))},
                "harmful": {"single_try": iv(sum(first[c] for c in HARM)), "with_retry": iv(sum(final[c] for c in HARM)),
                            "aftermaths": harm},
                "by_floor_friction": by_mu,
                "by_face_down": {k: {"n": v[0], "single_try": v[1] / v[0], "with_retry": v[2] / v[0]} for k, v in sorted(by_face.items())},
            }
        res = [json.loads(l) for l in (OUT / "results.jsonl").read_text().splitlines()] if (OUT / "results.jsonl").exists() else []
        policy = None
        for r in res:
            if f"{r.get('design_key')}__{r.get('policy')}" == ddir.name:
                dd = r["design"]
                policy = {"name": r.get("policy"), "retries": r.get("retries"), "escalate": r.get("escalate"), "kscale": r.get("kscale")}
        if prims:
            worst = min(v["with_retry"]["success"]["rate"] for v in prims.values())
            designs[ddir.name] = {"design": dd, "policy": policy, "floor": "steel sheet" if (dd or {}).get("steel_floor") else "bare (non-magnetic)", "primitives": prims, "worst_move_success_with_retry": worst,
                                  "worst_move": min(prims, key=lambda m: prims[m]["with_retry"]["success"]["rate"])}
    out = {"name": "rw_cube_sim", "generated": time.strftime("%Y-%m-%d %H:%M"), "source": "hardware/scripts/module_phys/rw_sim.py (MuJoCo, hardware/src/beebot/hardware/rw_cube.py)",
           "note": "Outcome class probabilities per move primitive, design point and policy. Policy 'single': one try at the calibrated impulse (single_try == with_retry). Policy 'escalate': first try at 0.893 x the calibrated impulse, then after stayed / fell_back / held a re-try at 1.06 x the previous impulse, at most 5 re-tries; single_try is that policy's first try, with_retry its end state. Harmful outcomes included. 'lattice' maps the detailed classes to what the voxel grid shows (reached, stayed, harm, off).",
           "class_definitions": CLASS_TEXT, "lattice_map": LATTICE, "idealisations": rc.IDEALISATIONS, "designs": designs}
    (OUT / "outcomes.json").write_text(json.dumps(out, indent=1))
    print("wrote", OUT / "outcomes.json", list(designs))


def cmd_outcomes(args):
    compile_outcomes()


# ----------------------------------------------------------------------------------------------
# Media
# ----------------------------------------------------------------------------------------------

SIZE = (1280, 720)
FPS = 30


def _font(size):
    from PIL import ImageFont
    for path in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


class Film:
    def __init__(self, path, title):
        import subprocess
        path.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SIZE[0]}x{SIZE[1]}", "-r", str(FPS), "-i", "-",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(path)]
        self.pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.title, self.caption = title, ""
        self.f1, self.f2 = _font(30), _font(24)
        self.last = None
        self.raw = None

    def add(self, img, repeat=1):
        from PIL import Image, ImageDraw
        pic = Image.fromarray(img).convert("RGB")
        dr = ImageDraw.Draw(pic, "RGBA")
        dr.rectangle([0, 0, 60 + dr.textlength(self.title, font=self.f1), 56], fill=(0, 0, 0, 150))
        dr.text((30, 12), self.title, fill=(255, 255, 255, 255), font=self.f1)
        if self.caption:
            lines = self.caption.split("\n")
            top = SIZE[1] - 22 - 30 * len(lines)
            dr.rectangle([0, top - 10, SIZE[0], SIZE[1]], fill=(0, 0, 0, 170))
            for i, line in enumerate(lines):
                dr.text((30, top + 30 * i), line, fill=(255, 255, 255, 255), font=self.f2)
        self.last = pic
        self.raw = img
        self.pipe.stdin.write(pic.tobytes() * repeat)

    def close(self):
        self.pipe.stdin.close()
        self.pipe.wait()


def film_trial(move, dd, seed, film, caption, slow=0.04, opts=None):
    import mujoco
    opts = opts or {}
    d = design_from(dd)
    state = {"r": None, "cam": None}

    def cb(sim):
        if state["r"] is None:
            state["r"] = mujoco.Renderer(sim.m, SIZE[1], SIZE[0])
            cam = mujoco.MjvCamera()
            p = np.mean([sim.pose(k)[0] for k in range(sim.n)], axis=0)
            cam.lookat = p + np.array([0.3 * d.L, 0, 0.6 * d.L])
            cam.distance = 7.5 * d.L if sim.n > 1 else 6 * d.L
            cam.azimuth, cam.elevation = 110, -22
            state["cam"] = cam
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
        state["r"].update_scene(sim.data, camera=state["cam"], scene_option=opt)
        film.caption = caption + f"\nt = {sim.data.time * 1e3:6.1f} ms (slowed {1 / slow:.0f}x)"
        film.add(state["r"].render())

    every = slow / FPS
    if move == "yaw":
        tr = rc.run_yaw(d, seed, retries=opts.get("retries", 2), callback=cb, every=every)
    elif move == "bond":
        tr = rc.run_bond(d, seed, callback=cb, every=every)
    elif move.startswith("fall"):
        tr = rc.run_fall(d, seed, pair=(move == "fall_pair"), callback=cb, every=every * 0.25)
    else:
        tr = rc.run_move(d, move, seed, retries=opts.get("retries", 0), k_override=opts.get("k"), callback=cb, every=every)
    film.caption = caption + f"\nOutcome: {tr.cls} (first try: {tr.cls_first}, {tr.attempts} tr{'y' if tr.attempts == 1 else 'ies'})"
    if film.raw is not None:
        film.add(film.raw, repeat=int(1.5 * FPS))
    return tr


def pick_seeds(design_key, move, policy="single"):
    f = OUT / "trials" / f"{design_key}__{policy}" / f"{move}.jsonl"
    rows = [json.loads(l) for l in f.read_text().splitlines()] if f.exists() else []
    ok = success_set(move)
    good = [r for r in rows if r["cls_first"] in ok]
    bad = [r for r in rows if r["cls_first"] not in ok]
    picks = []
    if good:
        good.sort(key=lambda r: r["metrics"].get("offset_mm", 0.0))
        picks.append(("best", good[0]))
        picks.append(("typical", good[len(good) // 2]))
    if bad:
        common = Counter(r["cls_first"] for r in bad).most_common(1)[0][0]
        picks.append(("failure", next(r for r in bad if r["cls_first"] == common)))
    return picks, len(rows), (sum(r["cls"] in ok for r in rows) / len(rows) if rows else None)


def cmd_render(args):
    import subprocess
    from PIL import Image, ImageDraw
    dd = json.loads(args.design) if args.design else {}
    d = design_from(dd)
    moves = MOVES if args.moves == "all" else args.moves.split(",")
    clips = OUT / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    stills = {}
    names = []
    for move in moves:
        picks, n, rate = pick_seeds(d.key(), move)
        for label, row in picks:
            name = f"{move}_{label}"
            film = Film(clips / f"{name}.mp4", f"{move.replace('_', ' ')}: {label} seed (L = {d.L * 1e3:.0f} mm)")
            m = row["metrics"]
            var = m.get("var") or {}
            cap = (f"{move}: MC seed {row['seed']}, recorded {row['cls_first']} first try, {row['cls']} with retries"
                   + (f"; floor mu {var['mu_floor']:.2f}" if 'mu_floor' in var else ""))
            film_trial(move, asdict(d), row["seed"], film, cap, opts={"retries": 0})
            film.close()
            names.append(name)
            if label == "best" or move not in stills:
                stills[move] = (film.last, rate, n)
            print("filmed", name, flush=True)
    listing = clips / "list.txt"
    listing.write_text("".join(f"file '{clips / n}.mp4'\n" for n in names))
    MEDIA.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy",
                    str(MEDIA / "rw-cube.mp4")], check=True)
    cols, w, h, band = 4, 480, 270, 40
    grid = Image.new("RGB", (cols * w, math.ceil(len(stills) / cols) * (h + band)), (255, 255, 255))
    dr = ImageDraw.Draw(grid)
    f = _font(18)
    for i, (move, (img, rate, n)) in enumerate(stills.items()):
        x, y = (i % cols) * w, (i // cols) * (h + band)
        if img is not None:
            grid.paste(img.resize((w, h)), (x, y))
        dr.text((x + 8, y + h + 8), f"{move}: {'%.3f' % rate if rate is not None else '?'} single try, n = {n}", fill=(20, 20, 20), font=f)
    grid.save(MEDIA / "rw-cube-sheet.png")
    print("wrote", MEDIA / "rw-cube.mp4", MEDIA / "rw-cube-sheet.png")


# ----------------------------------------------------------------------------------------------
# NSGA-II over the design
# ----------------------------------------------------------------------------------------------

OPT_VARS = ["L_mm", "sphere_frac", "clearance_um", "rotor_fit", "rotor_t_mm", "log10_catch_T", "k_scale"]
OPT_LO = [6.0, 0.09, 5.0, 0.6, 0.3, -4.3, 0.9]
OPT_HI = [12.0, 0.20, 50.0, 1.0, 1.5, -2.3, 1.15]
OPT_MOVES = ["roll_floor", "roll_top", "climb_row", "descend_row", "split_row", "dock"]
OPT_N = int(os.environ.get("RW_OPT_N", "8"))
BASE = {}  # calibrated k per move, set by --base


def x_to_design(x, base=None):
    base = dict(base or {})
    L = x[0] * 1e-3
    t = x[4] * 1e-3
    inner = L - 2 * rc.WALL
    r_max = inner / 2 - t - 0.15e-3
    r = max(0.3e-3, x[3] * r_max)
    ks = {k: base.get(k, getattr(rc.Design, k)) * x[6] for k in ("k_roll", "k_top", "k_climb", "k_descend", "k_split")}
    base.update(dict(L=L, sphere_d=x[1] * L, clearance=x[2] * 1e-6, rotor_r=r, rotor_t=t, catch_T=10 ** x[5], **ks))
    return rc.Design(**base)


def energy_per_move(d, rows):
    der = rc.derive(d)
    es = []
    for move in set(r["move"] for r in rows):
        om = [r["metrics"].get("omega_cmd") for r in rows if r["move"] == move and r["metrics"].get("omega_cmd")]
        if om:
            sp = rc.spinup(d, der, float(np.mean(om)))
            es.append(sp["energy"] if sp["ok"] else 10.0)
    return float(np.mean(es)) if es else 10.0


def opt_objective(x):
    d = x_to_design(x, BASE_K)
    der = rc.derive(d)
    if not der.feasible:
        return [1.0, x[0], 10.0]
    rows = []
    worst = 1.0
    for move in OPT_MOVES:
        rr = [one_trial((move, asdict(d), 50_000 + s, {"retries": 5, "escalate": 1.06, "kscale": 0.893})) for s in range(OPT_N)]
        rows += rr
        rate = sum(r["cls"] in success_set(move) for r in rr) / len(rr)
        worst = min(worst, rate)
    e = energy_per_move(d, rows)
    rec = {"kind": "opt_eval", "x": [float(v) for v in x], "design_key": d.key(), "design": asdict(d), "worst_move_success": worst,
           "L_mm": x[0], "energy_J": e, "per_move": {m: sum(r["cls"] in success_set(m) for r in rows if r["move"] == m) / OPT_N for m in OPT_MOVES},
           "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    append_jsonl(OUT / "opt_evals.jsonl", rec)
    return [1.0 - worst, x[0], e]


BASE_K = json.loads(os.environ.get("RW_OPT_BASE", "{}") or "{}")


def _init_base(base):
    global BASE_K
    BASE_K = base


def cmd_opt(args):
    from beebot.hardware.mmsim import opt as mo
    base = json.loads(args.base) if args.base else {}
    _init_base(base)
    os.environ["RW_OPT_BASE"] = json.dumps(base)
    t0 = time.time()
    res = mo.run_search(opt_objective, OPT_LO, OPT_HI, n_obj=3, algo="nsga2", pop=args.pop, n_gen=args.gens, workers=args.workers, seed=args.seed)
    front = []
    for x, f in zip(res.X, res.F):
        d = x_to_design(x, base)
        front.append({"x": dict(zip(OPT_VARS, [float(v) for v in x])), "design_key": d.key(), "design": asdict(d),
                      "worst_move_success_n_small": 1 - float(f[0]), "L_mm": float(f[1]), "energy_J": float(f[2])})
    front.sort(key=lambda r: r["L_mm"])
    (OUT / "pareto.json").write_text(json.dumps({"n_evals": res.n_evals, "secs": round(time.time() - t0), "moves": OPT_MOVES,
                                                 "trials_per_move": OPT_N, "front": front}, indent=1))
    for r in front:
        print(f"L {r['L_mm']:.2f} mm  worst {r['worst_move_success_n_small']:.2f}  E {r['energy_J'] * 1e3:.1f} mJ  {r['x']}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    a = sub.add_parser("mc")
    a.add_argument("--moves", default="all")
    a.add_argument("--n", type=int, default=200)
    a.add_argument("--seed0", type=int, default=0)
    a.add_argument("--retries", type=int, default=2)
    a.add_argument("--escalate", type=float, default=1.08)
    a.add_argument("--design", default="")
    a.add_argument("--policy", default="single")
    a.add_argument("--kscale", type=float, default=1.0)
    b = sub.add_parser("calibrate")
    b.add_argument("--moves", required=True)
    b.add_argument("--n", type=int, default=24)
    b.add_argument("--ks", default="1.0,1.1,1.2,1.35,1.5,1.7")
    b.add_argument("--tcs", default="5e-5,3e-4,1e-3,3e-3,8e-3")
    b.add_argument("--design", default="")
    o = sub.add_parser("opt")
    o.add_argument("--pop", type=int, default=18)
    o.add_argument("--gens", type=int, default=6)
    o.add_argument("--seed", type=int, default=0)
    o.add_argument("--base", default="")
    o.add_argument("--workers", type=int, default=WORKERS)
    c = sub.add_parser("outcomes")
    e = sub.add_parser("render")
    e.add_argument("--moves", default="all")
    e.add_argument("--design", default="")
    args = ap.parse_args()
    {"mc": cmd_mc, "calibrate": cmd_calibrate, "outcomes": cmd_outcomes, "render": cmd_render, "opt": cmd_opt}[args.cmd](args)


if __name__ == "__main__":
    main()
