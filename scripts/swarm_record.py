"""Record sessions of the live-steerable 2D swarm (swarm_steer.py) for the static demo page to replay
when no backend answers: each session is a fresh field and a few suggested commands, each run until
the swarm rests, exactly as swarm_steer.py's loop steps a visitor.

    python scripts/swarm_record.py --init_run RUN [--checkpoint policy_latest.pt] [--sessions 6] [--commands 4]
        [--out site/2d-demo/recordings.json]

Frames carry the robot mass and walls as hex strings (32 x 32 bits each, 256 hex digits), so a
recording of a few thousand frames stays around a megabyte.
"""

import argparse
import json
import secrets
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import swarm_steer as S  # noqa: E402

from beebot.vla import steer  # noqa: E402
from beebot.vla.lm_rl import RLConfig, load_policy  # noqa: E402
from beebot.vla.world import COLOUR_NAMES, COLOUR_RGB  # noqa: E402


def hexbits(bits):
    """A '0'/'1' string as hex, four bits per digit."""
    return "".join(f"{int(bits[i:i + 4], 2):x}" for i in range(0, len(bits), 4))


def record_command(site, sid, text, max_steps):
    """The frames of one command, from the frame it is typed until the swarm rests."""
    frames = [site.command(sid, text)]
    since = site.swarms.state[sid]["t"]
    while site.swarms.state[sid]["text"]:
        if site.swarms.state[sid]["t"] - since >= max_steps:
            site.swarms.instruct(sid, "")
            site.visitors[sid]["events"].append("done")
        else:
            eaten = site.swarms.advance([sid])
            colours = eaten.get(sid) or []
            site.visitors[sid]["events"] += [f"ate {c}" for c in colours]
        frames.append(site.frame(sid))
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_run", required=True)
    parser.add_argument("--checkpoint", default="policy_latest.pt")
    parser.add_argument("--kinds", default=str(Path(__file__).with_name("steer_kinds.json")))
    parser.add_argument("--min_success", type=float, default=0.6)
    parser.add_argument("--max_steps", type=int, default=140)
    parser.add_argument("--sessions", type=int, default=6)
    parser.add_argument("--commands", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--out", default="site/2d-demo/recordings.json")
    args = parser.parse_args()
    device = torch.device(args.device)
    policy, tokenizer = load_policy(RLConfig(init_run=args.init_run, checkpoint=args.checkpoint), device)
    policy.eval()
    fresh = S.reliable_kinds(args.kinds, "fresh", args.min_success)
    after = S.reliable_kinds(args.kinds, "after", args.min_success)
    site = S.Site(steer.Swarms(policy, tokenizer, device), fresh, after, fps=7.0, max_visitors=4, max_steps=args.max_steps)
    site.rng = np.random.default_rng(args.seed)
    sessions = []
    for n in range(args.sessions):
        sid = secrets.token_hex(8)
        start = site.new(sid)
        frames = [start]
        for _ in range(args.commands):
            text = site.visitors[sid]["suggestion"]
            if not text:
                break
            frames += record_command(site, sid, text, args.max_steps)
        for f in frames:
            f["mass"] = hexbits(f["mass"])
            f.pop("suggestion", None)
            f.pop("first", None)
        sessions.append({"walls": hexbits(start["walls"]), "boxes": start["boxes"], "frames": [{k: v for k, v in f.items() if k not in ("walls", "boxes", "sid")} for f in frames]})
        site.visitors.pop(sid)
        site.swarms.state.pop(sid, None)
        print(json.dumps({"run": "swarm-record", "session": n, "frames": len(frames)}), flush=True)
    palette = {"colours": COLOUR_NAMES, "rgb": {name: [round(float(v), 3) for v in rgb] for name, rgb in zip(COLOUR_NAMES, COLOUR_RGB.tolist())}, "size": 32}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"palette": palette, "fps": 7.0, "sessions": sessions}, separators=(",", ":")))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, {sum(len(s['frames']) for s in sessions)} frames)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
