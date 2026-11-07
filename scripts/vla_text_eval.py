"""Score a relative-position text run on what its held-out captions ask for (requested events,
labelled by scripts/vla_requested_events.py), next to the same windows driven by goal frames.

    uv run python scripts/vla_text_eval.py RUN [RUN ...] [--split test] [--guidance 1 2 4]
        [--play_cache data/vla/play_cache_g32] [--requested data/vla/captions/requested_g32.json]

Per run and guidance: requested_best (the best fraction of the asked events present at one
moment of the rollout), requested_end (the fraction at the end), unasked_disruptive (rollouts
that end having eaten or pushed a cheese nobody asked about), each for the caption, the caption
of another window (shuffled: what the scene alone gets), and the window's goal frame. Rows are
appended to RUN/text_eval.jsonl.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from beebot.vla import goal_eval
from beebot.vla.caption import TIERS
from beebot.vla.play_data import load_play_cache
from beebot.vla.swarm import RelativePolicy
from beebot.vla.train import CaptionHalf, Config


def load_policy(run, device):
    stored = json.loads((Path(run) / "config.json").read_text())
    cfg = Config(**{k: v for k, v in stored.items() if k in Config.__dataclass_fields__})
    policy = RelativePolicy(cfg.view, width=cfg.width, layers=cfg.layers, crop=cfg.crop, text_dim=1024, text_mode=cfg.text_mode, text_tokens=cfg.text_tokens)
    policy.load_state_dict(torch.load(Path(run) / "model_best.pt", map_location="cpu"))
    return policy.to(device).eval(), cfg


def by_tier(result, tier):
    out = {}
    for level, name in enumerate(TIERS):
        mine = (tier == level).numpy()
        if mine.any():
            out[f"requested_{name}"] = round(float(np.nanmean(result["requested_best"].numpy()[mine])), 4)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--guidance", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    parser.add_argument("--play_cache", default="data/vla/play_cache_g32")
    parser.add_argument("--captions", default="data/vla/captions/play_g32_set.pt")
    parser.add_argument("--requested", default="data/vla/captions/requested_g32.json")
    parser.add_argument("--windows", type=int, default=0, help="score only the first N labelled captions (0: all)")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    device = torch.device(args.device)

    store = load_play_cache(args.play_cache)
    held = getattr(CaptionHalf(args.captions, store, 512), args.split)
    labels = json.loads(Path(args.requested).read_text())[args.split]
    keep = [i for i, x in enumerate(labels) if x is not None]
    keep = keep[: args.windows] if args.windows else keep
    windows, emb, tier = held.windows[keep], held.emb[keep], held.tier[keep]
    requested = [labels[i] for i in keep]
    print(f"{args.split}: {len(keep)} labelled captions, {np.mean([len(r) for r in requested]):.2f} events asked per caption", flush=True)
    for run in args.runs:
        policy, cfg = load_policy(run, device)
        for guidance in args.guidance:
            row = {"run": Path(run).name, "split": args.split, "guidance": guidance, "n": len(keep)}  # n < 512: the first n labelled captions
            for source, text in (("text", emb), ("shuffled", emb.roll(1, dims=0)), ("goal", None)):
                torch.manual_seed(0)
                result = goal_eval.run(policy, store, windows, device, guidance=guidance, text_emb=text, execute=cfg.execute, requested=requested)
                s = goal_eval.summary(result)
                row.update({f"{source}_{k}": s[k] for k in ("requested_best", "requested_end", "unasked_disruptive", "fact_score")})
                if source == "text":
                    row.update({f"text_{k}": v for k, v in by_tier(result, tier).items()})
            print(json.dumps(row), flush=True)
            with open(Path(run) / "text_eval.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
