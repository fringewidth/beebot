"""Score a Qwen LM policy (lm_train's checkpoints) on requested events, on the same held-out test
captions as scripts/vla_text_eval.py scores the swarm runs (CaptionHalf's test sample and the
labels from scripts/vla_requested_events.py), next to the same windows driven by goal frames.

    uv run python scripts/vla_lm_text_eval.py RUN_DIR [--checkpoint policy_best.pt] [--windows 128]
        [--guidance 2] [--play_cache data/vla/play_cache_g32]

RUN_DIR holds lm_train's config.json and the checkpoint. --windows N scores the first N labelled
test captions (Qwen rollouts are slow on a laptop GPU). Rows go to RUN_DIR/text_eval.jsonl.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from vla_lm_play import load  # noqa: E402

from beebot.vla import goal_eval  # noqa: E402
from beebot.vla.lm_policy import EvalAdapter  # noqa: E402
from beebot.vla.play_data import load_play_cache  # noqa: E402
from beebot.vla.train import CaptionHalf  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--checkpoint", default="policy_best.pt")
    parser.add_argument("--windows", type=int, default=128)
    parser.add_argument("--guidance", type=float, nargs="+", default=[2.0])
    parser.add_argument("--play_cache", default="data/vla/play_cache_g32")
    parser.add_argument("--captions", default="data/vla/captions/play_g32_set.pt")
    parser.add_argument("--requested", default="data/vla/captions/requested_g32.json")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    device = torch.device(args.device)

    policy, tokenizer = load(args.run, device, args.checkpoint)
    tokenizer.padding_side = "right"  # as lm_train
    store = load_play_cache(args.play_cache)
    held = CaptionHalf(args.captions, store, 512).test
    labels = json.loads(Path(args.requested).read_text())["test"]
    keep = [i for i, x in enumerate(labels) if x is not None][: args.windows]
    windows, requested = held.windows[keep], [labels[i] for i in keep]
    enc = tokenizer([re.sub(r"</?[a-z]>", "", held.texts[i]) for i in keep], return_tensors="pt", padding=True, truncation=True, max_length=96)
    ids = torch.where(enc.attention_mask.bool(), enc.input_ids, torch.full_like(enc.input_ids, -1)).float()
    adapter = EvalAdapter(policy)
    for guidance in args.guidance:
        row = {"run": Path(args.run).name, "checkpoint": args.checkpoint, "split": "test", "guidance": guidance, "n": len(keep)}
        for source, text in (("text", ids), ("shuffled", ids.roll(1, dims=0)), ("goal", None)):
            torch.manual_seed(0)
            with torch.no_grad():
                result = goal_eval.run(adapter, store, windows, device, guidance=guidance, execute=1, batch_size=args.batch, text_emb=text, requested=requested)
            s = goal_eval.summary(result)
            row.update({f"{source}_{k}": s[k] for k in ("requested_best", "requested_end", "unasked_disruptive", "fact_score")})
            print(source, {k: s[k] for k in ("requested_best", "fact_score")}, flush=True)
        print(json.dumps(row), flush=True)
        with open(Path(args.run) / "text_eval.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
