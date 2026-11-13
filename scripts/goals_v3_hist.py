"""Cell-count histograms per source before (data/goal_library, data/goal_library_v2, no cap) and after
(data/goal_library_v3), and the total per bin, as JSON lines.

    nice -n 10 .venv/bin/python scripts/goals_v3_hist.py
"""

import json
from pathlib import Path

import numpy as np

BINS = [4, 16, 32, 64, 128, 192, 257]
LABELS = ["4-15", "16-31", "32-63", "64-127", "128-191", "192-256"]
BEFORE = {
    "gen_simple": "data/goal_library",
    "gen_medium": "data/goal_library",
    "stabletext2brick": "data/goal_library_v2",
    "craftassist_parts": "data/goal_library_v2",
    "craftassist_houses": "data/goal_library_v2",
    "iglu_single": "data/goal_library_v2",
    "iglu_multiturn": "data/goal_library_v2",
    "hack337": "data/goal_library_v2",
    "oga_vox": "data/goal_library_v2",
    "itch_vox": "data/goal_library_v2",
    "text2shape": "data/goal_library_v2",
}


def counts(name, root):
    path = Path(root) / f"{name}.jsonl"
    if not path.exists():
        return None
    values = [json.loads(text)["cells"] for text in open(path)]
    return np.histogram(values, BINS)[0].tolist()


def main():
    totals = {"before": np.zeros(len(LABELS), dtype=int), "after": np.zeros(len(LABELS), dtype=int)}
    for name, root in BEFORE.items():
        for stage, folder in (("before", root), ("after", "data/goal_library_v3")):
            hist = counts(name, folder)
            if hist is None:
                continue
            totals[stage] += np.asarray(hist)
            print(json.dumps({"source": name, "stage": stage, "bins": dict(zip(LABELS, hist))}))
    for stage, hist in totals.items():
        print(json.dumps({"source": "TOTAL", "stage": stage, "bins": dict(zip(LABELS, hist.tolist()))}))


if __name__ == "__main__":
    main()
