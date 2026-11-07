"""Build site/vla.html, the pitch page for the swarm VLA: demos, graphs, what worked and what
didn't. Same theme as site/explain.html. Data comes from the runs' val histories, the frozen
comparison (logs/vla-final-table.out) and site/media/vla/demos.json.

    uv run python scripts/vla_site.py
"""

import json
from pathlib import Path

RUNS = [
    # (run, label, role): "now" accent, "alt" blue, "base" dashed neutral; "table" only in the
    # table under the chart (three lines is the most the palette keeps apart on a line chart).
    ("vla-d-g32-play-readout", "Readout, no expert (now)", "now"),
    ("vla-d-g32-play-masked-ki", "Masked expert + insulation", "alt"),
    ("vla-d-g32-play-goal", "Flow expert (pi0-style, before)", "base"),
    ("vla-d-g32-play-insulate", "Flow expert + insulation", "table"),
]
BARS = [
    # (label, test facts, eat recall, role), each policy at its best setting on the same 512
    # frozen test windows (logs/vla-final-table.out).
    ("Flow expert (pi0-style)", 0.552, 0.12, "before"),
    ("Masked expert + insulation", 0.654, 0.37, "before"),
    ("Readout, no expert", 0.768, 0.63, "now"),
    ("Readout + future states", 0.776, 0.68, "now"),
]
REPLAY = {"facts": 0.879, "ate": 0.68}
LEAK_RATIO = [(1.0, 1.885), (0.9, 0.826), (0.7, 0.682), (0.5, 0.642), (0.3, 0.625), (0.1, 0.617)]


def curves():
    out = []
    for run, label, role in RUNS:
        rows = [json.loads(line) for line in open(f"runs/{run}/val_history.jsonl")]
        out.append({"label": label, "role": role, "points": [[r["step"], round(r["minutes"], 1), round(r["fact_score"], 3)] for r in rows]})
    return out


def leak_curve():
    rows = [json.loads(line) for line in open("runs/vla-d-g32-play-masked/val_history.jsonl")]
    return [[r["step"], round(r["fact_score"], 3)] for r in rows]


def main():
    demos = json.loads(Path("site/media/vla/demos.json").read_text())
    data = {"curves": curves(), "bars": [dict(zip(("label", "facts", "ate", "role"), b)) for b in BARS], "replay": REPLAY, "leak_ratio": LEAK_RATIO, "leak_curve": leak_curve(), "demos": demos}
    page = Path("scripts/vla_site_template.html").read_text().replace("__DATA__", json.dumps(data))
    Path("site/vla.html").write_text(page)
    print("wrote site/vla.html")


if __name__ == "__main__":
    main()
