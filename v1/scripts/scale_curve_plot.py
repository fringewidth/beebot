"""Held-out demonstration loss against step for the model-size arms of the scale curve.

    uv run python v1/scripts/scale_curve_plot.py --log logs/scale-curve.out --out runs/scale-curve

Reads the JSON lines expert_sft train prints with --arm (evaluations carry demo_val_loss, the
50-step rows carry minutes and mps_driver_gb) and writes curve.png and summary.json.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# arm -> (label, colour, dash): a single blue ramp, darker for bigger models; the control in grey
ARMS = {
    "0.8b": ("Qwen3.5-0.8B", "#5598e7", "-"),
    "2b": ("Qwen3.5-2B", "#256abf", "-"),
    "4b": ("Qwen3.5-4B", "#0d366b", "-"),
    "0.8b-random": ("0.8B, random init", "#8a8985", "--"),
}


def read(log):
    rows = {}
    for line in Path(log).read_text().splitlines():
        if not line.startswith("{"):
            continue
        row = json.loads(line)
        if row.get("arm") in ARMS:
            rows.setdefault(row["arm"], []).append(row)
    return rows


def summarise(rows):
    out = {}
    for arm, arm_rows in rows.items():
        evals = {r["step"]: r["demo_val_loss"] for r in arm_rows if "demo_val_loss" in r}
        timed = [r for r in arm_rows if "demo_loss" in r]
        last = max(timed, key=lambda r: r["step"]) if timed else None
        out[arm] = {
            "model": ARMS[arm][0],
            "demo_val_loss": {str(s): v for s, v in sorted(evals.items())},
            "final_step": max(evals) if evals else None,
            "final_demo_val_loss": evals[max(evals)] if evals else None,
            "seconds_per_step": round(last["minutes"] * 60 / last["step"], 2) if last else None,
            "peak_mps_driver_gb": max((r["mps_driver_gb"] for r in timed if "mps_driver_gb" in r), default=None),
            "peak_cuda_max_gb": max((r["cuda_max_gb"] for r in timed if "cuda_max_gb" in r), default=None),
        }
    steps = set.intersection(*[set(map(int, a["demo_val_loss"])) for a in out.values()]) if out else set()
    return {"matched_steps": sorted(steps), "arms": out}


def plot(summary, path):
    plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=200)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for arm, (label, colour, dash) in ARMS.items():
        if arm not in summary["arms"]:
            continue
        points = sorted((int(s), v) for s, v in summary["arms"][arm]["demo_val_loss"].items() if int(s) > 0)  # step 0 (untrained, ~3.4-3.9) would flatten the rest
        xs, ys = zip(*points)
        ax.plot(xs, ys, dash, color=colour, lw=2, marker="o", ms=5, label=label)
        ax.annotate(f"{label}  {ys[-1]:.3f}", (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points", va="center", fontsize=11, color="#0b0b0b")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("held-out demonstration loss")
    ax.set_title("2D swarm policy: held-out demonstration loss by model size", loc="left", fontsize=15, color="#0b0b0b", pad=22)
    ax.text(0, 1.015, "fresh LoRA rank 64 on each Qwen3.5 model; identical data, order, batch (16) and schedule; step 0 not shown", transform=ax.transAxes, fontsize=10, color="#52514e", va="bottom", ha="left")
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    ax.tick_params(colors="#52514e")
    ax.legend(frameon=False, loc="upper right")
    ax.margins(x=0.02)
    right = max(max(int(s) for s in a["demo_val_loss"]) for a in summary["arms"].values())
    ax.set_xlim(right=right * 1.32)  # room for the end labels
    ax.set_xticks(range(0, right + 1, 100))
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="logs/scale-curve.out")
    parser.add_argument("--out", default="runs/scale-curve")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = summarise(read(args.log))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    plot(summary, out / "curve.png")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
