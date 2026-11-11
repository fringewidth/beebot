"""TPU smoke test of the think trainer on a Colab TPU runtime (one cell: clone tpu-port, run this).

    !git clone -q --depth 1 -b tpu-port https://github.com/fringewidth/beebot.git /content/beebot
    !cd /content/beebot && python scripts/kaggle_tpu/colab_smoke.py

Stages, each printing one JSON line {"smoke": <stage>, ...}:
  setup     torch_xla and libtpu as the Kaggle kernel installs them (think_kernel.py), the TPU chips
  inverse   think.chunk_gated_delta_rule_matmul on the TPU against transformers' reference on the
            CPU, with correlated keys (the case that broke the old inverse)
  probe     think_train on fake data (the kernel's probe arguments), 40 steps on one chip
  real      think_train on real data: a small store drawn here (demos.write, the training stores'
            options), rendered views, templates only, 20 steps on one chip
A stage that fails prints its error and the next stage still runs.
"""

import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO, "src")
MODEL = "Qwen/Qwen3.5-0.8B"
TORCH_XLA = "2.9.0"


def say(stage, **fields):
    print(json.dumps({"smoke": stage, **fields}), flush=True)


def pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=False)


def setup():
    try:
        import torch_xla  # noqa: F401
    except ImportError:
        pip(f"torch=={TORCH_XLA}", "--index-url", "https://download.pytorch.org/whl/cpu")
        pip(f"torch_xla=={TORCH_XLA}", "libtpu==0.0.21.1" if sys.version_info >= (3, 13) else "libtpu==0.0.21")
    pip("transformers==5.22.1", "datasets")
    import glob

    import torch
    import torch_xla

    chips = glob.glob("/dev/vfio/[0-9]*") + glob.glob("/dev/accel*")
    say("setup", python=sys.version.split()[0], torch=torch.__version__, torch_xla=torch_xla.__version__, chips=len(chips))


def inverse():
    sys.path.insert(0, SRC)
    import torch
    import torch_xla.core.xla_model as xm
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

    from beebot.vla import tpu
    from beebot.vla.voxel import think

    tpu.xla_setup(0)
    torch.manual_seed(0)
    seq, heads, dim = 192, 4, 128
    shared = torch.randn(1, 1, heads, dim)
    k = 0.9 * shared + 0.1 * torch.randn(1, seq, heads, dim)
    q, v = torch.randn(1, seq, heads, dim), torch.randn(1, seq, heads, dim)
    beta = 0.5 + 0.5 * torch.rand(1, seq, heads)
    g = -0.05 * torch.rand(1, seq, heads)
    reference, _ = torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, output_final_state=True, use_qk_l2norm_in_kernel=True)
    device = xm.xla_device()
    out, _ = think.chunk_gated_delta_rule_matmul(*(t.to(device) for t in (q, k, v)), g=g.to(device), beta=beta.to(device), output_final_state=True, use_qk_l2norm_in_kernel=True)
    error = float((out.cpu() - reference).abs().max())
    say("inverse", max_error=round(error, 6), mean_size=round(float(reference.abs().mean()), 4), passed=error < 1e-3)


def train(stage, args, steps):
    env = {**os.environ, "PYTHONPATH": SRC, "PJRT_DEVICE": "TPU", "BEEBOT_XLA_PROCESSES": "1", "TOKENIZERS_PARALLELISM": "false"}
    env.update({"TPU_PROCESS_BOUNDS": "1,1,1", "TPU_CHIPS_PER_PROCESS_BOUNDS": "1,1,1", "TPU_VISIBLE_CHIPS": "0", "TPU_PROCESS_PORT": "8476", "TPU_PROCESS_ADDRESSES": "localhost:8476"})
    command = [sys.executable, "-c", "from beebot.vla.voxel.think_train import main; main()", *args, "--device", "xla", "--chips", "1"]
    started = time.time()
    process = subprocess.run(command, env=env, cwd=REPO, capture_output=True, text=True)
    rows = []
    for line in process.stdout.splitlines():
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    logged = [r for r in rows if "examples_per_s" in r]
    last = logged[-1] if logged else {}
    say(stage, exit=process.returncode, minutes=round((time.time() - started) / 60, 1), steps=last.get("step", 0), loss=last.get("loss"), text=last.get("text"), action=last.get("action"), examples_per_s=last.get("examples_per_s"), passed=process.returncode == 0 and last.get("step", 0) >= steps, tail=(process.stdout + process.stderr)[-1500:] if process.returncode else "")


COMMON = ["--model", MODEL, "--batch", "4", "--accumulate", "1", "--checkpointing", "true", "--warmup_steps", "5", "--log_every", "10", "--val_examples", "0", "--val_every_minutes", "100000", "--save_every_minutes", "100000", "--save_resume", "false", "--distill_weight", "0", "--widths", "384,512"]
FAKE = {"images": 2, "image_size": 256, "side": 16, "scene": False, "act_share": 0.95, "instruction_words": 18, "thought_words": 40}


def probe():
    train("probe", ["--name", "probe", "--data", "fake", "--data_kwargs", json.dumps(FAKE), "--steps", "40", *COMMON], 40)


def real():
    store = "/tmp/smoke-store"
    if not os.path.isdir(store):
        subprocess.run([sys.executable, "-m", "beebot.vla.voxel.demos", "--out", store, "--scenes", "48", "--seed", "77", "--cube", "--rule", "switch_cube", "--robots", "6", "115", "--counts", "log", "--coverage", "0.3", "--objects", "1.0", "--fit", "--keep"], env={**os.environ, "PYTHONPATH": SRC}, cwd=REPO, check=False, capture_output=True)
    train("real", ["--name", "real", "--data", "think", "--stores", store, "--data_kwargs", json.dumps({"mismatch": "drop", "held_share": 0.1}), "--views", "scene", "--steps", "20", *COMMON], 20)


if __name__ == "__main__":
    for stage in (setup, inverse, probe, real):
        try:
            stage()
        except Exception as error:  # report and go on
            say(stage.__name__, error=f"{type(error).__name__}: {error}"[-800:], passed=False)
    say("done")
