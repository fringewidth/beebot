"""Kaggle TPU v5e-8 kernel: the think-then-act policy on Qwen3.5-0.8B (beebot.vla.voxel.think_train,
--device xla), a timing gate and then the real run, in one session (TPU sessions can queue a day).

    1. timing, one chip, --data fake (shapes as the real data's: two 256 px images, a 16^3 grid):
       compile minutes, seconds per step, examples/s, HBM; per-chip batch 16 with fp32 weights,
       else 8, else 8 with bf16 weights (the first that runs);
    1. timing, eight chips data-parallel, the same settings;
    1. if the eight-chip rate is at least GATE examples/s: the real run on the real data (the
       Mac run's settings, scripts in the beebot-think-data dataset) for the rest of the session,
       step-named checkpoints and resume.pt every hour under /kaggle/working/runs/<name>/;
       otherwise exit at once.

Inputs: the private datasets beebot-src-tpu (src/ as beebot-src-tpu.tar.gz) and beebot-think-data
(data/vla/voxel_full/s0-s7, voxel_build/s0-s1, haiku/run-s*, build-s* as beebot-think-data.tar.gz);
internet for pip, the model and the distillation text. Every measurement is a JSON line
{"run": "tpu-probe", ...} on stdout. No evaluation here: the Mac scores the checkpoints.
"""

import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time

T0 = time.time()
SESSION_MINUTES = 9 * 60  # Kaggle's TPU session limit
MARGIN_MINUTES = 25  # the real run's last validation, last save and the output upload
GATE = 5.0  # examples/s on eight chips to go on to the real run
MODEL = "Qwen/Qwen3.5-0.8B"
WORK = "/kaggle/working"
DATA = "/tmp/beebot-data"  # outside /kaggle/working, so the data is not saved as output
NAME = "think-08b-tpu"
PROBE_STEPS = 40
FIRST_LOG_MINUTES = 25  # a probe without its first step by then is stuck compiling: killed
FAKE = {"images": 2, "image_size": 256, "side": 16, "scene": False, "act_share": 0.95, "instruction_words": 18, "thought_words": 40}
ATTEMPTS = [(16, "fp32"), (8, "fp32"), (8, "bf16")]  # (per-chip batch, LM weights)
STORES = [f"voxel_build/s{i}" for i in range(2)] + [f"voxel_full/s{i}" for i in range(8)]
HAIKU = ["build-s0", "build-s1"] + [f"run-s{i}{later}" for i in range(8) for later in ("", "-later")]


def note(**row):
    print(json.dumps({"run": "tpu-probe", "minutes": round((time.time() - T0) / 60, 1), **row}), flush=True)


def find(name, max_depth=7):
    found = []
    for depth in range(1, max_depth + 1):
        found += glob.glob("/kaggle/input/" + "*/" * (depth - 1) + name)
    return sorted((p for p in found if "/._" not in p), key=len)


# ---- code, data, packages, model

code = find("beebot-src-tpu.tar.gz")
if code and os.path.isfile(code[0]):
    tarfile.open(code[0]).extractall(f"{WORK}/code")
else:
    package = find("beebot/vla/voxel/think_train.py", 9)[0]
    shutil.copytree(package.rsplit("/beebot/vla/voxel/think_train.py", 1)[0], f"{WORK}/code/src")
data = [p for p in find("beebot-think-data.tar.gz") if os.path.isfile(p)]
if data:
    tarfile.open(data[0]).extractall(DATA)
    root = f"{DATA}/data/vla"
else:  # Kaggle unpacked the archive: the directory holding voxel_full/
    root = find("voxel_full/s0/compact.json", 9)[0].rsplit("/voxel_full/", 1)[0]
stores = [f"{root}/{s}" for s in STORES]
haiku = [f"{root}/haiku/{h}" for h in HAIKU]
missing = [p for p in stores + haiku if not os.path.isdir(p)]
assert not missing, f"missing data: {missing}"
print("data root", root, flush=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.22.1", "datasets"], check=False)
import torch  # noqa: E402
import torch_xla  # noqa: E402

note(phase="setup", torch=torch.__version__, torch_xla=torch_xla.__version__, cpus=os.cpu_count())
from huggingface_hub import snapshot_download  # noqa: E402

snapshot_download(MODEL)
from datasets import load_dataset  # noqa: E402

load_dataset("openai/gsm8k", "main")  # cached once, not by eight processes at a time
note(phase="ready")


# ---- runs


def train(phase, chips, args, first_log_minutes=FIRST_LOG_MINUTES):
    """Run think_train on chips TPU chips; its output goes to logs/<phase>.out and stdout. Killed
    when no step was logged within first_log_minutes. Returns the JSON rows it logged."""
    env = {**os.environ, "PYTHONPATH": f"{WORK}/code/src", "PJRT_DEVICE": "TPU", "BEEBOT_XLA_CACHE": "/tmp/xla_cache", "BEEBOT_XLA_PROCESSES": str(chips), "TOKENIZERS_PARALLELISM": "false"}
    if chips == 1:
        env.update({"TPU_PROCESS_BOUNDS": "1,1,1", "TPU_CHIPS_PER_PROCESS_BOUNDS": "1,1,1", "TPU_VISIBLE_CHIPS": "0", "TPU_PROCESS_PORT": "8476", "TPU_PROCESS_ADDRESSES": "localhost:8476"})
    command = [sys.executable, "-c", "from beebot.vla.voxel.think_train import main; main()", *args, "--device", "xla", "--chips", str(chips)]
    print(f"==== {phase}: {' '.join(command)}", flush=True)
    os.makedirs(f"{WORK}/logs", exist_ok=True)
    log = open(f"{WORK}/logs/{phase}.out", "w")
    process = subprocess.Popen(command, env=env, cwd=WORK, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    rows, started = [], time.time()

    def read():
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
            if line.startswith("{"):
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    while process.poll() is None:
        time.sleep(10)
        if not any("examples_per_s" in r for r in rows) and time.time() - started > first_log_minutes * 60:
            note(phase=phase, chips=chips, error=f"no step after {first_log_minutes} minutes: stuck compiling, killed")
            os.killpg(process.pid, signal.SIGKILL)
    reader.join(timeout=60)
    note(phase=phase, chips=chips, exit=process.returncode, wall_minutes=round((time.time() - started) / 60, 1))
    return rows


def probe_args(batch, weights):
    return ["--name", "probe", "--model", MODEL, "--data", "fake", "--data_kwargs", json.dumps(FAKE), "--batch", str(batch), "--accumulate", "1", "--weights_dtype", weights, "--checkpointing", "true", "--steps", str(PROBE_STEPS), "--warmup_steps", "50", "--log_every", "10", "--val_examples", "0", "--val_every_minutes", "100000", "--save_every_minutes", "100000", "--save_resume", "false", "--widths", "384,512"]


def summary(phase, chips, rows, batch, weights):
    """The steady rate: the last logged window (steps 30-40, after the compilations of steps 1 and 2)."""
    steps = [r for r in rows if "examples_per_s" in r]
    if not steps or steps[-1]["step"] < PROBE_STEPS:
        note(phase=phase, chips=chips, batch=batch, weights=weights, failed=True, steps=steps[-1]["step"] if steps else 0)
        return None
    last = steps[-1]
    note(phase=phase, chips=chips, batch=batch, weights=weights, examples_per_s=last["examples_per_s"], step_s=last.get("step_s"), hbm_gb=last.get("gpu_gb"), compiles=last.get("compiles"), compile_minutes=round(last.get("compile_s", 0) / 60, 1), first_step_minutes=steps[0]["minutes"], data_wait=last.get("data_wait"), cpu_fallbacks=last.get("cpu_fallbacks"), loss=last.get("loss"))
    return last


chosen = None
for batch, weights in ATTEMPTS:
    result = summary("one-chip", 1, train(f"one-chip-b{batch}-{weights}", 1, probe_args(batch, weights)), batch, weights)
    shutil.rmtree(f"{WORK}/runs/probe", ignore_errors=True)
    if result:
        chosen = (batch, weights)
        break
if chosen is None:
    note(phase="gate", passed=False, reason="no one-chip setting ran")
    sys.exit(0)
batch, weights = chosen
eight = summary("eight-chip", 8, train(f"eight-chip-b{batch}-{weights}", 8, probe_args(batch, weights), first_log_minutes=30), batch, weights)
shutil.rmtree(f"{WORK}/runs/probe", ignore_errors=True)
rate = eight["examples_per_s"] if eight else 0.0
if rate < GATE:
    note(phase="gate", passed=False, examples_per_s=rate, gate=GATE)
    sys.exit(0)

# ---- the real run: the Mac run's settings (bugcatch-08b) at the TPU's batch, for the rest of the session
minutes = SESSION_MINUTES - (time.time() - T0) / 60 - MARGIN_MINUTES
global_batch = 8 * batch
steps = max(100, int(0.85 * rate * minutes * 60 / global_batch))
note(phase="gate", passed=True, examples_per_s=rate, gate=GATE, batch_per_chip=batch, weights=weights, global_batch=global_batch, steps=steps, wall_minutes=round(minutes, 1))
real = ["--name", NAME, "--model", MODEL, "--data", "think", "--stores", ",".join(stores), "--haiku_dirs", ",".join(haiku), "--data_kwargs", json.dumps({"mismatch": "drop"}), "--views", "scene", "--batch", str(batch), "--accumulate", "1", "--weights_dtype", weights, "--steps", str(steps), "--warmup_steps", "50", "--wall_minutes", str(round(minutes, 1)), "--checkpointing", "true", "--val_examples", "64", "--val_every_minutes", "60", "--save_every_minutes", "60", "--log_every", "10", "--widths", "384,512"]
rows = train("real", 8, real, first_log_minutes=40)
done = [r for r in rows if "examples_per_s" in r]
note(phase="real", chips=8, steps=done[-1]["step"] if done else 0, examples_per_s=done[-1]["examples_per_s"] if done else 0, checkpoints=sorted(os.path.basename(p) for p in glob.glob(f"{WORK}/runs/{NAME}/policy_step*.pt")))
