"""Kaggle kernel: fine-tune Qwen3 as the readout policy (beebot.vla.lm_train), one run per T4.
Needs internet (the model, FineWeb-Edu and GSM8K come from the Hugging Face Hub) and two
datasets: the code (beebot-src) and the play data with its captions (beebot-play-g32).
The driver (scripts/vla_kaggle.py lm-run) fills in RUNS. Outputs: /kaggle/working/runs/<name>/
and logs/<name>.out.
"""

import glob
import os
import shutil
import subprocess
import sys
import tarfile
import time

RUNS = __RUNS__  # [[arg, ...], ...], one list of trainer arguments per GPU (DDP: per run, in order)
DDP = __DDP__  # True: the runs one after another, each on every GPU through torchrun
MODULE = "__MODULE__"  # the default trainer: beebot.vla.lm_train or beebot.vla.lm_rl; a run's own "--module X" overrides it

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.22.1", "datasets"], check=False)
t_setup = time.time()


def find(name, max_depth=6):
    """Paths named name under /kaggle/input, at most max_depth levels down. Fixed-depth globs: a
    recursive glob walks every play-run directory on the slow input disk (minutes)."""
    found = []
    for depth in range(1, max_depth + 1):
        found += glob.glob("/kaggle/input/" + "*/" * (depth - 1) + name)
    return [p for p in found if "/._" not in p]


archives = find("beebot-src.tar.gz")
if archives:
    tarfile.open(archives[0]).extractall("/kaggle/working/code")
else:
    package = find("beebot/vla/train.py", 8)[0]
    shutil.copytree(package.rsplit("/beebot/vla/train.py", 1)[0], "/kaggle/working/code/src")
play = find("beebot-play-g32.tar.gz")
if play:
    tarfile.open(play[0]).extractall("/kaggle/working/data")
    play_root = "/kaggle/working/data/beebot-play-g32"
else:
    # Kaggle unpacks the uploaded tarball, sometimes several directories deeper: the root is the
    # directory holding the player directories (haiku-1, luna-2, ...).
    play_root = sorted(find("haiku-1", 7), key=len)[0].rsplit("/", 1)[0]
captions = find("captions_g32.jsonl")[0]
# Files from a previous kernel or a checkpoint dataset, resolved now so a missing one fails before
# the play cache is built.
RUNS = [[find(a[len("@input/") :], 8)[0] if a.startswith("@input/") else a for a in args] for args in RUNS]
print(f"play root {play_root}, captions {captions}, found in {time.time() - t_setup:.0f} s", flush=True)
import torch, transformers  # noqa: E402

print("torch", torch.__version__, "transformers", transformers.__version__, "gpus", torch.cuda.device_count(), flush=True)
# One copy of the play store on local disk, memory-mapped by both runs: two in-memory copies
# (about 15 GB each) do not fit in the machine's RAM. Not under /kaggle/working, which is saved.
sys.path.insert(0, "/kaggle/working/code/src")
from beebot.vla.play_data import cache_play, run_dirs  # noqa: E402

t0 = time.time()
cache = "/tmp/play_cache"
episodes = cache_play(run_dirs(play_root, size=32), cache)
print(f"play cache: {episodes} episodes in {(time.time() - t0) / 60:.1f} min", flush=True)
os.makedirs("/kaggle/working/logs", exist_ok=True)
noise = ("Warning", "Loading weights", "it/s]", "split:", "warn", "s/it]")


def launch(gpu, args):
    """Start one run: on GPU gpu, or (DDP) on every GPU through torchrun. A run's own --module and
    --captions (e.g. '@input/...') take the place of the defaults."""
    module = MODULE
    if "--module" in args:
        i = args.index("--module")
        module, args = args[i + 1], args[:i] + args[i + 2 :]
    name = args[args.index("--name") + 1]
    env = {**os.environ, "PYTHONPATH": "/kaggle/working/code/src", "TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    if not DDP:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log = open(f"/kaggle/working/logs/{name}.out", "w")
    full = [*args, "--play_root", play_root, "--play_cache", cache, *([] if "--captions" in args else ["--captions", captions]), "--device", "cuda"]
    launcher = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(torch.cuda.device_count())] if DDP else [sys.executable]
    print(f"{'all gpus' if DDP else f'gpu {gpu}'}: {name} {' '.join(full)}", flush=True)
    return name, subprocess.Popen([*launcher, "-m", module, *full], env=env, cwd="/kaggle/working", stdout=log, stderr=subprocess.STDOUT)


def relay(started):
    """Kaggle shows only this script's output while it runs: relay the runs' progress lines live
    until every run has exited."""
    seen = {name: 0 for name, _ in started}
    while True:
        for name, _ in started:
            lines = open(f"/kaggle/working/logs/{name}.out", errors="replace").read().split("\n")
            for line in lines[seen[name] : len(lines) - 1]:
                if line.strip() and not any(n in line for n in noise):
                    print(f"[{name}] {line[:600]}", flush=True)
            seen[name] = len(lines) - 1
        if all(p.poll() is not None for _, p in started):
            break
        time.sleep(30)
    for name, p in started:
        print("exit", name, p.returncode, flush=True)


if DDP:  # one run at a time, each on every GPU
    for args in RUNS:
        relay([launch(None, args)])
else:  # one run per GPU, side by side
    relay([launch(gpu, args) for gpu, args in enumerate(RUNS)])
for path in glob.glob("/kaggle/working/logs/*.out"):
    print(f"==== {path}", flush=True)
    print("".join(open(path).readlines()[-60:]), flush=True)
shutil.rmtree("/kaggle/working/data", ignore_errors=True)  # keep outputs small
