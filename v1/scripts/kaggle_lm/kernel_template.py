"""Kaggle kernel: fine-tune Qwen3 as the readout policy (beebot.vla.lm_train), one run per T4.
Needs internet (the model, FineWeb-Edu and GSM8K come from the Hugging Face Hub) and two
datasets: the code (beebot-src) and the play data with its captions (beebot-play-g32).
The driver (v1/scripts/vla_kaggle.py lm-run) fills in RUNS, the control topic and the kill switch
(v1/scripts/kernel_control.py: kill, kill NAME, stop). Outputs: /kaggle/working/runs/<name>/ and
logs/<name>.out.
"""

import glob
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import time

RUNS = __RUNS__  # [[arg, ...], ...], one list of trainer arguments per GPU (DDP: per run, in order)
DDP = __DDP__  # True: the runs one after another, each on every GPU through torchrun
MODULE = "__MODULE__"  # the default trainer: beebot.vla.lm_train or beebot.vla.lm_rl; a run's own "--module X" overrides it
FAST_LINEAR_ATTENTION = __FLA__  # True: try the fused kernels for Qwen3.5's linear-attention layers (else transformers' plain PyTorch path)
CONTROL_URL = "__CONTROL_URL__"  # filled in at push time (v1/scripts/run_control.py): the run's private ntfy topic; never printed

# __KERNEL_CONTROL__
CONTROL = KernelControl(CONTROL_URL).start()  # polls from the first minute: a kill during setup ends the session too

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.22.1", "datasets"], check=False)
if FAST_LINEAR_ATTENTION:
    t_fla = time.time()
    # --no-deps: never let it replace the image's torch. causal-conv1d is left out: it compiles from source
    # on Kaggle's torch (over 10 minutes, 2026-11-17), and its plain PyTorch fallback is a cheap 4-tap conv.
    # flash-linear-attention's Triton kernels compile on first use: about 10-15 minutes before the first steps.
    fla = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "fla-core", "flash-linear-attention", "einops"], check=False).returncode
    print(f"fast linear attention: flash-linear-attention exit {fla}, in {time.time() - t_fla:.0f} s", flush=True)
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
def needs_play(args):
    """expert_sft with --loss_only and --caption_share 0 never reads the play data."""
    return not ("--loss_only" in args and "--caption_share" in args and float(args[args.index("--caption_share") + 1]) == 0)


play = find("beebot-play-g32.tar.gz")
if play and any(needs_play(a) for a in RUNS):
    tarfile.open(play[0]).extractall("/kaggle/working/data")
    play_root = "/kaggle/working/data/beebot-play-g32"
elif not play:
    # Kaggle unpacks the uploaded tarball, sometimes several directories deeper: the root is the
    # directory holding the player directories (haiku-1, luna-2, ...).
    play_root = sorted(find("haiku-1", 7), key=len)[0].rsplit("/", 1)[0]
else:
    play_root = ""  # no run reads it
captions =find("captions_g32.jsonl")[0]
# Files from a previous kernel or a checkpoint dataset, resolved now so a missing one fails before
# the play cache is built.
# Extra data packed as a beebot-extra*.tar archive (expert stores, a starting checkpoint): unpacked here
# unless Kaggle already unpacked it, and searched after /kaggle/input.
for archive in [p for p in find("beebot-extra*.tar*") if os.path.isfile(p)]:
    tarfile.open(archive).extractall("/kaggle/working/extra")


def resolve(name):
    found = find(name, 8) or sorted(glob.glob(f"/kaggle/working/extra/**/{name}", recursive=True), key=len)
    assert found, f"no {name} in the inputs"
    return found[0]


RUNS = [[resolve(a[len("@input/") :]) if a.startswith("@input/") else a for a in args] for args in RUNS]


def local(path):
    """An expert store on the input mount, copied once to local disk: the trainer memory-maps it and
    reads at random, which from the mount is several times slower until the page cache is warm
    (the 30k-demo run lost its first two hours to it)."""
    if not path.startswith("/kaggle/input/"):
        return path
    target = "/tmp/stores/" + path.rstrip("/").rsplit("/", 1)[1]
    if not os.path.exists(target):
        shutil.copytree(path, target)
        print(target, "labels md5", hashlib.md5(open(target + "/labels.jsonl", "rb").read()).hexdigest(), flush=True)
    return target


t_copy = time.time()
for args in RUNS:
    if "--store" in args:
        i = args.index("--store") + 1
        while i < len(args) and not args[i].startswith("--"):
            args[i] = local(args[i])
            i += 1
print(f"expert stores on local disk in {time.time() - t_copy:.0f} s", flush=True)
print(f"play root {play_root}, captions {captions}, found in {time.time() - t_setup:.0f} s", flush=True)
import torch, transformers  # noqa: E402

print("torch", torch.__version__, "transformers", transformers.__version__, "gpus", torch.cuda.device_count(), flush=True)
# One copy of the play store on local disk, memory-mapped by both runs: two in-memory copies
# (about 15 GB each) do not fit in the machine's RAM. Not under /kaggle/working, which is saved.
sys.path.insert(0, "/kaggle/working/code/src")
from beebot.vla.play_data import cache_play, run_dirs  # noqa: E402

t0 = time.time()
cache = "/tmp/play_cache"
if any(needs_play(a) for a in RUNS):
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
    return name, CONTROL.track(name, subprocess.Popen([*launcher, "-m", module, *full], env=env, cwd="/kaggle/working", stdout=log, stderr=subprocess.STDOUT), module)


# Kaggle shows only this script's output while it runs: relay() prints the runs' lines live as "[name] ...".
if DDP:  # one run at a time, each on every GPU
    for args in RUNS:
        if CONTROL.stopping:  # "stop" was sent: no further runs
            break
        relay([launch(None, args)], noise=noise)
else:  # one run per GPU, side by side
    relay([launch(gpu, args) for gpu, args in enumerate(RUNS)], noise=noise)
for path in glob.glob("/kaggle/working/logs/*.out"):
    print(f"==== {path}", flush=True)
    print("".join(open(path).readlines()[-60:]), flush=True)
shutil.rmtree("/kaggle/working/data", ignore_errors=True)  # keep outputs small
shutil.rmtree("/kaggle/working/extra", ignore_errors=True)
