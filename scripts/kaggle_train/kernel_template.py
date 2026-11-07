"""Kaggle kernel: run up to one train-vla experiment per GPU, in parallel, on scripted data that
each run generates on its own GPU, or (PLAY) on the play data (beebot-play-g32) with the caption
set (beebot-caption-set-g32), cached once on local disk and memory-mapped by every run. The driver
(scripts/vla_kaggle.py) fills in RUNS and PLAY. Outputs: /kaggle/working/runs/<name>/ (config,
val history, summary, checkpoints) and logs/<name>.out.
"""

import glob
import os
import shutil
import subprocess
import sys
import tarfile
import time

RUNS = __RUNS__  # [[arg, ...], ...], one list of train-vla arguments per GPU
PLAY = __PLAY__  # True: train on the play data and caption set (both appended to every run's arguments)


def find(name, max_depth=7):
    """Paths named name under /kaggle/input, at most max_depth levels down. Fixed-depth globs: a
    recursive glob walks every play-run directory on the slow input disk (minutes)."""
    found = []
    for depth in range(1, max_depth + 1):
        found += glob.glob("/kaggle/input/" + "*/" * (depth - 1) + name)
    return [p for p in found if "/._" not in p]


# Kaggle may unpack an uploaded .tar.gz; accept either form and copy the tree somewhere writable.
archives = find("beebot-src.tar.gz")
if archives:
    tarfile.open(archives[0]).extractall("/kaggle/working/code")
else:
    package = find("beebot/vla/train.py", 8)[0]
    shutil.copytree(package.rsplit("/beebot/vla/train.py", 1)[0], "/kaggle/working/code/src")
os.makedirs("/kaggle/working/logs", exist_ok=True)
if PLAY:
    t0 = time.time()
    play = find("beebot-play-g32.tar.gz")
    if play:
        tarfile.open(play[0]).extractall("/tmp/play_data")
        play_root = "/tmp/play_data/beebot-play-g32"
    else:  # unpacked by Kaggle: the root holds the player directories (haiku-1, luna-2, ...)
        play_root = sorted(find("haiku-1"), key=len)[0].rsplit("/", 1)[0]
    caption_set = find("play_g32_set.pt")[0]
    print(f"play root {play_root}, caption set {caption_set}, found in {time.time() - t0:.0f} s", flush=True)
    # One copy of the store on local disk (not /kaggle/working, which is saved), memory-mapped by
    # every run: one in-memory copy per run (about 15 GB each) does not fit in the machine's RAM.
    sys.path.insert(0, "/kaggle/working/code/src")
    from beebot.vla.play_data import cache_play, run_dirs  # noqa: E402

    episodes = cache_play(run_dirs(play_root, size=32), "/tmp/play_cache")
    print(f"play cache: {episodes} episodes in {(time.time() - t0) / 60:.1f} min", flush=True)
    RUNS = [[*args, "--data", "play", "--play_cache", "/tmp/play_cache", "--captions", caption_set] for args in RUNS]
processes = []
for gpu, args in enumerate(RUNS):
    name = args[args.index("--name") + 1]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": "/kaggle/working/code/src"}
    log = open(f"/kaggle/working/logs/{name}.out", "w")
    print(f"gpu {gpu}: {name} {' '.join(args)}", flush=True)
    processes.append(subprocess.Popen([sys.executable, "-m", "beebot.vla.train", *args, "--device", "cuda"], env=env, cwd="/kaggle/working", stdout=log, stderr=subprocess.STDOUT))
for p in processes:
    p.wait()
    print("exit", p.returncode, flush=True)
for path in glob.glob("/kaggle/working/logs/*.out"):
    print(f"==== {path}", flush=True)
    print("".join(open(path).readlines()[-40:]), flush=True)
