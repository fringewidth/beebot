"""Kaggle TPU v5e-8 kernel: up to 8 independent train-vla runs, one per TPU chip, in parallel.
The driver (scripts/vla_kaggle.py tpu-run) fills in RUNS. The model trains on its chip through
torch_xla (--device xla); data, batch sampling and evaluation run on the host CPU
(beebot.vla.tpu). Scripted stores come pre-generated from the beebot-scripted-g32 dataset.
Outputs: /kaggle/working/runs/<name>/ and logs/<name>.out.
"""

import glob
import os
import shutil
import subprocess
import sys
import tarfile
import time

RUNS = __RUNS__  # [[arg, ...], ...], one list of train-vla arguments per chip
PORT = 8476

archives = glob.glob("/kaggle/input/**/beebot-src.tar.gz", recursive=True)
if archives:
    tarfile.open(archives[0]).extractall("/kaggle/working/code")
else:
    package = glob.glob("/kaggle/input/**/beebot/vla/train.py", recursive=True)[0]
    shutil.copytree(package.rsplit("/beebot/vla/train.py", 1)[0], "/kaggle/working/code/src")

# Pre-generated scripted stores, where train.load_or_generate looks for them.
cache = "/kaggle/working/data/vla/scripted/robots"
os.makedirs(cache, exist_ok=True)
for path in glob.glob("/kaggle/input/**/g*_n*_s*.pt", recursive=True):
    target = os.path.join(cache, os.path.basename(path))
    if not os.path.exists(target):
        os.symlink(path, target)
print("stores:", sorted(os.listdir(cache)), flush=True)

try:
    import torch
    import torch_xla

    print("torch", torch.__version__, "torch_xla", torch_xla.__version__, flush=True)
except ImportError as error:
    print("torch_xla missing:", error, flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "torch_xla[tpu]", "-f", "https://storage.googleapis.com/libtpu-releases/index.html"], check=False)
print("cpus", os.cpu_count(), flush=True)

os.makedirs("/kaggle/working/logs", exist_ok=True)
threads = max(1, (os.cpu_count() or 8) // max(1, len(RUNS)))
processes = []
for chip, args in enumerate(RUNS):
    name = args[args.index("--name") + 1]
    env = {
        **os.environ,
        "PYTHONPATH": "/kaggle/working/code/src",
        "PJRT_DEVICE": "TPU",
        "TPU_PROCESS_BOUNDS": "1,1,1",
        "TPU_CHIPS_PER_PROCESS_BOUNDS": "1,1,1",
        "TPU_VISIBLE_CHIPS": str(chip),
        "TPU_PROCESS_PORT": str(PORT + chip),
        "TPU_PROCESS_ADDRESSES": f"localhost:{PORT + chip}",
        "OMP_NUM_THREADS": str(threads),
    }
    log = open(f"/kaggle/working/logs/{name}.out", "w")
    print(f"chip {chip}: {name} {' '.join(args)}", flush=True)
    processes.append(subprocess.Popen([sys.executable, "-m", "beebot.vla.train", *args, "--device", "xla"], env=env, cwd="/kaggle/working", stdout=log, stderr=subprocess.STDOUT))
    time.sleep(5)
for p in processes:
    p.wait()
    print("exit", p.returncode, flush=True)
for path in glob.glob("/kaggle/working/logs/*.out"):
    print(f"==== {path}", flush=True)
    print("".join(open(path).readlines()[-40:]), flush=True)
