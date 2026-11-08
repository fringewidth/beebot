"""Kaggle kernel: voxel-world LM policy runs (beebot.vla.voxel.lm), one run per T4. The scripted
play and its hindsight labels are generated here (minutes of CPU), so only the code is uploaded.
Needs internet (the model from the Hugging Face Hub) and the code dataset (beebot-src); runs that
start from a 2D checkpoint ('--init_lora @input/<path>') also need that checkpoint's dataset.
The driver (scripts/vla_kaggle.py voxel-run) fills in RUNS, EPISODES and DEMO_SCENES. Outputs:
/kaggle/working/runs/<name>/ and logs/<name>.out.
"""

import glob
import os
import subprocess
import sys
import tarfile
import time

RUNS = __RUNS__  # [[arg, ...], ...], one list of voxel.lm arguments per GPU
EPISODES = __EPISODES__  # scripted play episodes to generate (runs may train on a prefix: --episodes N)
DEMO_SCENES = __DEMO_SCENES__  # scenes of expert demonstrations of the suite (voxel/demos.py); 0 for none

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.22.1"], check=False)


def find(name, max_depth=8):
    found = []
    for depth in range(1, max_depth + 1):
        found += glob.glob("/kaggle/input/" + "*/" * (depth - 1) + name)
    return [p for p in found if "/._" not in p]


tarfile.open(find("beebot-src.tar.gz")[0]).extractall("/kaggle/working/code")
RUNS = [[find(a[len("@input/") :])[0] if a.startswith("@input/") else a for a in args] for args in RUNS]
env = {**os.environ, "PYTHONPATH": "/kaggle/working/code/src", "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4"}
os.makedirs("/kaggle/working/logs", exist_ok=True)
store = "/tmp/voxel_play_g16"
t0 = time.time()
subprocess.run([sys.executable, "-m", "beebot.vla.voxel.play_store", "--out", store, "--episodes", str(EPISODES)], env=env, cwd="/kaggle/working", check=True)
subprocess.run([sys.executable, "-m", "beebot.vla.voxel.relabel", "--store", store, "--out", f"{store}/labels.jsonl", "--workers", "4"], env={**env, "OMP_NUM_THREADS": "1"}, cwd="/kaggle/working", check=True)
demos = "/tmp/voxel_demos_g16"
if DEMO_SCENES:
    subprocess.run([sys.executable, "-m", "beebot.vla.voxel.demos", "--out", demos, "--scenes", str(DEMO_SCENES)], env=env, cwd="/kaggle/working", check=True)
print(f"play, labels and demos in {(time.time() - t0) / 60:.1f} min", flush=True)


def launch(gpu, args):
    name = args[args.index("--name") + 1]
    log = open(f"/kaggle/working/logs/{name}.out", "w")
    full = [*args, "--store", store, "--labels", f"{store}/labels.jsonl", *(["--demo_store", demos, "--demo_labels", f"{demos}/labels.jsonl"] if DEMO_SCENES else []), "--device", "cuda"]
    print(f"gpu {gpu}: {name} {' '.join(full)}", flush=True)
    return name, subprocess.Popen([sys.executable, "-m", "beebot.vla.voxel.lm", *full], env={**env, "CUDA_VISIBLE_DEVICES": str(gpu)}, cwd="/kaggle/working", stdout=log, stderr=subprocess.STDOUT)


started = [launch(gpu, args) for gpu, args in enumerate(RUNS)]
seen = {name: 0 for name, _ in started}
while True:
    for name, _ in started:
        lines = open(f"/kaggle/working/logs/{name}.out", errors="replace").read().split("\n")
        for line in lines[seen[name] : len(lines) - 1]:
            if line.strip() and not any(n in line for n in ("Warning", "Loading weights", "warn")):
                print(f"[{name}] {line if '"card"' in line else line[:900]}", flush=True)  # scorecards whole
        seen[name] = len(lines) - 1
    if all(p.poll() is not None for _, p in started):
        break
    time.sleep(30)
for name, p in started:
    print("exit", name, p.returncode, flush=True)
