"""Run train-vla experiments on Kaggle's free GPUs (two T4s per kernel, one run per GPU).

    uv run python scripts/vla_kaggle.py code                      # upload src/ as a private dataset
    uv run python scripts/vla_kaggle.py run SLUG "--name a --grid 16 ..." "--name b ..."
    uv run python scripts/vla_kaggle.py run SLUG --play "--name a ..."   # on the play data and caption set
    uv run python scripts/vla_kaggle.py tpu-run SLUG "--name a ..." ...   # TPU v5e-8, one run per chip, up to 8
    uv run python scripts/vla_kaggle.py voxel-run SLUG --episodes 12000 "--name a ..." "--name b ..."  # voxel LM runs, play generated in the kernel
    uv run python scripts/vla_kaggle.py status SLUG
    uv run python scripts/vla_kaggle.py logs SLUG                 # the running kernel's log so far
    uv run python scripts/vla_kaggle.py pull SLUG                 # outputs to runs/kaggle/SLUG/
    uv run python scripts/vla_kaggle.py quota                     # weekly GPU and TPU hours left

On GPU each run generates its scripted data on its own GPU, so nothing but code is uploaded. On
TPU the scripted g32 stores come from the private dataset SCRIPTED_SLUG (data generation on the
TPU host's CPU is slow), and the model trains through torch_xla (beebot.vla.tpu). Kaggle kernels
run up to 12 hours (9 on TPU); give each run a --minutes budget that fits.
"""

import argparse
import json
import re
import shlex
import subprocess
import tarfile
from pathlib import Path

WORK = Path("data/vla/kaggle_train")
TEMPLATE = Path(__file__).with_name("kaggle_train") / "kernel_template.py"
TPU_TEMPLATE = Path(__file__).with_name("kaggle_tpu") / "kernel_template.py"
LM_TEMPLATE = Path(__file__).with_name("kaggle_lm") / "kernel_template.py"
VOXEL_TEMPLATE = Path(__file__).with_name("kaggle_voxel") / "kernel_template.py"
PLAY_SLUG = "beebot-play-g32"
CODE_SLUG = "beebot-src"
SCRIPTED_SLUG = "beebot-scripted-g32"
CAPTION_SET_SLUG = "beebot-caption-set-g32"  # the embedded caption set (data/vla/captions/play_g32_set.pt)


def kaggle(*args, check=True):
    print("kaggle", " ".join(args), flush=True)
    return subprocess.run(["uvx", "--with", "six", "kaggle", *args], check=check)


def username():
    view = subprocess.run(["uvx", "--with", "six", "kaggle", "config", "view"], capture_output=True, text=True, check=True).stdout
    return re.search(r"- username: (\S+)", view).group(1)


def code():
    dataset = WORK / "code"
    dataset.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dataset / "beebot-src.tar.gz", "w:gz") as tar:
        tar.add("src", filter=lambda info: None if "__pycache__" in info.name else info)
    user = username()
    (dataset / "dataset-metadata.json").write_text(json.dumps({"title": CODE_SLUG, "id": f"{user}/{CODE_SLUG}", "licenses": [{"name": "CC0-1.0"}]}))
    exists = subprocess.run(["uvx", "--with", "six", "kaggle", "datasets", "status", f"{user}/{CODE_SLUG}"], capture_output=True).returncode == 0
    if exists:
        kaggle("datasets", "version", "-p", str(dataset), "-m", "code", "--delete-old-versions")
    else:
        kaggle("datasets", "create", "-p", str(dataset))


def run(slug, run_args, tpu=False, lm=False, from_kernel=None, from_dataset=None, data_owner=None, ddp=False, play=False, module="beebot.vla.lm_train", voxel_episodes=0):
    user = username()
    owner = data_owner or user  # whose datasets to mount (shared privately when a teammate runs the kernel)
    kernel = WORK / slug
    kernel.mkdir(parents=True, exist_ok=True)
    runs = [shlex.split(a) for a in run_args]
    assert len(runs) <= (8 if tpu else 2), "one run per GPU (2) or per TPU chip (8)"
    assert not ddp or lm, "ddp: lm-run only (each run on both GPUs, one after another)"
    template = VOXEL_TEMPLATE if voxel_episodes else TPU_TEMPLATE if tpu else LM_TEMPLATE if lm else TEMPLATE
    assert not play or template == TEMPLATE, "play: train-vla runs on GPU only"
    (kernel / "kernel.py").write_text(template.read_text().replace("__RUNS__", json.dumps(runs)).replace("__DDP__", str(ddp)).replace("__PLAY__", str(play)).replace("__MODULE__", module).replace("__EPISODES__", str(voxel_episodes)))
    meta = {
        "id": f"{user}/{slug}",
        "title": slug,
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": [f"{owner}/{CODE_SLUG}"],
    }
    if tpu:
        # Internet only so the kernel can pip-install torch_xla if the image lacks it.
        meta.update({"enable_gpu": False, "enable_tpu": True, "machine_shape": "TpuV5E8", "enable_internet": True})
        meta["dataset_sources"].append(f"{owner}/{SCRIPTED_SLUG}")
    if voxel_episodes:
        meta.update({"enable_internet": True})  # the model from the Hub; play is generated in the kernel
    elif lm:
        # Internet for the model, FineWeb-Edu and GSM8K from the Hub; the play data as a dataset.
        meta.update({"enable_internet": True})
        meta["dataset_sources"].append(f"{owner}/{PLAY_SLUG}")
    if play:
        meta["dataset_sources"] += [f"{owner}/{PLAY_SLUG}", f"{owner}/{CAPTION_SET_SLUG}"]
    if from_kernel:
        # A previous kernel's outputs (checkpoints) mounted under /kaggle/input.
        meta["kernel_sources"] = [f"{user}/{from_kernel}"]
    if from_dataset:
        meta["dataset_sources"].append(from_dataset if "/" in from_dataset else f"{owner}/{from_dataset}")
    (kernel / "kernel-metadata.json").write_text(json.dumps(meta))
    kaggle("kernels", "push", "-p", str(kernel))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["code", "run", "tpu-run", "lm-run", "voxel-run", "status", "logs", "pull", "quota"])
    parser.add_argument("slug", nargs="?")
    parser.add_argument("runs", nargs="*")
    parser.add_argument("--from-kernel", help="lm-run: mount this kernel's outputs; run arguments '@input/<path>' resolve to files in them")
    parser.add_argument("--data-owner", help="run: the account owning the datasets, when it is not the account running the kernel (a teammate's; set KAGGLE_API_TOKEN to theirs)")
    parser.add_argument("--ddp", action="store_true", help="lm-run: one run on both GPUs (torchrun)")
    parser.add_argument("--module", default="beebot.vla.lm_train", help="lm-run: the trainer module (beebot.vla.lm_rl for RL)")
    parser.add_argument("--play", action="store_true", help="run: train on the play data and caption set (datasets beebot-play-g32, beebot-caption-set-g32)")
    parser.add_argument("--from-dataset", help="lm-run: mount this dataset too (e.g. checkpoints of a cancelled kernel, which Kaggle will not mount)")
    parser.add_argument("--episodes", type=int, default=12000, help="voxel-run: scripted episodes the kernel generates")
    args = parser.parse_args()
    if args.command == "code":
        code()
    elif args.command in ("run", "tpu-run", "lm-run"):
        run(args.slug, args.runs, tpu=args.command == "tpu-run", lm=args.command == "lm-run", from_kernel=args.from_kernel, from_dataset=args.from_dataset, data_owner=args.data_owner, ddp=args.ddp, play=args.play, module=args.module)
    elif args.command == "voxel-run":
        run(args.slug, args.runs, from_dataset=args.from_dataset, data_owner=args.data_owner, voxel_episodes=args.episodes)
    elif args.command == "status":
        kaggle("kernels", "status", f"{username()}/{args.slug}")
    elif args.command == "logs":
        kaggle("kernels", "logs", f"{username()}/{args.slug}")
    elif args.command == "quota":
        kaggle("quota")
    else:
        out = Path("runs/kaggle") / args.slug
        out.mkdir(parents=True, exist_ok=True)
        kaggle("kernels", "output", f"{username()}/{args.slug}", "-p", str(out))


if __name__ == "__main__":
    main()
