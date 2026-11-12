"""scripts/ops_launch_guard.py: tracked jobs need an open ops node; detached work stays out of /tmp.

The commands are real ones from earlier sessions (Bash calls and ops claims), shortened."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ops_launch_guard.py"
spec = importlib.util.spec_from_file_location("ops_launch_guard", SCRIPT)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


@pytest.fixture(autouse=True)
def nodes(tmp_path, monkeypatch):
    folder = tmp_path / "nodes"
    folder.mkdir()
    (folder / "frame-into-training.yaml").write_text("title: x\nstatus: running\n")
    (folder / "cube-harm-down.yaml").write_text("title: x\nstatus: todo\n")
    (folder / "think-08b-mac.yaml").write_text("title: x\nstatus: done\n")
    monkeypatch.setattr(guard, "NODES", folder)


BLOCKED = [
    "INIT=runs/think-08b-mac/resume.pt nohup caffeinate -i scripts/mac_chain.sh think-08b-cam > logs/mac-chain-think-08b-cam.out 2>&1 &",
    "cd /Users/hrishik/beebot && time .venv/bin/python -m beebot.vla.voxel.basics --arms still,expert --per 2 2>&1 | tail -12",
    "PYTHONPATH=src nohup caffeinate -i /Users/hrishik/beebot/.venv/bin/python -m beebot.vla.voxel.haiku testset --cube --workers 4 > logs/haiku-testset.out 2>&1 &",
    ".venv/bin/python -m beebot.vla.voxel.haiku_frame generate --limit 40 --workers 4 2>&1 | tail -4",
    "PYTHONPATH=src /Users/hrishik/beebot/.venv/bin/python -m beebot.hardware.cube_outcomes calibrate --draw runs/cube_outcomes/draw --per 20 --workers 4",
    "OMP_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python -m beebot.hardware.closed_loop measure --per-kind 2 --mode simultaneous --workers 4",
    "for s in 0 1; do PYTHONPATH=src nohup nice -n 5 .venv/bin/python -m beebot.vla.voxel.demos --out data/x/s$s > logs/full-demos-s$s.out 2>&1 & done",
    "uv run python -m beebot.vla.voxel.think_train --name smoke --device mps",
    "python src/beebot/vla/voxel/think_train.py --name smoke",
    "kaggle kernels push -p kaggle/beebot-sft-long",
    "bash -c 'python -m beebot.vla.voxel.basics --arms policy'",
    "cat > scripts/chain.sh <<'EOF'\npython -m beebot.vla.voxel.think_train --name x\nEOF\nnohup zsh scripts/chain.sh > logs/chain.out 2>&1 &",
    # a node that is done, or does not exist
    "OPS_NODE=think-08b-mac nohup caffeinate -i scripts/mac_chain.sh think-08b-cam > logs/x.out 2>&1 &",
    "nohup caffeinate -i scripts/mac_chain.sh think-08b-cam > logs/x.out 2>&1 &  # ops:no-such-node",
    # detached work under /tmp, tracked or not
    "OPS_NODE=frame-into-training nohup caffeinate -i /private/tmp/chain.sh > logs/chain.out 2>&1 &",
    "S=/private/tmp/claude-501/x/scratchpad; nohup uv run python $S/size_probe.py Qwen/Qwen3-0.6B > logs/size-probe.out 2>&1 &",
    "nohup scripts/mac_throttle.sh > /tmp/throttle.out 2>&1 &",
]

ALLOWED = [
    "OPS_NODE=frame-into-training INIT=runs/think-08b-mac/resume.pt nohup caffeinate -i scripts/mac_chain.sh think-08b-cam > logs/mac-chain.out 2>&1 &",
    "nohup caffeinate -i scripts/mac_chain.sh think-08b-cam > logs/mac-chain.out 2>&1 &  # ops:frame-into-training",
    "PYTHONPATH=src .venv/bin/python -m beebot.hardware.cube_outcomes measure --set isolated --trials 200  # ops: cube-harm-down",
    "export OPS_NODE=cube-harm-down; python -m beebot.hardware.cube_outcomes right --per 4",
    "OPS_NODE=frame-into-training bash -c 'python -m beebot.vla.voxel.basics --arms policy'",
    # tests, help, imports, reads, dry runs, read-only subcommands
    ".venv/bin/python -m pytest -q tests/test_closed_loop.py tests/test_vla_voxel_haiku.py",
    ".venv/bin/python -m beebot.vla.voxel.think_train --help",
    "python -c 'import beebot.vla.voxel.basics as b; print(b.__doc__)'",
    "grep -n 'def main' src/beebot/vla/voxel/basics.py src/beebot/hardware/closed_loop.py",
    "sed -n 1,40p scripts/mac_chain.sh; cat src/beebot/vla/voxel/demos.py | head",
    "pgrep -fl 'mac_chain|think_train' | cut -c1-200; pkill -f 'beebot.vla.voxel.basics'",
    "git log --oneline -- src/beebot/vla/voxel/haiku_frame.py scripts/mac_chain.sh",
    "PYTHONPATH=src .venv/bin/python -m beebot.vla.voxel.haiku generate --dry-run --dry-items 4",
    "PYTHONPATH=src .venv/bin/python -m beebot.vla.voxel.haiku report --name run-s0",
    ".venv/bin/python -m beebot.vla.voxel.haiku_frame camera --out data/vla/haiku/frame",
    "PYTHONPATH=src .venv/bin/python -m beebot.hardware.closed_loop report runs/closed_loop/simultaneous.jsonl | tail -60",
    "PYTHONPATH=src .venv/bin/python -m beebot.hardware.cube_outcomes tally runs/cube_outcomes/x.jsonl",
    "kaggle kernels status fringewidth/beebot-sft-long",
    "python3 - <<'EOF'\nimport subprocess\nprint('python -m beebot.vla.voxel.think_train --name x')\nEOF",
    "cat > docs/run.md <<'EOF'\nnohup caffeinate -i scripts/mac_chain.sh think-08b-cam &\nEOF",
    "S=/private/tmp/claude-501/x/scratchpad; python3 $S/plot.py > $S/out.txt",
    "nohup caffeinate -i .venv/bin/python scripts/vla_demo_media.py --out site/media > logs/media.out 2>&1 &",
    "git commit -m 'scripts/mac_chain.sh: run think_train under caffeinate -i'",
]


@pytest.mark.parametrize("command", BLOCKED)
def test_blocked(command):
    assert guard.problems(command), command


@pytest.mark.parametrize("command", ALLOWED)
def test_allowed(command):
    assert guard.problems(command) == [], command


def test_background_tool_call_with_tmp_log_of_a_tracked_job():
    command = "OPS_NODE=frame-into-training .venv/bin/python -m beebot.vla.voxel.basics --arms policy > /tmp/basics.out 2>&1"
    assert guard.problems(command, background=True)
    assert guard.problems(command, background=False) == []


def test_hook_protocol():
    def run(command):
        event = {"tool_name": "Bash", "tool_input": {"command": command}}
        return subprocess.run([sys.executable, str(SCRIPT)], input=json.dumps(event), capture_output=True, text=True)

    blocked = run("kaggle kernels push -p kaggle/x")
    assert blocked.returncode == 2 and "ops launch guard: blocked" in blocked.stderr
    assert run("ls -la").returncode == 0
    broken = subprocess.run([sys.executable, str(SCRIPT)], input="not json", capture_output=True, text=True)
    assert broken.returncode == 0
