"""One-off: the 2026-11-17 split of the repo into v0/ (neural CA), v1/ (2D LM policy and shared code),
v2/ (3D voxel) and hardware/ (cube modules). Kept as the record of where every file went.

    python3 v1/scripts/reorg_2026_11_17.py plan     # print the moves
    python3 v1/scripts/reorg_2026_11_17.py move     # git mv them (directories move whole, untracked files included)
    python3 v1/scripts/reorg_2026_11_17.py rewrite  # rewrite old paths to new ones in every tracked text file

The map is also written to v1/scripts/reorg_2026_11_17.json (old path -> new path).
"""

import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MAP_PATH = os.path.join(HERE, "reorg_2026_11_17.json")

V0_MODULES = "bench ca chase evaluate feast final_test layout model probe render report rivals rivals_probes score train train_feast train_rivals".split()

V0_SCRIPTS_TESTS = "queue_rivals.sh supervise.sh".split()
V0_TESTS = "chase_feasible.py test_chase.py test_feast.py test_physics.py test_rivals.py".split()

V1_SCRIPTS = """field_demo_record.py field_demo_video.py steer_dwell_chain.sh move37_chain.sh move37_search.py steer_kinds.json
swarm_record.py swarm_steer.py swarm_steer.html swarm_steer_serve.sh run_control.py kaggle_follow.py kaggle_rest.py
ops_launch_guard.py ops_session_brief.sh vla_kaggle.py vla_embed_remote.py vla_caption_audit.py vla_cheese_diagnostic.py
vla_demo_media.py vla_fact_breakdown.py vla_final_table.sh vla_goal_oracle.py vla_grid_reading_probe.py vla_grounding_eval.py
vla_herd_diagnostic.py vla_inference_sweep.py vla_lm_fair_2d.py vla_lm_play.py vla_lm_text_eval.py vla_requested_events.py
vla_rollout_strips.py vla_site.py vla_site_template.html vla_size_transfer.py vla_snap_test.py vla_swarm_demo.py
vla_swarm_live.py vla_swarm_live.html vla_swarm_web.py vla_task_wordings.py vla_text_eval.py vla_train_vs_val.py
kaggle_embed kaggle_lm kaggle_train kaggle_tpu/kernel_template.py""".split()
V1_TESTS = """test_control.py test_kaggle_follow.py test_ops_launch_guard.py test_mujoco_world.py test_vla_caption.py
test_vla_caption_aug.py test_vla_controller.py test_vla_facts.py test_vla_goal_eval.py test_vla_model.py test_vla_play.py
test_vla_relabel.py test_vla_swarm.py test_vla_tasks.py test_vla_world.py""".split()

HW_SCRIPTS = """cube_turning_runs.sh joint_requirements.py joint_validate.py module_c_demos.py module_demos.py module_phys
module_phys_push.sh physics_primitive_probe.py physics_rig.py rotating_cube_video.py small_joint_machines.py""".split()
HW_TESTS = """test_cube.py test_cube_hinge.py test_cube_swarm.py test_cube_turning.py test_geared_joint.py test_joint_strength.py
test_mmsim.py test_module_b.py test_module_c.py test_rw_cube.py test_sma_cube.py test_closed_loop.py test_hardware.py""".split()
HW_RUNS = "closed_loop cube cube_outcomes cube_redesign cube_swarm module_pairs module_phys".split()
HW_HANDOFF = """gear14-machines.md joint-strength.md module-demos.md module-pairs.md rotating-cube-poc.md rw-cube.md
small-joint.md cube-control.md cube-redesign.md harm-down.md hinge-adopt.md sma-cube.md physics-options.md
physics-primitive.md compose.py fit_pooled.py pareto.py prefix.py trunc.py walkgap.py tests.log tests2.log media""".split()

V1_HANDOFF = ["remote-control.md"]

V2_RUNS = "build coverage free_cube objects randomize views think-08b-cam quarantine.json task_mix.json frame_verified.jsonl.gz".split()


def tracked():
    return subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).split("\n")


def plan():
    """(old, new) pairs; a directory moves whole."""
    moves = []
    add = lambda old, new: moves.append((old, new))
    # v0: the neural cellular automata
    for m in V0_MODULES:
        add(f"src/beebot/{m}.py", f"v0/src/beebot/{m}.py")
    for d in "runs_v1 runs_v2 runs_v3a runs_v3b runs_v4 fog_hack".split():
        add(d, f"v0/{d}")
    for f in sorted(f for f in tracked() if f.startswith("logs/")):
        add(f, "v0/" + f)
    for f in sorted(f for f in tracked() if f.startswith("runs/v6-rivals")):
        add(f, "v0/" + f)
    for f in "HILLCLIMB.md JOURNAL.md REPORT.md".split():
        add(f, f"v0/docs/{f}")
    add("README.md", "v0/README.md")
    for f in V0_SCRIPTS_TESTS:
        add(f, f"v0/scripts/{f}")
    for f in V0_TESTS:
        add(f"tests/{f}", f"v0/tests/{f}")
    # v1: the 2D language-steered policy, and the code the later versions share
    add("src/beebot/control.py", "v1/src/beebot/control.py")
    for f in sorted(f for f in tracked() if f.startswith("src/beebot/vla/") and not f.startswith("src/beebot/vla/voxel/")):
        if f != "src/beebot/vla/__init__.py":
            add(f, "v1/" + f)
    for f in V1_SCRIPTS:
        add(f"scripts/{f}", f"v1/scripts/{f}")
    for f in V1_TESTS:
        add(f"tests/{f}", f"v1/tests/{f}")
    for f in sorted(f for f in tracked() if f.startswith("VLA") and f.endswith(".md")):
        add(f, f"v1/docs/{f}")
    add("docs/vla_explainer", "v1/docs/vla_explainer")
    add("assets", "v1/assets")
    add("site", "v1/site")
    for f in V1_HANDOFF:
        add(f"runs/handoff/{f}", f"v1/handoff/{f}")
    # hardware: cube modules (the CAD folders hardware/module_phys, rotating_cube, sma_cube stay where they are)
    add("src/beebot/hardware", "hardware/src/beebot/hardware")
    for f in HW_SCRIPTS:
        add(f"scripts/{f}", f"hardware/scripts/{f}")
    for f in HW_TESTS:
        add(f"tests/{f}", f"hardware/tests/{f}")
    for d in HW_RUNS:
        add(f"runs/{d}", f"hardware/runs/{d}")
    for f in HW_HANDOFF:
        add(f"runs/handoff/{f}", f"hardware/handoff/{f}")
    add("HARDWARE.md", "hardware/docs/HARDWARE.md")
    add("hardware/MODULE_SPEC.md", "hardware/docs/MODULE_SPEC.md")
    # v2: the 3D voxel world and the think-then-act policy (everything left in scripts/, tests/, runs/handoff/)
    add("src/beebot/vla/voxel", "v2/src/beebot/vla/voxel")
    add("MICROBOTS.md", "v2/docs/MICROBOTS.md")
    for d in V2_RUNS:
        add(f"runs/{d}", f"v2/runs/{d}")
    taken = {old for old, _ in moves}
    under = lambda f: any(f == o or f.startswith(o + "/") for o in taken)
    rest = [f for f in tracked() if f and not under(f)]
    for f in sorted(rest):
        for prefix, new in (("scripts/", "v2/scripts/"), ("tests/", "v2/tests/"), ("runs/handoff/", "v2/handoff/")):
            if f.startswith(prefix):
                add(f, new + f[len(prefix):])
    return moves


def move(moves):
    for old, new in moves:
        if not os.path.exists(os.path.join(ROOT, old)):
            print("missing", old)
            continue
        os.makedirs(os.path.dirname(os.path.join(ROOT, new)), exist_ok=True)
        subprocess.run(["git", "mv", old, new], cwd=ROOT, check=True)
    with open(MAP_PATH, "w") as f:
        json.dump(dict(moves), f, indent=1)
        f.write("\n")


TEXT = re.compile(r"\.(py|sh|md|json|html|yml|yaml|toml|txt|js|css)$")


def rewrite():
    moves = json.load(open(MAP_PATH))
    keys = sorted(moves, key=len, reverse=True)
    # an old path counts when it is not glued to a longer path on the left or a longer name on the right
    pattern = re.compile(r"(?<![\w./-])(" + "|".join(re.escape(k) for k in keys) + r")(?![\w-])")
    changed = 0
    for f in tracked():
        if not f or not TEXT.search(f) or f == "v1/scripts/reorg_2026_11_17.json" or f.startswith("v1/site/media"):
            continue
        path = os.path.join(ROOT, f)
        if os.path.islink(path) or os.path.getsize(path) > 3_000_000:
            continue
        try:
            text = open(path).read()
        except UnicodeDecodeError:
            continue
        new = pattern.sub(lambda m: moves[m.group(1)], text)
        if new != text:
            open(path, "w").write(new)
            changed += 1
    print("rewrote", changed, "files")


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "plan":
        for old, new in plan():
            print(old, "->", new)
    elif command == "move":
        move(plan())
    elif command == "rewrite":
        rewrite()
