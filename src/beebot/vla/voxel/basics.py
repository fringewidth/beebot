"""Basics eval: does the swarm go where the user points?

    python -m beebot.vla.voxel.basics --checkpoint runs/<run>/policy_step<k>.pt [--per 8] [--arms policy,still,expert]

Plain direction instructions, said by a user standing somewhere around the board, seen by a camera
from a drawn side: "go to my left", "head to the far right corner", "go north", "take three steps
toward me". Nothing in training is a direction family of its own: whether these work is what the
model has made of every other task's directions, so the eval measures that, and nothing here
feeds training.

Frames. The camera's board side and the user's place relative to the camera (think_data.
USER_POSITIONS) are drawn per episode; together they put the user at one side of the board
(haiku_frame.user_side). The user's directions map onto the board as haiku_frame.mapping does
and as people talk: "left" and "right" are the user's own; "the front", "near", "toward me" and
"south" are the side nearest the user; "the back", "far", "forward", "away" and "north" the side
away from them; east is their right, west their left. The instruction goes to the policy exactly
as said (rollout.Episode.as_said), with the user turn saying where the user stands.

Scoring. Each instruction becomes the suite task it means on the board: go to a side or corner
area (coverage goto, whole swarm) or take three steps toward a side (coverage move); the suite's
checker scores it and the pass mark is the suite's (rollout.passed). As the suite's own draw does
(suite.make_motion), a scene where the task already scores half marks, or a move has no room for
its steps (suite.has_room), is drawn again (feasible). Arms: the policy (argmax
actions), standing still (the floor of the scale) and the scripted expert given the board task
(the ceiling: what the scene and the move failures allow).

Output: one JSON line per arm and category (side, corner, compass, steps) and one per arm over
all: {"run": "basics", "arm", "category", "episodes", "pass", "score"}. ops follows these lines
(basics_direction_pass{arm=...} is pass over all).
"""

import argparse
import json
import sys

import numpy as np
import torch

from beebot.vla.voxel import rollout as R
from beebot.vla.voxel import suite as S
from beebot.vla.voxel.haiku_frame import mapping, user_side
from beebot.vla.voxel.think_data import ORDER, USER_NAMES

SEED = 4242
REDRAWS = 200  # scene draws an episode may take to find one its board task is feasible in
PHRASES = (  # (category, what the user types, the user's directions it names, "goto" or "move")
    ("side", "go to my left", ("left",), "goto"),
    ("side", "everyone over to my right", ("right",), "goto"),
    ("side", "go to the side nearest me", ("toward",), "goto"),
    ("side", "head to the far side", ("away",), "goto"),
    ("side", "gather at the front", ("toward",), "goto"),
    ("side", "move over to the back", ("away",), "goto"),
    ("corner", "go to the far left corner", ("away", "left"), "goto"),
    ("corner", "head for the near right corner", ("toward", "right"), "goto"),
    ("corner", "everyone to the back right corner", ("away", "right"), "goto"),
    ("corner", "go to the front left corner", ("toward", "left"), "goto"),
    ("compass", "go to the north side", ("away",), "goto"),
    ("compass", "everyone head west", ("left",), "goto"),
    ("compass", "gather in the southeast corner", ("toward", "right"), "goto"),
    ("compass", "go to the northwest corner", ("away", "left"), "goto"),
    ("steps", "take three steps forward", ("away",), "move"),
    ("steps", "move three steps toward me", ("toward",), "move"),
    ("steps", "shift three steps to my left", ("left",), "move"),
    ("steps", "go three steps to my right", ("right",), "move"),
)
CATEGORIES = ("side", "corner", "compass", "steps")


def board_region(directions, side):
    """The suite floor area (suite.FLOOR_REGIONS) the user's directions name, for a user at side."""
    table = mapping(side)
    sides = [table[d] for d in directions]
    fb = [s for s in sides if s in ("front", "back")]
    lr = [s for s in sides if s in ("left", "right")]
    return " ".join(fb + lr)


def board_task(world, i, text, region, job):
    """The suite task the instruction means on the board: the whole swarm to an area, or three
    steps toward a side; no cheese may be eaten."""
    forbid = {c for _, c in S.present(world, i)}
    if job == "goto":
        spec = {"kind": "goto", "who": dict(S.ALL), "region": region, "forbid": forbid}
    else:
        spec = {"kind": "move", "who": dict(S.ALL), "toward": region, "steps": S.MOVE_STEPS, "forbid": forbid}
    return S.coverage_task("motion", text, spec, "")


def feasible(task, world, meta, i):
    """The suite's own draw filters (suite.make_motion): the task does not already score half
    marks at the start, and a move has room for nearly all robots (suite.has_room)."""
    return S.has_room(task, world, i) and not S.already_done(task, world, meta, i)


def scene(rng):
    """(world of one scene, its meta) drawn as the eval's scenes are."""
    world, metas, _ = S.sample_coverage(rng, 1, family="motion", robots=S.ROBOTS, counts="log", job="goto", keep=True)
    return world, metas[0]


def episodes(per=8, seed=SEED, redraws=REDRAWS):
    """per episodes of every phrase: a fresh motion scene each, a drawn camera side and user place.
    A scene where the board task is not feasible (feasible: already half done, or a move without
    room) is drawn again, from a generator of its own seeded by (seed, 1), so the other episodes
    keep their draws; ValueError after `redraws` tries."""
    rng = np.random.default_rng(seed)
    again = np.random.default_rng((seed, 1))
    n = per * len(PHRASES)
    world, metas, _ = S.sample_coverage(rng, n, family="motion", robots=S.ROBOTS, counts="log", job="goto", keep=True)
    out = []
    for i in range(n):
        category, text, directions, job = PHRASES[i % len(PHRASES)]
        camera = ORDER[int(rng.integers(4))]
        place = USER_NAMES[int(rng.integers(len(USER_NAMES)))]
        region = board_region(directions, user_side(camera, place))
        board, meta = world.index(i), metas[i]
        task = board_task(board, 0, text, region, job)
        tries = 0
        while not feasible(task, board, meta, 0):
            tries += 1
            if tries > redraws:
                raise ValueError(f"no feasible scene for {text!r} ({region}) in {redraws} draws")
            board, meta = scene(again)
            task = board_task(board, 0, text, region, job)
        key = {"category": category, "text": text, "camera": camera, "place": place, "region": region, "i": i, "redraws": tries}
        out.append(R.Episode(board, meta, task, [text], [0], [S.canonical_record(task)], R.STEPS, seed * 1_000 + i, key=key, user=place, camera=camera, as_said=True))
    return out


class Still:
    def __call__(self, world, t):
        return torch.full(world.mass.shape, R.STAY, dtype=torch.long)


def play_arm(arm, eps, policy=None, batch=16):
    played = [None] * len(eps)
    for idx in R.batches(eps, batch):
        some = [eps[k] for k in idx]
        if arm == "policy":
            result = R.play_policy(policy, some)
        elif arm == "expert":
            result = R.play_expert(some, seed=SEED)
        else:
            result = R.play(some, Still())
        for k, p in zip(idx, result):
            played[k] = p
    return played


def summary(arm, eps, played):
    rows = []
    for category in CATEGORIES + ("all",):
        pick = [k for k, e in enumerate(eps) if category in ("all", e.key["category"])]
        if not pick:
            continue
        rows.append({"run": "basics", "arm": arm, "category": category, "episodes": len(pick), "pass": round(float(np.mean([played[k].passed for k in pick])), 3), "score": round(float(np.mean([played[k].score for k in pick])), 3)})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--arms", default="policy,still,expert")
    parser.add_argument("--per", type=int, default=8, help="episodes per phrase")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--out", default="", help="also write every episode's row here (JSON lines)")
    args = parser.parse_args(argv)
    eps = episodes(args.per)
    policy = None
    arms = args.arms.split(",")
    if "policy" in arms:
        if not args.checkpoint:
            parser.error("the policy arm needs --checkpoint")
        from beebot.vla.voxel import think

        policy, _ = think.load_policy(args.checkpoint, args.device, torch.float32)
        policy.eval()
    for arm in arms:
        with torch.no_grad():
            played = play_arm(arm, eps, policy, args.batch)
        for row in summary(arm, eps, played):
            print(json.dumps(row), flush=True)
        if args.out:
            with open(args.out, "a") as f:
                for e, p in zip(eps, played):
                    thought = next((t["text"] for t in p.thoughts if t["t"] == 0), "")
                    f.write(json.dumps({"arm": arm, **e.key, "passed": p.passed, "score": round(p.score, 4), "thought": thought, "spoke": p.spoke}) + "\n")


if __name__ == "__main__":
    sys.exit(main())
