# grounding-formats: four ways to tie the cubes in the pictures to the cubes the policy moves

Written 2026-11-13. These are opt-in example kinds for the think-then-act policy. Training does not read them
yet: nothing turns them on.

Code:
1. `src/beebot/vla/voxel/think_ground.py`: the store builder, marks, words and the sampler.
1. `src/beebot/vla/voxel/think_data.py`: one new argument, `ThinkData(..., ground=None)`.

Tests:
1. `tests/test_think_ground.py`.
1. `tests/think_data_digest.py` with the fixture `tests/fixtures/think_data_default_digest.json`.

Data (not committed): `data/vla/ground/v0/s0..s3`. Build logs: `logs/ground-build-v0-s*.out`.

    nice -n 10 .venv/bin/python -m beebot.vla.voxel.think_ground build --out data/vla/ground/v0/s0 --goals 60 --moves 80 --seed 0 --max-robots 40
    # train with them (no trainer change): think_train --data_kwargs '{..., "ground": {"dirs": ["data/vla/ground/v0/s0", ...], "shares": {"goal_image": 0.05, "point": 0.05, "views": 0.05, "predict": 0.05}}}'

## Design

Every kind keeps think_data's plain causal layout:

    <|im_start|>user\n<IMAGES>\n{instruction}<|im_end|>\n<|im_start|>assistant\n{reply}

All of the example's images sit in the one `<IMAGES>` place, as consecutive vision blocks in a fixed
order. Only their order separates them, which the manager approved for now. The reply is one of two
forms:

1. The usual act reply: `<think>\n{thought}\n</think>\n\n<tool_call>`, then the 64 action slots. A
   thought appears only at the episode's first step, and later steps carry it as memory.
1. A text reply: `<think>\n\n</think>\n\n{answer}<|im_end|>`.

The user stands at the scene camera. The observation, the action codes and every direction word are
in that camera's frame (camera_frame.py), as in think_data. Images are 256 x 256 at
views.DEFAULT_STRENGTH randomization; P is the perspective camera, T the top view, 64 visual tokens each.

| kind | images, in order | instruction (examples) | reply | scored by |
|---|---|---|---|---|
| goal_image | scene P, scene T, goal P from another side | "build what the last picture shows", "copy the last picture" | act; first thought "the last picture, in words: <goal text in the camera's words>" | action loss; on the Mac, rollout end state equal to the goal |
| point:mark | scene P with a ring on the cube, an arrow and a cross where it goes; scene T | "move this one there", "put this one on top of that one" | act | action loss; rollout: that cube at the cross, nothing else moved |
| point:words | scene P with a ring; scene T | "put the circled one into the front left corner", "... on top of the tallest stack" | act | as point:mark |
| views:first / views:second | P from the user's side and P from another side, in either order | "my view is the second picture. move the top cube of the stack nearest me into the back right corner" | act; the thought also names the cube as seen from the other camera | as point |
| predict:moved | P at t with letters on 3-4 cubes; P at t + k (k = 1-3) from the same camera and look | "which of the lettered cubes moved between the first and the second picture?" | "b." / "a and c." | exact letter set (`score_text`) |
| predict:where | P at t, a ring on one cube, letters on 3-4 cells it could go to | "the circled cube moves one cell up and to the left. which letter marks where it ends up?" | "c." | exact letter |

Where the episodes come from:
1. goal_image: goal-library goals of at most 40 robots, one quarter each from gen_simple, gen_medium,
   gen_complex and gen_ground (tricks and replies skipped), walked backwards by `reverse.take_apart`.
   End states start from a natural scene; gen_complex and gen_ground edits start from their A. An
   episode is kept only if every step replays and the start is what was asked.
1. point and views: one-robot moves (the "move" episodes). The start scene is a frame of a goal
   episode, either its natural start or its end state. The mover is the top cube of a stack. The
   destination is one of three kinds:
   1. a free floor corner;
   1. the top of another stack;
   1. a free floor cell 2-6 columns away.

   The episode is the guided walk to the moved scene, kept only if it reaches it exactly, every step
   replays and no other robot moved. The ring and the arrow follow the cube as it walks.
1. predict: the ground store's episodes and ThinkData's own demo stores (swarm demos with objects and
   cheeses), half each.

Marks: `cell_spots` finds where a cube shows in the randomized render. For an empty cell it uses the
faces a cube there would show, without changing the render. It goes through the view's lens warp,
crop and resolution, so a mark lands on its cube under any camera draw (tested against a render with
and without the cube). A mark needs at least 10 visible pixels. A cross that would touch the ring is
replaced by words, or the draw is retried.

Viewer-relative words: "nearest me", "farthest from me", "furthest to my left" and "furthest to my
right" are used only when that stack beats every other stack by 2 cells in the camera's frame; "the
tallest" is used when it beats every other by 1. Corners are said as think_data says them: front is
the side nearest the camera. Goal texts are turned by `think_data.in_user_frame`. A text it cannot
turn, or one that speaks of "me", "nearer" or the viewer, puts the scene camera at the world's front.

## Token cost per example

Measured with the Qwen3.5-0.8B tokenizer and image processor on CPU, no model weights; `slots` is
the 64 action slots of an acting row. The default row is today's ThinkData on data/vla/voxel_full/s0
with the scene views.

| kind | images | visual tokens | other tokens | reply tokens | slots | row total (max) | CPU s per example |
|---|---|---|---|---|---|---|---|
| default (today) | 2 | 132 | 91 | 25 | 64 x act share | 287 with slots | - |
| goal_image | 3 | 198 | 59 | 14 | 64 | 321 (354) | 0.08 |
| point:mark | 2 | 132 | 48 | 18 | 64 | 244 (254) | 0.10 |
| point:words | 2 | 132 | 57 | 14 | 64 | 253 (267) | 0.10 |
| views | 2 | 132 | 96-104 | 26-35 | 64 | 292-300 (334) | 0.06 |
| predict:moved | 2 | 132 | 31 | 7 | 0 | 163 (165) | 0.12 |
| predict:where | 1 | 66 | 35 | 7 | 0 | 101 (103) | 0.12 |

The "other tokens" column counts every non-visual token in the row, reply included. Visual tokens
include the two boundary tokens per image.

## Trainer changes needed: none

1. `think.prepare` already takes any number of images at the one `<IMAGES>` place: today's rows have
   two (perspective and top). The processor's grid gives exactly `think_data.image_tokens` for every
   example measured.
1. `get_rope_index` of a meta-device Qwen3.5-0.8B (no weights) took a 3-image goal_image row: shape
   (3, 1, 284).
1. Batching: rows are right-padded and bucketed by length. All kinds are 100-354 tokens, about
   today's length. A goal_image row sends 3 images through the frozen vision encoder instead of 2.
1. XLA `pack` counts image features from the grid, so the image count per row does not matter.
1. Text rows (predict) are `act=False`, like the clarification refusals the trainer already takes:
   text loss only, no slots.
1. Turned on through the existing `--data_kwargs` JSON (`ground`). `scripts/mac_chain.sh` is unchanged.

## Defaults unchanged

`ThinkData(ground=None)` (the default) is byte for byte what it was. The fixture holds 60 digests
written before this change (48 sample() draws and 12 validation examples; token ids with the
0.8B tokenizer, images, obs, codes, texts, on the synthetic store of test_vla_voxel_think_data.py with
Haiku labels and refusals). It still matches. A second test checks that `ground=None` and
`{"shares": {}}` draw no extra random numbers with the randomized scene views.

## The store (data/vla/ground/v0)

Four shards, seeds 0-3, at most 40 robots, CPU at nice 10, 4-7.5 minutes each:
1. Goals: 54, 56, 55 and 57 of 60 kept (222 in all; about a quarter per source).
1. Moves: 80 of 80 kept per shard (320).
1. Held out by a hash of the source goal: 18 moves on the held side.

Episodes run 4-20 steps (10-90 % range) for moves. Only 39 of the 302 training moves can be said
from some side in views (both the cube and its destination nameable), and none of the held ones, so
views has little variety. The fix is a bigger build (more moves) or a builder that draws views-ready
moves on purpose. A kind with no usable episode on its side of the split gives an ordinary example
instead, counted as `counts()["ground"]["drawn"]["unavailable:<kind>"]`.

## Render check

12 per kind in the browse page: section "Grounding formats", files `ground-fmt-<kind>-<k>.png`, cards
`cards-ground-fmt.json`.

What reads at a glance:
1. Rings sit on their cubes under every camera draw, including the wide-angle and tilted ones.
1. Arrows and crosses read as a drag.
1. Corner and stack words are computed in the camera's frame (unit-tested for all four quarter
   turns). In the renders I checked, the named stack ("furthest to my right") is the one the
   picture shows there.
1. Letter answers check out against the pictures; for example, "to the right" from a camera at the
   back is the letter to the screen right of the ring.

Weak points:
1. With strong camera randomization the arena can be small (cubes about 8 px), so letters on
   neighbouring cells crowd. They fan out from the ring, but predict:where stays hard to read on far
   cameras.
1. The goal picture is drawn from another side with its own lighting. Small changes, such as one
   cube moved in gen_ground edits, are hard to see at 256 px. Cutouts are off in goal pictures.
1. gen_ground goal texts such as "Fix it" give the thought "the last picture shows the end state".
1. point:words is rare (1 of 12 drawn): only corner and nameable-stack destinations can be said.

## Decisions for the user

1. Images separated by order only: accepted for now. Text between images ("now:" / "goal:") would
   need several `<IMAGES>` places in `think.prepare` and `think_data.tokenize`.
1. Shares of each kind when switched on (nothing switches them on yet).
1. Goal picture: as randomized as the scene camera (now), or a clean "design" render?
1. Moves come only from library scenes of at most 40 robots, not from the demo stores with objects.
1. predict answers by letters drawn on the picture (exact scoring) rather than in words.
1. Act kinds have no rollout scorer yet. Proposed: end state equal to the goal for goal_image; the
   cube at its destination with nothing else moved for point and views.
1. Vague goal texts in gen_ground ("Fix it"): keep them as goal pictures (now), or drop them.
1. A bigger, views-oriented build before any training use.
