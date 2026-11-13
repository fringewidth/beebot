# reverse-loader: basic tasks generated live by reversal

Written 2026-11-13. Opt-in example source for the think-then-act policy. Training does not read it yet: nothing
turns it on.

Code:
1. `src/beebot/vla/voxel/reverse_live.py`: end-state samplers per family, the walk back (reverse.take_apart), the
   words (Haiku rewordings or templates), background workers, examples for ThinkData.
1. `src/beebot/vla/voxel/think_data.py`: one new argument, `ThinkData(..., live=None)`.
1. `scripts/reverse_live_measure.py` (throughput), `scripts/reverse_live_browse.py` (GIFs for the browse page).

Tests: `tests/test_reverse_live.py` (9 passed: default examples byte-identical to the fixture
`tests/fixtures/think_data_default_digest.json` with `live=None` and with share 0; every family replays; end states;
the text filter; no simulator nouns in the words; ThinkData mixes live examples at their share; background workers).
Full suite on the working tree: 860 passed, 1 failed (`test_vla_voxel_reverse.py::test_reversed_episodes_replay_forward[tower]`,
which fails with another agent's uncommitted reverse.py; at HEAD's reverse.py it and this file's tests pass, 43 of 43).

    # train with it (no trainer change):
    think_train --data_kwargs '{..., "live": {"share": 0.2, "workers": 4}}'
    # measure
    OMP_NUM_THREADS=1 nice -n 10 .venv/bin/python scripts/reverse_live_measure.py --per 24 --procs 4 --replay --out runs/reverse_live/measure.json

## How an episode is made

1. Draw a family (equal weights by default; `weights=` changes them), a world (terrain as the stores draw it: flat,
   low hills or a trench between two plateaus; 0-2 loose blocks; gravity on) and a robot count (log-uniform 6-115,
   as `data/vla/haiku/run-s*/config.json`; a family that needs fewer narrows it, below).
1. Put every robot where the instruction wants it (the end state; table below) and the spec exactly as the
   stores write it (`{"kind": "goto", "region": "front left"}`), so the Haiku rewordings of that spec apply.
1. Walk back from the end with `reverse.take_apart`. Natural mode (the robots walk out to a start that matches
   natural scenes of that terrain and count statistically) for most families; guided mode (walk to a given start
   A) where the start must be of a kind: spread (A: one to three piles), move (A: the end shifted three columns
   back), climb (A: robots in the trench and on the other plateau), stay (A = end: a hold). Every recorded step is
   verified by the world's own forward step (take_apart); the measurement and tests also replay each episode with
   `world.step`.
1. Check start and end against the instruction (strict, below); a draw that fails is drawn again (up to 8 times).
1. Pack it (bit-packed frames, codes on robot cells, 13-70 kB) with its record, template wordings and the start's
   scene summary. The words are chosen per example, so one episode gives different rewordings on different draws.

A training example from an episode: step 0 (with the first thought) with probability 1/3, else a step from 1 to
T, where step T is the goal with every robot staying (so the policy also sees "done, hold still"). The camera,
the turned observation and codes, and the words in the camera's frame work as in think_data (`views`,
`in_user_frame`; words that cannot be turned get a camera at the world's front).

## End states and how strict

Strict means a stranger would agree at a glance; the check runs on every episode's start and end, and a draw that
fails it is drawn again. Robots: the stores' draw (log-uniform 6-115) unless the family says otherwise.

| family | spec (as the stores) | end state | start | walk |
|---|---|---|---|---|
| goto | goto region (9 ninths) | every robot piled in the region's box (one cell margin, as suite's goto); low piles | at most a third in the box | natural |
| gather | gather | one pile anywhere: exactly one 26-connected group | at least two groups, none half the robots | natural |
| groups | groups count 2 or 3 | 2-3 piles at suite.GROUP_PAIRS places, two empty columns apart: exactly that many groups, each 2+ | largest group under half | natural |
| spread | spread | nobody within one cell of anybody (corners count), all on the ground (6-~60 robots: as many as fit apart) | 1-3 piles, at least n/2 touching pairs | guided |
| line | line axis x/y | 3-14 robots (suite's LINE_MAX), one contiguous row on level ground | no straight run of half of them | natural |
| move | move toward side, 3 steps | the swarm in a window about 2n columns; start = end shifted 3 columns back, each stack resting on the ground | by construction: mean shift exactly 3 toward the side | guided |
| stay | stay | a natural scene | the same (reverse.HOLD_STEPS = 4 steps of everyone staying) | hold |
| climb | climb side (trench scenes) | every robot piled on that plateau, near its edge | nobody on that plateau (trench and other plateau, natural heaps) | guided |
| split_regions | parallel region+region | half the robots piled in each of two named ninths, two columns apart: two groups, each inside one box | at most 40 % in the two boxes | natural |
| split_tower | parallel region+tower (h 3-6) | a tower h high (single column, or 2 x 2 when the swarm has room) outside the region, everyone else in the region | tallest stack under h | natural |
| tower | tower h (3-8) | one tower h high (1 x 1 or 2 x 2), every other robot on the ground (one layer, not next to the tower) | tallest stack under h | natural |
| bridge | bridge (trench scenes) | the trench filled solid across in a lane 1 or 2 wide, flush with the plateaus; everyone else on the plateaus, one layer | under half of the crossing's cells filled | natural |
| zone | goto_zone colour | every robot on the patch (3-4 a side, at most two layers, so robots at most twice its cells); 0-2 other patches or strips | at most a fifth on it | natural |
| zone_line | line_zone colour | 3 to strip-length robots in a row on the strip (1 x 6-10), one per cell | at most a fifth on it | natural |
| push | none in the stores (template only) | reverse.push_goal: a box, tile or bar on a coloured patch, 2-3 pushers behind it, pushed 3-6 cells, 0-3 others (flat floor, 2-6 robots) | the object 3-6 cells off the patch | push phase then natural |

Left out: construction "fill the trench". At 6-115 robots no end state passes at a glance (a trench is 3-4 wide,
3-4 deep and 16 long: 144-256 cells). Also left out are herd (the pushed object is a cheese in the stores'
words), subsets ("the group on the left: ..."), the eat and logic families, and follow and approach (they need a
cheese). Haiku texts exist for none of those that would fit.

Loose blocks (0-2 per scene) are obstacles. No pile is put on a block's column, and the walk never moves one: a
step that moves an object does not verify.

## Words

For each draw, the record's spec (canonical JSON) is looked up among the verified Haiku rewordings (close,
indirect, casual) of `data/vla/haiku/run-s*` and `run-s*-later` (the parts of switch episodes included; subset
rows with "who" left out). A rewording is used when it passes `text_ok`, which allows no colour or area word the
record does not name ("corner" only for corner regions, "side" and "edge" for named sides, "middle" for the center),
no number the spec does not hold (one, and two where the task's own words have it), no robot count ("all 38 of
you"), no coordinates ("the row with the smallest y"), no trench, plateau or hill unless the task is a bridge or
a climb, and none of the simulator's nouns (cheese, zone, stripe). Of the examples whose spec has kept texts, 75 %
use a Haiku rewording and 25 % the suite's template wording (`TEMPLATE_SHARE`).

A Haiku thought is rebuilt for the live scene: "you asked: <the rewording>. i see <think_data.sight of this
scene>. <Haiku's meaning sentences, if all pass text_ok with the scene words filter (towers, clumps, runs,
already, ...)>, else "that means: <plain>". <Haiku's first: line if it passes, else the record's>". Then
`think_data.visible` as for stored labels, and zone -> patch, stripe -> strip.

| family | specs drawn | Haiku texts with the same spec | kept by the filter | with cheese / zone / stripe | meaning sentences kept | first: line kept |
|---|---|---|---|---|---|---|
| goto | 9 | 730 | 511 | 19 | 71% | 100% |
| gather | 1 | 462 | 282 | 0 | 51% | 99% |
| groups | 2 | 448 | 347 | 0 | 63% | 99% |
| spread | 1 | 306 | 208 | 0 | 74% | 100% |
| line | 2 | 213 | 99 | 0 | 37% | 96% |
| move | 4 | 24 | 16 | 1 | 44% | 100% |
| stay | 1 | 999 | 676 | 0 | 72% | 100% |
| climb | 4 | 166 | 131 | 17 | 37% | 100% |
| split_regions | 72 | 1380 | 1254 | 33 | 81% | 100% |
| split_tower | 36 | 2150 | 2031 | 28 | 62% | 100% |
| tower | 6 | 5651 | 5196 | 12 | 51% | 100% |
| bridge | 1 | 1974 | 1635 | 1 | 69% | 100% |
| zone | 8 | 1951 | 461 | 1372 | 12% | 0% |
| zone_line | 8 | 2119 | 270 | 1587 | 13% | 1% |
| push | (no stores spec) | 0 | 0 | 0 | - | - |

The simulator's nouns: 1,372 of 1,951 zone texts (70 %) and 1,587 of 2,119 strip texts (75 %) say zone or stripe;
elsewhere at most 10 % (climb 17 of 166, goto 19 of 730). Those are dropped, so the plain worded ones that stay are
"the white area", "the red square", "la zona roja" (Spanish casual rungs keep their own words). Move has only 24
Haiku texts (16 kept) and leans on the template. Templates: the suite's training wordings; for zones, new ones
with "patch" and "strip" (the suite's say "zone" and "stripe").

## Throughput

`scripts/reverse_live_measure.py --per 24 --procs 4 --replay` (log `logs/reverse-live-measure.log`, numbers
`runs/reverse_live/measure.json`, not committed), seed 0, the stores' robot draw, one thread per process, nice 10, on
a Mac loaded by other jobs (load 12-20). CPU seconds count the walk, its checks and the draws that failed, not the
replay. All 360 episodes replay exactly under `world.step`. Measured before `MOVE_MOST` = 40 (move then drew up
to 115 robots; capped, it is cheaper).

| family | CPU s per episode | 6-20 robots | 21-50 | 51-115 | steps (mean) | draws per episode | why draws failed |
|---|---|---|---|---|---|---|---|
| goto | 3.80 | 0.9 | 2.0 | 6.0 | 129 | 1.17 | start: a third already in the region 4 |
| gather | 2.34 | 1.6 | 1.9 | 3.9 | 67 | 1.29 | end: not one group 6, start: already mostly together 1 |
| groups | 2.02 | 1.1 | 1.2 | 3.6 | 56 | 1.12 | end: not exactly that many groups 3 |
| spread | 3.43 | 1.7 | 5.7 | 19.4 | 111 | 1.04 | walk max_steps 1 |
| line | 0.31 | 0.4 | - | - | 23 | 1.08 | no end state 2 |
| move | 1.91 | 0.4 | 3.8 | 4.1 | 81 | 1.00 | none |
| stay | 0.01 | 0.0 | 0.0 | 0.0 | 4 | 1.00 | none |
| climb | 5.25 | 3.6 | 3.5 | 8.7 | 184 | 1.12 | walk stalled 3 |
| split_regions | 3.30 | 1.0 | 2.9 | 4.8 | 81 | 1.75 | end: not two groups 9, start: already in the regions 9 |
| split_tower | 1.66 | 0.8 | 1.7 | 3.7 | 71 | 1.08 | end: the others not all in the region 2 |
| tower | 1.16 | 0.5 | 1.2 | 2.2 | 52 | 1.04 | walk stalled 1 |
| bridge | 2.72 | 0.6 | 1.4 | 4.3 | 88 | 1.00 | none |
| zone | 0.62 | 0.6 | 0.9 | - | 46 | 1.00 | none |
| zone_line | 0.27 | 0.4 | - | - | 19 | 1.12 | start: a fifth already on it 3 |
| push | 0.14 | 0.1 | - | - | 11 | 1.00 | none |

All families at equal weights: 1.93 CPU s per episode, 0.52 episodes per CPU second per
worker; an episode is 39 kB packed. The slowest single episodes: spread at 51 robots 19 s
(guided walk into piles, 682 steps), goto at 113 robots 10 s, climb at 80-110 robots 9 s on average.

Does it keep up:
1. Mac trainer (~0.3 examples per second, `logs/think-08b-cam.out`): one worker makes about 0.5 episodes per second,
   more than the whole trainer eats even at share 1.0. `workers=0` (generation in the prefetch thread) also keeps up
   at share 0.2 with reuse 8 (0.06 live examples per second, one new episode per 8 of them: a few CPU seconds a
   minute). Recommended on the Mac: `workers=1` or 2, so the prefetch thread never waits.
1. Kaggle T4 pair (10-20 examples per second): 4 workers give about 2 episodes per second (less on Kaggle's 4 vCPUs,
   which also render the scene views). At share 0.2 (2-4 live examples per second) each episode is used 1-2 times;
   at share 1.0, 5-10 times (different steps, words and cameras). No family is too slow for that; the cache
   (`cache=64`, the latest episodes, drawn uniformly) is the pre-generated buffer refreshed in the background, so the
   trainer never waits after the first episode. If more freshness is wanted on Kaggle, pre-generate on the Mac
   (`generate` is a plain function of a seed; pickle the rows) and load them as a cache to start from; not built.

## Render check

45 GIFs, 3 per family (robot bins 6-20, 21-50, 51-115 where the family allows), seed 7, in the browse page
section "Basic tasks made live by reversal" (`live-<family>-<k>.gif`, `cards-live.json`; caption under the picture:
instruction, first thought, robots, steps; two grey isometric views). All 45 replay. Looked at the first and last
frames of most of them:

1. Obvious at a glance: goto (a pile in the named corner or side, robots stream in), gather, groups (2 or 3 clearly
   separate piles), spread (piles open out into a lattice with gaps, hills included), line, stay, climb (everyone
   ends piled on the named plateau, from the trench and the other plateau), split_regions, split_tower (a pile in
   the region and a lone column), tower (a single column rises among robots lying flat), zone and zone_line, push
   (the object ends on the patch).
1. Readable but not instant: move. At 25-40 robots the cluster visibly shifts three cells; above about 60 robots
   (first render: 96 and 107) the swarm covers most of the floor and the shift could not be told from the start, so
   move is now capped at 40 robots (`MOVE_MOST`). The guided walk fills A's cells by rank, not by identity: some
   robots stay and others hop past them, so it is the formation that moves three cells, not each robot.
1. Bridge at 78-96 robots: the crossing is clear (the only robots in the trench) but the plateaus are crowded with
   the other robots; at 20-46 robots it reads at once.
1. A zone pile of two layers covers its patch at the end, so the colour is only seen in the earlier frames.
1. Words: Haiku rewordings include Spanish and French casual rungs ("vayan al fondo del piso amigos", "fais un
   pont sur la tranchée stp"); they pass when their area words match the spec (FOREIGN_AREAS: a first render had
   "ve a la zona cian pls la de adelante izq", a place from Haiku's own scene; now refused). Some kept Haiku meaning
   sentences are generic but true ("three steps, all to the right"); scene-specific ones are replaced by "that means:
   <plain>".

## Open questions

1. Weights: families are equal by default (stay and push are trivially short; push has 2-6 robots and only
   template words). Set `weights=` once the trainer's mix is decided.
1. Subsets, herd, approach and follow are left out (their words need a cheese or a group selector); subsets ("the
   group on the left: go to the back; everyone else stay") would fit this design (guided walk where the others
   hold) if wanted next.
1. Validation uses `held=16` fixed episodes generated from `held_seed` at the first validation call (15-60 s
   inside the trainer, or on the workers).
1. Each DDP rank that builds a ThinkData with `live` starts its own workers; on a Kaggle T4 pair that is 2 x
   `workers` processes on 4 vCPUs: use `workers=2` per rank there.
1. Stand-in views (`think_data.stand_in_views`) do not draw floor zones; zone and zone_line need the scene views
   (`think_train --views scene`), which do.
1. reverse.py was not changed (another agent is editing it). Nothing here needs a change there; a cheaper walk
   for guided piles (spread at 50+ robots takes 5-20 s) would help most.
