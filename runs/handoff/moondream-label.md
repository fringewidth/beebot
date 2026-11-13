# moondream-label: does a rendered goal still look like its text

Written 2026-11-13. Code: `scripts/goals_label.py` (stub, local, api backends; yes/no and multiple choice, wrong-text
controls, stratified sampling, resumable, progress lines), test `tests/test_goals_label.py` (4, stub backend).
Output: `data/goal_library/labels_<source>.jsonl` (gitignored), `calibration_scale9.jsonl`, `calibration_scale18.jsonl`,
progress `logs/moondream-label.out`.

## Setup

1. Model `vikhyatk/moondream2`, revision `5d6c926f44e26b07957b0dd315bbedcb4c17a5fe` (2026-09-23, "Support
   transformers 5"), pinned in the script (`REVISION`). The newest revision already works with the venv's
   transformers 5.22.1 and torch 2.14.1, so nothing was pinned back and neither was upgraded.
1. Packages added with `uv pip install --python .venv/bin/python`: einops, pyvips, pyvips-binary 8.16.0, pyvips 2.2.3,
   torchvision 0.29.2 (the model code imports it; torch stayed 2.14.1).
1. fp16 on mps, no `PYTORCH_ENABLE_MPS_FALLBACK`, `nice -n 5`. Weights 3.6 GB in the HF cache. Image encoded once per
   goal (`encode_image`) and reused for the caption and every question.
1. Render: `goal_library.render_views` at scale 9 (two isometric views side by side, robot-grey, no colour).

## Speed (20 goals, Mac otherwise loaded)

Caption 0.73 s, yes/no question 0.17 s, 4-way multiple choice 0.15 s, peak mps memory 5.1 GB. Whole goal (render,
caption, own + control yes/no, multiple choice) 1.5-2.4 s in the labelling run (the Mac was shared with other jobs).

## Calibration (280 goals, 40 per source, scale 9)

| | own text | control (wrong text) |
|---|---|---|
| yes rate, all 7 sources | 0.63 | 0.51 |
| yes rate, descriptive sources (stabletext2brick, gen_medium, gen_simple) | 0.71 | 0.31 |
| multiple choice accuracy (chance 0.25), all / descriptive | 0.38 / 0.50 | |

The yes/no question is biased toward yes but not hopeless: on descriptive texts it separates (0.71 vs 0.31). It does
not separate where the text is not a shape description (hack337 names, CraftAssist part labels such as "roof",
CraftAssist house names: control yes rate equals or exceeds own), and the IGLU instructions talk about colours and compass
directions that the grey render cannot show. The wrong-text rate is above 0.3 almost everywhere, so per-goal answers
are noisy. Multiple choice is about as good as yes/no on descriptive sources (0.50 vs 0.25) and at chance
elsewhere; both were recorded for every goal (`--method both`), neither clearly wins.
Larger render (scale 18, 120 goals on the three descriptive sources): own 0.83, control 0.41, MC 0.49, against
0.71 / 0.31 / 0.50 at scale 9 on the same goals. No gain, render kept at 9.

## Labelling (stratified sample, up to 500 per source over cell-count bins and generators, both methods)

| source | goals | own yes | control yes | own yes and control no | MC accuracy |
|---|---|---|---|---|---|
| gen_medium | 500 | 0.78 | 0.31 | 0.53 | 0.54 |
| gen_simple | 500 | 0.81 | 0.49 | 0.42 | 0.54 |
| gen_complex | 500 | 0.82 | 0.58 | 0.35 | 0.38 |
| stabletext2brick | 500 | 0.61 | 0.41 | 0.35 | 0.47 |
| iglu_single | 500 | 0.48 | 0.40 | 0.28 | 0.27 |
| iglu_multiturn | 353 (all) | 0.42 | 0.35 | 0.26 | 0.24 |
| hack337 | 498 | 0.53 | 0.49 | 0.26 | 0.30 |
| craftassist_parts | 500 | 0.52 | 0.62 | 0.16 | 0.29 |
| craftassist_houses | 500 | 0.76 | 0.74 | 0.06 | 0.25 |

Overall (4351 goals): `goal_looks_like_rate` 0.644 own yes, control yes 0.492, MC accuracy 0.367. The own-yes rate on its own
is not a measure of fidelity (control yes is 0.31-0.74); "own yes and control no" and MC accuracy above 0.25 are the
usable signals. Only gen_medium, gen_simple, stabletext2brick (and weakly gen_complex) carry one.

## My check of 20 examples (scale 12 renders, mixed yes/no answers, descriptive sources plus iglu_single)

1. 5 of 20 cannot be judged from a grey render (IGLU texts about colours and compass corners): excluded.
1. 2 of the rest are ambiguous (counts, a 3x3x4 brick that looks like a cube).
1. Of 13 clear cases Moondream agrees with me on 9. Where it said yes it was right 7 of 8 (table, cross, column, bench,
   pyramid, 3x3 checker posts, jar-ish tower; the "tall bottle" was a stretch). Where it said no it was right only 2 of 5:
   it said no to a clearly correct arrow sign on a pole, a birdhouse with an overhanging roof and a short two-high wall.
   So yes is mostly trustworthy, no is unreliable (too many false rejects), which fits the 0.7-0.8 own-yes ceiling.
1. Captions are generic ("two identical black cubes ... symmetrical composition") and carry little: the two views read
   as two objects. A single view might help.

## Cost of the whole library

About 300k goals at 1.5-2.4 s with both methods is 125-200 GPU hours (about 80-100 hours with yes/no only; rotations
of one item sharing a label would roughly halve it again). Not worth it: the answers only separate good from bad on
the short descriptive sources (generators, StableText2Brick), and there the generators are correct by construction and
the 500-goal samples already give the rates. What would help instead: label from structure (generator parameters) and
use Moondream only as a spot check on StableText2Brick, or a stronger model (Haiku with the render) for the
sources where text and shape agree loosely.

## GPU use

12:40 to 15:17 on 2026-11-13 (speed 1 min, calibration 11 + 6 min, labelling 2 h 18 min, hack337 stopped at the 2.3 h
deadline with 498 of 500 done). All mps processes exited at 15:17.
