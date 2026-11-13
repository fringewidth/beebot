# goal-library: end states from public block datasets

Written 2026-11-13. Code: `src/beebot/vla/voxel/goal_library.py` (parsing, re-voxelising, acceptance, storage,
`to_goal`, `render_views`), `scripts/goals_convert.py` (source -> library), `scripts/goals_reverse.py` (reversal on
a sample), `scripts/goals_render.py` (browse sample), `scripts/goals_label.py` (Moondream labelling, written, not
run), tests `tests/test_goal_library.py` (14). Library `data/goal_library/` and raw downloads
`data/goal_sources/` are gitignored. Progress and summaries: `logs/goal-library.out` (one JSON line each).

## Method

1. Parse each source to a boolean occupancy (z up, y, x), plus text and labels. Items: whole structures, labelled
   parts (CraftAssist), 2 random crops per structure (Hack337 and CraftAssist houses; crops rest on the
   structure's bottom layer 70% of the time).
1. Re-voxelise each item at scales 2^(-k/2) for k from -3 to 12 (box average, filled when >= 30% of the source
   voxels under a coarse cell are; upsampling is nearest). A variant is accepted when it fits 16 per axis, has
   4-115 cells and every cell is face-connected to z = 0 through cells of the structure (`grounded`; tested equal to
   `world.supported` on random 4^3 fills). Per item the accepted scales nearest to 100, 45 and 15 cells are kept,
   each in the original orientation and one random quarter turn about the vertical axis (skipped when identical).
   Exact duplicates (same kind, label, text, shape) are dropped.
1. Stored per source as `<source>.npz` (`cells` (N,3) uint8 z,y,x; `starts`) and `<source>.jsonl` (id, source,
   source_id, item, kind, scale, rot, dims, cells, text, label, extra). `goal_library.load(name)`, `to_goal(cells,
   text)` (a `reverse.Goal`, family "library", centred in y, x).
1. Reversal: `scripts/goals_reverse.py` stratifies a sample over cell-count bins (4-7, 8-15, 16-31, 32-63,
   64-115; equal per bin, so it is NOT the library's own cell distribution), runs `reverse.take_apart` (96 steps)
   and `reverse.replay`. Results in `data/goal_library/reverse_<source>.jsonl` (id, cells, fixed_point, end,
   natural, steps, replay). "ok" = replay passed and the start is natural. Not used to filter.

## Numbers (raw size is on disk under data/goal_sources)

| source (library file) | raw size | structures | candidates (whole/part/crop) | with >= 1 goal | accepted | duplicates dropped |
|---|---|---|---|---|---|---|
| StableText2Brick (stabletext2brick) | 42 MB parquet | 47,389 | 47,389 | 46,724 | 158,840 | 60,624 |
| IGLU multiturn (iglu_multiturn) | 71 MB (zip + extracted) | 142 sessions | 129 | 108 | 353 | 7 |
| IGLU singleturn targets (iglu_single) | 1.0 GB repo clone (731 MB targets) | 5,762 | 5,593 | 4,984 | 17,896 | 180 |
| CraftAssist houses (craftassist_houses) | 1.1 GB house_data | 2,537 | 7,364 (2,537 / 0 / 4,827) | 6,693 | 25,478 | 3,616 |
| CraftAssist parts (craftassist_parts) | 59 MB instance segmentation + 110 MB repo | 83,787 annotation instances | 39,673 | 39,615 | 76,874 | 85,465 |
| Hack337 Minecraft-Schematics (hack337) | 81 MB | 669 | 1,895 (669 / 0 / 1,226) | 1,802 | 7,957 | 23 |

Total accepted 287,398 goals; library 135 MB on disk (the jsonl text dominates). Cell-count histogram per source
is in the `candidates_with_goal` summary lines of `logs/goal-library.out` (`cells_hist`, bins 4-7, 8-15, 16-23,
24-31, 32-47, 48-63, 64-79, 80-99, 100-115). Reject tallies per scale tried are there too (oversize, small, big,
ungrounded). Ungrounded rejects are a minority of the rejects (e.g. StableText2Brick 11,962 of ~600k rejected scale tries).

Reversal today (sample size, ok share, ok by cell bin 4-15 / 16-31 / 32-63 / 64-115):

| source | sample | ok | 4-15 | 16-31 | 32-63 | 64-115 |
|---|---|---|---|---|---|---|
| stabletext2brick | 30 | 0.67 | 12/12 | 6/6 | 2/6 | 0/6 |
| hack337 | 20 | 0.60 | 8/8 | 4/4 | 0/4 | 0/4 |
| craftassist_houses | 20 | 0.65 | 8/8 | 4/4 | 1/4 | 0/4 |
| craftassist_parts | 20 | 0.45 | 8/8 | 1/4 | 0/4 | 0/4 |
| iglu_multiturn | 60 | 0.25 | 15/24 | 0/12 | 0/12 | 0/12 |
| iglu_single | 60 | 0.30 | 17/24 | 1/12 | 0/12 | 0/12 |

1. Every sampled goal is a fixed point of the forward step (magnets hold the overhangs) and every episode replays
   exactly. What fails is the walk: it stalls or reaches 96 steps before the start looks natural. Size is the
   driver: ok is near 1.0 up to 31 cells (solid blobs, IGLU is the exception: thin 3D lattices of coloured
   blocks and single-block-wide arms), falls at 32-63 and is 0 from 64 cells in every source (the 96 step cap
   and the natural-start thresholds bind; "max_steps" ends are the majority there).
1. Sample sizes are small (the reversal costs 10-50 s per goal with the machine loaded, load average 100); the
   shares are indicative. Re-run with `scripts/goals_reverse.py --per-source N` once the other agent's overhang
   fix is in.

## Sources: notes and what was skipped

1. StableText2Brick: bricks "h x w (x,y,z)" fill h cells in x and w in y at height z (20^3). Captions: 5 per
   structure; the first is the goal text, the others stay in the parquet (`extra.captions` counts them), label =
   ShapeNet category id (synset). Licence MIT; derived from ShapeNet objects, whose own terms apply to the
   underlying shapes (note for any release).
1. IGLU multiturn (microsoft/iglu-datasets, MIT): the zip is a git-lfs object, fetched from
   media.githubusercontent.com (4 KB pointer in the clone). One goal per session: the world state after its last
   step; text = the architect's instructions joined by " | ". Colours (6) are dropped; `extra.colours` counts them.
   Intermediate steps are not used. IGLU singleturn: 5,762 target worlds (world after one instruction, built on a
   given initial world), text = the instruction from clarifying_questions_train.csv. The 6,109 `actionHit` game
   tapes are not used. Targets contain the initial structure too, so many are near-duplicates with different
   instructions.
1. CraftAssist (facebookresearch/craftassist, MIT; data from the official S3 bucket pubr/): house_data (2,537
   crowd-built houses: schematic, placement log) and instance_segmentation_data (1,845 houses with labelled
   parts, 46 labels such as wall, window, roof, door, stairs, fence; most instances repeat, hence the 85k
   duplicates dropped). `extra.placed` and `extra.broken` carry the count of placement events per house; the
   build ORDER is not stored in the library (it does not survive rescaling); it remains in
   `data/goal_sources/craftassist_s3/house_data/houses/*/placed.json`. The dialogue/command data (collected_dataset
   9 MB, dialogue_data 22 MB) are text-to-action pairs with no voxel output, so not converted; the 482 MB
   `datasets_folder` tarball held only templated command text and was deleted after inspection.
   Part labels: the label is the part name; a part counts as a goal after moving it to z = 0 (a lone roof
   floating over the house becomes a standalone roof on the floor).
1. Hack337/Minecraft-Schematics (MIT): the 669 parsed blueprints (JSON grids) are used; the 873 raw
   schematic/litematic files are not (same builds). Air and fluids are empty, everything else is a block. Text =
   title + description from schematics_mapping.json (English short query in `extra.short`). Builds go up to
   192 x 65 x 172, so most only survive at the coarse scales (many look like blobs); the 1,226 crops recover
   details.
1. No hand heuristics beyond the acceptance rule: terrain blocks (grass, dirt) are blocks like any other.
   Houses and builds that include ground layers show as slabs.

## Render sample and labelling

1. `scripts/goals_render.py --out <browse dir> --per-source 10` renders 60 goals (10 per source, spread across
   cell counts, preferring ones with text), two views each (usual corner and turned 180 degrees), as
   `goal-<source>-<k>.png` and `cards-goals.json`; `browse_page.py` in the session scratchpad has a "Goal library"
   section (heading id `goals`).
1. `scripts/goals_label.py <source> --backend local|api|stub [--device mps] [--limit N]`: for each goal renders
   the two views, asks Moondream 2 for a short caption and "Does this look like <text or label>? Answer yes or
   no.", writes `data/goal_library/labels_<source>.jsonl` (id, caption, answer, looks_like), resumable. Local
   uses `AutoModelForCausalLM.from_pretrained("vikhyatk/moondream2", trust_remote_code=True)` (pin a revision with
   `--revision`; `caption(image, length="short")` and `query(image, question)["answer"]`); api uses the
   `moondream` package with MOONDREAM_API_KEY. Neither model weights nor the package are installed; only the
   `stub` backend was run, to check the plumbing. The robot-grey isometric render gives the model little to
   read (no colour, no ground); judge yes-rates accordingly.

## Next

1. Reversal at scale with the overhang fix; larger sizes need a longer step cap and a natural-start threshold that
   scales with the structure (a 100-cell goal cannot be scattered in 96 steps).
1. Use the library: sample by text-bearing sources for language data, by cell count for a curriculum; goals with
   `kind` part or crop carry weaker text (labels, "part of a house").
1. Dependency note: no new packages (scipy was avoided; flood fill is numpy). pyarrow and pandas come with datasets.

## v2: bigger shapes

The 115-cell cap was a convention, not a limit. `MAX_CELLS` is now 800 and `KEEP_TARGETS` (100, 250, 500, 800), both
parameters (`goal_library.variants(..., keep=, max_cells=)`, `scripts/goals_convert.py --max-cells --keep-targets
--out`). Every other acceptance rule is unchanged. v2 lives in `data/goal_library_v2/` (the old library is untouched);
`scripts/goals_v2_chain.sh` rebuilt it from `data/goal_sources/` (about 5 minutes, 4 procs, CPU).

Voxeliser (`goal_library.voxelise`, `best_variant`, `repair`): per item and scale it searches fill thresholds
(0.15, 0.3, 0.5) and sampling-grid phases (0, 1/3, 2/3 of a coarse cell in y and x), scores each by the mean IoU of
the three axis silhouettes against the source's silhouettes at the same grid plus half the soft 3D IoU against the
source fill, and keeps the best that passes acceptance. A floating candidate is reconnected to the floor through the
cheapest chain of coarse cells that hold some source voxels (legs, posts, rails; hollow interiors are never filled).
Grounding is now judged on the uncropped result: the old rule judged the cropped shape, so a table whose legs had
dropped was cropped to a legless slab at z = 0 and passed. `variants(..., search=False)` is the old rule.
`scripts/voxeliser_eval.py` measures old against new and renders the comparison.

Old vs new, pairs of (item, scale < 1) accepted by both (StableText2Brick 200 items, 892 pairs; CraftAssist houses
50 items, 159 pairs). The IoU is the search objective, so its gain is partly by construction; component survival and
floor-contact survival are not optimised directly.

| | silhouette IoU old / new | soft 3D IoU old / new | components kept | floor contacts kept |
| --- | --- | --- | --- | --- |
| StableText2Brick | 0.920 / 0.990 | 0.627 / 0.627 | 1.00 / 1.00 | 0.861 / 0.964 |
| CraftAssist houses | 0.840 / 0.950 | 0.476 / 0.478 | 0.803 / 0.899 | 0.744 / 0.919 |

Most StableText2Brick sources are already about 18 voxels across, so the change is small for them; the gain is in
legs and thin parts, and in houses.

Counts (accepted goals; cell bins 4-115, 116-250, 251-500, 501-800):

| source | goals | 4-115 | 116-250 | 251-500 | 501-800 |
| --- | --- | --- | --- | --- | --- |
| stabletext2brick | 141,290 | 56,803 | 57,772 | 22,846 | 3,869 |
| craftassist_parts | 55,944 | 30,864 | 17,710 | 6,660 | 710 |
| craftassist_houses | 25,946 | 11,998 | 7,644 | 4,641 | 1,663 |
| iglu_single | 13,504 | 8,068 | 3,731 | 1,636 | 69 |
| hack337 | 8,001 | 3,005 | 2,466 | 1,762 | 768 |
| iglu_multiturn | 282 | 178 | 59 | 45 | 0 |

Reversal at the new sizes (`scripts/goals_reverse_big.py`, StableText2Brick, 3 goals per bin, step cap 2000, 15
minute limit, one thread): 0 of 9 ok. Eight stalled (replay passes, the start is not natural) after 72-1735 steps
and 28-737 CPU s; one 457-cell goal hit the 15 minute timeout. Take-apart does not reach a natural start at these
sizes and costs 5-12 CPU minutes per goal above 200 cells; it needs work (reverse.py was not touched).

## Cap: 256 robots (2026-11-13)

Goals (end states and starts) hold at most 256 robots (`goal_library.MAX_CELLS`, one per floor cell of the 16x16x16 world); the world stays 16^3. At 500-800 robots a natural start is a 2-5 high carpet over the whole floor (reverse-v4.md); 256 keeps about one layer on average while shapes still read. Default kept sizes are `KEEP_TARGETS = (64, 128, 256)`. `goal_library.load`, `goal_sources.load_rgb` and `goal_gen_complex.load` drop larger entries by default (`max_cells=None` keeps all), so data already written is filtered without regenerating; `goal_gen_complex.MAX_ROBOTS` and `machine_parts.PART_MAX_CELLS` follow the constant. `reverse.py` is unchanged and still handles larger goals.

## v3: sizes spread over 4-256 (2026-11-13)

`data/goal_library_v3/` (v2 untouched; `scripts/goal_library_v3.sh` rebuilds the dataset sources): kept size targets 8, 16, 32, 64, 128, 256 with `--tolerance 2` (a target is skipped when no accepted scale has a count within a factor 2 of it, so targets do not all repeat the nearest variant), search voxeliser, rotations as before. Sources: stabletext2brick, craftassist_parts, craftassist_houses, iglu_single, iglu_multiturn, hack337, oga_vox, itch_vox. Not redone: text2shape (raw zip `nrrd_256_filter_div_32_solid.zip` is gone from disk and its URL (kaldir.vc.in.tum.de) answers 404) and objaverse_cap3d (about 1.9 GB of meshes to re-stream); their v2 files stay as they are. Generators `gen_simple` and `gen_medium` draw sizes log-uniformly up to nine per dimension (texts state counts below ten, so sizes stay below ten): bigger houses, walls, staircases, boxes, sofas, beds, bookcases, benches; 20,000 goals each written to v3 (seed 0). `scripts/goals_v3_hist.py` prints the per-source and total histogram before and after, `scripts/goals_render_v3.py` renders 6 per bin. Small sizes (4-31) of houses, parts and meshes are often blobs that do not read as their text; the Moondream filter should drop those.
