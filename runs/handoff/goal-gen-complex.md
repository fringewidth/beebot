# goal-gen-complex: scenes, edits, structure to structure, tricks

Written 2026-11-13. The complex tier of the goal generators, beside goal_gen_simple (shapes) and goal_gen_medium
(everyday objects). Not read by training yet.

Code: `src/beebot/vla/voxel/goal_gen_complex.py`; tests `tests/test_goal_gen_complex.py` (38 passed).
Data (not committed, `data/` is ignored): `data/goal_library/gen_complex.{npz,jsonl}` (end states B, goal_library
layout) and `gen_complex_start.npz` (starts A, same layout, empty for take-apart goals). Logs:
`logs/goal-gen-complex-{write,measure,reverse-*}.out`.

    .venv/bin/python -m beebot.vla.voxel.goal_gen_complex --n 10000 --seed 0 --render-dir <dir>
    .venv/bin/python -m beebot.vla.voxel.goal_gen_complex --measure 1000 --seed 1
    .venv/bin/python -m beebot.vla.voxel.goal_gen_complex --reverse 3 --max-robots 115 --only <generators>

## Method

1. Parts are small programs with random parameters, built in a local frame facing the viewer (+y) and turned by
   quarter turns: tower, wall (optionally with a gate), house (door side, roof, second floor), table, chair,
   bench, arch, stepped pyramid, stairs, figure (two legs, hips, torso, head, arms out / up / down), platform,
   post with an arm. Words follow camera_frame.py at quarter turn 0 and reverse.SIDES: left is low x, right high x,
   back low y, front (nearest the viewer) high y.
1. A part can be scaled: every cell of the shape becomes an s x s x s block of robots (s = 2 or 3). Big builds
   read better (manager, 2026-11-13: end states up to 800 robots). Texts never count a scaled part; it is
   "a big chair", "a large house with a doorway".
1. Generators compose parts with relations the viewer checks at a glance: footprints separated by 1-3 empty
   columns along the relation's axis and overlapping across it, parts never touching (Chebyshev distance 2 or
   more) unless one stands on the other. Left/right of a part that has a front of its own is always worded in
   the viewer's frame ("to the left of the house as I look at it"); front/behind relations face such parts toward
   the viewer, so both readings agree.
1. Cells are stored in the world frame (not centred), so "in the back left corner" and "nearer me" stay true.
1. Every record: `cells` (B, every robot), `start_cells` (A or None), `turns` (user texts, oldest first;
   with a start, the last turn asks for B and the earlier ones built A), `reply`, `kind`, `params` (parts with
   their parameters, turns and anchors; parking; `changed` = cells that fill and empty; overhang; span; scale;
   trick behaviour).
1. Spare robots stand in parked rows along one edge (back, front, left or right), filled from the middle of the
   edge outward, up to three rows of 14. A's row and B's row are prefixes of the same slot order, so robots that
   both share never move. Turn 1 says where spares wait when A has any; when only B has them, the request says
   where the leftovers go. With no words (conditionals), A always has a row and leftovers join it.
1. Checks on every state (A and B): 4-800 robots, inside 16^3, every robot face-connected to the floor
   (goal_library.grounded), at rest (reverse.fixed_point), overhang at most 4 x scale and unsupported span at
   most 4 x scale. Edits and retargets: same robot count in A and B, A differs from B. Texts: no cheese,
   stripes, zones, voxels or cells; no count of ten or more (digits or words).
1. Overhang: sideways steps through the structure from the nearest ground pillar (a column filled down to the
   floor); up and down steps are free (a hanging arm's lever is its shoulder's). Span: longest straight row of
   cells with nothing directly under them. Both recorded per goal for the joint-strength filter.

## Generators

Scenes (start None: reverse takes B apart to a natural start):
1. `scene_pair`: X left of / right of / in front of / behind Y; one in four has Y built big.
1. `scene_row3`: three parts in a row, left to right or back to front.
1. `scene_facing`: two benches or chairs facing each other, back to back, side by side facing me or away;
   a table with chairs on two or four sides facing it.
1. `scene_wall_towers`: a wall (optionally with a gate) with a tower at each end or at a named end.
1. `scene_between`: a middle part with matching parts on both sides (towers, figures, pyramids, posts with arms
   pointing out).
1. `scene_on_top`: a tower, pyramid, figure, chair or post standing on a table, platform, house, arch or wall,
   in the middle or at an end.
1. `scene_compare`: towers or walls by height (taller, same, twice as tall, increasing, tallest in the middle).
1. `scene_around`: towers off the corners of a table, house, pyramid or platform (four, or two opposite).
1. `scene_corners`: parts in named corners of the floor or its middle.
1. `scene_big`: a walled yard with a gate and corner towers, a big house with towers at its sides, a street of
   small houses.

Edits (start A):
1. `edit_figure`: arms out / up / down, both or one ("raise the arm on the right and leave the one on the left as
   it is"), longer or shorter legs. Figures face the viewer (side-on ones did not read).
1. `edit_house`: roof off / on, door moved to another wall, door filled, door added, walls one higher, a second
   floor (with or without its own roof).
1. `edit_towers`: one taller / shorter by k, match another, all to the tallest / shortest, swap heights, remove,
   add one at an end, move an end tower out.
1. `edit_wall`: longer at a named end or both, shorter, higher, lower, gate cut / filled, tower added at an end.
1. `edit_table`: legs longer / shorter, table to chair (back on a named edge), chair back off / taller / shorter.
1. `edit_turn`: chair, bench, stairs or post turned (around, or a quarter turn; "turn the chair to face the
   tower"); a post swings its arm about its column.
1. `edit_scene_move`: in a pair, X to the other side, swap, closer (one-space gap), farther by k, new relation.
1. `edit_arch`: opening wider / narrower by moving a named post, taller / lower, beam off, two posts joined by a
   beam.
1. `edit_stairs`: one step more / fewer, wider / narrower on a named side.
1. `edit_undo`: three turns (build A, an edit, "no, undo that"): start is the edited state, end is A.

Structure to structure (start A):
1. `retarget_part`: any part into any other kind, in its place (robots on shared cells stay) or beside where it
   stands; sizes within a factor three.
1. `retarget_compose`: wall into a row of towers, towers into a wall, anything flattened into a platform, two
   towers into an arch, pyramid into stairs, tower into a wall.
1. `retarget_external`: one side from goal_gen_simple / goal_gen_medium (their text is the turn that builds A, or
   the request), the other from this module.

Tricks (12 % of the written set):
1. `trick_too_many`: more robots asked for than exist (4-8 robots): the closest possible (a tower or wall as long
   as there are robots, two towers of half) with a reply, or nothing at all when the text says so.
1. `trick_already`: request already satisfied (tower at least m high, house has a door, relation holds, arms
   already up, left already taller, the same thing asked again): end state = A, reply says so.
1. `trick_negation`: "anything except a tower" (any other part, reply names it), house without roof / door,
   wall without gaps, two unequal towers, a tower no taller than m, a table that is not square.
1. `trick_false_premise`: roof off a roofless house, a part that is not there, "the second tower" when there is
   one, "the red tower" (everyone is the same colour): end state = A, reply.
1. `trick_conditional`: "if there's a tower make it k taller, otherwise build one", "if it has a roof take it
   off", "if the towers differ make the shorter match": both branches drawn, reply says which.

## Numbers

Seed 1, 1,000 draws per generator, CPU, nice 10. Acceptance counts draws that pass every check including rest
(rejects are mostly layouts that do not fit or touch, parking that needs more than three rows, size ratio for
retargets). Distinct = distinct (A, B) cell pairs. Reverse: `reverse.take_apart` on 3 samples per generator with at
most 115 robots (seed 0): natural take-apart for scenes, the guided walk to A (`Goal.start`) for edits, retargets
and changing tricks; ok = every step replays and the start is natural (scenes) or exactly A (guided).

| generator | acceptance | distinct / accepted | distinct texts | robots min / mean / max | changed mean | overhang max | span max | reverse ok / n | steps | CPU s |
|---|---|---|---|---|---|---|---|---|---|---|
| scene_pair | 0.87 | 866 / 866 | 866 | 8 / 60.0 / 496 | - | 7 | 8 | 3 / 3 | 129, 213, 103 | 26.2 |
| scene_row3 | 0.73 | 729 / 729 | 729 | 12 / 42.7 / 80 | - | 4 | 4 | 3 / 3 | 64, 285, 183 | 42.0 |
| scene_facing | 0.92 | 877 / 921 | 280 | 32 / 116.2 / 416 | - | 3 | 6 | 2 / 3 | 211, 215, 119 | 48.3 |
| scene_wall_towers | 1.00 | 976 / 1000 | 458 | 6 / 17.6 / 36 | - | 1 | 2 | 3 / 3 | 20, 77, 41 | 4.2 |
| scene_between | 0.85 | 841 / 847 | 792 | 10 / 31.5 / 86 | - | 4 | 4 | 3 / 3 | 146, 128, 47 | 18.4 |
| scene_on_top | 0.89 | 882 / 890 | 884 | 4 / 25.4 / 58 | - | 4 | 4 | 3 / 3 | 77, 24, 25 | 3.7 |
| scene_compare | 1.00 | 990 / 1000 | 221 | 4 / 12.9 / 42 | - | 0 | 0 | 3 / 3 | 14, 20, 52 | 2.2 |
| scene_around | 1.00 | 968 / 1000 | 639 | 8 / 29.6 / 62 | - | 2 | 4 | 3 / 3 | 70, 41, 15 | 3.8 |
| scene_corners | 1.00 | 969 / 996 | 995 | 7 / 29.3 / 68 | - | 4 | 4 | 3 / 3 | 60, 48, 73 | 8.0 |
| scene_big | 0.97 | 957 / 971 | 573 | 30 / 69.0 / 127 | - | 2 | 3 | 3 / 3 | 337, 180, 317 | 93.1 |
| edit_figure | 0.82 | 822 / 825 | 825 | 18 / 51.5 / 171 | 7.2 | 7 | 6 | 1 / 3 | 40, 26, 0 | 4.6 |
| edit_house | 0.88 | 878 / 881 | 874 | 15 / 118.3 / 624 | 12.0 | 3 | 8 | 2 / 3 | 63, 0, 105 | 10.4 |
| edit_towers | 0.92 | 916 / 916 | 916 | 5 / 14.0 / 26 | 3.2 | 0 | 0 | 3 / 3 | 79, 26, 52 | 4.5 |
| edit_wall | 1.00 | 1000 / 1000 | 1000 | 4 / 14.0 / 31 | 3.6 | 1 | 2 | 2 / 3 | 51, 65, 58 | 6.3 |
| edit_table | 0.88 | 880 / 880 | 877 | 14 / 26.8 / 47 | 7.9 | 2 | 4 | 2 / 3 | 170, 23, 110 | 16.8 |
| edit_turn | 0.98 | 975 / 976 | 976 | 4 / 81.4 / 630 | 27.6 | 9 | 9 | 1 / 3 | 79, 163, 6 | 23.1 |
| edit_scene_move | 0.66 | 664 / 664 | 664 | 7 / 50.8 / 341 | 21.5 | 7 | 8 | 0 / 3 | 221, 199, 215 | 40.2 |
| edit_arch | 0.91 | 909 / 910 | 910 | 7 / 12.0 / 18 | 4.0 | 2 | 3 | 3 / 3 | 21, 25, 63 | 3.3 |
| edit_stairs | 0.92 | 920 / 920 | 917 | 6 / 28.2 / 66 | 9.6 | 0 | 0 | 2 / 3 | 73, 183, 26 | 10.6 |
| edit_undo | 0.88 | 880 / 880 | 880 | 5 / 40.1 / 624 | 8.1 | 7 | 8 | 3 / 3 | 16, 29, 21 | 2.4 |
| retarget_part | 0.45 | 453 / 453 | 453 | 6 / 40.0 / 648 | 31.9 | 7 | 12 | 2 / 3 | 229, 124, 81 | 43.6 |
| retarget_compose | 1.00 | 998 / 1000 | 998 | 4 / 17.2 / 48 | 10.5 | 4 | 4 | 3 / 3 | 24, 33, 48 | 2.6 |
| retarget_external | 0.54 | 538 / 538 | 538 | 6 / 28.0 / 72 | 21.6 | 4 | 4 | 2 / 3 | 42, 198, 76 | 14.7 |
| trick_too_many | 0.92 | 826 / 916 | 334 | 4 / 6.1 / 8 | 0 | 0 | 0 | 3 / 3 | 31, 18, 41 | 2.1 |
| trick_already | 0.95 | 910 / 955 | 927 | 4 / 36.1 / 409 | 0 | 7 | 8 | no-op (A = B) | - | - |
| trick_negation | 0.82 | 810 / 823 | 214 | 4 / 33.9 / 621 | - | 9 | 12 | 3 / 3 | 30, 43, 50 | 3.4 |
| trick_false_premise | 0.88 | 868 / 877 | 806 | 4 / 30.0 / 333 | 0 | 7 | 8 | no-op (A = B) | - | - |
| trick_conditional | 1.00 | 995 / 1000 | 996 | 5 / 24.0 / 74 | 4.0 | 4 | 4 | 3 / 3 | 65, 47, 55 | 4.9 |

1. Reverse ok over the 26 generators that change something: 64 / 78 (0.82). Scenes 29 / 30, edits 19 / 30,
   retargets 7 / 9, changing tricks 9 / 9. Every episode replays exactly (78 / 78); failures are walks that stall.
1. Retarget in place vs beside, same 8 draws (seed 11, before the beside variant existed): 1 / 8 reached A in
   place, 5 / 8 with B moved off A's footprint. `retarget_part` now draws both; 6 more draws: 3 / 6.
1. Large goals (one draw each, at least 150 robots, CPU seconds): edit_turn 434 robots (scale 3 chair) stalled
   after 255 steps, 49 s, 25 cells of A short; edit_figure 171 robots stalled at 124 steps, 33 s, 6 short;
   edit_house 242 robots stalled at 139 steps, 19 s, 4 short; scene_facing 256 robots (two big chairs) stalled at
   869 steps after 330 s; scene_pair 178 robots stalled at 1,024 steps after 390 s. None of the five large draws
   closed (every episode still replays). The retarget_part draw was stopped unfinished to bound the check.
1. Written set (`--n 10000 --seed 0`, 100 s): 10,000 goals, 9,969 distinct, kinds scene 3,533, edit 3,417,
   retarget 1,824, trick 1,226 (12.3 %). Robots per goal:

   | 4-7 | 8-15 | 16-23 | 24-31 | 32-47 | 48-63 | 64-79 | 80-99 | 100-115 | 116-250 | 251-500 | 501-800 |
   |---|---|---|---|---|---|---|---|---|---|---|---|
   | 506 | 2,154 | 2,184 | 1,434 | 1,776 | 826 | 262 | 181 | 38 | 428 | 177 | 34 |

1. Renders: 30 per write, A and B side by side (an isometric view, a view from the viewer, a view from above
   with the viewer at the bottom; cells that fill orange, cells that empty blue, parked robots grey). First write:
   24 of 30 read at a glance; the six that did not (2-wide tables and chairs that are solid blocks, a side-on
   figure, towers one or two high, a post on a table that looked like a chair back, a garbled "with ... with"
   sentence, towers round a flat figure) were fixed. Second write: 28 of 30; the two left were a 2-wide table
   (fixed: tables are at least 3 by 3) and a figure in front hiding the stairs behind it (a viewpoint matter:
   front/behind relations between a tall part and a low one). Four scaled samples (163-621 robots) read clearly.

Five multi-turn texts from the written set:
1. "build a stick figure with the arm on the left raised and the arm on the right down" / "Raise the arm on the
   right and leave the one on the left as it is"
1. "A column two tall at the back and a straight wall four long, three high in front of it, nearer to me" / "Now
   move the tower so it's to the left of the wall" / "No, undo that"
1. "a straight wall eight long, three high, and line the rest of you up along the front edge of the floor." /
   "Turn the wall into a row of four towers, each two high, with gaps between"
1. "A small house, with the spare ones parked in a row by the back edge of the floor." / "if the house has a roof,
   take it off." Reply: "It had a roof, so I took it off."
1. "Build a chair with a backrest." / "make the figure taller". Reply: "There's no figure here, only a chair."

## Gaps in reverse.py

1. No-op episodes. `take_apart` cannot return a zero-step episode: `finished()` needs at least one recorded step,
   and with nothing to move the guided walk wanders. Tricks whose right answer is "do nothing" (A = B; 48 % of
   the tricks) need a STAY-only demo built directly (frames [A] x T, every code STAY), not a reversal.
1. Robots on A and B's shared cells never move in the guided walk. That makes enclosed fills impossible: a
   doorway filled under a lintel or roof (`edit_house` stalls at 0 steps, distance 2: the robot in the doorway has
   walls on both sides and the roof above, so no pivot), a wall's ground cell under a robot that stays, a chair's
   seat under a shared cell. The fix is to let shared robots move away and come back (plan through an
   intermediate state that is a subset of both), or to fall back to take-apart-then-build when the guided walk
   stalls. Evidence: the same 8 retargets reached A 1 / 8 in place and 5 / 8 with B moved off A's footprint.
1. Destination order in the guided walk. `_extract(state, movers, starts & ~occ)` sends a robot to any empty
   cell of A. Filling A's upper cells first leaves its lower cells enclosed (a stalled retarget left three ground
   cells of a house wall empty with the robots meant for them still in the parked row). Choosing destinations
   lowest layer first, and among them the most enclosed first (the reverse of a top-down, outside-in take-apart),
   should close most of the 1-3 cell stalls (`edit_wall`, `edit_table`, `retarget_*` distances 1-3).
1. Thick (scaled) parts. Guided edits of scale 2-3 parts all stalled (3 / 3 large draws): a robot inside an s x s
   x s block has no pivot until its neighbours leave, and the shared-cell freeze keeps them. Natural take-apart of
   178- and 256-robot scenes also stalled (1,024 and 869 steps, 390 and 330 CPU s). Big goals need the two fixes above and probably
   layer-by-layer extraction of solids; until then, scaled goals are end states without verified demos.
1. Speed: 2.6 steps per CPU second at 178-256 robots (4.7 at 60-102 robots in reverse-v2). `world.overhangs` inside
   verification dominates (reverse-v2's note).

## Next

1. Fix the guided walk (shared robots may move; lowest, most enclosed destinations first) and rerun
   `--reverse 3`; edits and retargets should then approach the scenes' 29 / 30.
1. Build no-op demos (STAY frames) for the A = B tricks and pair them with their replies.
1. Haiku paraphrases of the turns (the templates give 200-1,000 distinct texts per generator; `scene_facing`,
   `scene_compare`, `trick_too_many` and `trick_negation` are the narrowest at 214-334).
1. Filter on overhang and span once the joint-strength study gives the limits (recorded per goal, scaled parts
   reach 9 and 12).
1. More readable large builds: scale more part kinds in scenes (houses with towers, yards) once reversal handles
   them.
