# reverse-v4: big edits by taking apart and building again, dense natural starts

Written 2026-11-13. Follows `runs/handoff/reverse-v3.md` on its two open gaps: guided edits and retargets above 115
robots, and natural starts at 500-800 robots. Still not read by training (`reverse_live.py` calls `take_apart` for
its basic tasks of 6-115 robots; nothing below changes for those but two swarm options that stay off for them).

Code: `src/beebot/vla/voxel/reverse.py`; tests `tests/test_vla_voxel_reverse.py` (38 passed: the 34 of v3 and 4 new);
measurement `scripts/reverse_v3_measure.py` (new `--libraries all`: every library in `data/goal_library_v2`); GIFs
`scripts/reverse_v3_browse.py` (new `--min-robots`).

## What changed

1. The swarm runs forward in time too (`_swarm(..., forward=True)`). A single pivot runs both ways through the same
   cells (support and swing cells are the same set), so the breadth-first walks of `_reverse_path` serve both
   directions; every joint step is verified from the earlier state with the same `verify` (rule, at rest,
   world.step). Played forward that is a take-apart, top down, many robots at once.
1. Guided goals above `REBUILD_ROBOTS` = 115 robots go through `_rebuild`:
   1. K = the robots A and B have in common that hold themselves up (`_kept`: supported with nothing else). K never
      moves.
   1. B side, reverse time: B's other robots walk out to a heap (the natural swarm, `_drop_rank` against A or B:
      never under either, a heap up to `heap_cap + 2`). Forward: B's changed part is built from the heap, bottom up.
   1. A side, forward time: A's other robots walk out of A to the same heap cells, in the order B's robots arrived
      there (`_arrivals`, `_meet_rank`: an order that walking in is known to manage). Forward: A's changed part
      comes apart onto the heap, top down. Robots that ended off the heap walk on to its cells still empty.
   1. Where the two heaps still differ, a join walk (`take_apart(..., rebuild=False)`, the v3 fill, at most
      64 + 16 steps per robot off its place) joins them.
   1. Robots that cannot get out shrink K round them (`_unkeep`: the kept robots in the columns within r of the
      stuck ones, from their level up, then whatever no longer holds itself up; r = 1, 1, 1, 2, 2, 2, ...): the
      dig is local and only as big as needed. Both sides walk again.
   1. The episode, forward: A, dismantle steps, join steps, extract steps, B. Kinds `dismantle`, `join`,
      `extract`. When `_rebuild` cannot finish, the episode ends `stalled` with no steps (v3's fill and digs, the
      old fallback, reached none of the 8 such goals in two measurements and spent up to 15 CPU minutes).
1. Dense natural goals (`heap_cap > HEAP`: more than 512 robots on 16 x 16) plan their start first:
   1. `_level_plan(..., unbiased=True)`: 64 natural scenes drawn (`natural_scenes`, as many robots, same terrain);
      of those inside the band and filling no more of the goal than the median draw, the one closest to the goal
      (fewest robots to move). Its robots per column are the plan.
   1. `_plan_movers`: a goal robot below its column's planned count, on a column of goal robots down to the ground,
      stays (it is already where the natural start has a robot); the others move. One over a gap moves: the cells
      under a table top at z = 1 are a gap no cube can enter.
   1. `_plan_rank`: robots walk to planned cells, the lowest layer first (the carpet grows a layer at a time, so
      walkers cross a level surface), never under, beside or on a robot still to come out (the carpet closed in
      the goal's lower layers when it grew beside them), and in a layer a cell whose filling would close in a
      neighbour still to fill half a layer later, the most enclosed first (`_profile_rank`). A cube enters a cell
      only from a free cell beside it on the same layer (a straight pivot leans on the two cells below, a diagonal
      one sweeps the cell beyond), so a 1-wide pit can neither be filled nor left; natural profiles are full of
      them, and filling round a pit first loses it.
   1. Once the goal is clear, the levelling swarm (`_level_movers`, `_level_rank`) walks the rest to the plan's
      cells still empty, until the start is in band; then v3's spread and greedy walk as before.
1. Two swarm options, off by default (v3 behaviour), on for the rebuild and the dense natural swarms:
   1. `first`: walkers whose first move verifies on its own are recruited first; one that does not (it holds up a
      robot hanging below it, which dense take-aparts leave) only in a round that recruits no other. Without it
      the highest robots, all holding something up, kept every walker slot and the swarm went idle.
   1. `patient`: a round in which nobody walks and nobody could be recruited no longer ends the swarm while robots
      are left; only 12 idle rounds do. On everywhere (an intermediate run) it cost one 72-robot natural start
      (stabletext2brick:26876), so it is on only for the rebuild (both sides and the join) and the dense natural
      swarms. Its effect on the rebuild alone was not isolated: the run with it off also capped the join walk, and
      the cap was what lost three rebuilds.

## Numbers

`scripts/reverse_v3_measure.py --libraries all`, seed 0, 5 goals per source and size bin, the same draw before and
after: gen_complex by kind in turn (its draw is the one of reverse-v3.md), gen_medium (no goal above 115 robots), and
v2 = every library in `data/goal_library_v2` pooled (the six of v3 plus itch_vox, oga_vox, objaverse_cap3d and
text2shape; the pooled draw picked stabletext2brick, text2shape, objaverse_cap3d, craftassist_parts and
iglu_multiturn). Before = reverse.py at HEAD 89a0f9f (v3), run from a copy of that tree; after = this code. CPU only, one
thread per process, nice 10, at most 6 processes; a 15-minute CPU budget per walk (none hit it). The after rows come from
one run of the final code (`runs/reverse_v4/final`), except the 8 guided goals above 115 robots, run again after the
join walk's step cap was removed (`runs/reverse_v4/guided`; no other goal reaches `_rebuild`); `runs/reverse_v4/merged`
has the table's rows (not committed), logs `logs/reverse-v4-*.log`, jobs `runs/reverse_v4/jobs_*.sh`.

ok = replay passed and the start is what was asked (natural: all six statistics in band; guided: exactly A). Steps
and CPU are means over the episodes (a stalled rebuild counts with 0 steps and replays trivially); robots moving per
step is the mean over episodes of more than 4 steps.

| robots | code | ok | natural ok | guided ok | replay | steps | CPU s walk | CPU s replay | robots moving per step |
|---|---|---|---|---|---|---|---|---|---|
| 4-115 | before | 14 / 15 | 11 / 12 | 3 / 3 | 15 / 15 | 151 | 3.0 | 3.4 | 4.09 |
| 4-115 | after | 14 / 15 | 11 / 12 | 3 / 3 | 15 / 15 | 151 | 2.6 | 2.8 | 4.09 |
| 116-250 | before | 5 / 10 | 5 / 8 | 0 / 2 | 10 / 10 | 868 | 49.3 | 105.4 | 5.17 |
| 116-250 | after | 7 / 10 | 5 / 8 | 2 / 2 | 10 / 10 | 521 | 26.3 | 77.0 | 5.37 |
| 251-500 | before | 6 / 10 | 5 / 7 | 1 / 3 | 10 / 10 | 1,280 | 98.2 | 122.8 | 8.30 |
| 251-500 | after | 8 / 10 | 5 / 7 | 3 / 3 | 10 / 10 | 449 | 23.5 | 30.2 | 8.40 |
| 501-800 | before | 0 / 10 | 0 / 6 | 0 / 4 | 10 / 10 | 2,429 | 241.4 | 436.9 | 8.25 |
| 501-800 | after | 5 / 10 | 3 / 6 | 2 / 4 | 10 / 10 | 1,130 | 304.4 | 177.6 | 11.85 |

By source (ok, before -> after, bins 4-115 / 116-250 / 251-500 / 501-800): gen_complex 5/5 -> 5/5, 2/5 -> 4/5, 3/5 ->
5/5, 0/5 -> 2/5; gen_medium 5/5 -> 5/5; v2 4/5 -> 4/5, 3/5 -> 3/5, 3/5 -> 3/5, 0/5 -> 3/5. Below 116 robots and for
natural goals up to 500 nothing changed but noise (same rows ok and missed).

1. Guided goals above 115 robots: 1 / 9 -> 7 / 9 (the 1 is a hold). Reached: edit_turn 134 (540 steps, 9 CPU s),
   retarget_part 131 (350, 6 s), edit_house 371 (562, 13 s), retarget_part 272 (987, 36 s), retarget_part 569
   (1,197, 80 s), edit_turn 597 (1,048, 58 s). Robots moving per step 4.9-8.5 (v3 on the same goals: 2.8-9.1, most
   below 5.5, with 900-6,000 steps).
1. Natural goals at 501-800 robots: 0 / 6 -> 3 / 6 (stabletext2brick 545, objaverse_cap3d 568 and 769); 8-16 robots
   move per step; 145 CPU s per reached start on average.
1. Every episode replays exactly under world.step, before and after (45 / 45).
1. Time: the two edit_undo goals still spend 735-786 CPU s before `_rebuild` gives up (under the 900 s budget, but
   most of an episode's worth); every other episode above 115 robots walks in 2-450 CPU s.

## What natural scenes look like at 500-800 robots

`natural_scenes` (the reference for the bands) drops robots in random columns, stacked, at most `heap_cap` =
max(3, ceil(1.5 n / 256)) above the floor: 3 up to 512 robots, 4 at 513-682, 5 at 683-800. `sample_scenes` itself
only draws 51-115 robots and caps at 3, so the bands at these counts are an extrapolation of the same rule. Means
of 32 scenes on the flat floor:

| robots | cap | empty columns (of 256) | columns 0/1/2/3/4/5 high | lifted | face pairs per robot | highest |
|---|---|---|---|---|---|---|
| 300 | 3 | 75 | 29/37/22/13 % | 0.40 | 1.40 | 2 |
| 500 | 3 | 27 | 10/23/27/39 % | 0.54 | 1.88 | 2 |
| 600 | 4 | 21 | 8/21/25/22/24 % | 0.61 | 1.91 | 3 |
| 660 | 4 | 15 | 6/16/23/22/32 % | 0.64 | 2.00 | 3 |
| 700 | 5 | 15 | 6/16/24/23/16/16 % | 0.66 | 1.96 | 4 |
| 800 | 5 | 9 | 3/12/20/23/18/24 % | 0.69 | 2.07 | 4 |

So the honest start at 500-800 robots is a carpet two to four (five) high over nearly the whole floor, the goal's
footprint included, with 9-27 empty columns. The band on "lifted" is tight at these counts (at 660: 0.64, i.e. at
most about 18 empty columns), and "height" allows nothing above cap - 1. An evenly levelled carpet is not natural
either: it has more face pairs than random columns (about 2.35 per robot at 660 against a band of 2.03), which is
why the plan copies a drawn scene's columns instead of filling columns evenly.

## What still fails

1. edit_undo of a scale-3 house, 624 robots, 54 changing (2 goals): both sides walk out, but A's robots never fill
   all of B's heap cells and the join walk (v3's fill) stalls on the last 15-40 robots; each round of digs re-walks
   A's side (50 s each at this size). Keeping the heap a cell away from A and B made it worse (B's side ran out of
   room). Next: a shared flat heap (cells neither side can close in), or iterate both sides on the cells that differ.
1. Natural starts at 501-800 robots, 3 of 6 miss, all on "lifted" and "height" (too many empty columns, a few
   robots above heap_cap - 1): trick_negation 621 (1,160 steps), text2shape 616 (2,792 steps, 677 of them
   scaffolding) and text2shape 800 (2,058). Not diagnosed one by one; the mechanism seen while developing on
   stabletext2brick 545: a plan cell stays empty when parallel walkers fill all its free sides first (the closing
   check sees one walker's reservation at a time), and the levelling swarm then finds no way in; robots left high
   after scaffolding have no path down over a dense carpet.
1. Natural goals of 116-500 robots, unchanged from v3: trick_negation 160 (height, fill, spread), text2shape 217
   (height, fill), objaverse_cap3d 404 (height). Three library goals (text2shape:11708, 83 robots;
   stabletext2brick:16656, 161; text2shape:40243, 269) stall within 2-4 CPU s before and after with every statistic
   out of band: they are at rest (`fixed_point`), but the extraction swarm makes no step at all (40243 is 16 high,
   the whole grid); not investigated.
Tests: `tests/test_vla_voxel_reverse.py` 38 passed; full suite 895 passed (one earlier run with -x stopped on
`tests/test_randomize.py`, another agent's uncommitted work at the time, which passes on its own with this reverse.py
and with HEAD's).

## Next

1. A rebuild that cannot join: walk B's heap and A's heap to a shared, flat set of cells instead of B's own heap,
   or iterate the two sides on the cells that differ.
1. Replay is now the larger part of the time for big episodes (world.step with the full rule table); the
   vectorised replay check of reverse-v3.md "Next" still applies.
1. Find out why the three library goals above make no extraction step (path depth, the top of the grid?).
1. Callers building data: guided goals above 115 robots that `_rebuild` cannot finish come back `stalled` with no
   steps; drop them.
