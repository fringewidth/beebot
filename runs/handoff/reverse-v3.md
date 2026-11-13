# reverse-v3: swarms, guided digs, holds, 4-800 robots

Written 2026-11-13. Upgrade of `runs/handoff/reverse-v2.md` for the gaps in `runs/handoff/goal-gen-complex.md` and the
robot cap lifted to 800. Still not read by training.

Code: `src/beebot/vla/voxel/reverse.py`; tests `tests/test_vla_voxel_reverse.py` (34 passed: the 27 of v2 and 7 new;
full suite 848 passed);
measurement `scripts/reverse_v3_measure.py`; GIFs for the browse page `scripts/reverse_v3_browse.py`.

## What changed

1. Verification is the same test, cheaper. `verify` asks `rule_ok` instead of `world.refuse`: the same answer
   (`world.refuse(S, a) == a`), but the overhang test (`world.overhangs`, one support flood per movable robot whose
   removal unsupports another, over the whole scene) runs for the moving robots only, and not at all for a mover with
   no robot on any face. Then `world.step` without the rule (identical when the rule refuses nothing), and S at rest.
   A test compares `rule_ok` with `world.refuse` on random states and legal and illegal codes. `replay` still uses
   `world.step` with the rule, and every number below counts an episode only if it replays.
1. Local pivot test at 14 cells. `sources` evaluated `world.pivot_moves` on the whole 7^3 crop (343 cells x 27
   codes) to read 14 entries; `_legal_into` gathers the same supports and swing cells for those 14 only. The v2 test
   that compares with `pivot_moves` on the full grid still passes. Together with the point above, one 416-robot scene
   went from 108 to 52 CPU s (profiled mid-way; later changes moved it again).
1. The swarm (`_swarm`) replaces extraction and its 4-walk pipeline. Every round, robots that must move and have a
   legal move are recruited highest first (up to 8 breadth-first walks a round), each at least 3 cells (Chebyshev)
   from every walker and its next cell, and each reserves its destination. All walkers take their next move; the
   joint step is verified; if it fails, the walkers that verify alone join a group at a time while the joint step
   still verifies (greedy bisection). A walker that cannot move for 2 rounds plans again from where it stands; after
   6 it gives up and is recruited again later (stranded). Played forward: robots stream in from where they stand and
   the structure rises bottom up, many at once.
1. Destinations by rank. `_reverse_path` takes a rank field instead of a mask: the reached cell of lowest rank,
   nearest first, looking a few moves past the first destination it finds.
   1. Natural (`_drop_rank`): 0 on the ground at least 3 from the goal touching no robot; 1 on a lone robot that far
      out; 2 on the ground off the goal and its faces; 3 heaped on anything up to `heap_cap` high. Never under the
      goal's cells (robots parked under a table top or a lintel closed in what was still to come out; that stalled
      big tables).
   1. Guided (`_fill_rank`): A's empty cells, the lowest layer only, in it the most enclosed first (faces closed by A
      or the floor), and a cell whose filling would close in an empty neighbour (a cube could pivot between it and a
      free cell before, not after; `_closing`) after the others. A cell that would close in a cell a walker is heading
      for is not offered until that walker is there (walkers in parallel filled a lintel's neighbour first and shut
      the lintel out). When the lowest layer cannot be reached at all, any empty cell of A (v2's order).
1. Digs (guided, `_dig_region`): when nothing moves (strict and relaxed fill, then scaffolding), the robots on A's
   cells in the columns within r of the stuck cells (robots that must move and have no move; else the lowest empty
   cells of A), from their level up, are taken out top down by the swarm to parking cells, then filled back bottom up
   (they are kept last for one fill). r grows 1, 2, 3, 5, the whole grid while digs do not bring the state nearer A,
   and starts at 1 again when they do. This is the take-apart-then-build fallback, local first. A doorway filled
   under a lintel (wall 3 or 4 high, door 1 or 2 high) now reaches A (test); before it stalled at 0 steps.
1. Spread (natural, after the goal is clear): v2 kept robots 3 cells from the goal while it came apart, so 150+
   robot starts were denser than natural heaps of the same count (crowd and face pairs above their bands) and the
   greedy walk, which charges +3 for standing on the goal's cells, never spread them back. Robots that put a
   statistic out of band (crowded, lifted, touching another) walk as a swarm to resting cells where they would be
   alone on the ground first, then touching, then on top of another, with fewer others within two columns than the
   crowd band, the goal's cells included up to the fill band. It stops as soon as the start is in band (checked
   every 4 rounds), and each robot walks at most once per spread (the movers are a set of cells: without that, a
   robot stopping where another had left walked again, 5,000 steps of churn at 600 robots). Then v2's greedy walk
   if still needed.
1. Natural mode also moves anything higher than a natural heap (heap_cap above the terrain): scaffold helpers stay
   where they held a part up (forward: they come from there), and at 500+ robots they were the robots that put
   "height" out of band. A 588-robot goal went from stalled (904 steps) to natural (392 steps, 31 CPU s).
1. Natural heaps for many robots: `natural_scenes` piles at most `heap_cap` = max(3, ceil(1.5 n / open columns))
   above the terrain (3 up to 512 robots on 16 x 16, 5 at 800); 3 high cannot hold 800 robots on 256 columns.
1. Holds: a guided goal with A = B gets an episode of 4 steps of every robot staying (`end == "hold"`, natural),
   built directly; it replays and lays out in the store like any other.
1. Scaffold attempts per episode: 8 plus one per 16 robots, and stranded robots count as robots that must move.
   Step cap 64 + 16 per robot, at most 6,000 (was 2,000).

## Numbers

`scripts/reverse_v3_measure.py`, seed 0, 5 goals per source and size bin, the same draw before and after. gen_complex is
drawn by kind in turn (scene, edit, retarget, trick; the plain draw gave 5 of 5 `edit_house` at 251-500), so its goals
include guided edits, retargets and tricks; gen_medium has no goal above 115 robots; v2 is `data/goal_library_v2`
restricted to the six libraries complete since 11:20 (craftassist houses and parts, hack337, iglu single and
multi-turn, stabletext2brick; another agent was still adding sources during the run). Before = reverse.py at HEAD
3d0521b (v2), after = this code. CPU only, one thread per process, nice 10, at most 6 processes on a loaded Mac; a
15-minute CPU budget per walk (only v2 hit it, 3 times). Guided rows of "after" come from the first after-run; the
natural rows were re-measured after the last fix (it touches natural mode only). Files: `runs/reverse_v3/{before,
after,final}/*.jsonl` (not committed), logs `logs/reverse-v3-*.log`.

ok = replay passed and the start is what was asked (natural: all six statistics in band; guided: exactly A).
Steps and CPU are means over the episodes that finished; robots moving per step is the mean over steps (holds left
out); steps per CPU s is the walk only.

| robots | code | ok | natural ok | guided ok | replay | steps | CPU s per episode | robots moving per step | steps per CPU s | timeouts |
|---|---|---|---|---|---|---|---|---|---|---|
| 4-115 | before | 12 / 15 | 11 / 12 | 1 / 3 | 15 / 15 | 244 | 41.9 | 2.40 | 5.8 | 0 |
| 4-115 | after | 15 / 15 | 12 / 12 | 3 / 3 | 15 / 15 | 150 | 4.4 | 4.73 | 34.1 | 0 |
| 116-250 | before | 0 / 10 | 0 / 8 | 0 / 2 | 10 / 10 | 636 | 278.6 | 2.11 | 2.3 | 0 |
| 116-250 | after | 7 / 10 | 7 / 8 | 0 / 2 | 10 / 10 | 730 | 36.0 | 6.70 | 20.3 | 0 |
| 251-500 | before | 0 / 10 | 0 / 7 | 0 / 3 | 9 / 9 | 806 | 427.1 | 2.31 | 2.3 | 1 |
| 251-500 | after | 8 / 10 | 7 / 7 | 1 / 3 | 10 / 10 | 1,335 | 121.6 | 8.89 | 11.0 | 0 |
| 501-800 | before | 0 / 10 | 0 / 6 | 0 / 4 | 8 / 8 | 642 | 494.7 | 2.36 | 1.9 | 2 |
| 501-800 | after | 1 / 10 | 1 / 6 | 0 / 4 | 10 / 10 | 2,316 | 226.1 | 7.99 | 10.2 | 0 |

By source (ok, before -> after): gen_complex 3/5 -> 5/5, 0/5 -> 2/5, 0/5 -> 3/5, 0/5 -> 0/5; gen_medium (4-115 only)
5/5 -> 5/5; v2 4/5 -> 5/5, 0/5 -> 5/5, 0/5 -> 5/5, 0/5 -> 1/5. The one guided ok at 251-500 is a hold (trick_already,
302 robots, A = B; v2 stalled on it with 0 steps). Natural episodes that reached their start took 4 CPU s on average
at 4-115, 13 at 116-250, 48 at 251-500 and 32 (one episode) at 501-800.

1. Every finished episode replays exactly under world.step, before and after (after: 45 / 45).
1. Replay itself (reverse.replay, world.step with the full rule table) runs at 35, 17, 10 and 7 steps per CPU s by
   bin, slower than the walk from 116 robots up: a 6,000-step 624-robot episode takes 14 minutes to replay.
1. Robots moving per step: 2-2.6 before in every bin, 4.7-9 after (up to 41 in one step at 621 robots); natural
   take-aparts at 251-800 average 9-15. The forward demo is many robots streaming in at once.

## What still fails

1. Natural starts at 501-800 robots (5 of 6 miss): extraction clears the goal (12-15 robots per step), but the start
   ends just out of band, "lifted" and usually "height" (example, 660 robots, heap_cap 4: lifted 0.65 against a band
   of 0.62, 6 robots at height 4 against 3, 18 columns still empty). The heap round the goal is 65 % full and the
   too-high robots find no path down; the spread then runs out of rounds. Drops go to the nearest cell of the best
   tier; a planned spread (assign drop cells so columns fill evenly, lowest first, before walking) is the next step.
   One 160-robot trick_negation start misses on height, fill and spread the same way.
1. Guided edits and retargets above 115 robots (0 of 8, not counting the hold): the fill and digs make progress (distance to A falls) but
   run out of steps (6,000 at most) or stall after 12 digs. They are thick scaled parts (edit_house, edit_undo of a
   scale-3 house, edit_turn, retarget_part of 130-570 robots): a closed-in cell needs a dig, a dig escalates to whole
   walls, and the rebuild fills a layer at a time at 3-5 robots per step (the strict lowest-layer order serialises
   it). Distances left: 4, 19, 35, 76, 93, 131, 414, 491. Scaffolding attempts run to 50-310 steps per episode.
   Ideas: dig the part the edit changes (A xor B) rather than columns round a cell; rebuild with the natural swarm
   (take B's changed part apart to a heap, then fill A as the reverse of a natural take-apart of A's changed part,
   which runs at 9-15 robots per step); allow two layers at once when the upper cells do not rest on the empty lower
   ones.
1. Time: guided episodes that fail use their whole budget (up to 7 CPU minutes, plus up to 14 minutes of replay).
   Callers that build data should cap steps lower for guided goals above 115 robots until the above is fixed.
1. The fill order is a heuristic (lowest layer, enclosure, closing check). It fixed the lintel and doorway cases
   (tests) but is not a planner; edits where an upper cell must be filled before a lower one (a cantilever whose
   support comes from the side) fall back to the relaxed order.

## Next

1. Planned spreading for 500+ robot natural starts (above); it is the one blocker for the 501-800 natural bin.
1. Guided edits of thick parts: dig A xor B, rebuild by swarm, two-layer fill.
1. A vectorised replay check (rule_ok in place of world.refuse inside replay is exact and about 10x faster on the
   rule part) would cut replay of big episodes from minutes to tens of seconds; replay is kept on world.step here on
   purpose, as the independent check.
1. Wire hold episodes into the trick data (A = B tricks now have a 4-step STAY demo).
