# Hill-climb log

Val score = mean of (first-eaten step / rollout length) over interp, extrap and damage; failure = 1; lower is better.
Baseline (`GreedyBaseline`, closeness 10, crowding 0.5): val score **0.365**.

| Run | Change | Minutes | Val score | Interp | Extrap | Damage | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sanity | defaults | 4 | 0.696 | 0.72 | 0.11 | 0.44 | Reached level 5 of 9. Learns. |
| r1-default | defaults | 30 | 0.274 | 1.00 | 0.91 | 0.98 | Reached the final level in about 10 minutes. |
| r1-lr2e3 | lr 2e-3 | 30 | **0.236** | 1.00 | 0.98 | 0.99 | Best of round 1. Clean wrap, no crowding. |
| r1-h128l1 | 1 hidden layer of 128 | 30 | 0.278 | 1.00 | 0.89 | 0.99 | About the same as default. |

All three beat the baseline (0.365). Videos checked: real wraps, no hacks. The learned shell is about 2 squares thick at 0.5 to 0.8 mass each, an artifact of continuous mass.

Round 2: is the round-1 gap bigger than seed noise? Seed repeat of lr 2e-3, lr 4e-3, and training at firing probability 0.5 (eval stays at 0.75).
| r2-lr2e3-seed1 | lr 2e-3, seed 1 | 30 | **0.226** | 1.00 | 1.00 | 1.00 | Seed repeat of r1-lr2e3. Seed noise is about 0.01. |
| r2-lr4e3 | lr 4e-3 | 30 | 1.000 | 0.00 | 0.00 | 0.00 | Collapsed. Stuck at level 2, never eats. |
| r2-lr2e3-fire05 | lr 2e-3, train firing 0.5 | 30 | 0.233 | 1.00 | 1.00 | 1.00 | Same as without it. |

Physical lower bound on val: every ring square needs at least its distance from the nearest starting mass in steps. That gives 0.13 if every square acted every step, and about 0.18 at the eval's 0.75 firing rate. The best run (0.226) is within about 25% of that.

Round 3: 60-minute runs of lr 2e-3: plain, hidden 128, and distance weight decayed by 0.7 per level. Robustness probes added outside the frozen score.

Round 3 (longer, bigger, faster distance decay) was stopped at 18 minutes: the val score was saturated, and the task was too easy to say anything about distributed behavior.

## v1 final test

Run once on the best val run, never used for selection.

| | Val | Test | Test interp | Test extrap | Test damage |
| --- | --- | --- | --- | --- | --- |
| r2-lr2e3-seed1 | 0.226 | **0.267** | 1.00 · 0.239 | 0.91 · 0.303 | 0.99 · 0.258 |
| Baseline | 0.365 | 0.405 | 1.00 · 0.367 | 0.98 · 0.364 | 0.92 · 0.485 |

Test is harder than val for both by about 0.04; the gap holds. Weak spot: 72x72 worlds, where the smell is fainter than anything seen in training.

Why v1 was easy: the smell field hands every square an arrow to the cheese, so no square needs information another square has. Also, empty squares ran the network and wrote notes, so the floor itself could compute. v1 is tagged `v1-smell`.

# v2: no smell

A robot only senses the cheese by touching it (it's in its 3x3 window). Notes are pheromones on the floor: only robots write them, in proportion to the mass on the square, and they fade 5% per step. The swarm starts as one nest of 6% of the world's area (at least 1.5 per ring square). Rollouts are 3x the grid size. Baseline: wander at random, stay put once touching the cheese.

Val score = same hard metric as v1. Fog = share of mass on squares holding under 0.5, reported but not scored.

Baseline val: 1.000 (0% solved anywhere; random wandering spreads the swarm too thin to fill a ring in time).

| Run | Change | Minutes | Val score | Interp | Extrap | Damage | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v2-sanity (first physics) | defaults, lr 2e-3 | 6 | 0.279 | 0.99 | 0.95 | 0.81 | Not a hack: a valid strategy under the rules as written. It spread into a thin fog over the world, then used pheromones to condense onto the cheese, much like a slime mold. The rules were the problem: fog puts some mass on nearly every square, so every square has a neighbor to relay through, a connected network that sparse physical robots don't get. Rule change, aiming at robots: a square moves and writes in proportion to its mass (a whole robot is 1). Under the new rule this model drops to about 30% solved. |
| v2-sanity | defaults, lr 2e-3 | 8 | 0.998 | 0.01 | 0.00 | 0.00 | Honest physics. Reaches level 5, then destabilizes: training success 97% to 0%, loss rising. |

v2 round 1: lr 1e-3, lr 5e-4, lr 1e-3 with hidden 128. 30 minutes each.
| v2r1-lr1e3 | lr 1e-3 | 30 | **0.723** | 0.45 | 0.41 | 0.35 | Honest learning: 45% of unseen layouts. Level 5, still climbing. Only 1,192 iterations. |
| v2r1-lr5e4 | lr 5e-4 | 30 | 0.865 | 0.25 | 0.11 | 0.20 | Too slow; level 2. |
| v2r1-lr1e3-h128 | lr 1e-3, hidden 128 | 30 | 0.877 | 0.20 | 0.27 | 0.16 | Only 335 iterations: starved by swapping. |

Round 1 runs swapped: one run at 40x40 with 120 steps holds 6.7 GB of activations, so three overflowed 24 GB. Gradient checkpointing cuts that to 1.2 GB for 24% more time per iteration. Now on by default; default lr is 1e-3.

v2 round 2 (45 minutes): warm start from v2r1-lr1e3, fresh lr 1e-3, fresh hidden 128.

Rule change: success now also needs the swarm to gather (ring full and at least 80% of the mass within the gather radius), with a matching loss term (w_gather 0.5). Round 1 re-scored: v2r1-lr1e3 0.723 to **0.909**, v2r1-lr5e4 0.865 to 0.950, v2r1-lr1e3-h128 0.877 to 0.932, baseline still 1.000.

v2 round 2 restarted on the gather objective (45 minutes): warm start from v2r1-lr1e3, fresh lr 1e-3, fresh hidden 128.
| v2r2-warm | warm start from v2r1-lr1e3 | 45 | 0.979 | 0.06 | 0.00 | 0.06 | Gather objective. Stuck at level 3. |
| v2r2-lr1e3 | lr 1e-3 | 45 | 0.928 | 0.12 | 0.06 | 0.14 | Gather objective. Stuck at level 3; loss spikes to 8.7 from the crowding term. |
| v2r2-h128 | hidden 128 | 45 | 1.000 | 0.00 | 0.00 | 0.00 | Gather objective. Stuck at level 3. |

v2 closed without solving the gather task (best 0.909, v2r1-lr1e3 re-scored). Runs archived in `runs_v2/`, tagged `v2-no-smell`.

# v3: terrain

Random cave-like walls (smoothed noise on the torus, 0 to 30% of the world), robots can't enter them. Distances are walking distances around walls; the gather radius uses them too. Every layout is checked so every ring square is reachable from the nest. Walls enter through the curriculum (none at the first levels, up to 30% at the last). New val and test sets include terrain. A physics test suite (`uv run pytest`) now runs before each round.

Feasibility check: a hand-written oracle that knows the walking distance solved only 69 to 97% of interp layouts and almost none of the bigger worlds, even with twice the time. The cause was the oracle, not the rule: it piled 10 to 28 robots on single ring squares that jumped back and forth between ring squares every step. Softer oracles did worse. So feasibility is tested with a learned control instead: a privileged swarm that also smells the walking distance, under the same physics and success rule.

v3 round 1 (45 minutes): privileged control, blind defaults, blind with w_crowding 0.3.

| Run | Change | Minutes | Val score | Interp | Extrap | Damage | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v3r1-privileged | control: smells walking distance | 45 | **0.418** | 0.91 | 0.75 | 0.84 | Feasibility control. Reached the last level; 92% gathered, fog 0.19. Videos: a clean, compact gather. The task and time limit are achievable. |
| v3r1-blind | defaults | 45 | 0.863 | 0.25 | 0.22 | 0.23 | Best blind so far, with terrain (v2's best was 12% without walls). Stuck at level 3. Videos: spreads out laying pheromone, finds the cheese, still gathering when time runs out. |
| v3r1-crowd03 | w_crowding 0.3 | 45 | 0.996 | 0.02 | 0.00 | 0.01 | Worse. |

The gap between privileged (0.418) and blind (0.863) is the cost of search and communication.

v3 round 2 (90 minutes): blind defaults with seed 1; blind warm-started from the privileged swarm with its scent input removed (inherits gathering, has to learn search; never sees the scent at test time); blind with batch 64.

New standard: every task must reach 100% on val before moving on, or come with evidence of why 100% is out of reach.

Round 2 (90-minute runs) was stopped after 20 minutes to restart on better infrastructure:
- the next batch of layouts is generated in a worker process while the GPU trains (generation was about 20% of each step);
- 30% of batches replay earlier curriculum levels, so easy cases aren't forgotten;
- checkpoints every 10 minutes, `--start_level` to resume a curriculum, and a STOP file for a graceful early end;
- `uv run score` scores checkpoints on val only, for choosing between runs mid-training.

Population (6-hour runs, managed at each check-in: the worst run is replaced by a copy of the best with a changed setting):
- v3p-teacher: blind, initialized from v3r1-privileged with the scent input removed.
- v3p-scratch: blind, from scratch, seed 1.
- v3p-privileged: the control, continued from v3r1-privileged at the last level, to find the ceiling.

Population check-in at 46 minutes (val scores of the 40-minute checkpoints):

| Run | Level | Val score | Interp | Extrap | Damage |
| --- | --- | --- | --- | --- | --- |
| v3p-teacher | 4 | 0.840 | 0.28 | 0.27 | 0.24 |
| v3p-scratch | 3 | 0.829 | 0.31 | 0.25 | 0.24 |
| v3p-privileged | 8 | **0.314** | 0.96 | 0.84 | 0.86 |

Diagnosis of the blind failures (v3p-teacher on val interp): success falls with the starting distance, 74% within 6 squares, 17% at 6 to 10, 3% at 10 to 15, 0% beyond. Most failures never fill the ring: the swarm doesn't find the cheese in time. The privileged swarm solves 100% up to 10 squares and 93% at 10 to 15.

Two causes. The level curriculum gates distance, and blind runs stall at level 3 (at most 8 squares), so they never practise far search. And search itself may be physically tight: a swarm moving as one blob sweeps the world in roughly 4 to 5 grid-lengths of steps, against a limit of 3; only a stretched-out sweep fits.

Change: adaptive curriculum. Full cheese and wall ranges from the start; the start distance is chosen in proportion to s(1-s) of its recent success, so training sits at the frontier. Swap: v3p-scratch stopped (stuck at level 3); replaced by v3p-adaptive, started from v3p-teacher's 40-minute checkpoint.

Population check-in at 95 minutes:

| Run | Val score | Interp | Extrap | Damage | Notes |
| --- | --- | --- | --- | --- | --- |
| v3p-privileged | 0.322 | 0.95 | 0.84 | 0.92 | Plateaued around 0.31 to 0.32. |
| v3p-adaptive | 0.893 | 0.27 | 0.06 | 0.24 | Same distance wall: 53% at 6 squares, about 5% beyond 10. |
| v3p-teacher | 1.000 | 0.00 | 0.00 | 0.00 | **Collapsed** after about 90 minutes (training loss 2.3); latest checkpoint worthless. Stopped. |

Diagnosis, with the time limit raised for diagnosis only (official limit unchanged):

| Blind swarm | 3x grid | 6x grid | 9x grid |
| --- | --- | --- | --- |
| v3p-adaptive interp | 0.27 | 0.40 | 0.45 |
| v3r1-blind interp | 0.25 | 0.29 | 0.30 |

More time barely helps: the blind swarm has no real search strategy. Of its failures, 44% never touch the cheese and 56% touch it but can't recruit the rest. Why gradient descent doesn't find search: the distance term is the only signal before contact, and on a torus with the cheese placed uniformly at random every square has the same expected distance to it, so the term carries no information about how to search.

Changes:
- `w_explore`: rewards the share of open floor visited, only until a robot first touches the cheese.
- Cosine learning-rate decay, and each run scores itself on val every 30 minutes and keeps its best checkpoint (`model_best.pt`), so a collapse no longer loses the run. Mid-run evaluation restores training's random state afterwards.

New population: v3p-adaptive stopped; v3x-explore (from v3p-adaptive's checkpoint) and v3x-explore-teacher (from v3r1-privileged's weights, scent input removed), both adaptive, w_explore 0.5, cosine. v3p-privileged continues.

Check-in, 45 minutes into the exploration runs:

| Run | Val (best checkpoint) | Interp | Extrap | Damage | Notes |
| --- | --- | --- | --- | --- | --- |
| v3x-explore | **0.745** (at 30 min) | 0.41 | 0.34 | 0.39 | Best blind result so far. By 45 minutes training success fell to about 2% at every distance: with w_explore 0.5, the swarm learned to spread out and never regroup. |
| v3x-explore-teacher | 1.000 | 0.00 | 0.00 | 0.00 | Never recovered under w_explore 0.5. |
| v3p-adaptive (final) | 0.804 | | | | Stopped earlier; best checkpoint. |

Both stopped. New runs from v3x-explore's 30-minute best checkpoint, adaptive, cosine: v3y-explore015 (w_explore 0.15) and v3y-explore005 (w_explore 0.05).

Check-in: collapses everywhere.

| Run | Val (best checkpoint) | Notes |
| --- | --- | --- |
| v3y-explore015 | 0.773 at 30 min | Training success fell to about 5% by 45 minutes. |
| v3y-explore005 | 1.000 | Collapsed within 30 minutes. |
| v3p-privileged | 1.000 at 190 min | **Collapsed** after more than 3 healthy hours (0.314 at 40 min, 0.322 at 90 min). It started before best-checkpoint keeping, so only the scores survived, not the weights. |

Every long run collapses, whatever the settings: the teacher run at 90 minutes, the explore runs within 30 to 45, the privileged control at about 190. A cause common to all of them is in the training loop. The prime suspect is per-parameter gradient normalization (from Growing NCA): it rescales every parameter's gradient to unit norm before Adam, so near-zero gradients on a plateau become full-size steps of noise, the weights random-walk, and eventually they drift into a saturated regime (one-hot moves, piled mass, vanishing gradient) with no way back. Adam already scales steps per weight, so the normalization is redundant.

Change: global gradient-norm clipping (max norm 1) is now the default (`--grad_mode clip`), and fine-tunes from good checkpoints use lower learning rates.

New population (180 minutes, clipping, cosine, best checkpoint kept): v3z-priv (privileged from v3r1-privileged, lr 3e-4), v3z-explore (blind from v3x-explore's best, w_explore 0.15, lr 3e-4), v3z-explore-slow (same, w_explore 0.1, lr 1e-4).

Bar revised (decided by the user at 04:00): a task is done at above 90% success on every val split (interp, extrap, damage), confirmed once on the test set.

Check-in at 60 minutes (clipping held; no collapses):

| Run | Val at 30 min | Val at 60 min | Interp | Extrap | Damage |
| --- | --- | --- | --- | --- | --- |
| v3z-priv | 0.326 | **0.291** | 0.96 | 0.95 | 0.89 |
| v3z-explore | 0.754 | 0.778 | 0.44 | 0.25 | 0.39 |
| v3z-explore-slow | 0.745 | **0.738** | 0.43 | 0.33 | 0.43 |

The privileged control holds at 89% damaged on both checks, one point under the bar: even a swarm that can smell the cheese barely clears 90% after a quarter of it is wiped mid-run. Stopped; it has shown the task is solvable to about 90% when search is free. v3z-explore stopped (bigger worlds fell from 38% to 25%). v3z-explore-slow continues.

New: blind networks 192 wide (from 96), from scratch, w_explore 0.15, clipping, cosine: v3w-big-levels (level curriculum) and v3w-big-adaptive (adaptive curriculum). Tests whether capacity limits blind coordination.

Paused at the user's request to free the machine. v3z-explore-slow, v3w-big-levels and v3w-big-adaptive were stopped mid-run; their latest checkpoints (`model.pt`, `checkpoint.json`) and best val checkpoints (`model_best.pt`, where one exists) are in their run folders. Resume with `--init runs/<name>/model.pt` and the same flags.

Resumed on a different machine (RTX 3070, CUDA, torch 2.11.0+cu128). The `.pt` files are gitignored, so the pushed run folders have no weights: v3z-explore-slow could not be resumed (it needs `runs/v3x-explore/model_best_min30.pt`), and v3w-big-levels and v3w-big-adaptive were restarted from scratch with identical flags as `-3070` runs (200 minutes, hidden 192, lr 1e-3, w_explore 0.15, clipping, cosine).

Check-in at 30 minutes:

| Run | Val | Interp | Extrap | Damage | Notes |
| --- | --- | --- | --- | --- | --- |
| v3w-big-levels-3070 | 0.779 | 0.35 | 0.30 | 0.30 | Level 3 of the curriculum. |
| v3w-big-adaptive-3070 | 0.960 | 0.09 | 0.06 | 0.06 | Training success only just started rising (about 47% smoothed at 28 minutes). |

Queue and bar unchanged: above 90% on every val split. Runs continue to 200 minutes.

## Resumed 2026-10-28

While paused, v3z-explore-slow's 90-minute val check had already set a new blind best: **0.724** (interp 0.44, extrap 0.39, damage 0.42), saved as its `model_best.pt`. v3w-big-adaptive was dead at 30 minutes (val 1.000, training success near 0 at every distance): from scratch, every distance sits near 0% success, so the adaptive weights s(1-s) are flat and the curriculum has no frontier to follow. It is not resumed.

Two folders from a CUDA machine also appeared, `v3w-big-levels-3070` and `v3w-big-adaptive-3070`: same configs, metadata only, no weights. In 30 minutes they ran about 2.4 times as many iterations (1743 vs 730). Val at 30 minutes: big-levels 0.779 (35/30/30%), big-adaptive 0.960, confirming that the adaptive curriculum fails from scratch.

New population (clipping, cosine; resumed runs pick their cosine schedule up at the step where the pause cut it, approximately):

| Run | Start | Settings | Question |
| --- | --- | --- | --- |
| v3z-explore-slow-r | v3z-explore-slow at 110 min | lr 4e-5 cosine to 1e-5, 70 min, w_explore 0.1, adaptive | Finish the best blind run's schedule |
| v3w-big-levels-r | v3w-big-levels at 30 min, level 3 | 192 wide, lr 9.5e-4 cosine, 170 min, w_explore 0.15 | Does width help blind coordination? |
| v3w-base-levels | scratch | 96 wide, otherwise identical to v3w-big-levels | Width control: same recipe, original width |

**Invalidated.** The laptop slept during all three runs. Training budgets were wall-clock time, so the sleep counted against them: the runs "finished" after about 37 minutes of real training each (786, 745 and 913 iterations), with their cosine schedules jumping to the floor across the gap. The folders are moved to `runs_invalid/` and excluded from the page. Fix: the training clock is now `awake_seconds()` (CLOCK_UPTIME_RAW on macOS, CLOCK_MONOTONIC on Linux), which stops while the machine sleeps.

One observation from before the sleep, worth re-checking: v3w-big-levels-r collapsed while the machine was awake, at about 33 minutes (training success EMA 0.76 to 0.06 within 100 iterations, val 0.823 to 1.000), at lr 9.5e-4 with gradient clipping. Clipping alone may not prevent collapse at full learning rate for the 192-wide network.

To resume in the morning: the same three runs, under the same names, with the same flags (see the table above).

## v3b: terrain with 6x time

Resumed 2026-10-29 with the same three runs under `caffeinate -i`. About 10 minutes in, the user pointed out that the blind swarm didn't reach the cheese in the demo and suggested more steps. Those runs were cancelled (folders in `runs_cancelled/`, not counted).

The existing best checkpoints, scored on val with the time limit doubled and no retraining:

| Checkpoint | 3x grid: score (interp / extrap / damage) | 6x grid: score (interp / extrap / damage) |
| --- | --- | --- |
| v3z-priv (scent-following control) | 0.291 (0.96 / 0.95 / 0.89) | **0.158 (0.98 / 0.97 / 0.98)** |
| v3z-explore-slow (best blind) | 0.724 (0.44 / 0.39 / 0.42) | 0.624 (0.58 / 0.45 / 0.53) |
| v3w-big-levels (192 wide, 30 min) | 0.846 (0.28 / 0.20 / 0.24) | 0.786 (0.30 / 0.19 / 0.30) |

At 6x, the control clears the 90% bar on every split, so the task is feasible at that limit. The blind swarm gains 6 to 14 points without ever having trained on a horizon that long. The earlier diagnosis still holds (going from 6x to 9x added little for a model trained at 3x), but no model has been trained at a longer horizon yet, and with BPTT the training horizon caps what a search strategy can be credited for.

Change: the time limit is 6 steps per grid square, in both training (`steps_per_cell` default) and evaluation (`STEPS_PER_CELL`). This changes the benchmark, so it is a new stage: the 3x runs are archived in `runs_v3a/` as stage v3a (closed), and stage v3b starts with an empty `runs/`. The val and test layouts are unchanged; the test set is still unused for v3. Videos still run to twice the limit, now 12x.

v3b population (clipping, cosine, `caffeinate -i`):

| Run | Start | Settings | Question |
| --- | --- | --- | --- |
| v3b-explore-slow | v3z-explore-slow best (val 0.724 at 3x) | lr 1e-4, 120 min, w_explore 0.1, adaptive | How far does the best blind swarm get when trained on the longer horizon? |
| v3b-big-levels | scratch | 192 wide, lr 5e-4, 200 min, w_explore 0.15, level curriculum | Width test |
| v3b-base-levels | scratch | 96 wide, otherwise identical | Width control |

The learning rate is 5e-4 rather than 1e-3: the 192-wide run collapsed at 9.5e-4 the day before, while the laptop was awake.

Note: the cancelled runs' stdout logs overwrote the stdout logs of the runs the laptop slept through (`runs_invalid/`). Those runs' `log.jsonl` and val histories are intact.

Check-in at 90 minutes:

| Run | Val at 30 / 60 / 90 min | Interp | Extrap | Damage | Notes |
| --- | --- | --- | --- | --- | --- |
| v3b-explore-slow | 0.627 / 0.627 / 0.627 | 0.59 | 0.50 | 0.51 | Flat on val. Training success at start distances 10 to 20 keeps rising (25-40% at 50 min, 32-44% at 92 min), but val doesn't follow. Ends at 120 min. |
| v3b-base-levels (96) | 0.885 / 0.808 / 0.795 | 0.38 | 0.17 | 0.32 | Level 4, still improving. |
| v3b-big-levels (192) | 0.972 / 0.887 / 0.867 | 0.21 | 0.08 | 0.20 | Level 3. Also behind at equal iterations: 1450 iterations here vs 0.808 for the 96-wide run at about 1350. |

So far, width doesn't help: the 192-wide network is slower per iteration and no better per iteration.

v3b-explore-slow finished: val **0.626** (interp 0.59, extrap 0.48, damage 0.54), flat from 30 minutes on (0.627, 0.627, 0.627, 0.626). The fine-tune didn't help: its starting checkpoint (v3z-explore-slow's best) scores 0.624 at 6x, slightly better, and evaluation is deterministic, so that's not noise. The run never saved its starting point as a candidate, because its first val check is at 30 minutes. The best blind checkpoint at 6x is still `runs_v3a/v3z-explore-slow/model_best.pt` (0.624).

Failure breakdown (val interp + extrap, no damage, 160 layouts, 55% success):

| Start distance | Layouts | Success | Touched the cheese |
| --- | --- | --- | --- |
| 2 to 6 | 69 | 0.88 | 0.99 |
| 6 to 10 | 53 | 0.32 | 0.74 |
| 10 to 15 | 22 | 0.41 | 0.68 |
| 15 to 25 | 15 | 0.07 | 0.20 |

- 49% of failures never touch the cheese.
- The other 51% touch it late (at 45% of the limit on average; successes touch at 8%), fill the ring (final surround 0.92), but gather only 58% of the swarm against the 80% needed: part of the swarm never gets recruited.
- Video (extrap): the swarm breaks into a few compact blobs that wander along walls. One blob sits for a long stretch in a concave pocket of wall a few squares from the cheese, on the wrong side of the wall. No systematic sweep.

Next: v3b-explore03, from v3b-explore-slow's best, with w_explore 0.3 (from 0.1), otherwise identical (lr 1e-4 cosine, adaptive, 120 min). At 3x, w_explore 0.5 made the swarm spread and never regroup; with twice the time to regroup, stronger exploration pressure may pay off.

**Rule (set by the user, 2026-10-29): every stage's results come from runs trained from scratch in that stage.** No starting from another run's or another stage's weights. Resuming a run from its own checkpoint (after a pause, or to extend it) is still the same run. Reasons: a stage's result has to be reproducible from scratch, and the evidence agrees. Fine-tunes from the old best plateaued at once (v3b-explore-slow flat at 0.626 for 90 minutes), while the from-scratch runs at 6x keep climbing (v3b-base-levels 0.885, 0.808, 0.795, 0.714, 0.702 at 30 to 150 minutes). The old checkpoint seems to carry habits learned under the 3x limit.

Consequences:
- v3b-explore-slow and v3b-explore03 (both started from v3a weights) are moved to `runs_finetuned/` and don't count for v3b. v3b-explore03 was stopped at 45 minutes (val 0.627 at 30).
- The 6x scores of v3a checkpoints (control 98/97/98%, blind 58/45/53%) were measured on fine-tune lineages: v3z-priv descends from v3r1-privileged, v3z-explore-slow from v3x-explore and v3p-adaptive. They motivated the 6x limit but aren't v3b results.
- So the feasibility control has to be redone from scratch: v3b-priv-scratch, identical to v3b-base-levels except that it smells the walking distance (96 wide, lr 5e-4, w_explore 0.15, level curriculum, 200 min).
- In v1 to v3a, several reported runs were fine-tunes or teacher starts; their configs (`init`) say which.

Note (user, 2026-10-29): v3b keeps its current setup (shaping terms and curriculum unchanged). From the next stage on, experiments follow the bitter lesson: as little human-engineered help as possible (shaping terms, hand-staged curricula, scripted structure), and more compute instead. Feast, Chase, Rivals and 3D are the groundwork before the language-conditioned control idea (VLA.md).

Width runs finished (200 minutes, from scratch, both stuck at level 5 of 9 at the end):

| Run | Iterations | Final val | Interp | Extrap | Damage |
| --- | --- | --- | --- | --- | --- |
| v3b-base-levels (96) | 4768 | **0.691** | 0.47 | 0.42 | 0.41 |
| v3b-big-levels (192) | 3222 | 0.716 | 0.40 | 0.33 | 0.35 |

At equal iterations they're about even (192: 0.726 at 2904 and 0.716 at 3222; 96: 0.714 at 2833 and 0.702 at 3590), and the 192-wide network runs about 1.5 times slower per iteration. Width isn't the bottleneck at this scale. The 96-wide run was still improving at the end.

v3b-priv-scratch (control, from scratch) at 30 minutes: 0.745 (41/34/38%), already at level 5.

New:
- v3b-base-levels-cont: v3b-base-levels continued from its own final checkpoint at level 5, fresh cosine from lr 5e-4, 200 minutes. Same run, more compute; its starting score (0.691) is the first best candidate.
- v3b-base-seed1: v3b-base-levels repeated with seed 1, to check that the climb isn't one lucky seed.

Scope (user, 2026-10-29): the point of each stage is an existence proof, that the task is achievable by swarm intelligence. Those proofs make the case for a general model that does all the tasks with the same weights. So no ablations or variants for their own sake: one from-scratch recipe per task, given compute until it clears the bar. v3b-base-seed1 cancelled (`runs_cancelled/`). Running: v3b-base-levels-cont (the main line) and v3b-priv-scratch (control, ends about 18:20; it shows whether the bar is reachable at 6x from scratch).

**v3b closed as a proof of concept (user: "if you have a proof of concept, you move on").** v3b-base-levels-cont and v3b-priv-scratch stopped (moved to `runs_cancelled/`; the control had reached 0.745 at 30 minutes). Runs archived to `runs_v3b/`.

Finalist v3b-base-levels (96 wide, from scratch, 200 minutes, level 5 of 9):

| | Score | Interp | Extrap | Damage |
| --- | --- | --- | --- | --- |
| Val | 0.691 | 0.47 | 0.42 | 0.41 |
| **Test** (once) | **0.695** | **0.52** | **0.38** | **0.44** |
| Baseline, test | 1.000 | 0.00 | 0.00 | 0.00 |

| Probe | Success |
| --- | --- |
| Reference | 0.50 |
| Pheromone blackout | 0.00 |
| Firing 0.5 / 1.0 | 0.42 / 0.50 |
| Wipe 50% / chunk cut | 0.38 / 0.50 |
| 2 / 3 separate nests | 0.25 / 0.06 |
| 128x128 world | 0.19 |
| No walls / walls 40-45% | 0.53 / 0.47 |

## v4: Feast

Two to four cheeses (1 to 5 squares each, never touching) on the same terrain (walls up to 30%), no smell. A cheese is eaten once every square of its ring holds more than tau, and then it disappears. Success is eating every cheese within the limit (6 steps per grid square). The whole-swarm gather rule is dropped: with several cheeses the swarm has to split. Swarm mass is 1.5 times the total ring size (or 6% of the open floor, if larger), enough to eat every cheese at once.

First stage under the bitter-lesson rule. The loss is the task only, the unfilled share of the remaining cheeses' rings at every step, plus the crowding penalty, which stands in for robots taking up space. No distance or exploration terms, no curriculum: every batch samples the full range of sizes, cheese counts and walls. Trained from scratch.

Frozen sets: val seed 3000 (interp sizes 22, 30, 36 with 2 to 4 cheeses; extrap 48, 64 with 5 or 6), test seed 4000 (interp 18, 26, 34; extrap 56, 72), plus the damage split. Score = mean time fraction over the three splits, as before. Code: `src/beebot/feast.py`, `src/beebot/train_feast.py`, tests in `tests/test_feast.py`.

Run: v4-feast, 96 wide, batch 64 (more compute per step, since the task-only loss gives a sparse signal), lr 5e-4 cosine, 240 minutes, alone on the GPU.

**v3b diagnostic: is the finalist weak because it runs out of time?** (Asked by the user. Val only; same rollouts read off at 6x, 12x and 24x the grid size.)

| Split | Success at 6x | 12x | 24x | 6x failures that succeed by 24x |
| --- | --- | --- | --- | --- |
| Interp | 0.46 | 0.55 | 0.58 | 23% |
| Extrap | 0.44 | 0.50 | 0.62 | 33% |

Mostly no. Four times the time recovers a quarter to a third of the failures. Far starts in bigger worlds gain the most (start 15 to 25 squares: 27%, 40%, 53%). Among the layouts still failing at 24x:
- The swarm usually found the cheese: only 12% (interp) and 22% (extrap) of all 6x failures never touch it even by 24x.
- 55% (interp) and 38% (extrap) filled the ring at some point, but never with 80% of the swarm gathered at the same time (best gathered share 0.54 and 0.37 on average). By 24x the ring has emptied again: final ring share 0.20 and 0.10, final gathered share 0.24 and 0.12.
- Even from starts 2 to 6 squares away, about 18% fail at every limit.

So the main weakness is recruiting the whole swarm and holding the formation, not search time. Behavior after 6x is past the training horizon, so the drift away from the cheese may just be untrained behavior; but the failure to gather is there within the limit too.

v4-feast, first 65 minutes: learned fast with the task-only loss. Training success (smoothed) reached 0.84 by 38 minutes, collapsed at about 42 (loss 1.6, above where it started), then recovered to about 0.56 by 65. But val checks were far below training: 0.990 at 30 minutes (4% interp) and 0.970 at 60 (15%).

Diagnosis (no code bug): the weights swing between good and bad within 10 to 20 optimizer steps.
- With the 30-minute checkpoint frozen, the exact training loop gets the same low success as val.
- From that checkpoint, 40 to 60 more steps at lr 5e-4 bring frozen success at size 24 to 0.68-0.77, with dips to 0.09 along the way.
- So the val checks landed in dips, and any raw snapshot is a lottery ticket.

Fix: checkpoints, val checks and the final model now use an exponential moving average of the weights (decay 0.995, about a 200-step window), and lr goes from 5e-4 to 3e-4. Both are standard, general-purpose training practice, not task knowledge. v4-feast stopped (`runs_cancelled/`). New run from scratch: v4-feast-ema, batch 64, 240 minutes.

v4-feast-ema at 30 minutes: val 0.808, with 72% of unseen sizes and 69% of bigger worlds fully eaten (damage 30%). The averaged weights closed the train/val gap. But the videos show the fog strategy: the swarm dissolves into a film over the whole world and condenses onto each ring. Measured on val layouts, 73 to 96% of the mass sits on squares holding less than half a robot, and mass covers 85% of the world. On terrain, the whole-swarm gather rule had kept swarms compact. Feast dropped it, and the proportional-mobility rule alone only slows fog down.

Decision (user): fog is not okay. Add ad hoc loss terms for the ideal behavior; splitting the swarm is fine. Train longer.
- **Fog term** (w_fog 0.1): a soft count of squares holding some mass but less than half a robot, per unit of swarm mass. It's a sine bump on [0, 0.5]: zero for empty squares and at half a robot, peaking at a quarter, so thin squares are pushed to empty or up to half a robot. It measures about 4 to 5 on the fog swarm (so 0.4 to 0.5 after weighting, more than the task loss it earns), 0 on a compact swarm, and 1.2 on an untrained network.
- **Time term** (w_time 0.5): 1 minus the product, over remaining cheeses, of their ring coverage. A cost at every step until the last cheese is eaten.
- Evaluation now also reports fog (time-averaged share of the mass on squares holding less than half a robot).

v4-feast-ema stopped (`runs_cancelled/`). New run from scratch: v4-feast-shaped, batch 64, averaged weights, lr 3e-4, 480 minutes. The time limit stays at 6 steps per grid square.

v4-feast-shaped at 100 minutes: learning is slow but real. Training batches: eaten share 0 to 0.38, full success 11%. Val at 90 minutes: 0.998 (interp 1% success, 20% eaten). But fog share on training batches rose from 0.35 to 0.54 as eating improved, so at w_fog 0.1 the net is relearning the fog strategy, only more slowly. A second run, from scratch, goes in parallel: v4-feast-fog1, identical except w_fog 1.0. Whichever one clears the bar without fog is the recipe; the other is cancelled. This is not an ablation: at 0.1 the fog weight fails the requirement.

v4-feast-fog1 froze. By 18 minutes the loss sat at exactly 1.5 (deficit 1 + time 0.5): every robot stays put, fog 0, nothing eaten, val 1.0 at 30 minutes. Moving an untrained swarm always makes thin squares, so at a flat weight of 1.0 each move costs more than it can earn, and standing still is a dead local minimum. Cancelled (`runs_cancelled/`).

v4-feast-shaped at 150 minutes: val 0.92 (interp 34% fully eaten, extrap 6%, damage 21%), but fog 0.66 to 0.69, rising with success. It is the fog strategy again.

Recipe change: the fog weight ramps linearly from 0.1 to 1.0 over the first half of training (`--w_fog_final 1.0`, `fog_ramp 0.5`). The swarm learns to move and eat while fog is cheap, then has to compact as fog gets expensive. New run from scratch: v4-feast-fogramp, otherwise identical. v4-feast-shaped keeps running as the flat-0.1 reference until fogramp shows whether the ramp works.

v4-feast-shaped collapsed at 189 minutes. Training success had peaked near 50% (val 0.887 at 181 minutes: interp 48%, extrap 17%, damage 25%, fog 0.6 to 0.7). Then within 20 steps the loss jumped from about 1 to 3 to 5 and eating fell to zero; it never recovered. No NaN, so the divergence check didn't fire. Stopped (`runs_cancelled/`, best checkpoint kept). It ate by fog anyway, so it was never a candidate.

v4-feast-fogramp at 49 minutes: w_fog 0.28, fog 0.12 to 0.18, but nothing eaten yet (the flat-0.1 run had 15% eaten by 60 minutes). The ramp may outrun learning and freeze the swarm like fog1.

Second candidate, from scratch: v4-feast-fogdual. Fog becomes a constraint (fog share at most 0.1), and w_fog is a Lagrange multiplier updated by dual ascent (`--fog_target 0.1 --dual_lr 0.01`, starting at 0.1). It grows while fog is over target and shrinks while it is under, so a frozen swarm gets the penalty lifted and can move again. This is the standard constrained-optimization treatment, and it replaces a hand-picked weight or schedule. A compact swarm can move without fog, since a whole robot moves whole (squares fire as a unit and act in proportion to their mass), so 0.1 is reachable.

Check at about 100 awake minutes (fogramp) and 50 (fogdual):
- fogdual froze within 13 minutes (fog 0, loss exactly 1.5). The multiplier then fell to 0 by 28 minutes as intended, but the swarm stayed frozen for another 20 minutes with no fog penalty at all. Around 46 to 50 minutes it started moving again (fog 0.33, first cheese eaten), and the multiplier is rising again (0.09).
- fogramp eats a little (2% at 100 minutes, against 36% for flat 0.1) with fog 0.1 to 0.23; val fog fell to 0.02 to 0.29 but nothing is fully eaten. w_fog is 0.48 and still rising.
- Reading: fog is not just a bad strategy, it is how the untrained swarm explores. Diffused mass reaches the rings, so the loss has a gradient through it. A compact frozen swarm far from the rings gets almost none, which is why the freeze is sticky even at w_fog 0. Any fog penalty strong enough early on removes the learning signal.

At about 150 awake minutes both fog-penalty runs had failed: fogramp ate 2 to 3% with fog 0.12 (w_fog 0.66 and rising), and fogdual cycled between frozen (fog 0, loss 1.5) and moving, with the multiplier going 0 to 0.3 and back. Both stopped (`runs_cancelled/`).

Recipe change, following the exploration reading: let the swarm learn to eat with fog first, then take fog away gradually. Fog is still a constraint with a dual-ascent multiplier, starting at 0, but the target now tightens linearly from 1.0 (no limit) to 0.1 over the first 75% of training (`--w_fog 0 --fog_target 0.1 --fog_target_start 1.0 --fog_ramp 0.75`). At the flat-0.1 run's pace the limit would start to bind around 150 minutes, about when that run had learned to eat. New run from scratch: v4-feast-tighten, 480 minutes, alone on the GPU.
