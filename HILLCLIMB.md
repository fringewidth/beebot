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
