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
