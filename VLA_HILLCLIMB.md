# VLA hill-climb log

Goal: an architecture whose closed-loop goal reaching keeps improving with training compute,
model size, data and grid size. A change that wins at one small scale but not at the next is
not adopted.

## Protocol

- Task for the climb: goal-conditioned (window start and end frames), scripted play, grid 16.
  The caption half joins once the goal half climbs; it reuses whatever goal pathway wins.
- Metric: closed-loop on eventful val windows (the do-nothing baseline fails all of them).
  - main: val overlap (1 - total variation distance between pooled mass and goal mass, at the
    end of the 1.5x horizon); continuous, so less noisy than reached;
  - also: reached (any step within the horizon), reached at end, direction agreement with the
    true vs a shuffled goal, and the z gap (flow loss with shuffled minus true goal).
  - Replay ceiling 1.0; do-nothing overlap about 0.10.
- Screening: 4000 steps, 2 jobs sharing the GPU (about 50 minutes). The incumbent plateaus from
  step 3000, so a change that isn't clearly ahead of it (val overlap about 0.40, reached about
  0.09) by then is counted as a failure and stopped. The val curve is read, not only its end.
- Scaling check before adopting a change: rerun the winner and the incumbent at the next scale
  (3x steps, and 2x width and depth), and adopt only if the gap holds or grows.
- Test windows are only for the final numbers, never for choosing.
- Bitter-lesson rule: prefer changes that remove a bottleneck or add compute or data over
  changes that add task knowledge.

## Diagnosis so far

At g16 with one goal token (mean or attention pooled), training loss falls smoothly (0.29 at
step 2000, 0.21 at 8000) but val overlap stops at about 0.39 from step 3000, and the z gap
stops growing at about 0.03. The extra fitting goes into goal-independent parts of the field.

Candidate causes:

1. goal bottleneck: one 64-d vector for a whole end frame;
1. closed-loop drift: the policy imitates states from the player's own trajectories and never
   sees how to recover from its own errors (covariate shift), compounding over 8 to 48 steps;
1. loss dominated by goal-independent structure (routing around walls, empty squares);
1. inference settings: 4 of 8 chunk steps executed open loop, guidance, Euler steps.

## Runs

| Run | Change | Steps | Val overlap | Val reached | Test overlap | Test reached | Agreement true / shuffled | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vla-r16-mean1 | 1 goal token, mean pool | 8000 | 0.36 (peak 0.41) | 0.09 (best 0.125) | 0.379 | 0.094 | 0.50 / 0.35 | Incumbent. Plateau from step 3000. |
| vla-r16-attn1 | 1 goal token, attention pool | 8000 | 0.38 | 0.09 | 0.363 | 0.096 | 0.51 / 0.36 | Tie with mean. |
| vla-r16-attn4 | 4 goal tokens, attention pool | 5575 (stopped) | 0.41 | 0.06 | 0.396 | 0.076 | 0.48 / 0.34 | Same plateau. More pooled tokens don't help. |
| vla-r16-mean4 | 4 goal tokens, mean pool | 5177 (stopped) | 0.40 | 0.07 | 0.357 | 0.100 | 0.51 / 0.35 | Same plateau. |

Queue, in order:

1. vla-r16-attn4, vla-r16-mean4: 4 goal tokens (running);
1. vla-r16-frames: the end frame as 64 goal tokens, no bottleneck (the test of cause 1);
1. vla-r16-dart10, vla-r16-dart30: 10% / 30% of squares execute a random action while the clean
   one is recorded (the test of cause 2);
1. vla-r16-big: width 384, expert 256, 8 layers, 18.8M parameters (model-size scaling);
1. grid probe 24, 32, 48, and g16 at 24000 steps (training-steps scaling).

### Inference sweep on vla-r16-mean1 (no training; 256 val windows, SE of reached about 0.02)

| Chunk steps executed | Guidance | Euler steps | Overlap | Reached | Reached at end |
| --- | --- | --- | --- | --- | --- |
| 4 (default) | 1 | 10 | 0.401 | 0.082 | 0.023 |
| 1 | 1 | 10 | **0.433** | **0.129** | 0.027 |
| 2 | 1 | 10 | 0.413 | 0.106 | 0.016 |
| 8 | 1 | 10 | 0.382 | 0.066 | 0.020 |
| 4 | 2 | 10 | 0.365 | 0.109 | 0.004 |
| 4 | 3 | 10 | 0.354 | 0.109 | 0.023 |
| 4 | 1 | 30 | 0.373 | 0.059 | 0.008 |
| 1 | 2 | 10 | 0.373 | 0.125 | 0.008 |

- Replanning more often helps monotonically (8 > 4 > 2 > 1 steps per decision): open-loop error
  compounds, which points at cause 2 (closed-loop drift). The gain is real but small; the
  training-side test is the DART runs.
- Guidance above 1 lowers overlap; more Euler steps don't help.
- Reached at end is far below reached: the policy passes through the goal and keeps going. Training
  never shows what to do once the goal is reached (chunk steps past the window's end get weight
  0), so it has no notion of stopping there. Noted; reached-at-any-step is the plan's success
  measure, so this is secondary for now.

## Metric calibration (same 256 val windows for every row)

The "reached" metric was miscalibrated. A goal-reading oracle with no learning (the scripted
controller, one attractor per goal mass blob from weighted k-means, shares by blob mass; it never
pushes on purpose) reaches only 7% of windows, and its mass alone meets the 0.75 overlap
threshold in only 11%. 54% of val windows involve a cheese push or eat. "Reached" asks for
pixel-level precision that the end frame barely pins down, so it can't rank architectures.

New main metric, the facts score: the goal's facts (window_facts from start to goal: the move's
direction and end region, number of groups, which cheeses were eaten or pushed which way, boxes
entered or left, gathering around or moving away from a cheese) against the rollout's facts, as
the Jaccard overlap of their atom sets; best within the 1.5x horizon ("fact score") and at its
end ("end"). It is what captions describe, so it is also what prompts will ask for. Per-window
spread puts its SE at about 0.02.

| Policy | Execute | Fact score | Fact score at end | Overlap | Reached |
| --- | --- | --- | --- | --- | --- |
| do nothing | | 0.00 | 0.00 | 0.115 | 0.00 |
| goal-reading oracle | | 0.486 | 0.460 | **0.577** | 0.074 |
| replay (ceiling) | | 1.00 | 1.00 | 1.00 | 1.00 |
| vla-r16-mean1 (8000 steps) | 4 | 0.561 | 0.407 | 0.414 | 0.086 |
| vla-r16-mean1 | 1 | **0.607** | 0.433 | 0.438 | 0.113 |
| vla-r16-attn1 (8000) | 4 | 0.554 | 0.398 | 0.372 | 0.066 |
| vla-r16-attn1 | 1 | 0.557 | 0.424 | 0.422 | 0.102 |
| vla-r16-mean4 (5177) | 4 | 0.558 | 0.406 | 0.355 | 0.055 |
| vla-r16-mean4 | 1 | 0.556 | 0.392 | 0.373 | 0.098 |
| vla-r16-attn4 (5575) | 4 | 0.587 | 0.415 | 0.397 | 0.074 |
| vla-r16-attn4 | 1 | 0.568 | 0.430 | 0.436 | 0.102 |
| vla-r16-frames (4000) | 4 | 0.584 | 0.390 | 0.341 | 0.051 |
| vla-r16-frames | 1 | **0.607** | 0.432 | 0.377 | 0.109 |
| vla-r16-dart10 (4000) | 4 | 0.586 | 0.398 | 0.356 | 0.051 |
| vla-r16-dart10 | 1 | 0.578 | 0.415 | 0.397 | 0.094 |

Reading:

vla-r16-dart30 (30% of squares execute a random action, 4000 steps; scored by its own run on
128 val / 512 test windows, not the 256-window set above): val fact score 0.59, end 0.41,
overlap 0.36; test fact score 0.573, end 0.395, overlap 0.325, reached 0.076. Same band.

vla-r16-hold (10% of goal-half samples start at the goal frame, labelled with the zero field,
4000 steps): val fact score 0.59 / end 0.42 at step 4000 (best checkpoint 0.61 / 0.39 at step
2246); test fact score 0.549, end 0.367, overlap 0.31. The end score did not improve: stopping
on the exact goal frame doesn't carry over to the near-goal states the policy actually reaches.
Failed as tried.

vla-r16-big (width 384, expert 256, 8 layers, 18.8M parameters, 4000 steps): training loss the
lowest of any run (0.18 vs 0.21 for the 3.75M model at 8000 steps), but test fact score 0.565,
end 0.409, overlap 0.36, direction agreement 0.46 / 0.35. Same band: 5x the parameters buys
loss, not closed-loop behaviour.

Two more checks on vla-r16-mean1:
- Train vs val windows (256 each): fact score 0.540 vs 0.583, overlap 0.389 vs 0.386. No
  overfitting; it can't do its own training windows better either.
- Snapping each generated vector to the stored action set (8 directions or zero) before
  executing: no help (execute 4: fact 0.561 to 0.551; execute 1: 0.582 to 0.566).

Rollout strips (scripts/vla_rollout_strips.py) show what goes wrong: recorded play makes crisp
shapes (lines along walls, solid groups) and pushes or eats specific cheeses; the policy moves
the right general way but its mass smears over partly filled squares (neighbouring squares
point different ways), and it sometimes goes for the wrong cheese. Smeared mass presses too
weakly to push and covers too little to eat.

Grid probe, same recipe (1 goal token, mean pool) and 8000 steps, test windows (512):

| Grid | Test fact score | End | Overlap | Reached | Do-nothing overlap |
| --- | --- | --- | --- | --- | --- |
| 16 (vla-r16-mean1, rescored) | 0.546 | 0.376 | 0.383 | 0.080 | 0.096 |
| 24 (vla-probe2-g24) | 0.564 | 0.393 | 0.367 | 0.080 | 0.126 |
| 32 (vla-probe2-g32) | 0.614 | 0.402 | 0.381 | 0.090 | 0.137 |
| 48 (vla-probe2-g48) | 0.506 | 0.359 | 0.381 | 0.059 | 0.186 |

g48 starts slowest and was still rising (val 0.30, 0.40, 0.43, 0.53, 0.48, 0.56), but at equal
compute it trails: fact score 93% of g16's, overlap above the do-nothing baseline only 68% of
g16's, eat recall 0.16 and push recall 0.26. Its play data would also be about 11 GB at 8000
episodes. Choice: g32, the best of the four at equal compute, with room for its data; g32 at
24000 steps is running to confirm it at the main-run budget.

g32 learns more slowly (val fact score 0.35 at step 1500 vs about 0.52 for g16; each action token
covers 4x4 squares, 128 numbers) but climbs steadily (0.40, 0.43, 0.52, 0.56) and ends level with
or above g16 on test, still rising. Bigger patches slow early learning without capping it.

g24 holds g16's level (fact score slightly higher, within noise), at the same compute: grid
resolution costs nothing so far. g32 and g48 are queued.

Next diagnostic, vla-r16-cmd: the same policy also sees the scripted player's commands at the
current step (attractor points, shares, straight flags) as 3 extra input channels: perfect
knowledge of intent. If its rollouts are still smeared, the limit is action generation; if they
turn crisp and the fact score jumps, the limit is that goal frames don't pin down the intended
behaviour, and the fix belongs in how goals are specified.

vla-r16-cmd result (4000 steps): direction agreement 0.74 with the true goal and 0.71 with a
shuffled one (vs 0.50 / 0.35 without commands), so the policy uses the commands; but test fact
score only 0.578 (vs 0.546), overlap 0.384, end 0.393. Even with perfect intent a quarter of the
moving mass steps the wrong way, and closed-loop barely improves. So the main limit is action
generation (reproducing the controller's field: shortest paths around walls, mass shares), and
goal ambiguity is secondary.

Next: is it flow matching's sampling? With commands the target is deterministic, so a direct
regression head (objective mse) should do well if the network can represent the field.
Queued: vla-r16-cmd-mse (commands + regression) and vla-r16-mse (the plan's regression-head
ablation, goal frames only).

### Training steps scale (vla-r16-mean1-long, 24000 steps, same recipe as vla-r16-mean1)

| Test (512 windows) | 8000 steps | 24000 steps |
| --- | --- | --- |
| Fact score | 0.546 | **0.607** |
| Fact score at end | 0.376 | **0.460** |
| Overlap | 0.383 | **0.457** |
| Reached | 0.080 | **0.145** |
| Reached at end | 0.014 | **0.068** |

Val was still rising at the last step (fact score 0.65, end 0.50, overlap 0.50; the goal-reading
oracle's overlap is 0.58). The plateau seen in the 8000-step runs was a compute limit, not an
architecture ceiling: most of the gain arrives late, as the learning rate decays over the longer
schedule. Every screening comparison above was made at 4000 to 8000 steps, so those changes may
deserve a second look at longer budgets. Queued: 48000 steps.

vla-r16-cmd-mse (commands as input, direct regression head, 4000 steps): direction agreement
0.91 (vs 0.74 for the flow head with the same input), test reached 0.209, overlap 0.515, end
0.465, fact score 0.571. With a deterministic target the network represents the controller's
field well; flow matching gets there much more slowly at this budget. The fact score stays at
0.57 to 0.61 even here while overlap and reached move a lot, so it is probably too coarse (3 to 4
atoms per window) or dominated by hard atoms; to inspect before leaning on it further.

vla-r16-mse (goal frames only, direct regression head, 4000 steps): test overlap 0.431 and end
0.440 (higher than the flow head's 0.383 / 0.376 at 8000), but fact score 0.497, and on the 256
val windows eat recall collapses to 0.05 (flow, 24000 steps: 0.43) and push recall is 0.23
(0.32). With an ambiguous target, regression averages possible behaviours into weak, blurry
fields: good for averaged mass overlap, unable to concentrate enough mass to eat or push. Flow
matching keeps the decisive behaviour, and language goals are more ambiguous still, so the flow
head stays. (With commands, where the target is deterministic, regression was far ahead at this
budget, so a flow head that trains faster would be worth looking for.)

### Fact breakdown (vla-r16-mean1-long, 256 val windows; scripts/vla_fact_breakdown.py)

| Atom type | In goals | Recall | Extra in rollout |
| --- | --- | --- | --- |
| move direction | 226 | 0.88 | 26 |
| end region | 226 | 0.84 | 34 |
| away from a cheese | 74 | 0.77 | 7 |
| number of groups | 104 | 0.62 | 22 |
| around a cheese | 59 | 0.53 | 10 |
| ate a cheese | 74 | 0.43 | 19 |
| pushed a cheese (and direction) | 94 | 0.32 | 63 |
| box entered / left | 13 | 0.8 to 1.0 | 1 |

Where the swarm goes is largely solved; cheese interactions are the weak spot, pushes worst.
That is why the overall fact score moves little: the easy atoms sit near their ceiling and the
hard ones are a small share. From here, push and eat recall are reported as their own lines.

- Semantically the policy already beats the oracle (0.56 to 0.61 vs 0.49): it does the right kind
  of thing (right direction and region, right cheese) more often than steering at goal blobs does,
  though its mass shape is looser (overlap 0.41 vs 0.58).
- Every variant so far lands in the same band (0.55 to 0.61, within about 2 SE). Goal pooling, goal
  tokens, removing the goal bottleneck, and 10% action noise all change little. The 4000-step runs
  match the 8000-step ones.
- The end score is well below the best score: the policy reaches the right outcome and then keeps
  going (no notion of stopping at the goal).
- Since no goal-side or data-side change moves the band, the next tests are the scaling axes:
  model size (vla-r16-big), training steps (vla-r16-mean1-long, 24000) and then grid size.
  Checkpoints are now chosen on the val fact score.

## Switch to discrete robots (2026-11-04)

Everything above ran on continuous mass (fractional mass per square, capacity 4). From here the
world holds discrete robots, one per square, which is what real hardware is. The model, tokens and
action space are unchanged: the policy still outputs a velocity field, and the world snaps each
robot's vector to the nearest of 8 directions, or stay below |v| = 0.5. Robots wanting the same
square are resolved in rounds with random priority; a line moving together advances; swaps and
cycles wait. Pushing still sums the robots pressing on a cheese against friction; a shove needs a
free square beyond the shoved robot.

- Controller: attractors fill outward from the inside (a robot is settled when no empty square
  is nearer the attractor), and among equally short moves robots prefer free squares.
- Replay is no longer exact (random conflict order): facts 0.92 to 0.95, reached 0.86 to 0.91 on
  scripted val, eat recall 0.73 to 0.87 (one late robot breaks a ring). This is the new ceiling
  for open-loop replay; the closed-loop policy can correct where replay cannot.
- Continuous results above are reference only. Old scripted caches moved to
  data/vla/scripted/continuous/; play runs now record "physics" in meta.json and loaders take
  only "robots". All LLM play programs are being re-run at g32 with discrete robots into
  /Users/hrishik/beebot-play-g32 (vla_rerun_play.sh in a loop, so new programs follow).
- vla-r16-mean1-48k (continuous) was left to finish for the steps-scaling curve;
  vla-probe3-g32-24k (continuous) was stopped for vla-d-g32-24k, the first discrete run (g32,
  scripted, 24000 steps).
- "around" fact redefined: at least half of the cheese's open ring squares hold a robot (eating
  takes three quarters), instead of 30% of the swarm on the ring. A ring holds at most one robot
  per square, so the old rule could almost never fire for a g32 swarm (recall_around was None on
  the first discrete val). Now 12% of scripted g32 val windows carry it. vla-d-g32-24k was
  restarted on the new definition.

### Steps scaling to 48000 (vla-r16-mean1-48k, continuous physics, g16; test windows, 512)

| Steps | Guidance | Fact score | End | Overlap | Reached |
|---|---|---|---|---|---|
| 24000 (vla-r16-mean1-long) | 1 | 0.607 | 0.460 | 0.457 | 0.145 |
| 24000 | 2 | 0.641 | 0.508 | 0.521 | 0.207 |
| 48000 | 1 | 0.663 | 0.539 | 0.513 | 0.213 |
| 48000 | 2 | 0.691 | 0.570 | 0.575 | 0.283 |

Doubling steps again still pays (+0.05 to +0.06 fact score, reached up by half), and guidance 2
beats 1 at both budgets. Val fact score was still at its best around step 37000 to 42000. So the
main run goes long (about 64000 steps, overnight) and is reported at guidance 2 as well as 1.

### Why eats fail (scripts/vla_cheese_diagnostic.py on vla-d-g32-play-goal, held-out test programs)

Closed loop, 217 cheeses the goal eats (160 windows): policy eats 14%, replay 73%. Of the policy's
failures, 90% got within 2 squares of the right cheese and 4% ate another cheese instead, so
localisation is fine; best ring cover stalls below half in 74% of failures (replay's failures are
all 50 to 75%, a late robot).

Ring closing from the recorded state a few steps before the recorded eat (run for lead + 8 steps):

| Lead | Recorded actions eat | Policy eats |
|---|---|---|
| 4 | 93% | 14% |
| 8 | 88% | 8% |
| 12 | 82% | 6% |
| 16 | 77% | 8% |

The policy can't close a ring even when it is nearly closed, so this is not compounding error.
The goal frame not showing the ring explains a little (lead 4: 21% when the eat is within 4 steps
of the window end, 12% otherwise; no difference at lead 12).

Open loop on recorded states: the policy's first-step direction matches the player's on 16% of
robots next to the target cheese and 21% elsewhere, close to chance (about 13 to 15% for 9
actions at these stay rates). Near the cheese the player holds 39% of robots still and the policy
27%, and the policy's stays overlap the player's on 25%, which is chance. The policy gets the
bulk flow right while each robot's action is close to noise: fine for moves, fatal for eats and
pushes, and the same flaw as not stopping at the goal.

Hypothesis: with discrete robots each square's action is a 9-way choice, but the expert still
generates a continuous vector that the world snaps; leftover sampling noise flips choices at
random. Next (2026-11-05): a discrete action head (masked discrete diffusion over 9 actions per
square), compared against the main run. The self-relabelling loop comes after it.

### Masked discrete action head: a context leak, and its fix (2026-11-04 evening)

The masked head (objective "masked", MaskGIT over the 9 actions per square and step) as first
written collapsed on play g32: val facts 0.17, 0.11, 0.06, 0.01 at 15, 30, 45, 60 minutes, while
flow climbed past 0.48 over the same time. The masked + world-model variant never left 0.03.
Training loss kept falling (0.54 nats at step 6000; chance is ln 9 = 2.2).

Cause: the action label is the program's field on every square, robot or not. Programs write
smooth fields, so the revealed slots on empty squares give away the program's intent next to
every robot. On the masked checkpoint, the loss on masked slots is 1.89 with everything masked
and 0.83 with 10% revealed: the model learned to copy its neighbours rather than read the scene
and the goal. At sampling, the first round sees nothing, so it is near the marginals, and every
later round copies it. More training meant more copying, so val fell.

A first fix (label stay on empty squares, train on every square) removed the leak but made the
loss mostly about predicting which squares stay empty. At 8% occupancy, empty squares outweigh
robots 12 to 1, and a one-batch memorisation test no longer learned the robots (36% agreement).

Fix kept: only robots' actions are ever context or targets. An action slot is revealed only
where a robot is known to stand before that step: at step 0 from the scene, and at step k > 0
from a revealed future state of step k - 1 (so only with world_model). The sampler follows the
same rule, using the future states it has itself decoded so far. The program's field on empty
squares is never seen. Memorisation agreement on robot squares is back to 99.7%. A regression
test checks that scrambling the field on empty squares leaves the loss unchanged.

The general lesson for any masked or autoregressive action model: the context must contain only
what the policy could know at decode time. A dense field label sneaks in information (the
program's intent at places where nothing acts) that no real controller would ever have.

Runs: vla-d-g32-play-masked2 and vla-d-g32-play-world2 (play g32, 24k steps), and on Kaggle
k16-masked2 vs k16-flow (scripted g16, 24k steps). Kaggle's k16-masked and k16-masked-world
use the leaky code; they could not be cancelled, so they stand as a negative control.

### Knowledge insulation, and the backbone readout as the policy (2026-11-05, night)

KI (pi0.5 Knowledge Insulation): the backbone gets its own per-square action cross-entropy
readout, and the expert reads the backbone's keys and values through a stop-gradient. With the
stop-gradient, only the readout trains the backbone and the goal encoders.

Val facts during training (play g32, 128 val windows; the play data grows between runs, so the
val windows differ slightly between runs):

| Run | Expert | 45 min | step 3.5k | step 9.5k | step 12k | end |
| --- | --- | --- | --- | --- | --- | --- |
| vla-d-g32-play-goal | flow, no KI | 0.36 | 0.41 | about 0.49 | | test 0.500 / 0.537 (w1 / w2) |
| vla-d-g32-play-insulate | flow, KI | 0.52 | 0.45 | about 0.53 | 0.53 | test 0.505 / 0.567 |
| vla-d-g32-play-masked2 | masked, no KI | 0.32 | 0.32 | | | stopped at 5.8k, flat at 0.30 |
| vla-d-g32-play-masked-ki | masked, KI | 0.43 | 0.45 | 0.57 | 0.55 | stopped at 14k; eat recall 0.25 to 0.36 |

KI speeds learning a lot (both experts), and the masked expert eats far better than the flow
expert (eat recall 0.25 to 0.36 vs about 0.10).

Then the surprise. With the stop-gradient, the backbone is trained only by the readout, so the
readout is itself a complete policy: a transformer whose scene tokens are read out directly
into each square's 8 actions, sampled once, independently per square. On the insulate
checkpoint, same 512 test windows (scripts/vla_inference_sweep.py):

| Decoder | Execute | Guidance | Facts | Reached | Pushed | Ate | Around | Groups |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flow expert | 4 | 1 | 0.530 | 0.30 | 0.37 | 0.18 | 0.28 | 0.72 |
| flow expert | 4 | 2 | 0.587 | 0.35 | 0.30 | 0.21 | 0.49 | 0.73 |
| backbone readout, one-shot | 4 | 1 | 0.678 | 0.47 | 0.38 | 0.44 | 0.50 | 0.79 |
| backbone readout, one-shot | 1 | 1 | 0.706 | 0.49 | 0.33 | 0.61 | 0.62 | 0.84 |

The simplest policy wins by a wide margin, and eats go from 0.18 to 0.61. The expert, reading the
same backbone, loses information: a randomly initialised second network over token space is a
worse decoder than a linear map from the scene tokens, at least at this scale. Independent
per-square sampling does not stop the ring from closing once the policy re-plans every step.

Follow-ups (running): objective "readout" (no expert at all; cheaper, and its val now scores
the readout itself), and readout plus a future-state readout (world_model with "readout": each
square's state after each step, read out from the same tokens, so the world-model loss now
shapes the backbone, which it could not through the insulated expert).
