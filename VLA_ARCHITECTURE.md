# Swarm VLA: architecture recommendation

Written overnight, 2026-11-04 to 05, from the hill-climb in VLA_HILLCLIMB.md, the cheese
diagnostic, and the pi0.5 / Knowledge Insulation / pi*0.6 / FAST / Discrete Diffusion VLA
papers. The results section is filled from the overnight runs; everything above it is the
argument.

## The claim

One transformer over one stream of discrete tokens: world state, actions, and text. It is trained
with a single objective, masked-token infilling (absorbing-state discrete diffusion). Policy,
world model, inverse dynamics, goal reaching and captioning are all the same operation: fill in
the masked tokens given the visible ones. They differ only in which tokens are masked at
inference.

| Use | Visible | Masked (decoded) |
| --- | --- | --- |
| goal-conditioned policy | now, goal | actions (and, optionally, future states) |
| language-conditioned policy | now, text | actions, future states |
| world model | now, actions | future states |
| inverse dynamics | now, future states | actions |
| captioning / hindsight labels | now, future states | text |
| planning by imagination | now, goal | future states first, then actions |

This is the "Attention is all you need" move for embodied control: take away the special-purpose
pieces (a separate flow expert, a separate world model, a separate goal encoder) and let one
model and one objective see every modality as tokens. Discrete diffusion LMs (LLaDA, Dream) give
the language side of the same objective, so a language model and this policy can be one network
rather than two stuck together.

## Why discrete and masked, not continuous flow

1. Our actions are categorical: a robot moves to one of 8 neighbouring squares or stays. Flow
   matching on a 9-way categorical variable fits a continuous density to something that has
   none, and then has to snap back. A cross-entropy on the categories is the natural loss.
   FAST makes the same point for continuous robot actions: tokenise, and you train 5x faster.
1. Knowledge Insulation (KI): a flow-only pi0 needs about 7.5x more steps than models whose backbone
   is trained with a discrete action CE; the randomly initialised flow expert's gradients slow
   the backbone and erode language following. Discrete CE is the signal that trains
   representations quickly.
1. Masked decoding models the joint distribution over robots. A ring around a cheese needs
   every square of the ring filled. If every robot samples independently from its marginal, the
   ring fails even when each robot's marginal is right. In MaskGIT-style decoding, later robots
   condition on earlier ones' committed moves, which is how the swarm can agree on who goes where.
1. The same objective covers world tokens (square states), so the world model costs nothing
   extra and shares the representation.
1. Scaling evidence outside our setup: Discrete Diffusion VLA matches continuous diffusion and
   beats FAST autoregression, and it is more robust under perturbation.

## The rule we learned the hard way: context is only what the policy could know

Our first masked head leaked. The action label was a dense field over all squares, including
empty ones, where the play program's smooth field gives away its intent. The model learned to
copy revealed neighbours instead of reading the scene, and got worse the longer it trained (see
VLA_HILLCLIMB.md, "Masked discrete action head: a context leak"). The fix generalises: in any
masked or autoregressive action model, a token may be visible only if a real controller could
have it at decode time. For us that means an action token exists only where an actuator (a
robot) is known to be. At step 0 that comes from the observation; at later steps it comes from
the model's own decoded future state. In the real world it is the same rule: per-joint action
tokens for actuators that exist, and never labels derived from a planner's internals.

## Architecture (the scalable form)

1. **Tokens.**
   - World: one token per spatial cell per time step, for the state of that cell (empty,
     robot, cheese of colour c, box, wall). In 2D, a patch of cells is one transformer position
     whose embedding sums its cells (the patch is the compression knob; at scale, a learned
     VQ/FSQ tokenizer replaces the hand-made cell states, as in video models).
   - Actions: one token per actuator per step (here, per robot per step, 9 classes).
   - Text: the LM's own tokens.
1. **Backbone.** One transformer. Positions get space-time embeddings (rotary over x, y, (z), t).
   The prefix (observation, goal, text) uses bidirectional attention. Today the action and
   future tokens attend to the prefix's keys and values, as in pi0's action expert. At scale
   they are the same transformer with a block-causal mask: prefix → future → actions.
1. **Objective.** Absorbing-state masked diffusion: draw a mask ratio, mask that share of the
   target tokens (actions, future states, and optionally text), and minimise cross-entropy on the
   masked ones. Classifier-free guidance by dropping the goal or text 10% of the time.
1. **Backbone grounding (KI).** The backbone also gets a direct per-cell action CE readout, and
   the action expert reads it through a stop-gradient. This is under test tonight
   (vla-d-g32-play-insulate). In the unified form this becomes unnecessary because there is one
   network and one CE.
1. **Decoding.** Iterative confidence-ordered unmasking (MaskGIT), 4 to 12 rounds, with the
   "known actuator" rule above. Futures can be decoded before actions, which is planning by
   imagination with no extra machinery.
1. **Execution.** Receding horizon: decode a chunk of 8 steps, execute the first few, re-plan.

## Path to 3D voxels and the real world

1. **3D voxels.** Cells become voxels, patches become 3D patches, rotary embeddings get a z
   axis. Nothing else changes. Token count grows as side cubed. That is the scaling cost: patch
   size and sparse tokens (only occupied or changing voxels, i.e. the same "known actuator /
   relevant cell" rule) keep it tractable. A 64^3 world with 4^3 patches is 4096 positions per
   frame, the size of a 64x64 image transformer.
1. **Real sensors.** Replace the hand-made cell states with a learned tokenizer (video VQ/FSQ
   or a frozen video encoder's discrete codes) over camera frames or point clouds. The policy
   never needs the true occupancy grid; it needs tokens that predict the future, which the
   world-model half of the objective enforces.
1. **Real actuators.** One token stream per actuator: discretised joint targets or FAST-style
   compressed action chunks (DCT + BPE) for smooth continuous control. The discrete objective
   is unchanged. For swarms of physical robots, each robot's command is its own token, so the
   joint decoding across robots is exactly what the masked objective gives.
1. **Language.** Start from a discrete diffusion LM (or keep an autoregressive LM for text and
   use the masked objective for the world and action tokens; pi0.5 shows the hybrid works).
   Text tokens are just more masked tokens: "describe what will happen" and "do what I say"
   are the same infilling problem in different directions.
1. **Data.** The model learns from unlabeled video (world tokens only: train the world model
   half), from teleoperation and play (world and action tokens), and from captioned windows
   (all three). Every source trains the same weights, which is what lets it scale past the robot
   data bottleneck.

## What stays from today's design

- Goal from frames and goal from text into the same conditioning slot, with the text encoder
  trained (pi0 trains its VLM; ours will move from a frozen Qwen embedding to a trained
  encoder once there is enough text).
- Classifier-free guidance.
- Hindsight relabelling of play windows, with captions at four tiers, and the facts score as the
  metric.
- Receding-horizon chunks.

## Results (overnight)

Filled in from the runs below. Test windows, held-out programs, facts score (higher is
better), w = guidance weight.

| Run | Head | Data | Steps | Test facts w1 | Test facts w2 | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| vla-d-g32-play-goal | flow | play g32 | 24000 | 0.500 | 0.537 | incumbent |
| vla-d-g32-play-big | flow, 18.8M | play g32 | 5159 | 0.493 | 0.449 | stopped by time budget |
| vla-d-g32-play-masked | masked (leaky) | play g32 | 6156 | | | collapsed; see the leak |
| vla-d-g32-play-masked2 | masked | play g32 | | | | |
| vla-d-g32-play-world2 | masked + world model | play g32 | | | | |
| vla-d-g32-play-insulate | flow + KI | play g32 | | | | |
| k16-flow (Kaggle) | flow | scripted g16 | | | | |
| k16-masked2 (Kaggle) | masked | scripted g16 | | | | |

## What I would run with more compute

1. The unified single-transformer form (no separate expert) at 2 sizes, to check the loss and
   the facts score keep improving with size.
1. Text conditioning on the winning head, plus the implicit-subtask loss from pi0.5: predict the
   play program's current step text from the backbone during training only.
1. Self-relabelling: roll the policy out, caption what it actually did, add those windows to
   training (GCSL). This is the loop that turns compute into data.
1. Quality conditioning (pi*0.6's advantage token): label each window by efficiency, condition
   on it with dropout, ask for "efficient" at test time.
