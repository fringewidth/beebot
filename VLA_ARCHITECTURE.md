# Swarm VLA: architecture recommendation

Written overnight, 2026-11-04 to 05, from the hill-climb in VLA_HILLCLIMB.md, the cheese
diagnostic, and the pi0.5 / Knowledge Insulation (KI) / pi*0.6 / FAST / Discrete Diffusion VLA
papers. Numbers are from play g32 (discrete robots, held-out play programs); single seeds.

## The finding

The policy is the backbone. A transformer over the scene's patch tokens plus the goal tokens,
with a linear readout from each scene token into a categorical over the 9 actions of each of its
squares for each of the next 8 steps, trained with plain cross-entropy, sampled once (every
square independently), re-planned every few steps. No action expert, no flow matching, no
iterative decoding.

On held-out test programs it nearly doubles the gap closed toward the replay ceiling. The
first rows come from each run's own test windows; the frozen-window comparison of every
checkpoint is in the next table.

| Policy | Test facts (w1 / w2) | Reached (w2) | Ate (w2) | Share of replay ceiling (w2) |
| --- | --- | --- | --- | --- |
| flow expert (pi0-style), baseline | 0.500 / 0.537 | 0.26 | about 0.15 | 61% |
| flow expert + KI | 0.505 / 0.567 | 0.29 | | 65% |
| backbone readout (objective "readout") | 0.697 / 0.740 | 0.57 | 0.52 | 84% |
| backbone readout + future-state readout | 0.709 / 0.748 | 0.55 | 0.51 | 85% |
| replay of the recorded actions (ceiling) | 0.879 | 0.87 | 0.68 | 100% |

It is also cheaper (no expert: about 2.4x the steps per minute of the flow model, each sharing the GPU with one other run) and learns faster
(0.61 val facts at 30 minutes; the flow model needed about 5 hours to reach 0.48).

The cheese problem, which the thesis rests on, is mostly the action head's fault. On the same backbone
and the same 512 windows, the flow expert recalls 18% of eats and the readout 44% (both
executing 4 of 8 steps per decision); re-planning every step, the readout recalls 61%. The diagnostic had
said localisation was fine and the ring was not being closed; a categorical per-square output
closes it.

Frozen comparison: every checkpoint rescored on the same 512 test windows (play data frozen at
one cutoff; scripts/vla_inference_sweep.py, scripts/vla_final_table.sh). Execute = chunk steps
run per decision (of 8).

| Checkpoint | Decoder | Execute | Guidance | Facts | Reached | Ate |
| --- | --- | --- | --- | --- | --- | --- |
| vla-d-g32-play-goal | flow expert | 4 | 1 | 0.510 | 0.27 | 0.12 |
| vla-d-g32-play-goal | flow expert | 4 | 2 | 0.552 | 0.30 | 0.12 |
| vla-d-g32-play-goal | flow expert | 1 | 1 | 0.524 | 0.31 | 0.09 |
| vla-d-g32-play-masked-ki | masked expert | 4 | 1 | 0.606 | 0.36 | 0.30 |
| vla-d-g32-play-masked-ki | masked expert | 4 | 2 | 0.654 | 0.45 | 0.37 |
| vla-d-g32-play-masked-ki | masked expert | 1 | 1 | 0.172 | 0.05 | 0.03 |
| vla-d-g32-play-masked-ki | backbone readout | 4 | 1 | 0.675 | 0.48 | 0.30 |
| vla-d-g32-play-masked-ki | backbone readout | 1 | 1 | 0.696 | 0.48 | 0.53 |
| vla-d-g32-play-readout | readout | 4 | 1 | 0.711 | 0.48 | 0.32 |
| vla-d-g32-play-readout | readout | 4 | 2 | 0.765 | 0.57 | 0.49 |
| vla-d-g32-play-readout | readout | 1 | 1 | 0.706 | 0.48 | 0.51 |
| vla-d-g32-play-readout | readout | 1 | 2 | 0.768 | 0.57 | 0.63 |
| vla-d-g32-play-readout-world | readout + future states | 4 | 1 | 0.723 | 0.51 | 0.38 |
| vla-d-g32-play-readout-world | readout + future states | 4 | 2 | 0.751 | 0.56 | 0.44 |
| vla-d-g32-play-readout-world | readout + future states | 1 | 1 | 0.723 | 0.51 | 0.53 |
| vla-d-g32-play-readout-world | readout + future states | 1 | 2 | 0.776 | 0.59 | 0.68 |
| vla-rel-g32-global (relative positions, whole grid) | readout | 1 | 1 | 0.834 | 0.82 | 0.89 |
| vla-rel-g32-global (relative positions, whole grid) | readout | 1 | 1.5 | 0.878 | 0.91 | 0.94 |
| vla-rel-g32-global (relative positions, whole grid) | readout | 1 | 2 | 0.885 | 0.94 | 0.96 |
| vla-rel-g32-global (relative positions, whole grid) | readout | 1 | 3 | 0.891 | 0.94 | 0.95 |
| vla-rel-g32-swarm (relative positions, one token per robot, 9x9 view) | readout | 1 | 1 | 0.773 | 0.68 | 0.76 |
| vla-rel-g32-swarm (relative positions, one token per robot, 9x9 view) | readout | 1 | 1.5 | 0.857 | 0.87 | 0.87 |
| vla-rel-g32-swarm (relative positions, one token per robot, 9x9 view) | readout | 1 | 2 | 0.884 | 0.90 | 0.88 |
| vla-rel-g32-swarm (relative positions, one token per robot, 9x9 view) | readout | 1 | 3 | 0.884 | 0.91 | 0.87 |
| replay of the recorded actions | | | | 0.879 | 0.87 | 0.68 |

The masked expert collapses when it re-plans every step (0.17): a fresh sample each step
probably does not commit to one plan, so robots dither (untested). The readout does not.
Best setting: re-plan every step, guidance 2. There the readout + future-state model recalls
eats as often as the recorded play itself (0.68); its lead over the readout alone (0.776 vs
0.768) is inside the noise.

The relative-position policies (src/beebot/vla/swarm.py: 2D rotary attention, the goal as
end-frame patch tokens) move the frontier: at guidance 2 both views score 0.885 on facts, at
the replay row (0.879), and recall eats more often than replay does (replay re-runs recorded
moves in a world whose outcomes can differ, so it is a reference, not a hard ceiling). The
swarm view, where each robot sees only its 9x9 neighbourhood and robots talk through
attention, matches the whole-grid view once guided; unguided it trails (0.773 vs 0.834). The
gain over vla-d-g32-play-readout comes from two changes at once (relative positions and patch
goal tokens), not separated. This benchmark is now saturated at guidance 2: harder tasks are
needed to separate designs further.

Bigger worlds, no retraining (scripts/vla_size_transfer.py: scripted test scenes, 256 eventful
windows, re-plan every step; both models trained on 32x32 play with about 110 robots):

| Model | Grid | Facts, guidance 1 | Facts, guidance 2 | Replay |
| --- | --- | --- | --- | --- |
| global view | 32 | 0.813 | 0.871 | 0.877 |
| global view | 48 (about 250 robots) | 0.720 | 0.810 | 0.856 |
| swarm view | 32 | 0.726 | 0.856 | 0.866 |
| swarm view | 48 (about 250 robots) | 0.602 | 0.738 | 0.875 |

Both carry over to an unseen grid size and robot count with a loss; the swarm view loses more
(guidance 2: 98% to 84% of replay, against 99% to 95% for the global view), which is the
opposite of what its size-agnostic design suggests. Two causes tested at 48, guidance 2, no
retraining:

| Probe | Swarm view | Global view |
| --- | --- | --- |
| none | 0.738 | 0.810 |
| sharper attention, queries x1.1 / x1.2 / x1.35 | 0.723 / 0.708 / 0.674 | 0.832 / 0.842 / 0.839 |
| position interpolation, positions x0.8 | 0.869 | 0.885 |
| position interpolation, positions x32/48 | 0.889 | 0.879 |
| replay | 0.875 | 0.856 |

Attention spread is not the cause. Distances beyond the trained range are: squeezing rotary
positions back into it (position interpolation, as in long-context LLMs) recovers both views to
the replay score with 2.3x the robots. Training on several world sizes would remove the need.

At 64x64 (about 380 robots, 3.5x training, guidance 2): swarm view 0.540 plain and 0.844 with
positions x0.5 (replay 0.837); global view 0.564 plain and 0.785 with positions x0.5 (replay
0.835). The further from the training size, the more the swarm view's design pays off.

## How we got there (each step is in VLA_HILLCLIMB.md)

1. Masked discrete diffusion head (MaskGIT over the 9 actions per square). The first version
   leaked: revealed slots on empty squares carried the play program's smooth field, so the
   model learned to copy neighbours and got worse the longer it trained (val 0.17 to 0.01). On
   Kaggle the same leak gave 0.20 and 0.01 test facts on scripted g16. Fixed by allowing only
   robots' actions as context or targets. The fixed version learned, but alone it trailed flow.
1. KI: a per-square action CE readout on the backbone, and a stop-gradient on the keys and
   values the expert reads. Learning got much faster for both experts (flow 0.52 vs 0.36 val at
   45 minutes; masked 0.57 at 9.5k steps, with eat recall 0.25 to 0.36).
1. With the stop-gradient, only the readout trains the backbone, so the readout is itself a
   policy. Decoding from it directly beat both experts on the same backbone (0.71 vs 0.53 test
   facts on 512 frozen windows). The expert, a second randomly initialised network over token
   space, was the weakest link.
1. Objective "readout" removes the expert entirely. A second readout of each square's future
   states (empty, robot, cheese after each step) adds a world-model loss that now reaches the
   backbone; on its own it is about neutral (+0.01, inside the noise).

## Why this matters beyond the toy

This is the same lesson as FAST, KI and Discrete Diffusion VLA, pushed one step further. The
action is the output of the sequence model, in the sequence model's own currency (categorical
tokens trained by cross-entropy), at the position of the thing it controls. The separate
continuous action expert that pi0 introduced, and that pi0.5 and pi0.7 keep, is there to output
smooth continuous trajectories at high rate. When actions are tokens, it is overhead.

## The scalable form

One transformer over one stream of discrete tokens, every position predicting the categorical
of what belongs there:

| Token type | Example here | At scale |
| --- | --- | --- |
| world state, now | the observation patches | VQ/FSQ video tokens, voxel patches, point-cloud tokens |
| goal | start and end frames, or text | a goal image, a subgoal, text |
| action per actuator | each square's next 8 moves | each joint's or each robot's next k commands (FAST-style compressed for smooth control) |
| future world state | each square's state after each step | next-frame video tokens |
| text | (frozen Qwen embedding today) | the LM's own tokens |

Trained with cross-entropy on whatever is a target in that example. One-shot readout is the
special case where every target is masked. Allowing some targets to be visible (masked-token
training with a random mask ratio) turns the same network into a world model (actions visible,
futures masked), inverse dynamics (futures visible, actions masked), or a captioner (text
masked). The masked experiment tonight shows the one rule that makes that safe: a token may be
visible only if a real controller could know it at decode time.

1. **3D voxels:** voxel patches instead of square patches, a z axis in the position embedding.
   The per-cell action readout is unchanged: each voxel patch reads out its occupants' actions.
   Cost grows with the number of occupied or changing cells, so sparse tokens (only cells that
   hold or may hold an actuator) keep it tractable, the same "known actuator" rule.
1. **Real world:** a learned tokenizer for the cameras; per-actuator action tokens. For smooth
   high-rate control, compress each actuator's chunk (DCT + BPE as in FAST) into a few tokens
   rather than adding a continuous expert. Unlabelled video trains the future-state half; play
   and teleop train the action half; captions train the text half. All of it trains the same
   weights.
1. **Language model hybrid:** the actions become tokens of the LM itself. With an
   autoregressive LM, actions for all actuators of one step can be read out in parallel from the
   current positions (as here), or decoded as text-like tokens. With a discrete diffusion LM
   (LLaDA-style), text, world and action tokens all share the masked objective. Either way there
   is no second network: the trillion-parameter model you describe outputs action tokens the way
   it outputs words.

## Caveats

1. Single seeds, one world. The readout's lead (+0.17 to +0.20 facts) is many times the run to
   run noise we have measured (about 0.02 to 0.03), but the readout vs readout + future-state
   comparison is not resolved.
1. The readout samples every square independently. That works here because the policy
   re-plans every 1 to 4 steps and the swarm's state carries the coordination. A task where two
   robots must commit to complementary moves within one step, with no feedback, would need joint
   decoding (the masked head is the tool; it is implemented and leak-free).
1. Play windows are what the play programs did; the policy is bounded by their quality (the
   replay ceiling 0.88). Self-relabelling and quality conditioning are the next lever.
1. The text pathway was not trained on the new head yet (see below).

## Running at the time of writing

- vla-d-g32-play-readout-world-big (width 384, 8 layers): 0.61 val at 6k steps, level with the
  small model per step so far and about 3x slower per step.
- vla-d-g32-play-readout-world-48k: 0.63 val at 6.7k steps.
- vla-d-g32-play-readout-world-text (captions): waiting on the caption set build.

## Next, in order

1. Text on the readout model: the caption half, then the implicit-subtask loss from pi0.5
   (predict the play program's current step text, training only).
1. Model and steps scaling of the readout model (2 sizes x 2 lengths minimum) to check that it
   keeps improving with compute, the bar set in VLA_HILLCLIMB.md.
1. Trained text encoder instead of the frozen Qwen embedding (pi0 trains its VLM).
1. Self-relabelling: roll the policy out, caption what it did, add those windows (GCSL); and
   pi*0.6-style quality conditioning.
1. The masked objective on the readout model (random mask ratio on actions and futures, with the
   known-actuator rule), which turns the same network into the policy, the world model and
   inverse dynamics at once.
