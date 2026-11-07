# VLA roadmap

A language-conditioned swarm policy, trained on one laptop. The original design is in
VLA_PLAN.md; the current architecture is in VLA_ARCHITECTURE.md; run results are in
VLA_HILLCLIMB.md. Ticked items are done; dates are when they landed.

## Phase 1: world and controller

- [x] Batched torch grid world: walls, pushable and edible cheeses, floor boxes, 10-channel observation
- [x] Attractor controller (up to 4 points, straight mode, shares via an auction), with tests
- [x] Facts extractor (moves, regions, groups, pushes, eats, around, away, boxes) with atoms for scoring
- [x] Switch to discrete robots, one per square (2026-11-04): snap to 8 directions or stay, random-priority conflicts, shove rule
- [x] Controller adapted to robots: fill attractors from the inside out, prefer free squares
- [x] "Around" redefined by ring coverage, so it works at any swarm size
- [ ] Re-check scripted behaviours at g32 with robots: split, push, eat rates

## Phase 2: data

Scripted play
- [x] Scripted player with DART noise and command-oracle variants
- [x] Discrete g32 caches (regenerated on the switch; continuous caches kept apart)

LLM play
- [x] Sandboxed program runner: subprocess per episode, time limit, no network or file writes
- [x] Long-running Haiku and Luna sessions in tmux, with a brief, an allowlist and restarts
- [x] Diversity audit (play-stats) and coverage-steered nudges (play-coverage)
- [x] Near-duplicate programs down-weighted in sampling
- [x] Physics recorded in each run's meta; loaders take only discrete runs
- [x] Re-run of every program at g32 with robots, looping to pick up new programs
- [ ] Re-run caught up with the live sessions
- [ ] Second diversity check on the discrete data
- [ ] Turn-by-turn LLM play (about 100 reactive episodes)
- [ ] Human teleop episodes (the tool exists and runs at g32; nothing recorded yet)

Captions
- [x] Captioner: 5 windows per call, alternating Haiku and Luna, resumable
- [x] Tags for directions, colours and unverified claims; caption rewriter for rotations and colour swaps, failing closed
- [x] Caption set builder: augmented variants, embedding cache
- [x] Text pre-flight probe on the frozen embedding
- [x] Brief updated for robots and pooled g32 grids
- [x] Fourth, casual register; loose names for cheeses ("the blue square", "the red block") and the swarm
- [x] Two calls in flight per provider
- [x] Set build tags bare colour words and drops captions that give distances
- [x] Automatic caption audit script (numbers, length, tags, openings, object names)
- [x] Captioning at scale: 87k windows, 348k captions by 2026-11-05 (Haiku only; the loop keeps running)
- [ ] Human audit of rendered windows against facts and captions (VLA_PLAN.md, about 1.5 hours)

## Phase 3: model and training

- [x] Goal encoder, text head, backbone, flow-matching action expert, classifier-free guidance
- [x] Contrastive loss between goal and text, with masked negatives
- [x] Train and eval loop: program splits, facts-score checkpoint selection, run queue
- [x] Calibration baselines: do-nothing, goal-reading oracle, replay
- [x] Hillclimb on continuous physics (VLA_HILLCLIMB.md): training steps are the strongest lever, flow beats MSE, grid g32 chosen
- [x] Text eval scored per caption register, plus a shuffled-caption baseline
- [x] First discrete scripted run, vla-d-g32-24k (test facts 0.54)
- [x] Goal-only run on discrete LLM play, vla-d-g32-play-goal (test facts 0.50 / 0.54 at guidance 1 / 2)
- [x] Rehearsal of the full recipe on early captions: text 0.37 vs shuffled-caption 0.23, casual register as good as the others
- [ ] Main run (tabled 2026-11-05 for the generalisation study): goal and caption halves on the winning architecture, longer budget, 2 seeds
- [ ] Ablations at a fixed budget: caption half only, scripted only, text encoder from scratch
- [x] Cheese diagnostic: eats fail on per-robot control (actions near chance per robot), not localisation
- [x] Discrete action head: masked discrete diffusion (leaked through empty squares at first; fixed: only robots' actions as context; ties flow alone)
- [x] Knowledge insulation (pi0.5 follow-up): backbone action readout plus stop-gradient into the expert; much faster learning
- [x] Backbone readout as the policy, no expert (objective "readout"): test facts 0.77 vs 0.55 for flow, eats 0.68 = replay (2026-11-05)
- [x] Future-state readout alongside actions: neutral (+0.01)
- [x] Frozen-window comparison of every head (VLA_ARCHITECTURE.md)
- [ ] Text on the readout model, vla-d-g32-play-readout-world-text (running; first val text 0.45 vs shuffled 0.26)
- [ ] Seeds 1 and 2 of the readout model (dropped for now; needed before any write-up)
- [ ] Model-size curve of the readout model: 4 sizes on Kaggle T4 (running), larger sizes on the TPU (being ported)
- [ ] Self-relabelling loop (goal-conditioned supervised learning on the policy's own rollouts), then outcome weighting
- [ ] Steps-remaining head for goal awareness (learned stopping)
- [x] 18.8M flow model: 0.49 test facts at 5k steps (more efficient per step, slower per minute); superseded by the readout
- [ ] If text lags goal-conditioned: bigger frozen encoder first (Qwen3-Embedding-4B), then token-level head

## Phase 4: evaluation

- [x] Facts score with recall per hard type (push, eat, around, groups)
- [x] Inference sweep: chunk steps executed, guidance, Euler steps
- [ ] Test prompt list written before seeing results, with novelty tags (paraphrase, new combination, new concept), including the user's wild prompts ("eat every square but the red one", "group the primary colours together", "sort the colours from cool to warm")
- [ ] Shuffled and null prompt baselines on the test prompts
- [ ] Results stratified by wall density

## Phase 5: language front-end

- [ ] LLM front-end: user message plus a text rendering of the scene in, a sequence of commands in caption style out
- [ ] Measured on the wild prompt list, against the policy alone

## Phase 6: generalisation study (current focus, from 2026-11-05)

How far the readout architecture carries, within our compute (local M4 Pro, Kaggle T4 x2 and
TPU v5e-8 until the Saturday 05:30 reset).

Swarm view
- [x] Relative-position policy (vla/swarm.py): rotary positions, so attention only sees offsets; goal as end-frame tokens with positions
- [x] Swarm view: one token per robot from its 9x9 local view, all robots attending to all
- [ ] Swarm view vs global view on g32 play, vla-rel-g32-swarm and vla-rel-g32-global (running)
- [ ] Local view size: 5x5, 9x9, 15x15
- [ ] Text goals in the relative policy (a broadcast added to every token, no position)

More cells
- [ ] Global view trained on scripted g16 and g32, tested on g48 and g64 without retraining
- [ ] Mixed-size training if zero-shot falls off

New worlds, in order (one new axis each; about a week each, bounded by building the world's
controller, facts extractor and captions, not by compute)

1. VMAS, 2D continuous physics (PyTorch, batched on the GPU, built for swarms): continuous motion
   and real dynamics. Binned force/velocity readout vs binned plus offset vs a flow head; facts
   score and final placement error. Fits locally.
   - [ ] Check VMAS scenarios and API; pick a pushing / transport / formation set
   - [ ] Play controller, facts extractor, captions
   - [ ] Readout policy runs
1. 3D voxel world (ours): 26 moves plus stay, optional gravity for assembly tasks ("build a
   tower"). Sparse tokens are required, not optional: the swarm view (one token per robot) plus
   goal tokens only for non-empty patches keeps it near 300 tokens (3 to 4x today's cost). A
   dense 32^3 global view (about 1024 tokens) would cost about 15x and does not fit.
   - [ ] World, scripted controller, 3D facts (around becomes a shell; 26-connected groups)
   - [ ] 3D rotary positions; captions from facts plus slices (Haiku cannot read 3D ASCII)
1. MuJoCo MJX (MuJoCo on JAX; GPU or TPU): articulated robots (arms, legs, quadrotors) for more
   degrees of freedom. One categorical per joint per robot token; compressed chunks (FAST) or
   joint decoding if coupled joints break independent sampling. Data generation needs the TPU
   (contact-rich bodies are too slow on the Mac CPU): planned for the TPU hours after the
   Saturday reset.
   - [ ] Pick a swarm-like task (quadrotor team, several legged robots, multi-arm transport)
   - [ ] Data generation on the TPU, training locally or on the T4s

Not planned: Isaac Lab (needs NVIDIA RTX hardware; revisit only with rented GPUs).

Scale
- [ ] Training on the TPU v5e-8 (torch_xla port in progress)
- [ ] Size curve to the largest model the TPU fits in 20 hours

## Phase 7: toward the real world

- [ ] Swarm view from real sensing: what the robots collectively observe, robots knowing only relative positions, all communicating
- [ ] Decentralisation with limited communication: not planned

## Published

- [x] Architecture explainer for the team, at https://hrishik.me/beebot/explain.html
