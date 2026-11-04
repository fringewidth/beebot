# VLA roadmap

A language-conditioned swarm policy, trained on one laptop. The design is in VLA_PLAN.md; run
results are in VLA_HILLCLIMB.md. Ticked items are done; dates are when they landed.

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
- [ ] Captioning at scale: 10 to 15k windows for the first main run, then all of them
- [ ] Human audit of rendered windows against facts and captions (VLA_PLAN.md, about 1.5 hours)

## Phase 3: model and training

- [x] Goal encoder, text head, backbone, flow-matching action expert, classifier-free guidance
- [x] Contrastive loss between goal and text, with masked negatives
- [x] Train and eval loop: program splits, facts-score checkpoint selection, run queue
- [x] Calibration baselines: do-nothing, goal-reading oracle, replay
- [x] Hillclimb on continuous physics (VLA_HILLCLIMB.md): training steps are the strongest lever, flow beats MSE, grid g32 chosen
- [x] Text eval scored per caption register, plus a shuffled-caption baseline
- [ ] First discrete scripted run, vla-d-g32-24k (running)
- [ ] Goal-only run on discrete LLM play, vla-d-g32-play-goal (queued)
- [ ] Rehearsal of the full recipe on the captions available around 13:00 (24k steps): catches caption-half bugs, first read on text vs goal and per register
- [ ] Main run, overnight: goal and caption halves on discrete g32 play with nearly all windows captioned, about 64k steps, 2 seeds side by side (the continuous 48k run was still improving past 40k)
- [ ] Ablations at a fixed budget: caption half only, scripted only, text encoder from scratch
- [x] Cheese diagnostic: eats fail on per-robot control (actions near chance per robot), not localisation
- [ ] Discrete action head: masked discrete diffusion over 9 actions per square (first experiment after the main run)
- [ ] Self-relabelling loop (goal-conditioned supervised learning on the policy's own rollouts), then outcome weighting
- [ ] Steps-remaining head for goal awareness (learned stopping)
- [ ] 18.8M model at a long budget (vla-d-g32-play-big running)
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

## Phase 6: toward the real world

- [ ] Swarm's fused view instead of the world's-eye view (observed mask, last-seen map with its age), only after the world's-eye policy is trained
- [ ] Decentralisation: not planned, one central policy throughout

## Published

- [x] Architecture explainer for the team, at https://hrishik.me/beebot/explain.html
