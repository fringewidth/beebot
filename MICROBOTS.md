# Microbots: plan

The destination: thousands of identical modules that form any shape on command, use those
shapes to do work, and move as one body, steered by language or a goal image (the microbots in
Big Hero 6). This is the plan for getting there from where Beebot is (2026-11-05): a 2D swarm
policy that is one transformer with a categorical readout per robot, reaching unseen goals,
with a swarm's-eye variant (one token per robot, local view, relative positions) in training.

Everything here is simulation. Real hardware is out of scope until the simulated capabilities
exist.

## Capabilities, in order (each is a milestone and a demo)

1. Form any shape on command: assemble a 3D structure (tower, arch, bridge, ring, hand) and hold
   it under gravity. Metric: overlap with the target (IoU), and success on shapes and words never
   seen in training.
1. Structures that do work: span a gap and carry an object across; make a ramp, a lever, a
   conveyor wave. Metric: task success rate.
1. Move as a body: roll, crawl or flow as one connected structure to a goal.
1. From intent: language or a goal image throughout, never a hand-written script.

## Data: generators that scale with compute, not with hand-written behaviours

LLM play programs worked in 2D but will not invent structural engineering. Three generators:

1. Shapes from shape libraries, assembled in reverse.
   - Targets: procedural primitives (arches, stairs, rings, bridges, overhangs) and voxelised
     ShapeNet / Objaverse objects for diversity.
   - A feasible build order comes from taking the target apart under gravity (remove one module
     at a time while the rest stays supported), then reversing it: assembly by disassembly,
     guaranteed feasible, no learning needed.
   - Every intermediate structure is also a goal (hindsight relabelling), as in 2D.
1. Functional tasks from search.
   - Tasks like "get the cheese across the gap" are solved in a fast batched simulator by RL or
     evolutionary search over structures; successful episodes are kept and relabelled.
   - The search invents, the policy imitates: the same play-then-relabel recipe with search
     as the player.
1. LLMs as architects.
   - An LLM writes construction programs over a shape-and-assemble API ("build an arch, then
     roll the ball over it"): compositional, language-rich episodes.
   - Captions from a VLM over multi-view renders (Haiku cannot read 3D ASCII), checked against
     the facts extractor as in 2D.

Rule carried over from 2D: no hand quotas or heuristics that skew the data; let each generator's
natural distribution through, and steer only by adding generators.

## Physics, three tiers

1. Kinematic voxels with gravity and support (built: src/beebot/vla/voxel/). One robot per
   voxel, 26 moves plus stay, robots must end each step supported. Measured on this Mac's CPU:
   0.8 to 2.9 s per 1000 episode-steps at 16^3 and 2.6 to 3.5 s at 24^3, so millions of
   episode-steps per hour. Scripted towers and bridges exist; no levers, captions or path planning
   yet.
1. A cheap strength rule: a limit on overhang or on load per link, so a bridge that is too long
   fails. Makes engineering matter without a full physics engine.
1. MuJoCo MJX (GPU or TPU) for a small set of hero demos with linked rigid modules and contact
   forces, and as the bridge toward real hardware. Too expensive per module for bulk data.

## Model

The architecture found in 2D carries over unchanged:

- one token per module, built from its local view (a small voxel neighbourhood);
- relative positions only (3D rotary embeddings), all-to-all attention;
- a categorical readout per module: 27 moves now, attach / detach later;
- the goal as shape tokens (the target structure's occupied voxels) or text tokens;
- optionally a future-state readout (a world model), which was neutral on 2D tasks but is
  expected to matter when actions have delayed physical consequences (structures, machines).

Cost grows with the number of modules, not the volume, so 1000 modules in a large world cost
about what 1000 tokens cost. Beyond a few thousand modules: hierarchical tokens (a token per
cluster of modules) or local attention plus a few global tokens.

## Compute

- This Mac and Kaggle (T4 x2, TPU v5e-8): the voxel world, data generators, and small models
  (up to 0.6B with LoRA) on simple tasks: towers, bridges, shapes from primitives.
- Rented compute for the real thing: millions of episodes and a 1B+ parameter backbone.
  - Plain rented GPUs (8 x H100, a few hundred dollars a day) run our own code as is.
  - Tinker (LoRA fine-tuning of open LLMs with custom training loops) may not allow our own
    input and readout layers; if not, the scene and actions would be serialised as text tokens
    ("the LLM speaks robot"). Verify before committing.

## First experiments

1. Voxel world with gravity: towers and bridges from the scripted generator; train the readout
   policy (global and swarm view) on them; measure goal reaching and the facts score.
1. Shape formation: assembly-by-disassembly data for procedural primitives; train with the
   target shape as goal tokens; test on held-out primitives and held-out sizes.
1. World model on structures: the same runs with and without the future-state readout. This
   is where it should start to matter; if it does not, drop it.
1. Language: captions from multi-view renders; the LM policy (Qwen3) on 3D goals in words.
1. First functional task: carry an object across a gap, with episodes from search.

## Pitch arc

- Now: a 2D swarm reaches goals it never saw; local sight holds up (pending the swarm-view
  rescore).
- Next: towers and bridges in 3D from words.
- Then: shapes from libraries, and machines on command.

Each step is shown with a demo, a number, and its caveat. Microbots stays the closing bet, not
a claim.
