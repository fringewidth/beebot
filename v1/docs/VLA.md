# Language-conditioned control (notes, to discuss)

Parked until the swarm queue (terrain, Feast, Chase, Rivals, 3D) is done. These
are notes from a design discussion on 2026-10-29, not a plan of record.

## Goal

Prompt to action: a natural-language command ("eat everything", "push the small
one to the top-left corner") turned into swarm motion, by a small model that
understands commands in this world, not a general-purpose one.

## Two versions

1. **v1, central brain.** One model sees the whole grid and drives the swarm.
   Accepted for v1: the model knows where the cheese is. The questions are
   whether it can move a cheese somewhere, and whether it splits the swarm
   across several cheeses for parallel work.
1. **v2, distributed.** The command is broadcast to every square; perception
   stays local (3x3 neighborhood plus pheromones), as in the swarm stages.

## Architecture (pi0 shape, sized for a Mac)

1. **Scene encoder** (our SigLIP): a small ViT over the grid's channels,
   trained contrastively on procedural (scene, caption) pairs against a frozen
   small LM's text embeddings (LiT, with the text side frozen instead of the
   image side). PaliGemma is no use here; Gemma 2B is too big to train on the Mac.
1. **Language backbone:** a small pretrained LM, frozen (SmolLM2-135M is the
   first candidate). It receives scene tokens through a projector trained by
   captioning scenes, LLaVA style.
1. **Action expert:** a small transformer that reads the backbone's states and
   generates action chunks by flow matching. The action is a 2D velocity per
   square (direction and intensity), projected onto flows to the two bracketing
   neighbors.
1. **Optional reasoning:** the backbone writes a short, checkable plan before
   acting ("group 1, 60% of mass: left cheese").

## Data and training: bootstrap, then RL

Copying an expert caps the model at the expert, and in 3D or real life there is
little supervised data. So imitation is only the cold start; RL does the rest.

1. **One objective throughout:** reward-weighted flow matching. Rung 0 imitates
   privileged scripts. Every later rung samples several rollouts per layout and
   command from the current policy, scores them (checker or language scorer),
   and trains on the ones that beat the group mean, weighted by the margin
   (group-relative, as in GRPO). Works with non-differentiable physics.
1. **No selecting luck:** rollouts in a group share random seeds, so they
   compare actions, not noise.
1. **Optional sources of demos:** Claude-written policy programs per task family
   (evolutionary search: write, score, revise; keep several diverse strong
   ones), run on unlimited layouts. Programs output velocities only, and the
   simulator applies them.

## Ladder

Each rung must be reachable: the previous policy has to succeed sometimes.

| Rung | Task | Skill from |
| --- | --- | --- |
| 0 | Surround the named cheese | Imitating scripts |
| 1 | Same, faster than the script | RL |
| 2 | Eat every cheese (splitting) | RL, no script |
| 3 | Gather on one side of a cheese | RL |
| 4 | Push a cheese to a place | RL; needs rung 3 (a ring pushes from every side, so the forces cancel) |
| 5 | A task family defined only in language | RL against the scorer |

New physics needed: several cheeses at once, and pushable cheese (rigid block;
net push from ring mass toward the cheese, moves when it beats friction scaled
by size; walls block; mass in the way is shoved ahead, so mass is conserved).

## What would show it scales

1. Ladder vs RL from scratch on each rung (bootstrapping does the work).
1. Ladder vs imitation only (RL goes beyond the scripts).
1. Samples needed per new rung: falling cost per rung means skills transfer.

## Out of reach for the toy

Real-world trial cost and the sim-to-real gap. For real-world RL, trials would
have to happen in a learned world model; here the simulator is the world model.
