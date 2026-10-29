# Progress report

*Last updated: 2026-10-29. This page is updated after every training round.*

## The question

Can a swarm of identical units, each running the same tiny neural network and
seeing only its immediate neighbors, learn to act as one organism: find food,
and gather around it, with no leader, no map, and no global information?

Each square of a grid runs the same ~20k-parameter network on its 3x3
neighborhood. The only action is passing mass to neighbors, and mass is never
created or destroyed. The whole rule is trained end to end with gradient
descent through the full rollout.

## Where it stands

| Stage | What the swarm gets | Result |
| --- | --- | --- |
| **v1: Smell** | Every square smells the cheese | Solved. Test score 0.267 vs a hand-written rule's 0.405, near the physical limit. Robust to damage, noise, asynchrony, and split starts it never trained on. Failed only in 128x128 worlds, where the smell is fainter than anything it trained on. |
| **v2: No smell** | Touch only, plus pheromones it learns to use | Learned honestly: 45% of unseen layouts surrounded, against 0% for a hand-written wanderer. Cutting off pheromones drops it to 0%, so it really communicates. When we required the *whole* swarm to gather, not just enough to fill the ring, it dropped to 12%. Not solved. |
| **v3: Terrain** | v2 plus random cave-like walls, 6x time limit | Proof of concept, closed. Trained from scratch: test 0.695, 52% of unseen sizes, 38% of bigger worlds, 44% damaged, against 0% for a hand-written wanderer. Blocking pheromones drops it to 0%. Not limited by time: with 4x the time, val success only rises from 46/44% to 58/62%. It usually finds the cheese but fails to bring the whole swarm in and hold it there. Weak at reuniting split nests. |

## What we learned so far

1. **When the environment hands each unit the answer, the task is easy.** v1's
   smell gives every square an arrow to the cheese, so no square needs
   information another square has. It saturated within an hour.
1. **Removing the smell forced real communication.** The physics provides 8
   pheromone channels that only robots can write and that fade over time. What
   to write, when, and how to react are all learned; writes start at exactly
   zero. Blocking pheromone reads drops success from 50% to 0%.
1. **The swarm finds loopholes in the rules, not bugs in its behavior.**
   - The first no-smell swarm dissolved into a thin fog over the whole world,
     then condensed onto the cheese, much like a slime mold. Valid under the
     rules, but impossible for physical robots: the fog gave every square a
     neighbor to relay through. We now make a partial robot move in proportion
     to its mass.
   - Swarms "ate" the cheese with 15% of their mass while the rest wandered,
     because success only required the ring to be filled. Success now requires
     80% of the swarm to gather.
1. **Generalization is tested, not assumed.** Every run is scored on grid
   sizes it never trained on, bigger worlds and bigger cheese than any in
   training, and a test where a quarter of the swarm is wiped mid-run. A
   separate test set is used once per stage, never to pick runs.

## Overnight: terrain

- Walls are smoothed noise on the torus, up to 30% of the world, with every
  layout checked so the whole ring is reachable from the nest.
- A physics test suite now guards the results: mass conservation, nothing
  entering walls or cheese, walking distance against brute-force search,
  empty floor unable to write pheromones, and deterministic evaluation.
- **Feasibility control.** A hand-written oracle that knows the walking
  distance couldn't satisfy the gather rule. It piled robots into stacks that
  jumped back and forth. Rather than loosen the rule by feel, a *privileged*
  swarm that smells the walking distance is trained under the same rules. Its
  result tells us whether the task is achievable, and the gap to the blind
  swarm measures what search and communication cost.

**Round 1 result: the task is feasible, and search is the bottleneck.**

| Swarm | Unseen sizes | Bigger worlds | Damaged | Pheromones blocked |
| --- | --- | --- | --- | --- |
| Privileged (smells the cheese) | 91% | 75% | 84% | 12% |
| Blind (touch + pheromones) | 25% | 22% | 23% | 0% |
| Hand-written wanderer | 0% | 0% | 0% | 0% |

- The blind swarm improved on v2 (25% vs 12%) despite the walls, but stalls
  at curriculum level 3.
- Both swarms depend on pheromones, even the one that can smell the cheese:
  packing into a shell takes coordination.
- The blind swarm can't reunite separate nests (3%); the privileged one can
  (92%). Without a shared signal, separated groups never find each other.
- Round 2 (running) tries longer training, a bigger batch, and a teacher
  start from the privileged swarm's weights.

**Overnight: chasing the bar.** A task is done at above 90% success on every
val split, confirmed once on the test set. (It was 100% for most of the night;
revised at 04:00.) What the night found:

1. **The blind swarm has no search strategy, and the loss explains why.** Its
   success falls steeply with starting distance (74% within 6 squares, 3% at 10
   to 15), and tripling the time limit barely helps. Before it touches the
   cheese, the only training signal is distance to the cheese, and on a
   wrap-around world with randomly placed cheese every square has the same
   expected distance to it. So the loss says nothing about where to search.
1. **An exploration signal fixes part of it.** Rewarding floor coverage until
   first contact gave the best blind result so far: 0.745 on val (41% of
   unseen sizes, 34% of bigger worlds, 39% damaged). Too strong, and the swarm
   learns to spread out and never regroup: the slime mold's dilemma between
   searching wide and staying connected enough to relay the news.
1. **Every long run eventually collapsed**, including the scent-following
   control after three good hours. The likely cause is a gradient
   normalization trick from the Growing NCA paper that turns near-zero
   gradients into full-size random steps. Training now uses standard gradient
   clipping and keeps each run's best checkpoint by val score.

The scent-following control topped out at 95 to 96% of unseen sizes, 84% of
bigger worlds and 86 to 92% damaged before collapsing, so **no run has reached
100% yet, blind or not.**

Round results are in the Terrain tab and the run log.

## Terrain closed (2026-10-29)

1. **More time.** The best blind swarm's demo was still searching when the
   clock ran out. Doubling the limit to 6 steps per grid square, with no
   retraining, took the scent-following control to 98/97/98% and the blind
   swarm from 44/39/42% to 58/45/53%. The limit is part of the task definition,
   so this became a new stage (v3b), with the 3x results archived as v3a.
1. **From scratch only.** Fine-tuning from the previous stage's best weights
   plateaued at once, while runs trained from scratch at 6x kept improving.
   Every stage's result now comes from a run trained from scratch in that stage.
1. **Width doesn't help.** A 192-wide network matched the 96-wide one at equal
   training steps and ran 1.5 times slower.
1. **The bar changed to proof of concept.** Each stage shows that a task is
   achievable by swarm intelligence, making the case for one general model that
   does every task with the same weights. Once a stage has a proof of concept,
   we move on.

| Finalist (from scratch) | Unseen sizes | Bigger worlds | Damaged | Score |
| --- | --- | --- | --- | --- |
| v3b-base-levels, val | 47% | 42% | 41% | 0.691 |
| v3b-base-levels, **test** (used once) | **52%** | **38%** | **44%** | **0.695** |
| Hand-written wanderer, test | 0% | 0% | 0% | 1.000 |

Stress tests (reference 50%): pheromone blackout 0%, half the robots firing
42%, every robot firing 50%, half the swarm wiped 38%, a chunk cut out 50%, two
separate nests 25%, three 6%, a 128x128 world 19%, no walls 53%, walls covering
40 to 45% of the world (more than in training) 47%. The swarm
depends on its pheromones, shrugs off damage and denser walls, but can't
reunite separate groups and loses most of its success in much bigger worlds.

## Next

1. **Feast:** several cheeses that vanish when eaten.
1. **Chase:** cheese that flees toward the side with the least mass.
1. **Rivals:** two swarms under different payoff rules (zero-sum, shared, stag
   hunt), testing whether cooperation or competition emerges, and whether
   swarms trained apart can cooperate as strangers. Then a variant where both
   swarms can push the cheese: tug of war, cooperative transport, and a heavy
   cheese only the two together can move.
1. **3D:** every task again in voxel space: 3x3x3 neighborhoods, cave-like 3D
   terrain, a shell instead of a ring, and an interactive 3D viewer on this page.
1. **Language-conditioned control (to discuss):** prompt to action. A scene
   encoder, a small frozen language model and a flow-matching action expert,
   bootstrapped by imitating scripts, then trained with RL up a ladder of
   harder tasks. Notes in VLA.md.
