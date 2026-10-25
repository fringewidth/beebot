# Progress report

*Last updated: 2026-10-25, overnight. This page is updated after every training round.*

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
| **v3: Terrain** | v2 plus random cave-like walls | In progress, see below. |

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

Round results are in the Terrain tab and the run log.

## Next

1. Solve gathering with terrain. The bar: 90% on unseen sizes and bigger
   worlds, 80% under damage.
1. **Feast:** several cheeses that vanish when eaten.
1. **Chase:** cheese that flees toward the side with the least mass.
1. **Rivals:** two swarms under different payoff rules (zero-sum, shared, stag
   hunt), testing whether cooperation or competition emerges, and whether
   swarms trained apart can cooperate as strangers. Then a variant where both
   swarms can push the cheese: tug of war, cooperative transport, and a heavy
   cheese only the two together can move.
1. **3D:** every task again in voxel space: 3x3x3 neighborhoods, cave-like 3D
   terrain, a shell instead of a ring, and an interactive 3D viewer on this page.
