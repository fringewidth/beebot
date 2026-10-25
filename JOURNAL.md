# Lab journal

The story of the project as it happened: what we tried, what surprised us, and
why each decision was made. Run-by-run numbers live in `HILLCLIMB.md`; this file
is the narrative.

## 2026-10-25

### Origin

A chat about von Neumann probes and whether cellular automata could lead to
general robotics. The toy problem that came out of it: a creature on a grid has
to surround a block of cheese, with local rules only. The question was what the
training signal would be, and whether self-organization could be optimized
rather than hand-designed.

### Design decisions

- **Neural cellular automaton.** Every square runs the same small MLP on its
  3x3 neighborhood (written as a 3x3 conv followed by 1x1 convs). Trained end to
  end with backprop through the whole rollout.
- **Conserved mass.** If the creature can create mass, the optimal strategy is
  to flood the grid in every direction, which is near the speed limit of a
  CA and leaves nothing to learn. So each square splits its mass across itself
  and its 8 neighbors with a softmax, and mass is never created or destroyed.
- **Loss.** Soft surround (share of ring squares holding at least 0.5) plus
  mass-weighted distance to the cheese (normalized to start at 1) plus a crowding
  penalty for squares over 1 unit. Averaged over all steps so finishing earlier
  scores better. The crowding term is what stops the distance term from rewarding
  a single stacked pile.
- **Generalization by construction.** Training randomizes grid size (16 to 40)
  and cheese shape and size (1 to 9 squares). The frozen val set uses unseen
  sizes, bigger worlds and cheeses, and a damage test (25% of mass wiped
  mid-rollout). A separate test set is run once at the end.
- **Runs are compared on the hard metric only**, never the loss.
- **PyTorch over JAX**, so the code is readable to the people working on it.
  On an M4 Pro, MPS was 13x faster than CPU.

### v1: smell

- Smell = exp(-distance / 8), fixed per layout.
- A 4-minute sanity run already learned. All three 30-minute round-1 runs beat
  the hand-written baseline (walk up the smell, avoid crowding). The best was
  lr 2e-3 at 0.236 val vs the baseline's 0.365.
- **Seed noise is about 0.01**, measured by repeating the best recipe with
  another seed. Differences smaller than that aren't signal.
- lr 4e-3 diverged to NaN. Its NaN scores made the report invalid JSON and
  blanked the website. Fixed by sanitizing the report and stopping training on a
  non-finite loss.
- **Physical lower bound.** No rule can fill the farthest ring square before
  mass can reach it, which gives about 0.18 at the eval's firing rate. The best
  run was at 0.226, so the score was saturated.
- **Stress tests.** Robust to asynchrony, to half the mass wiped, to chunks cut
  out, and to 20% smell noise. It handled 2 or 3 separate starting blobs (they
  found each other and merged) without ever training on that. It failed in
  128x128 worlds (59%): the smell there is fainter than anything seen in
  training. Fold-change sensing (reading neighbor/own smell ratios) was the
  proposed fix, but v1 was closed before it was tested.
- **v1 final test:** 0.267 vs the baseline's 0.405. Tagged `v1-smell`.
- **Why v1 was easy.** The smell field hands every square an arrow to the
  cheese, so no square needs information that another square has. The
  environment did the hard part.

### v2: no smell

- A robot only senses the cheese by touching it. Notes became pheromones on the
  floor: only robots write them, in proportion to the mass on the square, and
  they fade 5% per step. The swarm starts as a nest sized to 6% of the world's
  area. Rollouts are 3x the grid size.
- Before v2, empty squares also ran the network and wrote notes, so the floor
  itself could compute. Closed by making only robots write.
- **The fog strategy.** The first v2 run solved 99% in 6 minutes. It spread
  into a thin fog covering most of the world, laid pheromone everywhere, then
  condensed onto the cheese once any part of the fog touched it. This was a
  valid strategy under the rules, much like a slime mold. But it isn't
  available to physical robots: the fog put some mass on nearly every square, so
  every square had a neighbor to relay signals through. New rule: a square moves
  and writes in proportion to the mass it holds (a whole robot is 1), so a
  scrap of a robot is nearly stranded. Under that rule the fog model drops to
  about 30%. Fog (share of mass on squares under 0.5) is now reported on every
  eval.
- The new hand-written baseline (wander at random, stick once touching the
  cheese) solves 0%: random wandering spreads too thin to fill a ring in time.
- lr 2e-3 destabilized under the new physics (training success 97% to 0%).
  lr 1e-3 learned honestly: 45% of unseen layouts in 30 minutes.
- **Swapping.** One run at 40x40 with 120 steps holds 6.7 GB of activations;
  three in parallel overflowed 24 GB and starved the runs (one got 335
  iterations in 30 minutes). Gradient checkpointing cut that to 1.2 GB for 24%
  more time per iteration.
- **Pheromones are learned and essential.** The physics only provides the
  channels: 8 floor channels, robot-only writes, decay, 3x3 reads. What to
  write, when, and how to react are learned, and the write strength starts at
  exactly zero. With pheromone reads blacked out, success drops from 50% to 0%.
  Caveat: all-zero pheromones are also out of distribution, so a scrambled
  pheromone test would be a cleaner claim.
- **The swarm wasn't gathering.** Success only required the ring to be filled,
  which takes about 8 units of a 54-unit swarm. In successful runs only about
  half the swarm was within 2 squares of the cheese. The swarm did what was
  asked; the ask was wrong. New rule: the cheese counts as eaten only when the
  ring is full and at least 80% of the mass is within the gather radius (the
  smallest radius whose squares could hold the whole swarm at 0.8 per square),
  with a matching loss term. The best model drops from 45% to 12% under it.
  Old ring-only scores are kept as `val_ring_only` in each summary.

### Planned tasks (written before any results)

After Hunt (the current task) is solved, meaning val success of at least 90%
on unseen sizes and bigger worlds and at least 80% on damage, or two rounds in
a row without beating the best by more than seed noise:

1. **Feast:** several cheeses; an eaten cheese vanishes and the swarm has to
   find the next. Score is time to eat all. Tests forgetting stale trails and
   splitting up without a leader.
1. **Chase:** the cheese moves one square toward the side with the least mass
   every few steps, slower than the swarm (curriculum on its speed). A fully
   surrounded cheese can't move, so eaten means trapped. Tests pursuit and
   cutting off rather than chasing.
1. **Rivals, as a test of collaboration vs competition.** Two swarms with their
   own mass, colour and pheromones, sharing square capacity, each able to read
   the other's pheromones. The physics stays fixed and the payoff rules vary:
   - Zero-sum: one cheese, the swarm owning most of the ring takes all.
     Competitive control.
   - Shared: both get the full reward whenever it's eaten. Cooperative control.
   - Stag hunt: small cheeses either swarm can eat alone for a small reward,
     and one big cheese whose ring needs more mass than either swarm has, for a
     big reward split by contribution. The central social dilemma.
   - Commons (optional): cheese regrows only if not eaten too fast.

   Each swarm optimizes only its own payoff, and each swarm's update only
   touches its own parameters, so neither trains the other through the shared
   differentiable world.

   Evidence for real cooperation, beyond outcomes: payoffs, welfare and
   equality; partner swaps (paired with a non-adapting partner, does it do as
   well?); cross-species pheromone blackout; mixed-colour rings in the stag
   hunt.

   Stages: strangers (trained separately, then paired: zero-shot coordination),
   co-trained partners, then swapped partners after co-training (did they learn
   a general skill or a private handshake?).
