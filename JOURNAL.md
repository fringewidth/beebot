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

### The queue (updated 2026-10-27)

In order, each task finishing before the next starts:

1. **Terrain without smell.** The blind swarm has to clear 90% on every val split, then the test set is run once and the result tagged. This is the blocker. The scent-following control got 96% on unseen sizes, 95% in bigger worlds and 89% damaged, so the task is solvable once search is free. The blind swarm is at about 40%.
2. **Feast:** several cheeses that vanish when eaten.
3. **Chase:** cheese that moves away toward the side with the least mass.
4. **Rivals:** two swarms under zero-sum, shared and stag-hunt payoffs, to see whether they cooperate or compete. Swarms trained apart are tested together as strangers, against co-trained pairs and swapped partners. Then a variant where both swarms can push the cheese.
5. **3D voxels:** all of the above again in 3D.

### Planned tasks (written before any results; the queue above supersedes the order)

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

   Follow-up, after the regular games: **pushable cheese.** Robots touching the
   cheese push it away from themselves; every few steps it moves one square in
   the direction of the net push if that beats friction, blocked by walls and
   robots. Each swarm has a home nest and payoff depends on where the cheese
   ends up: zero-sum becomes tug of war, shared becomes cooperative transport to
   a common target, and the stag hunt uses a heavy cheese no single swarm can
   move.

### Closing v2: gathering is hard

- Round 2 trained on the gather objective (warm start, fresh, and 128 wide).
  All three stalled at curriculum level 3 and scored 0.93 to 1.00 on val. The
  best v2 result under the gather rule stays round 1's 0.909 (12% of unseen
  layouts). One run's loss spiked to 8.7 from the crowding term: gathering pushes
  robots into piles, which are penalized quadratically.
- v2 closed unsolved and is tagged `v2-no-smell`.

### v3: terrain, and making sure the results can be trusted

- **Terrain.** Smoothed noise on the torus thresholded at a random wall share
  (0 to 30%). The open space is the largest connected region; cut-off pockets
  become wall. Distances are walking distances (8 neighbors, around walls, never
  through the cheese), and the gather radius uses them. Layouts are rejected
  unless every ring square is reachable from the nest without passing through
  the cheese. Walls enter through the curriculum.
- **Physics tests.** Before trusting any more results, a test suite now checks
  that the world behaves as described: neighbor lookups; walking distance
  against a plain breadth-first search; layout invariants (nest off the ring,
  ring definition, reachability, swarm size); mass conservation under random
  policies for 60 steps, with no mass ever entering walls or cheese; mass moving
  to the chosen neighbor; a quarter-robot moving at a quarter strength; empty
  floor and resting squares writing nothing; the gather radius against brute
  force; the eaten rule needing both a full ring and a gathered swarm; finite
  loss with gradients reaching every layer; and deterministic evaluation. The
  first version of the eaten test failed because the test itself built an
  impossible state; once fixed, all tests passed.
- **Is gathering even achievable?** A hand-written oracle that knows the walking
  distance to the cheese solved only 69 to 97% of unseen-size layouts and almost
  none of the bigger worlds, even with twice the time. The time limit wasn't the
  bottleneck, so it was diagnosed directly: the oracle piled 10 to 28 robots on
  single ring squares, and those stacks jumped wholesale to the empty ring square
  next door every step, so the ring never settled. Softer oracles did worse.
  Hand-writing a good gathering policy for this physics is itself hard, so a
  failed oracle proves nothing about feasibility.
- **Privileged control instead.** A swarm that also smells the walking distance
  (v1's smell, routed around walls) is trained under the same physics and the
  same success rule. If it succeeds, the task is feasible and the gap to the
  blind swarm is the cost of search and communication. If it fails, the rule or
  time limit is too strict, and that's evidence for changing it rather than
  changing it by feel.

### Terrain round 1: the task is feasible; the blind swarm is the bottleneck

- The privileged control (smells walking distance) reached the last curriculum
  level and solved 91% of unseen-size layouts, 75% of bigger worlds, and 84%
  under damage, with 92% of the swarm gathered and little fog. Videos show a
  clean, compact gather. So the gather rule and time limit are achievable, with
  terrain.
- The blind swarm reached 25%, up from v2's 12% without walls, but stalled at
  level 3. Its videos show honest search: it spreads out laying pheromone, finds
  the cheese, and is still gathering when time runs out. Lowering the crowding
  penalty made it worse.
- **Stress tests.**
  - Pheromone blackout: blind 25% to 0%, and privileged 95% to 12%. Even a
    swarm that knows where the cheese is depends on pheromones, presumably to
    pack into a shell without collisions.
  - Split nests: the blind swarm almost never succeeds from 2 or 3 separate
    nests (3% and 0%), while the privileged one does (92%). The blind strategy
    depends on starting as one body; separated groups have no way to find each
    other.
  - 128x128 worlds break both (blind 6%, privileged 22%), the latter for v1's
    reason: the scent is too faint far away.
  - Denser walls than training (40 to 45%) didn't hurt; both did better, likely
    because less open space means shorter searches and a smaller swarm.
- **A bug the stress tests caught.** The pheromone-blackout wrapper didn't pass
  each new batch to the model inside it, so a wrapped privileged swarm would
  smell the previous batch's scent. It crashed when the grid size changed;
  with equal sizes it would have silently used the wrong scent. Fixed, with a
  test.
- Round 2 tries longer training, a bigger batch, and a teacher start: the blind
  swarm initialized from the privileged swarm's weights with the scent input
  removed, so it inherits gathering and only has to learn search. It never
  sees the scent at test time.

### Why the blind swarm can't learn to search

New bar from here on: every task has to reach 100% on val before moving on,
or come with evidence of why 100% is out of reach. Training was scaled up for
it: batches are generated in a worker process while the GPU trains, 30% of
batches replay earlier curriculum levels, runs checkpoint every 10 minutes, and
a population of 6-hour runs is managed at each check-in (the worst run replaced
by a variant of the best).

- The privileged control climbed to 95 to 96% on unseen sizes, 84% on bigger
  worlds and 86 to 92% under damage, then plateaued around 0.31 to 0.32.
- The blind swarm stayed near 30%, across a teacher start from the privileged
  weights, training from scratch, and an adaptive curriculum that samples the
  start distance where success is near 50%. The teacher run collapsed after
  90 minutes (training loss jumped to 2.3), which is why runs now decay their
  learning rate and keep their best val checkpoint.
- **Diagnosis.** Success falls steeply with starting distance: 74% within 6
  squares, 17% at 6 to 10, 3% at 10 to 15, 0% beyond. Giving the blind swarm
  three times the time limit (as a diagnostic only) barely helps: 27% to 45%
  for one model, 25% to 30% for another. So it isn't the clock; the swarm has no
  real search strategy. Of its failures, 44% never touch the cheese and 56%
  touch it but fail to recruit the rest.
- **The reason is in the loss.** Before contact, the only signal is the
  distance term. On a torus with the cheese placed uniformly at random, every
  square has the same expected distance to the cheese, so for a blind swarm the
  term carries no information about where to search. Spreading out, staying
  compact, or heading anywhere in particular all score the same in expectation.
  The swarm only learns from episodes where it happens to bump into the cheese,
  which is why it masters short distances and nothing further.
- **Fix: an exploration signal.** A new loss term rewards the share of open
  floor visited, only until a robot first touches the cheese, then switches
  off so gathering takes over. It doesn't touch evaluation. The tension the
  swarm now has to resolve is the slime mold's: spread wide enough to find food
  while staying connected enough to relay the news back.

### Exploration helps, then everything collapses

- With w_explore 0.5, the blind swarm reached 0.745 on val at 30 minutes: 41%
  of unseen sizes, 34% of bigger worlds, 39% damaged, its best result so far.
  Then exploration took over: by 45 minutes it had learned to spread out and
  never regroup, and training success fell to about 2%. Best-checkpoint keeping
  saved the 30-minute model. Lower weights (0.15, 0.05) started from that
  checkpoint also degraded within 30 to 45 minutes.
- The privileged control collapsed too, after more than three healthy hours.
  Its good weights were lost, because it started before best-checkpoint keeping
  existed.
- **Every long run had collapsed**, whatever its settings, so the cause had to
  be shared: the training loop. The suspect is the per-parameter gradient
  normalization taken from Growing NCA. It rescales each parameter's gradient to
  unit norm before Adam, so on a plateau, where the true gradient is near zero,
  noise becomes full-size steps; the weights random-walk until they drift into a
  saturated regime (one-hot moves, piled mass, vanishing gradient) with no way
  back. Adam already scales steps per weight, so the normalization is
  redundant. Training now clips the global gradient norm instead, and fine-tunes
  from good checkpoints use lower learning rates. Whether this was the cause is
  the next result.

### Clipping fixed the collapses; what blind search looks like

- With global gradient clipping, no run collapsed by 45 minutes, the point by
  which every earlier exploration run had. The privileged control, fine-tuned
  at a lower learning rate, reached 96% unseen sizes, 94% bigger worlds (up
  from 84%) and 89% damaged.
- The bar was revised to above 90% on every val split (from 100%).
- Watching the blind swarm fail on far layouts: its search is roughly "drift
  in one direction as a band and follow walls". In one far layout it streamed
  along a wall, sent a filament to the cheese, filled the ring, and ran out of
  time with 71% gathered (80% needed). In another it wrapped itself around a
  wall as an arc and never came near the cheese. A drifting band covers a strip
  of the world rather than the world, which matches the steep fall-off with
  distance. Stretching into a long line and sweeping would cover far more, but
  nothing yet points training there.

### More time: stage v3b

The demo of the best blind swarm showed it still searching when the clock ran out, even in the videos that run to twice the limit. Doubling the time limit with no retraining took the scent-following control from 96/95/89% to 98/97/98%, clearing the 90% bar on every split, and the blind swarm from 44/39/42% to 58/45/53%. A time limit is part of the task definition, not a loophole, as long as it's fixed in advance and applied the same way to every run. So the limit is now 6 steps per grid square, and the terrain stage restarts as v3b with the 3x results archived. The open question is whether a swarm trained on the longer horizon learns a real search strategy: with backprop through the rollout, a search that pays off after the training horizon can never be credited.

### Terrain closed as a proof of concept

Three things changed how the project runs. Fine-tuning from the previous stage's best weights stalled at once, while runs trained from scratch at the 6x limit kept improving, so every stage now trains from scratch. A 192-wide network was no better per training step than the 96-wide one and ran slower, so width was not the bottleneck. And the goal is now an existence proof per task: show that swarm intelligence can do it, then move on. The finalist, trained from scratch, scored 0.695 on the test set (52% of unseen sizes, 38% of bigger worlds, 44% damaged; val was 0.691, so no overfitting to val), against 0% for the hand-written wanderer. Blocking pheromones drops it to 0%. It handles damage and denser walls than it trained on, but can't reunite separate nests (25% with two, 6% with three) and weakens in a 128x128 world (19%). From the next stage on, experiments follow the bitter lesson: minimal hand-engineered help, more compute.
