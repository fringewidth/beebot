# beebot

Neural cellular automata that learn to surround a block of cheese.

Every square of a grid runs the same small neural network. Each square sees
only its 3x3 neighborhood, and the only thing it can do is split its mass
among itself and its 8 neighbors. Mass is never created or destroyed. The
cheese gives off a smell that fades with distance. The network has to learn
a rule that, run on every square at once, moves a blob of mass to the cheese
and wraps it completely, as fast as possible.

There is no central controller, no coordinates, and no global information.
Whatever coordination shows up comes out of the shared local rule.

## Layout

| File | What it holds |
| --- | --- |
| `src/beebot/layout.py` | Random layouts: grid size, cheese shape, starting blob, smell |
| `src/beebot/ca.py` | One CA step, rollouts, and the loss |
| `src/beebot/model.py` | The learned rule (`NCA`) and a hand-written baseline (`GreedyBaseline`) |
| `src/beebot/train.py` | Training with a curriculum; every `Config` field is a flag |
| `src/beebot/evaluate.py` | Frozen val and test sets and the hard metric |
| `src/beebot/final_test.py` | Scores a finished run, and the baseline, on val and test |
| `src/beebot/render.py` | GIFs of rollouts |
| `src/beebot/bench.py` | Iteration timing per device |

## The model

Inputs per square, over its 3x3 neighborhood: mass, smell, and 8 note
channels the network writes to and reads from, 90 numbers in total. Outputs:
9 scores, softmaxed into the fractions of mass that stay or go to each
neighbor, plus 8 note updates. Moves into the cheese are masked out. Each
step, every square acts with probability 0.75 and sits out otherwise, so the
rule can't rely on perfect synchrony.

## The loss

Averaged over every step of the rollout, so finishing earlier scores better:

- surround: fraction of ring squares (the squares touching the cheese) holding at least 0.5 mass, soft
- distance: mass-weighted average distance to the cheese, divided by its starting value
- crowding: penalty on squares holding more than 1 unit of mass

## Evaluation and anti-hacking rules

- Runs are compared on the hard metric only: the step at which every ring
  square first holds more than 0.5 mass, as a fraction of the rollout. A
  failure counts as 1. Lower is better. The loss is never used to compare runs.
- Training sizes are 16, 20, 24, 28, 32 and 40, with cheese of 1 to 9 squares.
  The val set uses grid sizes 22, 30 and 36 (interp), 48 and 64 with cheese of 10 to 16 squares
  (extrap), and interp layouts with 25% of the mass wiped mid-rollout (damage).
- Rollout length, firing probability and the occupancy threshold are fixed in
  `evaluate.py`, so a training config can't make the eval easier.
- `evaluate.py` and `layout.py` are frozen during the hill-climb.
- The test set uses other grid sizes (18, 26, 34; 56, 72) and seeds. It is
  run once per finalist at the end, never to choose between runs.

## Running

```
uv run bench
uv run train --name my-run --minutes 30
uv run final-test runs/my-run
```

The hill-climb log is in `HILLCLIMB.md`.
