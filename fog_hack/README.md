# The fog hack

Kept for the writeup: the strategy the swarm found on Feast when the loss was the task alone.

Run v4-feast-ema (2026-10-29), trained from scratch with the loss "unfilled share of each remaining cheese's ring, plus crowding", averaged weights, batch 64, lr 3e-4. `model_best.pt` is its 60-minute checkpoint, copied here from `runs_cancelled/v4-feast-ema/`.

What it does: the swarm dissolves. Within a few dozen steps it spreads into a thin film over the whole world, most squares holding a few hundredths of a robot. Wherever the film touches a cheese's ring it thickens until every ring square holds more than half a robot, and the cheese is eaten. It never searches, because it is already everywhere.

Why it's a hack: robot mass in the simulation is continuous, so a square can hold 0.03 of a robot. Real robots are whole. The task as written (fill each ring) says nothing about how the rest of the swarm looks, so gradient descent found the cheapest way to touch everything at once.

Why it's interesting:
- It scores better than any fog-free swarm trained so far. Val at 60 minutes: 81% of unseen sizes, 78% of bigger worlds, 46% with a quarter of the swarm wiped. Rescored on the CPU by `make.py` (`val.json`, which the page shows): 82%, 73%, 47%, with 72 to 92% of the swarm in fog. The two differ only in the random firing pattern, which the CPU and GPU generate differently.
- It generalizes to bigger worlds almost for free: a film doesn't care how far the cheese is.
- It is also how an untrained swarm learns at all. Spread-out mass reaches the rings, so the loss has a gradient to follow. Penalizing fog from the first step froze the swarm in place (v4-feast-fog1); what worked was allowing fog early and shrinking a fog allowance over training (v4-feast-tighten).

Files: `config.json`, `log.jsonl` (training), `val_history.jsonl` (val at 30 and 60 minutes), `val.json` (rescored with the fog measurement added later), `videos/` (the hardest val layouts it solves, from `make.py`), `picks.json` (what the clips show).
