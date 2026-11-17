"""Ten-cube walks: MuJoCo's lattice outcome shares vs the grid's (an outcome table replayed on the same moves)."""
import json, sys
import numpy as np
from collections import Counter
from beebot.hardware import cube_outcomes as C
from beebot.vla.voxel.world import OutcomeModel
model = OutcomeModel.load(sys.argv[1])
rows = [json.loads(l) for p in sys.argv[2:] for l in open(p)]
rows = [r for r in rows if r.get("movers", 1) == 1 and r["detail"] != "refused"]
grid = C.grid_outcomes(rows, model, runs=16)
mj = Counter(C.lattice_end(r)[0] for r in rows)
mj = np.array([mj[k] / len(rows) for k in C.LATTICE])
g = np.asarray(grid).mean(0)
print("n", len(rows), "lattice", C.LATTICE)
print("mujoco", np.round(mj, 3), "harm", round(1 - mj[0] - mj[1], 3))
print("grid  ", np.round(g, 3), "harm", round(1 - g[0] - g[1], 3), "TV", round(0.5 * np.abs(mj - g).sum(), 3))
