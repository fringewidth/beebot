"""A scripted random player: the stand-in for play in the grid-size probe and the scripted-play ablation.

Each scene runs segments of random length. Each segment picks a random intent and drives the
attractor controller with it:
    goto:      one attractor somewhere open
    split:     two to four attractors (corners or anywhere), with equal shares half the time
    surround:  an attractor on a cheese, so the swarm fills its ring
    push:      gather behind a cheese, then push it straight toward a far point
    away:      head for the open square farthest from a cheese
    half:      half the swarm surrounds a cheese, half goes elsewhere
    stay:      no attractors
Attractors wander by a square now and then, so the same intent plays out differently. The phase
label changes at every segment and stage, so windows can be cut on them.
"""

import numpy as np

from beebot.vla.world import cheese_side, friction, zoom

INTENTS = ("goto", "split", "surround", "push", "away", "half", "stay")
INTENT_WEIGHTS = np.array([2, 2, 2, 3, 1, 1, 0.5])
SEGMENT_STEPS = (8, 24)  # at the 16-grid
JITTER_PROB = 0.1


class ScriptedPlayer:
    def __init__(self, rng, world):
        self.rng = rng
        self.size = world.mass.shape[-1]
        self.side = cheese_side(self.size)
        b = world.mass.shape[0]
        self.walls = world.walls.cpu().numpy()
        self.segment_end = np.zeros(b, dtype=int)
        self.segment = np.zeros(b, dtype=int)
        self.plans = [None] * b

    def _open_square(self, i, cheese_cells):
        free = np.argwhere(~self.walls[i] & ~cheese_cells)
        return [int(v) for v in free[self.rng.integers(len(free))]]

    def _cheese_cells(self, pos, alive):
        cells = np.zeros((self.size, self.size), dtype=bool)
        for (y, x), a in zip(pos, alive):
            if a:
                cells[y : y + self.side, x : x + self.side] = True
        return cells

    def _new_plan(self, i, pos, alive, cells):
        intent = str(self.rng.choice(INTENTS, p=INTENT_WEIGHTS / INTENT_WEIGHTS.sum()))
        live = [k for k in range(len(alive)) if alive[k]]
        if intent in ("surround", "push", "away", "half") and not live:
            intent = "goto"
        plan = {"intent": intent, "stage": 0}
        n = self.size - 1
        if intent == "goto":
            plan["points"] = [self._open_square(i, cells)]
        elif intent == "split":
            k = int(self.rng.integers(2, 5))
            corners = [[0, 0], [0, n], [n, 0], [n, n]]
            use_corners = self.rng.random() < 0.4
            plan["points"] = [corners[j] for j in self.rng.permutation(4)[:k]] if use_corners else [self._open_square(i, cells) for _ in range(k)]
            plan["shares"] = self.rng.random() < 0.5
        elif intent in ("surround", "half", "away"):
            plan["cheese"] = int(self.rng.choice(live))
            plan["points"] = [self._open_square(i, cells)]
        elif intent == "push":
            plan["cheese"] = int(self.rng.choice(live))
            plan["direction"] = [(0, 1), (0, -1), (1, 0), (-1, 0)][int(self.rng.integers(4))]
        return plan

    # One handler per intent; each returns (commands, phase label).

    def _points(self, i, plan, y, x, mass, t):
        share = 1.0 / len(plan["points"]) if plan.get("shares") else None
        return [{"row": r, "col": c, "share": share} for r, c in plan["points"]], plan["intent"]

    def _stay(self, i, plan, y, x, mass, t):
        return [], "stay"

    def _surround(self, i, plan, y, x, mass, t):
        return [{"row": y, "col": x}], "surround"

    def _half(self, i, plan, y, x, mass, t):
        r, c = plan["points"][0]
        return [{"row": y, "col": x, "share": 0.5}, {"row": r, "col": c, "share": 0.5}], "half"

    def _away(self, i, plan, y, x, mass, t):
        rows, cols = np.indices((self.size, self.size))
        far = np.maximum(np.abs(rows - y - self.side / 2), np.abs(cols - x - self.side / 2))
        far = np.where(self.walls[i], -1, far)
        r, c = np.unravel_index(np.argmax(far), far.shape)
        return [{"row": int(r), "col": int(c)}], "away"

    def _push(self, i, plan, y, x, mass, t):
        """Stage 0 gathers on the side opposite the direction; stage 1 pushes straight through."""
        side, n = self.side, self.size - 1
        dy, dx = plan["direction"]
        mid = side // 2
        behind = [min(max(y + mid - dy * (side // 2 + 2), 0), n), min(max(x + mid - dx * (side // 2 + 2), 0), n)]
        if plan["stage"] == 0:
            plan.setdefault("start", t)
            window = mass[max(behind[0] - side, 0) : behind[0] + side + 1, max(behind[1] - side, 0) : behind[1] + side + 1]
            if window.sum() >= 2 * friction(self.size) or t - plan["start"] > 12 * zoom(self.size):
                plan["stage"] = 1
        target = [min(max(y + mid + dy * self.size, 0), n), min(max(x + mid + dx * self.size, 0), n)]
        gather = [{"row": behind[0], "col": behind[1]}]
        push = [{"row": target[0], "col": target[1], "straight": True}]
        return (gather, "push-gather") if plan["stage"] == 0 else (push, "push")

    HANDLERS = {
        "goto": _points,
        "split": _points,
        "stay": _stay,
        "surround": _surround,
        "half": _half,
        "away": _away,
        "push": _push,
    }

    def _commands(self, i, plan, pos, alive, mass, t):
        cheese = plan.get("cheese")
        gone = cheese is not None and not alive[cheese]
        y, x = (int(v) for v in pos[cheese]) if cheese is not None else (0, 0)
        handler = self._stay if gone else self.HANDLERS[plan["intent"]].__get__(self)
        return handler(i, plan, y, x, mass, t)

    def act(self, world, t):
        """(commands per scene, phase label per scene) for step t."""
        pos_all = world.cheese_pos.cpu().numpy()
        alive_all = world.cheese_alive.cpu().numpy()
        mass_all = world.mass.cpu().numpy()
        z = zoom(self.size)
        commands, phases = [], []
        for i in range(len(self.plans)):
            pos, alive = pos_all[i], alive_all[i]
            cells = self._cheese_cells(pos, alive)
            if t >= self.segment_end[i] or self.plans[i] is None:
                self.plans[i] = self._new_plan(i, pos, alive, cells)
                low, high = (round(v * z) for v in SEGMENT_STEPS)
                self.segment_end[i] = t + int(self.rng.integers(low, high + 1))
                self.segment[i] += 1
            plan = self.plans[i]
            if "points" in plan and self.rng.random() < JITTER_PROB:
                plan["points"] = [
                    [int(np.clip(r + self.rng.integers(-1, 2), 0, self.size - 1)), int(np.clip(c + self.rng.integers(-1, 2), 0, self.size - 1))]
                    for r, c in plan["points"]
                ]
            scene_commands, label = self._commands(i, plan, pos, alive, mass_all[i], t)
            commands.append(scene_commands)
            phases.append(f"{self.segment[i]}:{label}")
        return commands, phases
