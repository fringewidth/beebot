"""What is the best robot a given cube can form, and what cube does a wanted robot need?

    uv run python -m beebot.hardware.forms quadruped                 # Pareto front for hardware/cube50
    uv run python -m beebot.hardware.forms arm --mass 0.2 --twist 1.4
    uv run python -m beebot.hardware.forms size --want quadruped:step=0.06 --want arm:reach=0.2,payload=1c,lift=1c

A cube is a point on a lattice with a disc axis (x, y or z): its two faces across that axis carry
the discs, which are both the only connector and the only strong joint. So:

1. A design is a set of (x, y, z, axis). Two neighbours are bonded where a disc faces the other,
   and every bond can twist about its own normal. Nothing else is assumed: no legs, no templates.
1. The cube enters as three numbers, each after the safety margin: the twist a disc holds, the
   bending moment a bond holds and the pull a bond holds, in units of (cube weight x side),
   (cube weight x side) and cube weights. Two cubes with the same three numbers form the same
   robots, scaled.
1. One question is asked of every design, in every pose: `Body.spare`. With its own weight on,
   how many times a given extra load does it hold with no bond past its limits? Where it stands
   free it may push on the floor as it likes but never pull, so tipping over is the same check.
1. A form is a few lines that turn answers to that question into numbers to maximise (`arm`,
   `walker`, `span`). `search` grows designs one cube (and its mirror images) at a time and keeps
   the Pareto front; `size` runs it across cubes of different strength and reads off the weakest
   cube that reaches each target.

This is quasi-static: slow motion, rigid cubes, ideal bonds, no friction needed. It screens
designs; a front member still has to be run in MuJoCo (`cells` gives module_b.lattice rows).
"""

import argparse
import itertools
import json
import math
import time
from collections import deque
from dataclasses import dataclass, replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

GRAVITY = 9.81
PARAMS = Path(__file__).parents[3] / "cube50/sim/params.json"
RUNS = Path(__file__).parents[3] / "runs/forms"
E = np.eye(3)
SINK = 0.05  # sides: how far a swinging corner may dip into the floor (the body rides up that much)
NUDGE = math.radians(2)  # a joint's first move under load: enough to put a flat foot on its edge
RING = np.array([[math.cos(a), math.sin(a)] for a in np.arange(16) * math.pi / 8])  # bending is checked along 16 directions


@dataclass(frozen=True)
class Unit:
    """The cube. twist: torque a disc drive holds (N m); bend, pull: moment (N m) and tension (N) a bond holds."""

    side: float
    mass: float
    twist: float
    bend: float
    pull: float
    chamfer: float = 0.0075
    margin: float = 1.5
    grip: float = 0.6  # floor friction: a loaded foot that a joint drags pushes back sideways this hard

    @classmethod
    def load(cls, path=PARAMS, **changes):
        p = json.loads(Path(path).read_text())
        unit = cls(p["size_m"], p["mass_kg"], p["disc_torque_nm"], p["bond_moment_nm"], p["bond_hold_n"], p["edge_chamfer_m"])
        return replace(unit, **{k: v for k, v in changes.items() if v is not None})

    @property
    def caps(self):
        """(twist, bend, pull) in cube weights and sides, after the margin."""
        weight = self.mass * GRAVITY
        return np.array([self.twist / (weight * self.side), self.bend / (weight * self.side), self.pull / weight]) / self.margin


def turn(axis, angle):
    """Rotation by `angle` about the unit vector `axis`."""
    k = np.cross(E, axis)
    return E + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def corners(chamfer):
    """The 24 corners of a unit cube with chamfered edges."""
    a, b = 0.5, 0.5 - chamfer
    return np.array(sorted({p for s in itertools.product((1, -1), repeat=3) for p in itertools.permutations((a * s[0], b * s[1], b * s[2]))}))


class Body:
    """A design as a tree of cubes from `root`: its bonds, its joints, its pose, and what it holds."""

    def __init__(self, design, root, chamfer=0.15):
        self.cells = np.array([d[:3] for d in design])
        self.axes = [d[3] for d in design]
        self.root, self.shape, n = root, corners(chamfer), len(design)
        at = {tuple(c): i for i, c in enumerate(self.cells)}
        bonds = [(i, j) for i, c in enumerate(self.cells) for k in range(3)
                 if (j := at.get(tuple(c + E[k].astype(int)))) is not None and k in (self.axes[i], self.axes[j])]
        near = {i: [] for i in range(n)}
        for i, j in bonds:
            near[i].append(j)
            near[j].append(i)
        self.parent, self.order = {root: None}, [root]
        for i in self.order:  # breadth first: the list grows as it is read
            for j in near[i]:
                if j not in self.parent:
                    self.parent[j] = i
                    self.order.append(j)
        self.whole = len(self.order) == n
        self.sub = np.eye(n, dtype=bool)  # sub[i]: the cubes that hang from i's bond to its parent
        for i in reversed(self.order[1:]):
            self.sub[self.parent[i]] |= self.sub[i]
        self.loops = [(i, j) for i, j in bonds if self.parent.get(i) != j and self.parent.get(j) != i]
        self.joints = [i for i in self.order[1:] if self.swings(i)]

    def swings(self, i):
        """Turning i's bond to its parent moves something: no other bond bridges it, and not all it carries sits on its axis."""
        if any(self.sub[i][a] != self.sub[i][b] for a, b in self.loops):
            return False
        return bool(np.cross(self.cells[self.sub[i]] - self.cells[i], self.cells[i] - self.cells[self.parent[i]]).any())

    def pose(self, angles=()):
        """Centres (n, 3) and rotations (n, 3, 3), with cube i's bond to its parent turned by angles[i] (rad)."""
        angles = dict(angles)
        p, r = self.cells.astype(float), np.tile(E, (len(self.cells), 1, 1))
        for i in self.order[1:]:
            up = self.parent[i]
            step = r[up] @ (self.cells[i] - self.cells[up])
            r[i] = turn(step, angles[i]) @ r[up] if i in angles else r[up]
            p[i] = p[up] + step
        return p, r

    def points(self, p, r):
        """(n, 24, 3): every cube's corners."""
        return p[:, None] + np.einsum("nij,vj->nvi", r, self.shape)

    def lowest(self, p, r, cubes):
        """[(cube, point)]: the corners of `cubes` within 0.02 of their lowest: where they would stand."""
        pts = self.points(p, r)[cubes]
        low = pts[..., 2].min()
        return [(cubes[a], pts[a, b]) for a, b in zip(*np.nonzero(pts[..., 2] < low + 0.02))]

    def clash(self, p, r):
        """A cube is under the floor (the cells' z = 0 layer rests on it) or inside another cube."""
        pts = self.points(p, r)
        if pts[..., 2].min() < -0.5 - SINK:
            return True
        probe = p[:, None] + 0.92 * (pts - p[:, None])
        local = np.einsum("jivk,jkl->jivl", probe[None] - p[:, None, None], r)
        inside = (np.abs(local) < 0.48).all(-1)
        inside[np.arange(len(p)), np.arange(len(p))] = False
        return bool(inside.any())

    def spare(self, p, r, caps, live, feet=None):
        """The largest multiple of the load `live` (n, cube weights, pulling down at the centres) the
        posed design holds on top of its own weight with every bond within caps; None when it cannot
        hold itself. feet: [(cube, point)] where it stands on the floor (each point pushes up as much
        as needed, never pulls), or (cube, point, push) for a foot whose push leans (friction: the
        sideways part is not balanced here, the other feet take it); None: the root's lower face is
        bonded to the floor.

        Loads are wrenches about the origin (force, moment), one column per cause: own weight, live
        load, each foot, and each component of the wrench in every bond that closes a loop. What
        hangs from a bond is the sum over its subtree, so every bond's wrench is linear in the
        unknown columns, and the best split among them is a linear programme."""
        n, feet = len(p), feet or []
        down = np.concatenate([np.tile([0.0, 0, -1], (n, 1)), np.cross(p, [0.0, 0, -1])], axis=1)  # (n, 6): a unit weight on each cube
        cols = [down, down * np.asarray(live, float)[:, None]]
        for cube, point, *push in feet:
            c, push = np.zeros((n, 6)), push[0] if push else [0, 0, 1]
            c[cube] = np.concatenate([push, np.cross(point, push)])
            cols.append(c)
        for a, b in self.loops:
            mid = (p[a] + p[b]) / 2
            for k in range(6):
                c = np.zeros((n, 6))
                c[a] = np.concatenate([E[k], np.cross(mid, E[k])]) if k < 3 else np.concatenate([np.zeros(3), E[k - 3]])
                c[b] = -c[a]
                cols.append(c)
        cols = np.stack(cols, axis=-1)  # (n, 6, columns)
        kids = self.order[1:]
        ups = [self.parent[i] for i in kids]
        carried = [self.sub[kids].astype(float) @ cols.transpose(1, 0, 2).reshape(6, n, -1)]  # tree bonds
        at, normal = [(p[kids] + p[ups]) / 2], [p[kids] - p[ups]]
        for k, (a, b) in enumerate(self.loops):  # a loop bond carries its own unknown wrench
            own, mine = np.zeros((6, 1, cols.shape[2])), slice(2 + len(feet) + 6 * k, 8 + len(feet) + 6 * k)
            own[:, 0, mine] = cols[a][:, mine]
            carried.append(own)
            at.append([(p[a] + p[b]) / 2])
            normal.append([p[b] - p[a]])
        if not feet:  # the floor bond carries everything
            carried.append(cols.sum(0)[:, None])
            at.append([p[self.root] - [0, 0, 0.5]])
            normal.append([[0, 0, 1.0]])
        w = np.concatenate(carried, axis=1).transpose(1, 0, 2)  # (bonds, 6, columns)
        at, normal = np.concatenate(at), np.concatenate(normal)
        force, moment = w[:, :3], w[:, 3:] - np.cross(at[:, None], w[:, :3], axisb=1, axisc=1)  # moment about the bond
        t1 = np.cross(normal, E[np.abs(normal).argmin(1)])
        t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
        t2 = np.cross(normal, t1)
        along = lambda d, x: np.einsum("bk,bkc->bc", d, x)
        rows = [along(normal, moment), -along(normal, moment)]  # twist either way
        rows += [along(c * t1 + s * t2, moment) for c, s in RING]  # bending
        rows += [along(normal, force)] + [along(sign * t, force) for t in (t1, t2) for sign in (1, -1)]  # pull, shear
        limit = np.repeat([caps[0]] * 2 + [caps[1] * math.cos(math.pi / 16)] * len(RING) + [caps[2]] * 5, len(w))
        rows = np.concatenate(rows)  # (rows, columns): rows @ [1, mu, unknowns] <= limit
        dead, extra, free = rows[:, 0], rows[:, 1], rows[:, 2:]
        if free.shape[1] == 0:
            if (dead > limit + 1e-9).any():
                return None
            return float(np.min((limit - dead)[extra > 1e-12] / extra[extra > 1e-12], initial=np.inf))
        total = cols.sum(0)[[2, 3, 4]]  # standing free: vertical force and both tipping moments balance
        balance = np.zeros((3, 1 + free.shape[1])) if feet else None
        if feet:
            balance[:, 0], balance[:, 1:1 + len(feet)] = total[:, 1], total[:, 2:2 + len(feet)]
        bounds = [(0, 1e3)] + [(0, None)] * len(feet) + [(None, None)] * 6 * len(self.loops)
        cost = np.zeros(1 + free.shape[1])
        cost[0] = -1
        found = linprog(cost, A_ub=np.column_stack([extra, free]), b_ub=limit - dead, A_eq=balance, b_eq=-total[:, 0] if feet else None, bounds=bounds)
        return float(found.x[0]) if found.status == 0 else None


def reachable(joints, step, test, cap=300):
    """{setting: test(setting)} for the joint settings (whole multiples of `step` degrees) that can be
    reached from home one step of one joint at a time, through settings whose test is not None."""
    home = (0,) * len(joints)
    seen, queue = {home: test(home)}, deque([home])
    if seen[home] is None:
        return {}
    while queue and len(seen) < cap:
        s = queue.popleft()
        for k, d in itertools.product(range(len(joints)), (1, -1)):
            t = s[:k] + ((s[k] + d) % (360 // step),) + s[k + 1:]
            if t not in seen:
                seen[t] = test(t)
                if seen[t] is not None:
                    queue.append(t)
    return {s: v for s, v in seen.items() if v is not None}


def arm(design, unit, loads=(0.25, 1, 2, 4), step=45):
    """[(reach, payload, lift)]: the hand (the cube deepest in the tree from the base) carries
    `payload` cube weights between spots on the floor up to `reach` sides from the base and up to
    `lift` sides above the floor. The base is the first floor cube with an upright axis, bonded to
    the floor."""
    floor = [i for i, d in enumerate(design) if d[2] == 0 and d[3] == 2]
    body = Body(design, floor[0], unit.chamfer / unit.side) if floor else None
    if not floor or not body.whole or not body.joints:
        return []
    hand = body.order[-1]
    live, memo = np.eye(len(design))[hand], {}

    def look(s):
        if s not in memo:
            p, r = body.pose({j: math.radians(step * a) for j, a in zip(body.joints, s)})
            hold = None if body.clash(p, r) else body.spare(p, r, unit.caps, live)
            memo[s] = None if hold is None else (hold, tuple(np.round(p[hand] - p[body.root], 2)))
        return memo[s]

    out = []
    for load in loads:
        spots = {v[1] for v in reachable(body.joints, step, lambda s: look(s) if look(s) and look(s)[0] >= load else None).values()}
        floor = [math.hypot(x, y) for x, y, z in spots if z == 0]
        if len(floor) > 1:
            out.append((round(max(floor), 2), load, float(max(z for _, _, z in spots))))
    return out


def walker(design, unit, legs=4, step=30):
    """[(step, stride)]: with any one other foot in the air, a foot can be lifted (unloaded) to a
    spot `step` sides above the floor, take its share of the weight there and start to turn under
    it; the spots where it does span `stride` sides along x. A loaded foot that
    starts to turn stands on its leading edge and the floor drags against it with the cube's grip:
    this, not holding the leg up, is what a leg joint has to beat (MuJoCo shows the same).
    Legs are the `legs` limbs that hang from a joint and share the floor cubes equally; the rest
    must not touch the floor. One leg is tested: the others are its mirror images."""
    cells = np.array([d[:3] for d in design])
    middle = cells[:, :2].mean(0)
    root = min(range(len(design)), key=lambda i: (-cells[i, 2], np.sum((cells[i, :2] - middle) ** 2), design[i]))
    body = Body(design, root, unit.chamfer / unit.side)
    floor = cells[:, 2] == 0
    limbs = []
    for i in body.joints if body.whole else []:  # outermost first: a limb is the largest swinging part with one leg's share of the floor cubes
        if (body.sub[i] & floor).sum() * legs == floor.sum() and not any(body.sub[t][i] for t in limbs):
            limbs.append(i)
    if len(limbs) != legs or (floor & ~body.sub[limbs].any(0)).any():
        return []
    leg, others = limbs[0], limbs[1:]
    joints, mine = [j for j in body.joints if body.sub[leg][j]], np.nonzero(body.sub[leg])[0]
    stand = {t: body.lowest(*body.pose(), np.nonzero(body.sub[t])[0]) for t in others}
    nothing = np.zeros(len(design))

    def loaded(angles, still, j, way):
        """The leg takes weight as joint j starts to turn `way`, whichever other foot is in the air."""
        p, r = body.pose({**angles, j: angles[j] + way * NUDGE})
        moved = body.points(p, r)[mine]
        low = moved[..., 2] < moved[..., 2].min() + 0.005
        slide = (moved - still)[low][:, :2]
        slide /= np.maximum(np.linalg.norm(slide, axis=1, keepdims=True), 1e-9)
        foot = [(cube, point, (-unit.grip * x, -unit.grip * y, 1)) for cube, point, (x, y) in zip(mine[np.nonzero(low)[0]], moved[low], slide)]
        return all(body.spare(p, r, unit.caps, nothing, foot + sum((stand[t] for t in others if t != up), [])) is not None for up in others)

    def look(s):
        angles = {j: math.radians(step * a) for j, a in zip(joints, s)}
        p, r = body.pose(angles)
        if body.clash(p, r) or body.spare(p, r, unit.caps, nothing, sum(stand.values(), [])) is None:
            return None
        still = body.points(p, r)[mine]
        foot = still[still[..., 2] < still[..., 2].min() + 0.02]
        return all(loaded(angles, still, j, way) for j in joints for way in (1, -1)), foot[:, 2].min() + 0.5, foot[:, 0].mean()

    spots = [v for v in reachable(joints, step, look).values() if v[0]]
    if not spots or not look((0,) * len(joints))[0]:  # it must also get off the floor it stands on
        return []
    height, along = max(v[1] for v in spots), [v[2] for v in spots]
    return [(round(float(height), 2), round(float(max(along) - min(along)), 2))] if height > 0.05 else []


def span(design, unit):
    """[(overhang, payload)]: a rigid structure bonded to the floor reaches `overhang` sides sideways and holds `payload` cube weights at its far end."""
    floor = [i for i, d in enumerate(design) if d[2] == 0 and d[3] == 2]
    body = Body(design, floor[0], unit.chamfer / unit.side) if floor else None
    if not floor or not body.whole:
        return []
    p, r = body.pose()
    reach = np.linalg.norm((p - p[body.root])[:, :2], axis=1)
    hold = body.spare(p, r, unit.caps, np.eye(len(design))[reach.argmax()])
    return [(round(float(reach.max()), 2), round(min(hold, 1e3), 2))] if hold and reach.max() > 0 else []


@dataclass(frozen=True)
class Form:
    """What a robot is for: how to score a design, the names of its scores, and the mirror
    symmetries its designs keep (each (coordinate, offset) maps that coordinate c to offset - c)."""

    score: object
    names: tuple
    mirrors: tuple = ()
    warm: int = 4  # generations kept whole before the beam starts, since a seed cannot score yet


BOTH = [((0, a), (1, b)) for a in (0, 1) for b in (0, 1)]  # mirror planes through a cell or between two, in x and in y
FORMS = {
    "arm": [Form(arm, ("reach", "payload", "lift"))],
    "span": [Form(span, ("overhang", "payload"), warm=2)],
    "quadruped": [Form(walker, ("step", "stride"), m) for m in BOTH],
}


def images(cube, mirrors):
    """The cube and its mirror images."""
    out = {cube}
    for k, offset in mirrors:
        out |= {c[:k] + (offset - c[k],) + c[k + 1:] for c in out}
    return out


def settle(design, mirrors=()):
    """One name per design: lowest cubes on z = 0; with no mirrors to hold it in place, also slid to the corner and turned to the least of its 8 views from above."""
    def slid(d, ks):
        low = [min(c[k] for c in d) if k in ks else 0 for k in range(3)]
        return tuple(sorted((c[0] - low[0], c[1] - low[1], c[2] - low[2], c[3]) for c in d))

    if mirrors:
        return slid(design, (2,))
    views = []
    for swap, fx, fy in itertools.product((0, 1), repeat=3):
        views.append(slid([((-1) ** fx * c[swap], (-1) ** fy * c[1 - swap], c[2], 1 - c[3] if swap and c[3] < 2 else c[3]) for c in design], (0, 1, 2)))
    return min(views)


def grow(design, mirrors):
    """Every design one cube (and its mirror images) bigger, the new cube joined by a disc."""
    cells = {d[:3]: d[3] for d in design}
    for (cell, axis), k, sign in itertools.product(list(cells.items()), range(3), (1, -1)):
        new = cell[:k] + (cell[k] + sign,) + cell[k + 1:]
        for a in range(3):
            orbit = images(new + (a,), mirrors)
            if (k == axis or k == a) and not any(o[:3] in cells for o in orbit):
                yield settle(set(design) | orbit, mirrors)


def front(scored):
    """Keep the (design, scores) pairs no other pair beats: at least as good everywhere, better somewhere."""
    first = {s: k for k, s in sorted(scored.items(), reverse=True)}  # one design per score: the first by name
    return {k: s for s, k in first.items() if not any(all(x >= y for x, y in zip(t, s)) and t != s for t in first)}


def search(unit, forms, cubes=14, beam=12, log=None):
    """{(design, scores)}: the Pareto front over fewest cubes and the form's scores. Designs grow a
    generation at a time; the first `warm` generations are kept whole, after that the `beam` least
    beaten designs of each generation are grown."""
    found = {}
    for form in forms:
        layer, seen, generation = {settle(images((0, 0, 0, a), form.mirrors), form.mirrors) for a in range(3)}, set(), 0
        while layer:
            scored = {(d, s): (-len(d), *s) for d in sorted(layer) for s in form.score(d, unit)}
            found = front({**found, **scored})
            keep, rest = [], dict(scored)
            while rest and len(keep) < beam:  # peel fronts until the beam is full
                peel = front(rest)
                keep += sorted({d for d, _ in peel})
                rest = {k: v for k, v in rest.items() if k not in peel}
            keep = sorted(layer) if generation < form.warm else keep[:beam]
            seen |= layer
            layer = {c for d in keep for c in grow(d, form.mirrors) if len(c) <= cubes} - seen
            generation += 1
            if log:
                log(f"generation {generation}: {len(scored)} scored, front {len(found)}, next {len(layer)}")
    return found


def cells(design):
    """Rows for module_b.lattice (the MuJoCo model): each cube's cell, disc axis and tree parent, floor cube first."""
    body = Body(design, min(range(len(design)), key=lambda i: design[i][2]))
    place = {i: k for k, i in enumerate(body.order)}
    return [{"at": [int(v) for v in design[i][:3]], "axis": "+" + "xyz"[design[i][3]], **({"parent": place[body.parent[i]]} if body.parent[i] is not None else {})} for i in body.order]


def draw(design):
    """Layers from the top down, x to the right, y up the page; each cube shown by its disc axis."""
    c = np.array([d[:3] for d in design])
    lo, hi = c.min(0), c.max(0)
    at = {d[:3]: "xyz"[d[3]] for d in design}
    layers = [["".join(at.get((x, y, z), ".") for x in range(lo[0], hi[0] + 1)) for y in range(hi[1], lo[1] - 1, -1)] for z in range(hi[2], lo[2] - 1, -1)]
    return "\n".join("   ".join(layer[row] for layer in layers) for row in range(len(layers[0])))


def _at_strength(job):
    name, strength, base, cubes, beam = job
    scale = strength / (base.caps[0] * base.margin)
    return name, strength, search(replace(base, twist=base.twist * scale, bend=base.bend * scale, pull=base.pull * scale), FORMS[name], cubes, beam)


def size(wants, sides, base, strengths, cubes, beam, masses=None, log=print):
    """For each wanted form and each cube side, the weakest disc that reaches the targets.

    wants: {form: {score name: (target, in cubes)}}: a target is in metres or kilograms, or, when
    `in cubes` is set, in sides or cube weights. A design depends on the cube only
    through its three strength numbers, so each form is searched once per strength (the disc's
    twist over cube weight x side, before the margin; bend and pull keep `base`'s proportion to
    twist) and every side reads
    the same fronts: a target in metres is fewer sides on a bigger cube, a target in kilograms
    fewer cube weights. Mass is `masses` (one per side) or follows base's density. Returns rows of
    (form, side, mass, strength needed, twist N m, cubes, scores)."""
    rows, density = [], base.mass / base.side**3
    jobs = [(name, strength, base, cubes, beam) for name, strength in itertools.product(wants, strengths)]
    with Pool(min(len(jobs), 6)) as pool:
        fronts = {}
        for name, strength, found in pool.imap_unordered(_at_strength, jobs):
            fronts[name, strength] = found
            log(f"{name} at strength {strength}: front of {len(found)}")
    for name, (side, mass) in itertools.product(wants, zip(sides, masses or [density * side**3 for side in sides])):
        need = [value if whole else value / (mass if label == "payload" else side) for label in FORMS[name][0].names for value, whole in [wants[name].get(label, (0, True))]]
        hit = None
        for strength in strengths:  # weakest first
            fits = [(len(d), s) for (d, s) in fronts[name, strength] if all(x >= y for x, y in zip(s, need))]
            if fits:
                hit = (strength, *min(fits))
                break
        twist = hit[0] * mass * GRAVITY * side if hit else None
        rows.append((name, side, mass, *(hit or (None, None, None)), twist))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("form", choices=[*FORMS, "size"])
    parser.add_argument("--params", default=PARAMS)
    for name in ("side", "mass", "twist", "bend", "pull", "margin", "grip"):
        parser.add_argument(f"--{name}", type=float)
    parser.add_argument("--cubes", type=int, default=14)
    parser.add_argument("--beam", type=int, default=12)
    parser.add_argument("--want", action="append", default=[], help="size: form:score=target,... in metres and kilograms, or with a c in sides and cube weights (payload=1c)")
    parser.add_argument("--sides", type=float, nargs="+", default=[0.03, 0.04, 0.05, 0.056, 0.06, 0.072])
    parser.add_argument("--masses", type=float, nargs="+", help="size: kg, one per side (default: the cube's density)")
    parser.add_argument("--strengths", type=float, nargs="+", default=[3, 4.5, 6, 8, 11, 15, 20])
    args = parser.parse_args()
    unit = Unit.load(args.params, **{k: getattr(args, k) for k in ("side", "mass", "twist", "bend", "pull", "margin", "grip")})
    RUNS.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if args.form == "size":
        wants = {w.split(":")[0]: {k: (float(v.rstrip("c")), v.endswith("c")) for k, v in (pair.split("=") for pair in w.split(":")[1].split(","))} for w in args.want}
        rows = size(wants, args.sides, unit, sorted(args.strengths), args.cubes, args.beam, args.masses)
        print("form       side mm  mass g  strength  twist N m  cubes  scores (sides, cube weights)")
        for name, side, mass, strength, count, scores, twist in rows:
            print(f"{name:10s} {side * 1e3:7.0f} {mass * 1e3:7.0f}  " + (f"{strength:8.1f} {twist:10.2f} {count:6d}  {scores}" if strength else "not reached at any strength tried"))
        (RUNS / "size.jsonl").write_text("".join(json.dumps(dict(zip(("form", "side", "mass", "strength", "cubes", "scores", "twist"), row))) + "\n" for row in rows))
        return
    caps = unit.caps
    print(f"cube: {unit.side * 1e3:.0f} mm, {unit.mass * 1e3:.0f} g; after the {unit.margin} margin a disc holds {caps[0]:.1f} cube weights at one side, a bond bends at {caps[1]:.1f}, pulls off at {caps[2]:.0f} cube weights")
    found = search(unit, FORMS[args.form], args.cubes, args.beam, log=lambda line: print(f"  [{time.time() - started:5.0f} s] {line}", flush=True))
    names = FORMS[args.form][0].names
    with open(RUNS / f"{args.form}.jsonl", "w") as out:
        for (design, scores), _ in sorted(found.items(), key=lambda kv: kv[1], reverse=True):
            real = {n: (s * unit.mass if n == "payload" else s * unit.side) for n, s in zip(names, scores)}
            print(f"\n{len(design)} cubes: " + ", ".join(f"{n} {s:g} ({real[n] * 1e3:.0f} {'g' if n == 'payload' else 'mm'})" for n, s in zip(names, scores)))
            print(draw(design))
            out.write(json.dumps({"form": args.form, "cubes": len(design), **dict(zip(names, scores)), **{f"{n}_si": v for n, v in real.items()}, "cells": cells(design)}) + "\n")


if __name__ == "__main__":
    main()
