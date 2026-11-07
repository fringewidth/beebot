"""Facts about a window of a voxel episode: the 3D analogue of vla/facts.window_facts.

Atoms are hashable tuples, so the facts score of the 2D work (overlap of atom sets between a
rollout's window and the goal's) applies unchanged:

    ("move", direction)          the swarm's centroid moved at least MOVE_VOXELS
    ("end region", region)       where it ended, when it moved (one of 27 regions)
    ("groups", n)                the number of 26-connected groups changed (4 means 4 or more)
    ("pushed", colour, direction) a cheese slid (falls under gravity do not count)
    ("ate", colour)
    ("around", colour)           robots came to hold at least AROUND_SHARE of a cheese's open shell
    ("tower", bucket)            the tallest vertical run of robots rose into a higher bucket
    ("bridge",)                  a straight horizontal run of robots now spans air between two solids

Tower and bridge are only stated for scenes with gravity on (without it any blob has tall runs
and spans air).

Directions combine up/down (z), forward/back (y) and left/right (x): each axis whose share of
the displacement is at least AXIS_SHARE of the largest component is named, vertical first, e.g.
"up", "forward left", "down back right". Regions name each axis's third: low/high, back/front,
left/right ("center" when all three are middle), e.g. "low front left".
"""

from dataclasses import dataclass, field

import numpy as np

from beebot.vla.voxel.world import CHEESE_SIDE, DIRS, PALETTE_NAMES, SHELL_OFFSETS, SIZE

MOVE_VOXELS = 2.0  # at the 16-grid
AXIS_SHARE = 0.4
BLOB_SHARE = 0.1
AROUND_SHARE = 0.5
TOWER_BUCKETS = ((3, "low"), (5, "medium"), (8, "tall"))  # run lengths at the 16-grid
BRIDGE_AIR = 2  # voxels of the run with nothing beneath


@dataclass
class Facts:
    lines: list = field(default_factory=list)
    types: set = field(default_factory=set)
    data: dict = field(default_factory=dict)
    atoms: set = field(default_factory=set)

    @property
    def eventful(self):
        return bool(self.types)


def direction_name(dz, dy, dx):
    parts = np.abs([dz, dy, dx])
    top = parts.max()
    if top == 0:
        return "nowhere"
    names = (("down", "up"), ("back", "forward"), ("left", "right"))
    words = [names[a][int(v > 0)] for a, v in enumerate((dz, dy, dx)) if abs(v) >= AXIS_SHARE * top]
    return " ".join(words)


def region_name(z, y, x, size):
    third = size / 3
    names = (("low", "high"), ("back", "front"), ("left", "right"))
    words = []
    for a, v in enumerate((z, y, x)):
        if v < third:
            words.append(names[a][0])
        elif v >= 2 * third:
            words.append(names[a][1])
    return " ".join(words) or "center"


def centroid(mass):
    cells = np.argwhere(mass > 0.5)
    return cells.mean(axis=0) if len(cells) else np.zeros(mass.ndim)


def groups(mass):
    """26-connected groups of robots holding at least BLOB_SHARE of the swarm: (share, centroid)."""
    cells = {tuple(c) for c in np.argwhere(mass > 0.5)}
    total = len(cells)
    found = []
    while cells:
        stack = [cells.pop()]
        members = []
        while stack:
            c = stack.pop()
            members.append(c)
            for d in DIRS:
                n = (c[0] + d[0], c[1] + d[1], c[2] + d[2])
                if n in cells:
                    cells.remove(n)
                    stack.append(n)
        if len(members) >= BLOB_SHARE * total:
            found.append((len(members) / total, np.mean(members, axis=0)))
    return sorted(found, key=lambda g: -g[0])


def tallest_run(mass):
    """The longest vertical run of robots in any column."""
    occupied = mass > 0.5
    run = np.zeros(occupied.shape[1:], dtype=int)
    best = 0
    for z in range(occupied.shape[0]):
        run = np.where(occupied[z], run + 1, 0)
        best = max(best, int(run.max()))
    return best


def tower_bucket(height, size):
    scale = size / SIZE
    name = None
    for threshold, label in TOWER_BUCKETS:
        if height >= threshold * scale:
            name = label
    return name


def bridges(mass, solid):
    """Straight horizontal runs of robots (along x or y) whose two ends touch a solid voxel along
    the run and with at least BRIDGE_AIR robots over empty voxels. Returns the longest such run's
    length (0 for none)."""
    occupied = mass > 0.5
    below_empty = np.zeros_like(occupied)
    below_empty[1:] = ~occupied[:-1] & ~solid[:-1]
    best = 0
    for axis in (1, 2):
        occ = np.moveaxis(occupied, axis, -1)
        sol = np.moveaxis(solid, axis, -1)
        air = np.moveaxis(below_empty, axis, -1)
        n = occ.shape[-1]
        for idx in np.argwhere(occ.any(axis=-1)):
            row, srow, arow = occ[tuple(idx)], sol[tuple(idx)], air[tuple(idx)]
            i = 0
            while i < n:
                if not row[i]:
                    i += 1
                    continue
                j = i
                while j < n and row[j]:
                    j += 1
                ends = i > 0 and srow[i - 1] and j < n and srow[j]
                if ends and arow[i:j].sum() >= BRIDGE_AIR:
                    best = max(best, j - i)
                i = j
    return best


def solid_grid(walls, pos, alive):
    solid = walls.copy()
    for p, a in zip(pos, alive):
        if a:
            z, y, x = (int(v) for v in p)
            solid[z : z + CHEESE_SIDE, y : y + CHEESE_SIDE, x : x + CHEESE_SIDE] = True
    return solid


def shell_share(mass, pos, walls):
    size = mass.shape
    coords = np.asarray(pos, dtype=int) + SHELL_OFFSETS.numpy()
    inside = np.all((coords >= 0) & (coords < np.array(size)), axis=1)
    coords = coords[inside]
    open_ = ~walls[coords[:, 0], coords[:, 1], coords[:, 2]]
    covered = (mass[coords[:, 0], coords[:, 1], coords[:, 2]] > 0.5) & open_
    return covered.sum() / max(open_.sum(), 1)


def window_facts(static, start, end):
    """Facts about how the scene changed from snapshot `start` to snapshot `end`.

    static: walls (D, H, W) bool, cheese_color (K,), gravity bool. start, end: mass (D, H, W),
    cheese_pos (K, 3), cheese_alive (K,). numpy arrays."""
    facts = Facts()
    size = start["mass"].shape[-1]
    scale = size / SIZE
    m0, m1 = start["mass"], end["mass"]
    walls = static["walls"]

    c0, c1 = centroid(m0), centroid(m1)
    shift = c1 - c0
    facts.data["centroid_start"] = np.round(c0, 1).tolist()
    facts.data["centroid_end"] = np.round(c1, 1).tolist()
    if np.linalg.norm(shift) >= MOVE_VOXELS * scale:
        facts.types.add("move")
        direction, region = direction_name(*shift), region_name(*c1, size)
        facts.lines.append(f"The swarm's center moved {direction} by about {np.linalg.norm(shift):.0f} voxels, ending {region}.")
        facts.atoms.add(("move", direction))
        facts.atoms.add(("end region", region))

    g0, g1 = groups(m0), groups(m1)
    facts.data["groups_start"], facts.data["groups_end"] = len(g0), len(g1)
    if len(g1) != len(g0):
        verb = "split" if len(g1) > len(g0) else "merged"
        facts.types.add("split" if len(g1) > len(g0) else "merge")
        facts.lines.append(f"The swarm {verb} from {len(g0)} group(s) into {len(g1)}.")
        facts.atoms.add(("groups", min(len(g1), 4)))

    for k, colour in enumerate(static["cheese_color"]):
        if colour < 0 or not start["cheese_alive"][k]:
            continue
        name = PALETTE_NAMES[colour]
        p0, p1 = np.asarray(start["cheese_pos"][k], dtype=int), np.asarray(end["cheese_pos"][k], dtype=int)
        if not end["cheese_alive"][k]:
            facts.types.add("eat")
            facts.lines.append(f"The swarm ate the {name} cheese.")
            facts.atoms.add(("ate", name))
            continue
        d = p1 - p0
        if static["gravity"]:
            d[0] = max(d[0], 0)
        if d.any():
            facts.types.add("push")
            direction = direction_name(*d)
            facts.lines.append(f"The swarm pushed the {name} cheese {direction} by {int(np.abs(d).max())} voxels.")
            facts.atoms.add(("pushed", name, direction))
        s0, s1 = shell_share(m0, p0, walls), shell_share(m1, p1, walls)
        if s1 >= AROUND_SHARE > s0:
            facts.types.add("around")
            facts.lines.append(f"The swarm gathered around the {name} cheese (robots on {s1:.0%} of the voxels touching it).")
            facts.atoms.add(("around", name))

    h0, h1 = tallest_run(m0), tallest_run(m1)
    facts.data["tower_start"], facts.data["tower_end"] = h0, h1
    b0, b1 = tower_bucket(h0, size), tower_bucket(h1, size)
    rank = {None: 0, **{label: i + 1 for i, (_, label) in enumerate(TOWER_BUCKETS)}}
    if static["gravity"] and rank[b1] > rank[b0]:
        facts.types.add("tower")
        facts.lines.append(f"The swarm built a {b1} tower, {h1} robots high.")
        facts.atoms.add(("tower", b1))

    span0 = bridges(m0, solid_grid(walls, start["cheese_pos"], start["cheese_alive"]))
    span1 = bridges(m1, solid_grid(walls, end["cheese_pos"], end["cheese_alive"]))
    facts.data["bridge_start"], facts.data["bridge_end"] = span0, span1
    if static["gravity"] and span1 and not span0:
        facts.types.add("bridge")
        facts.lines.append(f"The swarm bridged a gap {span1} robots long.")
        facts.atoms.add(("bridge",))
    return facts


def store_facts(store, e, t0, t1):
    """window_facts for episode e of a store (voxel/scripted.generate) between steps t0 and t1."""
    static = {
        "walls": store["walls"][e].numpy(),
        "cheese_color": store["cheese_color"][e].tolist(),
        "gravity": bool(store["gravity"][e]),
    }

    def snap(t):
        return {"mass": store["mass"][e, t].numpy(), "cheese_pos": store["cheese_pos"][e, t].numpy(), "cheese_alive": store["cheese_alive"][e, t].numpy()}

    return window_facts(static, snap(t0), snap(t1))


def facts_score(goal, rollout):
    """Overlap of atom sets (Jaccard), 1 when both are empty."""
    if not goal.atoms and not rollout.atoms:
        return 1.0
    return len(goal.atoms & rollout.atoms) / len(goal.atoms | rollout.atoms)
