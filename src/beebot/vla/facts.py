"""Programmatic facts about a window of play: what changed between its start and its end.

The captioner gets these facts alongside the grids, and the audits check captions against them.
Directions use one fixed convention: row 0 is the top, column 0 is the left.

A snapshot is a dict with 'mass' (H, W), 'cheese_pos' (K, 2) and 'cheese_alive' (K,). The static
scene is a dict with 'walls' (H, W), 'cheese_color' (K,), 'box_rect' (B, 4) and 'box_color' (B,),
colors as palette indices (-1 for an empty slot). All numpy.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from beebot.vla.world import COLOUR_NAMES, DIRS, cheese_side, zoom

BLOB_MASS = 0.25  # a square belongs to a blob if it holds at least this much
BLOB_SHARE = 0.1  # blobs holding less than this share of the swarm are ignored
MOVE_SQUARES = 2.0  # a centroid shift at least this long counts as the swarm moving
BOX_IN = 0.5  # share of the swarm inside a box for the swarm to count as in it
BOX_OUT = 0.25
AROUND_SHARE = 0.5  # share of a cheese's open ring squares holding a robot to count as gathered around it (eating takes 0.75)
AWAY_SQUARES = 3.0  # rise in mean distance to a cheese to count as moving away from it


@dataclass
class Facts:
    lines: list = field(default_factory=list)  # one plain sentence per fact, for the captioner
    types: set = field(default_factory=set)  # fact types present, for audit stratification
    data: dict = field(default_factory=dict)  # the numbers behind the lines
    atoms: set = field(default_factory=set)  # hashable claims, for scoring a rollout against a goal

    @property
    def eventful(self):
        return bool(self.types)


def region_name(row, col, size):
    """Which of nine regions a point is in: 'top left', 'top', ..., 'center', ..., 'bottom right'."""
    third = size / 3
    vertical = "top" if row < third else ("bottom" if row >= 2 * third else "")
    horizontal = "left" if col < third else ("right" if col >= 2 * third else "")
    name = f"{vertical} {horizontal}".strip()
    return name or "center"


def direction_name(d_row, d_col):
    """One of eight directions for a displacement, with row 0 at the top."""
    angle = math.degrees(math.atan2(-d_row, d_col)) % 360  # up is +90
    names = ["right", "up and right", "up", "up and left", "left", "down and left", "down", "down and right"]
    return names[int(((angle + 22.5) % 360) // 45)]


def centroid(mass):
    total = mass.sum()
    rows, cols = np.indices(mass.shape)
    return (rows * mass).sum() / total, (cols * mass).sum() / total


def blobs(mass):
    """Connected groups (8-neighborhood) of squares holding at least BLOB_MASS, largest first.

    Each is (share of the total mass, centroid row, centroid col); groups under BLOB_SHARE are dropped.
    """
    size = mass.shape[0]
    occupied = mass >= BLOB_MASS
    seen = np.zeros_like(occupied)
    total = mass.sum()
    found = []
    for start in zip(*np.nonzero(occupied)):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        cells = []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in DIRS:
                ny, nx = y + dy, x + dx
                if 0 <= ny < size and 0 <= nx < size and occupied[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        ys, xs = np.array(cells).T
        weights = mass[ys, xs]
        share = weights.sum() / total
        if share >= BLOB_SHARE:
            found.append((share, (ys * weights).sum() / weights.sum(), (xs * weights).sum() / weights.sum()))
    return sorted(found, reverse=True)


def box_share(mass, rect):
    y, x, h, w = rect
    return mass[y : y + h, x : x + w].sum() / mass.sum()


def cheese_in_box(pos, rect, side):
    y, x, h, w = rect
    return y <= pos[0] and pos[0] + side <= y + h and x <= pos[1] and pos[1] + side <= x + w


def ring_share(mass, pos, walls):
    side = cheese_side(mass.shape[0])
    y, x = pos
    ring = np.zeros_like(walls)
    ring[max(y - 1, 0) : y + side + 1, max(x - 1, 0) : x + side + 1] = True
    ring[y : y + side, x : x + side] = False
    ring &= ~walls
    return (mass[ring] > 0.5).sum() / max(ring.sum(), 1)


def mean_distance(mass, pos):
    """Mass-weighted mean Chebyshev distance from the swarm to a cheese's footprint."""
    rows, cols = np.indices(mass.shape)
    side = cheese_side(mass.shape[0])
    dy = np.maximum(np.maximum(pos[0] - rows, rows - (pos[0] + side - 1)), 0)
    dx = np.maximum(np.maximum(pos[1] - cols, cols - (pos[1] + side - 1)), 0)
    return (np.maximum(dy, dx) * mass).sum() / mass.sum()


def window_facts(static, start, end):
    """Facts about how the scene changed from snapshot `start` to snapshot `end`."""
    facts = Facts()
    size = start["mass"].shape[0]
    side = cheese_side(size)
    scale = zoom(size)  # distance thresholds are in 16-grid squares
    m0, m1 = start["mass"].astype(np.float64), end["mass"].astype(np.float64)

    c0, c1 = centroid(m0), centroid(m1)
    shift = (c1[0] - c0[0], c1[1] - c0[1])
    distance = math.hypot(*shift)
    end_region = region_name(*c1, size)
    facts.data["centroid_start"] = [round(c0[0], 1), round(c0[1], 1)]
    facts.data["centroid_end"] = [round(c1[0], 1), round(c1[1], 1)]
    if distance >= MOVE_SQUARES * scale:
        facts.types.add("move")
        facts.lines.append(
            f"The swarm's center moved {direction_name(*shift)} by about {distance:.0f} squares, ending in the {end_region}."
        )
        facts.atoms.add(("move", direction_name(*shift)))
        facts.atoms.add(("end region", end_region))

    b0, b1 = blobs(m0), blobs(m1)
    facts.data["groups_start"] = len(b0)
    facts.data["groups_end"] = [[round(s, 2), region_name(r, c, size)] for s, r, c in b1]
    if len(b1) != len(b0):
        facts.types.add("split" if len(b1) > len(b0) else "merge")
        where = ", ".join(f"{region_name(r, c, size)} ({s:.0%})" for s, r, c in b1)
        verb = "split" if len(b1) > len(b0) else "merged"
        facts.lines.append(f"The swarm {verb} from {len(b0)} group(s) into {len(b1)}: {where}.")
        facts.atoms.add(("groups", min(len(b1), 4)))  # 4 means 4 or more

    for rect, color in zip(static["box_rect"], static["box_color"]):
        if color < 0:
            continue
        name = COLOUR_NAMES[color]
        s0, s1 = box_share(m0, rect), box_share(m1, rect)
        if s0 < BOX_OUT and s1 >= BOX_IN:
            facts.types.add("box")
            facts.lines.append(f"The swarm went into the {name} box ({s1:.0%} of it ends inside).")
            facts.atoms.add(("into box", name))
        elif s0 >= BOX_IN and s1 < BOX_OUT:
            facts.types.add("box")
            facts.lines.append(f"The swarm left the {name} box.")
            facts.atoms.add(("left box", name))

    for k, color in enumerate(static["cheese_color"]):
        if color < 0 or not start["cheese_alive"][k]:
            continue
        name = COLOUR_NAMES[color]
        p0, p1 = start["cheese_pos"][k], end["cheese_pos"][k]
        if not end["cheese_alive"][k]:
            facts.types.add("eat")
            facts.lines.append(f"The swarm ate the {name} cheese.")
            facts.atoms.add(("ate", name))
            continue
        d_row, d_col = int(p1[0] - p0[0]), int(p1[1] - p0[1])
        if d_row or d_col:
            facts.types.add("push")
            region = region_name(p1[0] + (side - 1) / 2, p1[1] + (side - 1) / 2, size)
            line = f"The swarm pushed the {name} cheese {direction_name(d_row, d_col)} by {max(abs(d_row), abs(d_col))} squares, to the {region}"
            for rect, box_color in zip(static["box_rect"], static["box_color"]):
                if box_color >= 0 and cheese_in_box(p1, rect, side) and not cheese_in_box(p0, rect, side):
                    line += f", into the {COLOUR_NAMES[box_color]} box"
            facts.lines.append(line + ".")
            facts.atoms.add(("pushed", name, direction_name(d_row, d_col)))
        r0, r1 = ring_share(m0, p0, static["walls"]), ring_share(m1, p1, static["walls"])
        if r1 >= AROUND_SHARE and r0 < AROUND_SHARE:
            facts.types.add("around")
            facts.lines.append(f"The swarm gathered around the {name} cheese (robots on {r1:.0%} of the squares touching it).")
            facts.atoms.add(("around", name))
        d0, d1 = mean_distance(m0, p0), mean_distance(m1, p1)
        if d1 - d0 >= AWAY_SQUARES * scale:
            facts.types.add("away")
            facts.lines.append(f"The swarm moved away from the {name} cheese (average distance {d0:.0f} to {d1:.0f} squares).")
            facts.atoms.add(("away", name))
    return facts
