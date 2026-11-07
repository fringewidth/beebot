"""Target masks for formation tasks: letters from a 5x7 bitmap font and simple shapes, drawn on a
size x size grid. Every mask is a (size, size) bool numpy array."""

import numpy as np

FONT = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10111", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "J": ["00111", "00010", "00010", "00010", "00010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "11011", "10001"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
}


def text_mask(text, size, scale=2, gap=1):
    """The letters of text side by side, each cell scale x scale squares, centred; None if it does
    not fit."""
    glyphs = [np.array([[c == "1" for c in row] for row in FONT[ch]]) for ch in text]
    cell = np.concatenate([np.concatenate([g, np.zeros((7, gap), bool)], axis=1) for g in glyphs], axis=1)[:, :-gap] if gap else np.concatenate(glyphs, axis=1)
    big = np.kron(cell, np.ones((scale, scale), bool))
    h, w = big.shape
    if h > size - 2 or w > size - 2:
        return None
    out = np.zeros((size, size), bool)
    r, c = (size - h) // 2, (size - w) // 2
    out[r : r + h, c : c + w] = big
    return out


def shape_mask(name, size, radius=None, thickness=2):
    """A named shape centred on the grid; radius defaults to about a third of the grid."""
    r = radius or size // 3
    yy, xx = np.mgrid[0:size, 0:size]
    cy = cx = (size - 1) / 2
    dy, dx = yy - cy, xx - cx
    dist = np.hypot(dy, dx)
    t = thickness
    if name == "ring":
        return (dist <= r) & (dist > r - t)
    if name == "disc":
        return dist <= r * 0.6
    if name == "square":
        inside = (np.abs(dy) <= r) & (np.abs(dx) <= r)
        return inside & ~((np.abs(dy) <= r - t) & (np.abs(dx) <= r - t))
    if name == "plus":
        return ((np.abs(dy) < t / 2 + 0.5) & (np.abs(dx) <= r)) | ((np.abs(dx) < t / 2 + 0.5) & (np.abs(dy) <= r))
    if name == "x":
        return ((np.abs(dy - dx) <= t / 2 + 0.2) | (np.abs(dy + dx) <= t / 2 + 0.2)) & (np.abs(dy) <= r) & (np.abs(dx) <= r)
    if name == "horizontal line":
        return (np.abs(dy) < t / 2 + 0.5) & (np.abs(dx) <= size // 2 - 2)
    if name == "vertical line":
        return (np.abs(dx) < t / 2 + 0.5) & (np.abs(dy) <= size // 2 - 2)
    if name == "diagonal line":
        return (np.abs(dy - dx) <= t / 2 + 0.2) & (np.abs(dy) <= size // 2 - 2)
    if name == "triangle":
        top, bottom = cy - r, cy + r
        half = (yy - top) / (2 * r) * r  # half width grows downward
        edge = (yy >= top) & (yy <= bottom) & (np.abs(dx) <= half)
        inner = (yy >= top + 2 * t) & (yy <= bottom - t) & (np.abs(dx) <= half - 1.5 * t)
        return edge & ~inner
    if name == "heart":
        x, y = dx / r, -dy / r
        f = (x**2 + y**2 - 1) ** 3 - x**2 * y**3
        inside = f <= 0
        x2, y2 = dx / (r - t), -dy / (r - t)
        inner = (x2**2 + y2**2 - 1) ** 3 - x2**2 * y2**3 <= 0
        return inside & ~inner
    raise ValueError(name)


SHAPES = ("ring", "disc", "square", "plus", "x", "horizontal line", "vertical line", "diagonal line", "triangle", "heart")


def formation_score(mass, target):
    """F1 between occupied squares (mass >= 0.5) and the target mask: 1 when the robots fill the
    target exactly; robots outside it and target squares left empty both cost."""
    occupied = mass >= 0.5
    hit = (occupied & target).sum()
    if hit == 0:
        return 0.0
    precision, recall = hit / occupied.sum(), hit / target.sum()
    return float(2 * precision * recall / (precision + recall))
