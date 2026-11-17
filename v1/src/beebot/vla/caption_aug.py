"""Rewrite a tagged caption to match an augmented scene, failing closed.

Rotations and reflections follow data.transform_grid: k quarter turns counterclockwise, then a
left-right flip. Under one counterclockwise quarter turn the top goes to the left, the left to the
bottom, the bottom to the right and the right to the top.

Only words inside <d>...</d> are rewritten, and only from a known table; a caption is refused
(None) when a direction word sits outside a tag or a tag holds a word the table can't map
(above, below, compass points, clockwise, ...). Colour words are swapped only inside
<c>...</c>, and a colour word outside a tag refuses the caption.

Vertical words come in two forms: places (top, bottom) and motions (up, down). A word keeps its
form when it has one (top stays a place, up stays a motion); a left or right that turns vertical
becomes a place when "the" comes right before it or a place noun (corner, edge, side, ...)
comes after it, and a motion otherwise ("move left" -> "move down", "the left" -> "the bottom").
After mapping, a horizontal word directly followed by a vertical one ("left bottom", "left and
down", "left and to the bottom") is swapped into English order.
"""

import re

from beebot.vla.caption import COLOUR_WORDS, DIRECTION_WORDS
from beebot.vla.world import PALETTE_NAMES

# Directions as (row, col) unit steps; row 0 is the top.
NORTH, SOUTH, WEST, EAST = (-1, 0), (1, 0), (0, -1), (0, 1)
WORD_DIRECTION = {"top": NORTH, "upper": NORTH, "up": NORTH, "bottom": SOUTH, "lower": SOUTH, "down": SOUTH, "left": WEST, "right": EAST}
PLACE_FORM = {NORTH: "top", SOUTH: "bottom", WEST: "left", EAST: "right"}
MOTION_FORM = {NORTH: "up", SOUTH: "down", WEST: "left", EAST: "right"}
PLACES = {"top", "upper", "bottom", "lower"}
MOTIONS = {"up", "down"}
PLACE_NOUNS = {"corner", "corners", "edge", "edges", "side", "sides", "half", "wall", "walls", "part", "area", "border", "end", "quarter", "region"}
AXIS_WORDS = {"horizontal": "vertical", "vertical": "horizontal", "horizontally": "vertically", "vertically": "horizontally"}
UNMAPPABLE = {"above", "below", "beneath", "under", "over", "north", "south", "east", "west", "clockwise", "counterclockwise", "anticlockwise", "leftmost", "rightmost", "topmost", "bottommost", "uphill", "downhill", "upward", "upwards", "downward", "downwards", "leftward", "leftwards", "rightward", "rightwards"}
TAG = re.compile(r"<(d|c|u)>(.*?)</\1>")
TOKEN = re.compile(r"[A-Za-z]+|[^A-Za-z]+")


def turn(direction, k, flip):
    row, col = direction
    for _ in range(k % 4):
        row, col = -col, row  # counterclockwise: top (-1, 0) -> left (0, -1)
    return (row, -col) if flip else (row, col)


def outside_words(caption):
    return re.findall(r"[a-z]+", TAG.sub(" ", caption).lower())


def match_case(word, like):
    return word.capitalize() if like[:1].isupper() else word


def rewrite_span(span, previous_word, k, flip):
    """The span with its direction words mapped, or None if it holds an unmappable word."""
    tokens = TOKEN.findall(span)
    words = [i for i, tok in enumerate(tokens) if tok[0].isalpha()]
    mapped = {}  # token index -> (direction, form) where form is "place", "motion" or "either"
    for position, i in enumerate(words):
        low = tokens[i].lower()
        if low in UNMAPPABLE:
            return None
        if low in AXIS_WORDS:
            if k % 2:
                tokens[i] = match_case(AXIS_WORDS[low], tokens[i])
            continue
        if low not in WORD_DIRECTION:
            continue
        form = "place" if low in PLACES else "motion" if low in MOTIONS else "either"
        if form == "either":
            before = tokens[words[position - 1]].lower() if position > 0 else previous_word
            after = [tokens[j].lower() for j in words[position + 1 : position + 3]]
            form = "place" if before == "the" or any(a in PLACE_NOUNS for a in after) else "motion"
        mapped[i] = (turn(WORD_DIRECTION[low], k, flip), form)

    # English puts the vertical word first: "left bottom" -> "bottom left", "right and up" -> "up and right".
    order = [i for i in words if i in mapped]
    for a, b in zip(order, order[1:]):
        between = [tokens[j].lower() for j in words if a < j < b]
        if mapped[a][0][0] == 0 and mapped[b][0][1] == 0 and between in ([], ["and"], ["and", "to", "the"]):
            # Swap the directions; each slot keeps its own form ("up and to the left" turned once
            # reads "down and to the left", not "left and to the bottom").
            (dir_a, form_a), (dir_b, form_b) = mapped[a], mapped[b]
            mapped[a], mapped[b] = (dir_b, form_a), (dir_a, form_b)
    for i, (direction, form) in mapped.items():
        table = PLACE_FORM if form == "place" else MOTION_FORM
        word = table[direction]
        if form == "place" and direction in (NORTH, SOUTH) and tokens[i].lower() in ("upper", "lower"):
            word = "upper" if direction == NORTH else "lower"
        tokens[i] = match_case(word, tokens[i])
    return "".join(tokens)


def rewrite(caption, k=0, flip=False, colour_map=None):
    """The caption for a scene turned k quarter turns counterclockwise then flipped, and with
    colour index i renamed to colour_map[i]; None if it can't be rewritten safely."""
    if re.search(r"</?[dcu]>", TAG.sub(" ", caption)):
        return None  # unbalanced or nested tags
    loose = outside_words(caption)
    if (k % 4 or flip) and any(w in DIRECTION_WORDS or w in UNMAPPABLE for w in loose):
        return None
    if colour_map is not None and any(w in COLOUR_WORDS for w in loose):
        return None
    names = {name: i for i, name in enumerate(PALETTE_NAMES)}
    pieces = []
    last = 0
    for m in TAG.finditer(caption):
        pieces.append(caption[last : m.start()])
        kind, span = m.group(1), m.group(2)
        before = re.findall(r"[a-z]+", caption[: m.start()].lower())
        previous_word = before[-1] if before else ""
        if kind == "d" and (k % 4 or flip):
            span = rewrite_span(span, previous_word, k, flip)
            if span is None:
                return None
        elif kind == "c" and colour_map is not None:
            if span.lower() not in names:
                return None
            span = match_case(PALETTE_NAMES[colour_map[names[span.lower()]]], span)
        elif kind == "u":
            inner = rewrite(span, k, flip, colour_map)  # a <u> claim may hold its own tags
            if inner is None:
                return None
            span = inner
        pieces.append(f"<{kind}>{span}</{kind}>")
        last = m.end()
    pieces.append(caption[last:])
    return "".join(pieces)
