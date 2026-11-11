"""Thoughts that map the user's words onto the board (format a), written by Haiku.

    python -m beebot.vla.voxel.haiku_frame items    [--out data/vla/haiku/frame] [--seed 0]
    python -m beebot.vla.voxel.haiku_frame generate [--out data/vla/haiku/frame] [--limit N]
    python -m beebot.vla.voxel.haiku_frame check    [--out data/vla/haiku/frame]

The earlier labels (haiku.py) were written in the board's frame: Haiku saw the scene and never a
user standing somewhere. ThinkData then turned the direction words mechanically into the user's
frame, which leaves the model to learn the mapping with no reasoning about it, and drops every
label with compass or foreign direction words. Here each verified item whose task has a direction
gets a camera side (the board side the oblique camera looks from) and a user position relative
to the camera (USER_POSITIONS); Haiku writes what that user would type and a thought that first
states where the user stands on the board and what their words mean there, then plans in the
board's terms:

    you asked: <instruction>. i see <scene>. the camera looks from the board's <side> and you
    stand <place>, so you face the board from its <side>. <what the words mean on the board>.
    first: <first sub-goal in the board's terms>.

Board terms are the world frame (views.py): left small x, right large x, back small y, front
large y. The user's words follow how people talk: "forward", "ahead", "north", "away from me" go
to the far side; "toward me", "backward", "south" to the near side; "the front" of the board is
the side nearest the user and "the back" the far side; "up" and "down" are height.

check: rows that pass the mechanical checks go to verified.jsonl (frame phrases exact; the first
sub-goal names the same board directions as the canonical one and no user-relative words; the
user's direction words, English, Spanish or French (USER_WORDS, accents ignored), mapped onto the
board, name the same directions as the task; each statement of what their words mean, "<their
words> is the board's <side>" or the other way round, maps them right).
"""

import argparse
import collections
import glob
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

from beebot.vla.voxel import haiku as H
from beebot.vla.voxel.think_data import ORDER, USER_POSITIONS

SOURCES = ("data/vla/haiku/run-s?", "data/vla/haiku/run-s?-later")
BATCH = 12
DIRECTION = re.compile(r"\b(front|back|left|right|forward|backward|behind|rear)\b")
BOARD_WORD = re.compile(r"\b(front|back|left|right)\b")
HIDDEN_WEIGHT = re.compile(r"\bheav|\blight(er|est)?\b")
USER_RELATIVE = re.compile(r"\b(your|my|you|me|i|north|south|east|west|forward|forwards|ahead|toward|towards|away|near|far)\b")
SIDES = r"((?:front|back|left|right)(?:(?:[ -]| and )(?:front|back|left|right)\b(?! (?:is|are|means|=)\b))?)"  # "back right", "right and back", not "front and right is ..."
STATEMENT = re.compile(r"([a-z' -]{2,40}?) (?:is|are|means|points to|=) (?:the board's |board )?" + SIDES + r"\b")  # "<their words> is the board's <side>"
REVERSED = re.compile(r"(?:the board's|board) " + SIDES + r"(?: side| corner| edge| end)? (?:is|are|means|=) ([a-z' -]{2,40}?)(?=[,.;:]|$| and | so | but )")  # "the board's <side> is <their words>"
PLACES = {  # how the prompt says the user's place relative to the camera
    "behind": "behind the camera",
    "left": "to the left of the camera",
    "right": "to the right of the camera",
    "opposite": "opposite the camera, facing it",
}
SIDE_NAMES = {"front": "front", "right": "right", "back": "back", "left": "left"}
NOT_A_THING = r"(?!\s+(?:del|de la|de las|de los|du|des|de l')\b)"  # "cerca del queso", "près du fromage": near a thing, no direction
SIDE_WORD = r"(?:droit|droite|gauche|derech[oa]|izquierd[oa]|du|de)"  # "l'avant droit", "l'avant du sol": the front, not forward
COMPASS_END = r"(?=\s*(?:corners?|area|side|part|end|of|and|n|&|[,.;!]|$))"  # what follows an abbreviated compass point
USER_WORDS = (  # (pattern in the user's text with accents taken off (plain), the user's directions it names), tried in order, each match taken out
    # compass words, compounds first: English "northeast" leaves "east" once "north" is taken out
    (r"\bnor(?:d-?)?(?:este|est)\b", ("away", "right")),  # noreste, nordeste, nord-est
    (r"\bnoroeste\b|\bnord-?ouest\b", ("away", "left")),
    (r"\bsu[rd]-?(?:este|est)\b", ("toward", "right")),  # sureste, sudeste, sud-est
    (r"\bsu[rd]-?oeste\b|\bsud-?ouest\b", ("toward", "left")),
    (r"\bne\b" + COMPASS_END, ("away", "right")),  # "the NE corner", "one goes sw, other goes se"; not French "ne" nor Spanish "se"
    (r"\bnw\b", ("away", "left")),
    (r"\bse\b" + COMPASS_END, ("toward", "right")),
    (r"\bsw\b", ("toward", "left")),
    (r"\b(?:my|our|your|ur) (?:side|way)\b|\b(?:to|by|at|near) (?:me|us|u)\b|\bclose (?:side|edge|end)\b|\bpa(?:ra)? aca\b|\bhacia aca\b|\b(?:hacia )?mi lado\b", ("toward",)),
    (r"\balej\w*", ("away",)),  # "se aleja de mi", "alejate"
    (r"\b(?:ahead|forwards?|away from (?:me|us|you|u)|far|farthest|furthest)\b|\bnorth|\bnorte\b|\bnord\b|\btout droit\b", ("away",)),
    (r"\b(?:backwards?|toward(?:s)? (?:me|us|you|u|ya)|toward(?:s)? (?:my|our|your|ur) side|near|nearest|closest)\b|\btowards?\b(?!\s+(?:the|a|an|it|them|each|my|your|our|ur|his|her|their)\b)|\bsouth|\bsud\b|\b(?:al|el|del|hacia el|lado|zona|esquina|parte)\s+sur\b", ("toward",)),  # "toward my left" is left
    (r"\beast(?:ern)?\b|\b(?:al|el|del|hacia el|lado|zona|esquina|parte)\s+este\b|\bl'est\b", ("right",)),  # not "least"; Spanish "este" is also "this", French "est" "is"
    (r"\bwest(?:ern)?\b|\boeste\b|\bouest\b", ("left",)),
    # Spanish and French: motion first ("hacia adelante" is forward, "de adelante" the front), then the sides
    (r"\bhacia (?:adelante|delante)\b|\bpa(?:ra)? (?:a)?lante\b|\bpalante\b|\ben avant\b|\bvers l'avant\b(?!\s+" + SIDE_WORD + r"\b)", ("away",)),
    (r"\bhacia atras\b(?!\s+(?:a\s+la\s+)?(?:derech|izquierd))|\ben arriere\b|\bvers l'arriere\b(?!\s+" + SIDE_WORD + r"\b)", ("toward",)),
    (r"\bhacia (?:mi|ti|nosotros|ustedes)\b(?!\s+(?:derech|izquierd|izq|lado))|\bvers (?:moi|toi|nous|vous)\b|\b(?:cerca|pres) de (?:mi|ti|nosotros|moi|toi|nous|vous)\b|\bcerca\b" + NOT_A_THING + r"|\bcercan[oa]s?\b|\bpres\b" + NOT_A_THING + r"|\bproches?\b" + NOT_A_THING + r"|\bpremier plan\b|\bde adelante\b|\bdelante\b|\bdelanter[oa]s?\b|\bdevant\b|\bfrente\b|\bfrontal\b|\b(?:coin|bord|cote|partie|a l') ?avant\b|\bl'avant\b(?=\s+" + SIDE_WORD + r"\b)|\bavant\b(?=\s+(?:droit|gauche)\b)", ("toward",)),  # near, the front
    (r"\blejos\b" + NOT_A_THING + r"|\blejan[oa]s?\b|\bloin\b" + NOT_A_THING + r"|\blointaine?s?\b|\beloignee?s?\b|\bl'autre cote\b|\bel otro lado\b|\bde enfrente\b|\bfondo\b|\bfond\b|\batras\b|\bdetras\b|\btraser[oa]s?\b|\barriere\b|\badelante\b", ("away",)),  # far, the back; "adelante" alone is forward
    (r"\bavant\b", ()),  # alone, forward (away) or the front (toward): not read
    (r"\bderech[oa]s?\b|\bdroite?s?\b", ("right",)),
    (r"\bizquierd[oa]s?\b|\bizq\b|\bgauche\b", ("left",)),
    (r"\bfront\b", ("toward",)),
    (r"\bback\b", ("away",)),
    (r"\bleft\b", ("left",)),
    (r"\bright\b", ("right",)),
)


def plain(text):
    """Lower case without accents ("atrás" -> "atras", "près" -> "pres")."""
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))


def mapping(user_side):
    """The board side each of the user's directions points to, for a user at user_side."""
    turns = ORDER.index(user_side)
    return {
        "toward": ORDER[turns % 4],
        "right": ORDER[(1 + turns) % 4],
        "away": ORDER[(2 + turns) % 4],
        "left": ORDER[(3 + turns) % 4],
    }


def user_side(camera_side, place):
    """The board side the user stands at: the camera's side turned by the place's offset."""
    return ORDER[(ORDER.index(camera_side) + USER_POSITIONS[place][0] // 90) % 4]


def board_words(text):
    return sorted(BOARD_WORD.findall(text.lower()))


def user_text_words(text, table):
    """The set of board directions the user's direction words name, in English, Spanish or French
    (USER_WORDS; empty when it has none the patterns read: another language, a phrase not listed)."""
    directions = set()
    rest = plain(text)
    for pattern, named in USER_WORDS:
        if re.search(pattern, rest):
            directions.update(table[d] for d in named)
        rest = re.sub(pattern, " ", rest)
    return directions


def first_clause(thought):
    head, sep, tail = thought.rpartition("first:")
    return tail.strip() if sep else ""


def select(seed=0):
    """One entry per verified item whose task names a direction: its record, scene, a verified
    instruction (board frame), and a drawn camera side and user place."""
    rng = random.Random(seed)
    entries = []
    for pattern in SOURCES:
        for directory in sorted(glob.glob(pattern)):
            items = {item["id"]: item for item in H.read_jsonl(Path(directory) / "items.jsonl")}
            seen = set()
            for pair in H.read_jsonl(Path(directory) / "verified.jsonl"):
                if pair.get("rung") == H.REFUSAL or pair["id"] in seen or pair["id"] not in items:
                    continue
                spec = json.dumps(pair.get("spec", {}))
                if not DIRECTION.search(spec) or HIDDEN_WEIGHT.search(pair["instruction"] + spec):
                    continue
                item = items[pair["id"]]
                goals = item["record"].get("subgoals") or []
                if not goals or not board_words(goals[0]):
                    continue
                seen.add(pair["id"])
                camera = ORDER[rng.randrange(4)]
                place = list(PLACES)[rng.randrange(4)]
                entries.append({
                    "id": f"{Path(directory).name}:{pair['id']}",
                    "source": directory,
                    "item": pair["id"],
                    "pair_id": pair["pair_id"],
                    "later": "later" in Path(directory).name,
                    "task": item["record"].get("plain"),
                    "sub_goals": goals,
                    "said": pair["instruction"],
                    "scene": item["scene"],
                    "camera": camera,
                    "place": place,
                    "user": user_side(camera, place),
                })
    rng.shuffle(entries)
    return entries


PROMPT = """You write training text for a swarm of small cube robots in a 16 x 16 x 16 voxel world, seen by a camera, and steered by a person standing somewhere around the square board who types instructions in their own words.

The board has four sides with fixed names: its left (small x), right (large x), back (small y) and front (large y); z is height. Each item's task, sub-goals, scene and "said" (a verified instruction) use these board names.

Each item also gives "camera": the board side the camera looks from; "place": where the person stands relative to the camera; "user": the board side the person stands at, facing the board; and "theirs": which board side each of the person's own directions points to ("toward": toward them, "away": away from them, "left" and "right": their left and right as they face the board).

For each item write two instructions the person would type for exactly this task, in their own frame, never with the board's names:
1. "plain": ordinary words. Their left and right are their own; "the front" or "the near side" is the side nearest them and "the back" or "the far side" is away from them; "forward", "ahead" and "away from me" go away from them, "toward me" and "backward" toward them.
2. "casual": the way people type in a chat: lower case, loose grammar, maybe a typo. In about half of the items use compass words (north = away from them, south = toward them, east = their right, west = their left) or write it in Spanish or French.
"up" and "down" mean height only. Each instruction asks for the same thing as the task: same targets, same places once mapped onto the board, nothing added. Never say "the board's".

For each instruction write the swarm's first thought, in lower case, at most 50 words besides the quoted instruction, in exactly this shape:
"you asked: <the instruction>. i see <the scene contents that matter, in board names>. the camera looks from the board's <camera> and you stand <place>, so you face the board from its <user>. <what their words mean on the board, e.g. your left is the board's front; north is the board's left>. first: <the first sub-goal in board names>."
The two phrases "the camera looks from the board's <camera>" and "you face the board from its <user>" must appear word for word. After "first:" use only board names (front, back, left, right of the board), never your, my, me, north, forward, toward, away.
Then "switches": one short lower-case line in board names for each later sub-goal, in order; none for a single sub-goal.
Items with "later": true arrive mid-task: never refer to where robots or pushed things are; "i see" names only terrain and which cheeses or objects there are.
Never state a number of robots of 10 or more (no one can count that many from a picture); smaller numbers are fine. Mention only colours, things and terrain the scene has. Never say which object is heavy.

Reply with JSON only, one object per item id and no other text:
{"<id>": {"instructions": [{"rung": "plain", "text": "...", "thought": "...", "switches": []}, {"rung": "casual", "text": "...", "thought": "...", "switches": []}]}}

Items (one JSON object per line):
@ITEMS@
"""


def prompt_entry(entry):
    return {
        "id": entry["id"],
        "task": entry["task"],
        "sub_goals": entry["sub_goals"],
        "said": entry["said"],
        "scene": entry["scene"],
        "later": entry["later"],
        "camera": entry["camera"],
        "place": PLACES[entry["place"]],
        "user": entry["user"],
        "theirs": mapping(entry["user"]),
    }


def prompt_of(batch):
    return PROMPT.replace("@ITEMS@", "\n".join(json.dumps(prompt_entry(e), ensure_ascii=False) for e in batch))


def problems(entry, row):
    """Why a generated instruction fails the mechanical checks (empty: it passes)."""
    out = []
    thought = row.get("thought", "").lower()
    text = row.get("text", "")
    if f"the camera looks from the board's {entry['camera']}" not in thought:
        out.append("camera_phrase")
    if f"you face the board from its {entry['user']}" not in thought:
        out.append("user_phrase")
    clause = first_clause(thought)
    if not clause:
        out.append("no_first")
    elif board_words(clause) != board_words(entry["sub_goals"][0]):
        out.append("first_directions")
    elif set(USER_RELATIVE.findall(clause)) - set(USER_RELATIVE.findall(entry["sub_goals"][0].lower())):
        out.append("first_user_words")
    meaning = plain(thought).split("you face the board from its")[-1].split("first:")[0]
    statements = [(phrase, sides) for phrase, sides in STATEMENT.findall(meaning) if "board" not in phrase]  # "<their words> is the board's <side>"
    statements += [(phrase, sides) for sides, phrase in REVERSED.findall(meaning)]  # "the board's <side> is <their words>"
    for phrase, sides in statements:
        named = user_text_words(phrase, mapping(entry["user"]))
        if named and named != set(board_words(sides)):
            out.append("wrong_statement")
            break
    if "board's" in text.lower():
        out.append("board_names_in_text")
    named = user_text_words(text, mapping(entry["user"]))
    if named and named != set(board_words(entry["task"] or entry["sub_goals"][0])):
        out.append("text_directions")
    if len(row.get("switches") or []) != len(entry["sub_goals"]) - 1:
        out.append("switch_count")
    return out


def generate(out, provider=H.haiku, limit=None, workers=H.WORKERS, passes=2):
    entries = H.read_jsonl(out / "items.jsonl")[:limit]
    by_id = {e["id"]: e for e in entries}
    done = {row["id"] for row in H.read_jsonl(out / "generated.jsonl")}

    def on_answers(batch, answers, call):
        for identifier, answer in answers.items():
            for row in (answer or {}).get("instructions", []):
                if not isinstance(row, dict):
                    continue
                entry = by_id[identifier]
                record = {**row, "id": identifier, "pair_id": f"{identifier}:{row.get('rung')}", "camera": entry["camera"], "place": entry["place"], "user": entry["user"], "call": call, "time": H.now()}
                record["problems"] = problems(entry, record)
                H.append(out / "generated.jsonl", record)
                if not record["problems"]:
                    H.append(out / "verified.jsonl", record)

    for _ in range(passes):
        done = {row["id"] for row in H.read_jsonl(out / "generated.jsonl")}
        pending = [e for e in entries if e["id"] not in done]
        if not pending:
            break
        with H.Meter("frame", len(pending), out / "meter.jsonl") as meter:
            H.run_batches(H.chunks(pending, BATCH), prompt_of, lambda b: [e["id"] for e in b], provider, on_answers, meter, out / "calls.jsonl", out / "failures.jsonl", workers=workers)


def report(out):
    rows = H.read_jsonl(out / "generated.jsonl")
    reasons = collections.Counter(p for row in rows for p in row["problems"])
    passed = sum(not row["problems"] for row in rows)
    print(json.dumps({"run": "haiku-frame", "generated": len(rows), "verified": passed, "keep": round(passed / max(len(rows), 1), 3), **reasons}))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("items", "generate", "check"))
    parser.add_argument("--out", default="data/vla/haiku/frame")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=H.WORKERS)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "items":
        entries = select(args.seed)
        (out / "items.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries))
        print(json.dumps({"run": "haiku-frame", "items": len(entries), **collections.Counter(e["user"] for e in entries)}))
    elif args.stage == "generate":
        generate(out, limit=args.limit, workers=args.workers)
        report(out)
    else:  # re-run the checks on every generated row and rewrite verified.jsonl
        by_id = {e["id"]: e for e in H.read_jsonl(out / "items.jsonl")}
        rows = H.read_jsonl(out / "generated.jsonl")
        for row in rows:
            row["problems"] = problems(by_id[row["id"]], row)
        (out / "generated.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        (out / "verified.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows if not r["problems"]))
        report(out)


if __name__ == "__main__":
    sys.exit(main())
