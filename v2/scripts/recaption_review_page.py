"""A review page for the re-caption pilot: one card per shape (grey render, category, original caption, new
caption, the vision check's two options and pick, KEEP or DROP), for the user to check Haiku's verdicts on a phone.

    .venv/bin/python v2/scripts/recaption_review_page.py --vision runs/label_audit/recaption/vision_pilot --out v1/site/media/label-review

Published with the demo site (the Pages build copies v1/site/media): https://hrishik.me/beebot/media/label-review/
"""

import argparse
import html
import json
from pathlib import Path

from beebot.vla.voxel import goal_library as G
from beebot.vla.voxel import haiku as H

ROOT = Path("data/goal_library_v2")

STYLE = """
:root { --bg: #f6f6f3; --card: #ffffff; --ink: #1d1d1d; --muted: #6b6b6b; --keep: #1f7a4d; --drop: #b3261e; --line: #dddddd; }
@media (prefers-color-scheme: dark) { :root { --bg: #151515; --card: #1f1f1f; --ink: #ececec; --muted: #a0a0a0; --keep: #5fcf95; --drop: #ff8a80; --line: #333333; } }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.45 -apple-system, system-ui, sans-serif; }
main { max-width: 760px; margin: 0 auto; padding: 16px; }
h1 { font-size: 22px; margin: 8px 0; }
p.lead, .muted { color: var(--muted); }
table.sum { border-collapse: collapse; margin: 12px 0; width: 100%; font-size: 14px; }
table.sum td, table.sum th { border-bottom: 1px solid var(--line); padding: 4px 6px; text-align: left; }
.filters { display: flex; gap: 8px; flex-wrap: wrap; margin: 12px 0; }
.filters button { border: 1px solid var(--line); background: var(--card); color: var(--ink); border-radius: 16px; padding: 4px 12px; font-size: 14px; }
.filters button.on { border-color: var(--ink); }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; margin: 12px 0; overflow: hidden; }
.card img { width: 100%; display: block; background: #f6f6f3; }
.body { padding: 10px 12px; }
.head { display: flex; justify-content: space-between; gap: 8px; font-weight: 600; }
.tag { font-weight: 700; }
.tag.keep { color: var(--keep); }
.tag.drop { color: var(--drop); }
.k { color: var(--muted); font-size: 13px; text-transform: uppercase; letter-spacing: .03em; margin-top: 8px; }
.opt.own { font-weight: 600; }
"""

SCRIPT = """
const buttons = document.querySelectorAll('.filters button');
buttons.forEach(b => b.addEventListener('click', () => {
  buttons.forEach(x => x.classList.toggle('on', x === b));
  document.querySelectorAll('.card').forEach(c => { c.hidden = !(b.dataset.f === 'all' || c.dataset.v === b.dataset.f || c.dataset.s === b.dataset.f); });
}));
"""


def card(row, image):
    e = html.escape
    if row["unclear"]:
        verdict, cls, value = "UNCLEAR (text pass)", "drop", "unclear"
        options = ""
    else:
        keep = row["keep"]
        verdict, cls, value = ("KEEP" if keep else "DROP") + f" (picked {e(str(row['pick']))})", "keep" if keep else "drop", "keep" if keep else "drop"
        own = row["own_slot"]
        options = "".join(f'<div class="opt{" own" if slot == own else ""}">{slot}: {e(row[slot])}{" (own)" if slot == own else " (control)"}</div>' for slot in "AB")
        options = f'<div class="k">vision check options</div>{options}'
    return f"""<div class="card" data-v="{value}" data-s="{e(row['source'])}">
<img loading="lazy" src="{image}" alt="render of shape {row['n']}">
<div class="body"><div class="head"><span>#{row['n']} {e(row['source'])} / {e(row['label'])}</span><span class="tag {cls}">{verdict}</span></div>
<div class="k">original caption</div><div>{e(row['original'])}</div>
<div class="k">new caption</div><div>{e(row['text'] or 'UNCLEAR')}</div>
{options}</div></div>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vision", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    rows = H.read_jsonl(Path(args.vision) / "vision.jsonl")
    summary = json.loads((Path(args.vision) / "summary.json").read_text())
    cells = {}
    for source in sorted({r["source"] for r in rows}):
        lines, arrays = G.load(source, root=ROOT)
        table = {}
        for line, array in zip(lines, arrays):
            table.setdefault(line["source_id"], array)
        cells[source] = table
    cards = []
    for row in rows:
        name = f"img/{row['n']:03d}.png"
        G.render_views(cells[row["source"]][row["source_id"]], scale=9).save(out / name, optimize=True)
        cards.append(card(row, name))
    per = []
    for source in ("text2shape", "objaverse_cap3d"):
        mine = [r for r in rows if r["source"] == source]
        judged = [r for r in mine if not r["unclear"]]
        per.append(f"<tr><td>{source}</td><td>{len(mine)}</td><td>{len(mine) - len(judged)}</td><td>{sum(r['keep'] for r in judged)}</td><td>{sum(r['pick'] == 'neither' for r in judged)}</td><td>{sum(r['pick'] in ('A', 'B') and not r['keep'] for r in judged)}</td></tr>")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Caption review</title><style>{STYLE}</style></head><body><main>
<h1>Re-caption pilot: {summary['shapes']} shapes</h1>
<p class="lead">Haiku rewrote each shape's caption as one sentence about its shape in our grey cube world, or UNCLEAR when it would not read as its caption. Then Haiku looked at the render and picked its own new caption (KEEP), another category's caption, or neither (DROP). Check the verdicts by number.</p>
<table class="sum"><tr><th>source</th><th>shapes</th><th>UNCLEAR</th><th>kept</th><th>neither</th><th>wrong pick</th></tr>{''.join(per)}</table>
<p class="muted">Known weakness: the control is another category, so a caption naming parts the render lacks can still win (#3, a solid box kept as a table with a shelf).</p>
<div class="filters"><button class="on" data-f="all">all</button><button data-f="keep">keep</button><button data-f="drop">drop</button><button data-f="unclear">unclear</button><button data-f="text2shape">text2shape</button><button data-f="objaverse_cap3d">objaverse</button></div>
{''.join(cards)}
</main><script>{SCRIPT}</script></body></html>
"""
    (out / "index.html").write_text(page)
    print(json.dumps({"cards": len(cards), "out": str(out / "index.html")}))


if __name__ == "__main__":
    main()
