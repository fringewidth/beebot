# Label audit fixes (2026-11-14)

Applies the verdicts of `runs/label_audit/report.md`.

## Dropped by default

1. `goal_library.DROPPED_SOURCES = ("oga_vox", "itch_vox", "fusion360")`. `goal_library.load` returns `([], [])` for them and `goal_sources.load_rgb` returns `[]`, unless `include_dropped=True`. Data stays on disk. Browsing scripts (`goals_render_more.py`, `machine_parts_render.py`) pass `include_dropped=True`; `reverse_v3_measure.py --libraries all` skips them; `goal_library_v3.sh` converts oga/itch only with `CONVERT_DROPPED=1`.
1. `reverse_live.DROP_REWORDING_KINDS = ("goto_zone", "line_zone")`: `haiku_index` and `Texts` ignore Haiku rewordings of the zone and zone_line families, so they use the template wording. `drop_kinds=()` brings them back.

## Generator wording fixes (source, not stored files)

1. gen_simple: `tower_row` "each one one cube taller" -> "each one cube taller than the one before"; `floor_cross` "seven wide one way and three the other" -> "seven wide left to right and three wide front to back"; `wall` "stretches nine across, front to back" -> "nine long that stretches front to back"; `frame_standing` "four left to right and five high" -> "four wide left to right and five high".
1. gen_medium: window parentheticals -> "a tall window", "three tall windows"; balcony "two out, three up" -> "sticking out two blocks, three blocks up from the ground"; person "seen face on from the right" -> "standing flat, facing the right"; letters "facing the back" -> "readable from the back" (same phrase in both letter wordings).
1. gen_complex: "right right in the centre" -> "exactly in the centre"; tower wordings use only "tower" (no "column", "stack of"); "stack N more onto" -> "put N more on top of"; "N farther" -> "N spaces farther"; house door edits (`door_move`, `door_close`) redraw the first turn until it mentions the door, else reject.
1. gen_ground: "finish it" family -> names the structure ("finish the structure"); lone-cube-to-tower wording says "the only tower".
1. gen_medium 1 vs 58 ("facing the back" vs "readable from the back") is unified as above.

## Regenerated

1. `data/goal_library_v3/gen_simple` (20000 goals, default seed 0). The previous files are kept as `gen_simple.{jsonl,npz}.pre-label-fix`. Spot count: 0 occurrences of the fixed phrases.
1. `data/goal_library_v3/gen_medium` (16562 goals, same count; backup `.pre-label-fix`). Parenthetical window and face-on wordings: 0 left (1015 before).
1. Not regenerated: gen_complex and gen_ground (`data/goal_library`, 10000 and 20000 goals; slower, reverse checks; ask before running). Their stored wording still has the old text until they are.

Regenerate with `nice .venv/bin/python -I -m beebot.vla.voxel.goal_gen_<name> --out-root data/goal_library_v3` (gen_simple took about 14 min wall under load, 3 min CPU; gen_complex and gen_ground write to `data/goal_library` by default and are the slow ones).

## FIX sources (not run)

`scripts/recaption_prepare.py --out runs/label_audit/recaption_batch.jsonl` writes one request per distinct shape (text2shape 15032, objaverse_cap3d 9190, 24222 total) and prints the cost. It sends nothing. About 7.3M input and 1.0M output tokens (chars/3.5 and 40 output tokens per item; assumptions). claude-haiku-5-5 ($0.10/$0.50 per M): about $1.22, $0.61 with the Batch API; claude-haiku-4-5 ($1/$5): about $12.19, $6.09 batch. Prices from the claude-api skill table cached 2026-11-03. Needs the user's yes. Objaverse category mismatches: the script lists rows whose category shares no word with any caption (4469 candidates, too noisy to trust). The ~15 real disagreements should be found by a Haiku yes/no judge ("does this caption describe a <category>?") over those candidates, about 4469 short calls, under 1 USD.

## MCB

KEEP with a caveat: the `text` field is empty in all rows, only class names. Left as is.
