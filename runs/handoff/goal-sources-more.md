# goal-sources-more: voxel art, Objaverse, Text2Shape, and a Minecraft survey

Written 2026-11-13. Code: `src/beebot/vla/voxel/vox.py` (MagicaVoxel and Slab6 readers), `src/beebot/vla/voxel/goal_sources.py`
(coloured variants, colour storage, coloured render), `scripts/goals_sources_more.py` (converters),
`scripts/objaverse_select.py`, `scripts/itch_fetch.py`, `scripts/goals_render_more.py`. Tests: `tests/test_vox_reader.py`,
`tests/test_goal_sources.py` (all pass). Output: `data/goal_library_v2/<source>.{npz,jsonl}`; progress and summaries in
`logs/goal-sources-more.out` (`goal_library_v2_added` per source). Raw: `data/goal_sources/<source>/` (gitignored).

## Format

Same as the v2 library (cells, starts, jsonl; same acceptance, `variants(search=True)` rules, keep sizes 100, 250, 500, 800,
two rotations). Sources with colour add an `rgb` array (N, 3) uint8 to the npz, row-aligned with `cells`
(`goal_sources.load_rgb(name, root)`; `goal_library.load` ignores it). Colour per cell is the mean colour of the source
voxels under the coarse cell; cells added by `repair` take the mean of coloured neighbours. Objaverse stores no `rgb`.
Licence, url and uid are in each jsonl line's `extra`.

## Numbers

| source | raw downloaded | items | accepted goals | 4-115 | 116-250 | 251-500 | 501-800 | licences (goals) |
|---|---|---|---|---|---|---|---|---|
| oga_vox | about 85 MB zips (35 packs with .vox) | 675 models | 2,302 | 945 | 776 | 427 | 154 | CC0 1,698; CC-BY-SA 4.0 385; CC-BY 3.0 161; CC-BY-SA 3.0 44; CC-BY 4.0 14 |
| itch_vox (padadu) | 1 MB, 216 models | 216 | 877 | 343 | 293 | 182 | 59 | CC0 877 |
| text2shape | 1.07 GB zip (deleted) + 11 MB csv | 15,032 shapes (chairs, tables) | 72,271 | 22,923 | 23,838 | 18,500 | 7,010 | ShapeNet terms (non-commercial research) |
| objaverse_cap3d | 77 MB metadata; 9,204 glb streamed in memory, about 1.9 GB transferred, none on disk | 9,204 objects | 42,140 | 14,337 | 13,092 | 9,829 | 4,882 | CC-BY 41,145; CC-BY-SA 995 |

Total added: 117,590 goals.

1. oga_vox: CC0 page licences for medieval (363 models, one Models.vox), 55 creatures (Slab6 `.vox`, read by `vox.slab6`),
   buildings, bathroom, mech, trooper, creature pack 1, farm, christmas, bow, sword and others; CC-BY and CC-BY-SA packs
   (alphabet, character starter pack, skeleton set, tangled voxel assets, truck, character models) keep their licence in
   `extra.licence`; CC-BY needs attribution (`extra.url` is the pack page). The medieval pack's page comment about the archive
   licence was fixed by the author ("Forgot I had a txt in there"); the archive we downloaded holds only `Models.vox`, page says CC0.
   The pack archives carry no licence file apart from the bathroom readme (free for any purpose) and `license.png` images.
   Skipped: the "- all.vox" duplicate of the creatures, tangled-pack tiles and test files by name, models without a usable name.
1. itch_vox: padadu packs 02 nature, 03 weapons, 04 scifi, 06 medieval, 08 modular dungeon, 08 spooktober, 11 christmas, 12 farm,
   12 wwii tanks, fantasy bundle, 15 modern warfare; their pages state CC0 (also in `manifest.json`). Not downloaded (page states no
   licence): 05 urban city, 07 home furniture, 14 90s. Their models are one palette entry (single colour), so the render is grey.
   Names are made readable from file names ("sturmtiger", "keg"); numbered variants (_32, _48) of one model are separate sources.
1. text2shape: solid 32^3 RGBA voxelisation (nrrd, channel fastest, up axis second), first description is the text, all
   descriptions in `extra.descriptions`. ShapeNet-derived: the Text2Shape page offers the data without sign-up, but ShapeNet's own
   terms (non-commercial research and education) apply, same caveat as StableText2Brick. Not suited to a public release.
1. objaverse_cap3d: LVIS-named objects (46,207), licences CC0 / CC-BY / CC-BY-SA only (dropped 1,379 NC-SA, 1,433 NC, 2,331
   animated), faces <= 6,000, glb <= 1 MB, caption present: 9,204 kept at random; surface voxelisation at 32 cells on the
   longest axis, filled when the mesh is watertight, y up. Captions are Cap3D (ODC-By 1.0). 7 meshes failed (Path3D objects). The
   LVIS label is stored but is noisy (a baseball bat labelled "wooden_leg"); the caption is the text. No colour (textures not sampled).
   Only one metadata shard in 160 held CC0 objects of this kind (1 CC0 object kept), so the set is nearly all CC-BY.

## Render judgement (12 per source; `more-<source>-<k>.png`)

1. text2shape: 11 of 12 read as their text (chairs, tables, coffee table, desk; one "basket top" table is ambiguous).
1. oga_vox: 10 to 11 of 12 (letters, bowl, piranha, shrub, crate, urn, building, police car; the "wooden bottom" box is generic).
1. itch_vox: about 9 of 12, all grey and chunky (keg, tank, barrels, robot, ambulance, bales read; "barrier half", "firewood", "console" do not clearly).
1. objaverse_cap3d: about 7 of 12 (bench, Glock, knight with bow, cup, bookcase, fruit and traffic light read; the captions that
   list colours or several objects do not, and the render is grey).

Moondream will filter these further; the counts above are my eye on 12 tiles per source, not a measurement.

## Minecraft survey (not scraped)

| source | size | text | terms on bulk download and reuse |
|---|---|---|---|
| Hack337/Minecraft-Schematics (HF) | 669 builds, already in our library | title and description (en, ru) | MIT on HF; builds originate from community sites, so creators' terms may still apply |
| rom1504/minecraft-schematics-dataset | schematics, download script on GitLab | none stated | no licence file seen in the README; unclear |
| CraftAssist / 3D-Craft house_data (S3) | 2,537 houses (2.8 M placement steps), already in our library | none (houses) | CraftAssist code MIT; 3D-Craft's VoxelCNN repo CC-BY-NC 4.0; dataset licence not stated on its page |
| GrabCraft | large blueprint library (the site calls itself the biggest), per-blueprint block lists | titles, categories | ToS: content "must be for personal use only ... not for commercial use"; accounts may not be created by script, bot or crawler; users grant GrabCraft rights, not third parties; robots.txt disallows only `/*?q=`. No explicit licence for ML use. |
| Planet Minecraft | tens of thousands of schematics and maps, user posted | titles, descriptions, tags | behind a Cloudflare bot challenge (HTTP 403 to scripted fetch); ToS page not readable without a browser; read it manually before any scrape; each author owns the work |
| minecraft-schematics.com | many thousands of schematics | titles, descriptions | same Cloudflare 403; ToS not read; empty robots.txt |
| GDMC (github.com/GDMC) | competition settlements, generator outputs | none | repos are code; no build dataset to bulk download |
| Hugging Face sets (Retoe/image10k, 15k; PartyStarter240/text-to-red-v3, v4; FallingStars123/Minecraft_Architecture_Generator) | captions "a minecraft structure of: [name]" or redstone DSL | very short or DSL | licences not stated in the search results; check each card |

Which sites allow bulk use: none states permission. GrabCraft's terms forbid commercial use and bot accounts (browsing
is not addressed except "personal use"); Planet Minecraft and minecraft-schematics.com actively block scripted access, so
scraping would mean evading a challenge, which I did not attempt. The permissive sets (Hack337 MIT, CraftAssist MIT) are in
the library already. The decision to scrape is the user's.
