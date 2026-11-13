# machine-parts: constituent parts of machines, voxelised to the swarm grid

Written 2026-11-13. Code: `src/beebot/vla/voxel/machine_parts.py` (names, measured shape class, orientation, mesh
voxelising, Fusion 360 assembly parsing, whole-assembly part-id voxels), `src/beebot/vla/voxel/remote_file.py`
(read a 7z over HTTP range requests, nothing stored), `scripts/machine_parts_mcb.py` + `machine_parts_mcb_stream.sh`,
`scripts/machine_parts_fusion.py`, `scripts/machine_parts_relabel.py`, `scripts/machine_parts_stats.py`,
`scripts/machine_parts_render.py`, tests `tests/test_machine_parts.py` (10). Outputs in `data/machine_parts/` (40 MB,
gitignored); progress and final lines in `logs/machine-parts.out` (`machine_parts_accepted` = 48,870 variants).

## Where the raw data lives

1. MCB (2.5 GB tarball) was never stored: curl pipes the Box download straight into the converter (disk was at 6-9 GB
   free for most of the session). Re-running needs the download again (about 1.5 hours; the voxelising dominates).
1. Fusion 360 archive `a1.0.0_04.7z` (1.2 GB) was read over HTTP range requests; only `assembly.json` of all 751
   assemblies and the body meshes of the first 152 were extracted (to `data/machine_sources/fusion360_assembly/a1.0.0_04/`,
   106 MB now: the json files; the meshes, 390 MB, were deleted after use). Nothing else is on disk.
1. `data/machine_sources/mcb/` is empty by design.

## Sources

| source | what was used | size read | licence |
|---|---|---|---|
| Fusion 360 Gallery Assembly (Autodesk; github.com/AutodeskAILab/Fusion360GalleryDataset) | archive 04 of 11 (751 of 8,251 assemblies) | 1.2 GB streamed, 106 MB kept | non-commercial research only; no redistribution of the whole dataset; portions or modifications only with attribution and the same restriction (LICENSE.md). Our `data/machine_parts` is a modification: keep it out of the public repo (it is gitignored) |
| Mechanical Components Benchmark, dataset A (Purdue; github.com/stnoah1/mcb, Box link) | whole tarball, `dataset_org_norm` (test and train) | 2.5 GB streamed | not stated on the repo or the project page |

Skipped: the other 10 Assembly archives (18.8 GB compressed, 146 GB extracted; the disk could not hold them and the
manager lowered the floor to 12 GB mid-run), the Assembly-Joint set (2.8 GB; it is pairs of parts cut out of these same
assemblies with the same joint records, so the assembly archive already carries the joints), MCB dataset B (a separate
18k-model Box link, not fetched), PartNet-Mobility and ABC (per instructions). MCB dataset A holds 24,215 parts, not the
58,696 of the paper (the paper's number covers both datasets). No better public dataset with functional labels and
joints turned up in the time available; PartNet-Mobility remains the obvious one (needs the user's sign-up).

## Method

1. Parse: Fusion `assembly.json` gives the tree (occurrence parents, depth), components, bodies (designer name,
   material, physical properties), occurrence transforms (composed down the tree), `joints` and `as_built_joints`
   (type, occurrence pair, axis and limits where given). A part is a body. A joint is attached to every part of both
   occurrences. MCB gives the class (directory name) and the mesh; there are no joints or assembly.
1. Orientation: rest on the largest planar face of the convex hull (coplanar hull triangles grouped, at least 5% of the
   hull area), else the shortest bounding-box axis down; plus the file's own frame (z up) when different. Each part is
   voxelised on a 64-cell fine grid along its longest axis (surface voxels of the subdivided mesh, interior filled), then
   re-voxelised at scales 2^(-k/2) up to 1 (never upsampled) through `goal_library.best_variant` (the v2 voxeliser with
   fill-threshold and phase search and floating-leg repair), accepted when it fits 16 per axis, holds 4 to 800 cells and
   is face-connected to z = 0 (the manager lifted the cap from 115 to 800 and the kept sizes to the variant nearest each
   of 100, 250 and 600 cells, each in one extra random quarter turn). Meshes over 40,000 triangles are tabled but not
   voxelised (memory): 182 MCB parts.
1. Storage: `<source>.npz/.jsonl` in goal_library layout (cells z, y, x; fields include `extra.orientation`,
   `cell_mm`, `thickness_cells` = shortest bounding extent in cells, `dims_mm`, `joint_kinds`), per-part tables
   `<source>_parts.jsonl` (every part, also those not voxelised), `fusion360_assembly_table.jsonl`, and
   `fusion360_assemblies.npz/.jsonl` (130 assemblies voxelised whole, cells z, y, x, part index, part names in the jsonl).
1. Units: the Fusion docs say cm and the data agree (the McMaster "M6" bolts come out 6.0 x 6.0 x 10 mm after x10). MCB
   meshes are normalised to a unit box: no real size, only `dims_norm` and aspect ratios.
1. Name rule: a name is auto-generated when empty, a bare number or hex id, or a CAD default (Body12, Component3:1,
   Part1, Untitled, root, prt0002, asm0002); everything else is "human" (it can still be uninformative). The part label
   is the first human name among occurrence, component and body name (":1" suffix removed).
1. Shape class (measured only): sorted bounding box a <= b <= c; cube when c/a <= 1.5; plate when a/b < 0.35; rod when
   c/b > 2.5; strip when both; else block.

## Numbers

| | Fusion 360 (archive 04) | MCB (dataset A) |
|---|---|---|
| assemblies | 751 (152 with meshes, 130 voxelised whole) | none |
| parts (bodies) | 13,207 (3,008 with meshes voxelised) | 24,215 in 68 classes (5,335 voxelised, first 120 per class in tar order; 182 skipped for size) |
| joints | 1,843 in 142 assemblies | none |
| parts with an accepted variant | 3,008 of 3,008 | 5,335 of 5,335 |
| variants accepted (4-800 cells) | 15,703 | 33,167 |
| cells, quartiles of variants | 76 / 138 / 290 | 96 / 174 / 341 |

Every voxelised part got at least one variant because some scale always fits; rejection happens per scale (Fusion:
oversize 25,305, small 4,859, big 554, ungrounded 7 scale tries; MCB: oversize 40,044, small 8,654, big 1,225,
ungrounded 7). "Small" (under 4 cells, or empty after rescale) is the thin and tiny failure; it affects scales, not
parts. Cell-count histogram of variants (4-15, 16-31, 32-63, 64-99, 100-149, 150-249, 250-399, 400-599, 600-800):
Fusion 1, 964, 1931, 2935, 2354, 3001, 1820, 1647, 1050; MCB 0, 131, 1890, 6996, 5629, 6542, 5189, 3302, 3488.
(The assemblies' own variants are the single scale nearest 250 cells, 4-800.)

### Joints (Fusion, archive 04 only; 142 of 751 assemblies have any joint, 19%)

| type | joints | share | assemblies containing it |
|---|---|---|---|
| rigid | 1,275 | 69.2% | 99 |
| revolute | 384 | 20.8% | 84 |
| slider | 113 | 6.1% | 32 |
| cylindrical | 40 | 2.2% | 12 |
| planar | 18 | 1.0% | 4 |
| ball | 11 | 0.6% | 4 |
| pin_slot | 2 | 0.1% | 2 |

1. Of the 142 jointed assemblies, 100 (70%) contain at least one non-rigid joint.
1. Parts: 78.6% take part in no joint at all (10,380 of 13,207; most assemblies are posed with no joint records),
   12.9% only rigid joints, 8.5% at least one free joint. Among parts that have any joint, 39.6% have a free one.
1. Only revolute (20.8%) and slider (6.1%) are common; all others together are 4.0%. Axis and limits are stored on each
   part's joint record (rotation_axis, slide_direction, limits); many limits are not enabled.

### Names (Fusion, 13,207 bodies)

1. Body names: 5.3% human. Occurrence and component names: 35.0% human. Any of the three: 37.9% (5,005 parts).
1. Human examples: "Upper Support Rod v1", "hose hook", "Left Casing D30", "Seger Ring D18", "chain pin v0", "fork",
   "Bolt M8x51", "Handle", "Silicone pad", "Hour Markers", "glass". Auto examples: Body25, Body40, Body1, Body53.
   Many human names are not English (oslona-tyl, wkrety, uchwyt_na_ostrze, a Chinese wheel name) or carry the designer's
   project name rather than the part's role ("rearsuspention2" on 148 bodies, "BAG" on 215, "keyshot for iphone" on 59).
   A name that says what a part does is a minority of the minority.
1. Depth in the tree: 88% of bodies sit at depth 0 (directly in the root), 9.5% at depth 1.
1. Materials: Steel 11,576 (88%, which is Fusion's default, not a decision), then Aluminum 267, Plastic 109, Stainless
   95, Glass 84, Brass 63.

### MCB classes (24,215 parts, 68 classes)

Top: Conventional rivets 3,806; Grooved pins 2,245; Cylindrical pins 1,895; Articulations/eyelets 1,632; Hex screws and
bolts 1,411; Eye screws 1,131; Hexagonal nuts 1,039; Keys and keyways 987; Studs 817; Bushes 764; Helical geared motors
732; Cylindrical-head screws 731. Smallest: Toothed 9, Plain guidings 9, Wingnuts 10, Square nuts 10, Rivet nut 10,
Spring washers 11, Locating pins 11. Full histogram in `data/machine_parts/mcb_parts.jsonl` and
`scripts/machine_parts_stats.py`. Measured shape: rod 47%, block 28%, cube 15%, plate 10%, strip 0.4%. Fasteners and
pins dominate the benchmark; gears, bearings, springs, wheels, pulleys are 1-3% each.

### Part size and thinness

1. Fusion bounding boxes (mm, quantiles 5/25/50/75/95): longest 3.5 / 17 / 65 / 176 / 2,500; shortest 0.7 / 3.0 / 10 /
   48 / 190; aspect (longest over shortest) 1.1 / 2.4 / 4.7 / 18 / 95. Shape: plate 28%, rod 28%, block 24%, cube 12%,
   strip 9%.
1. After voxelising to 16 per axis the cell is about 1/16 of the longest side, so thickness in cells = 16 / aspect at
   best. Thickness of the shortest extent in cells, quantiles 5/25/50/75/95: Fusion 0.32 / 1.8 / 3.4 / 5.6 / 10.0; MCB
   1.07 / 2.7 / 4.1 / 6.0 / 9.3.
1. Variants whose true thickness is under one cell (the voxel is thicker than the part): Fusion 15.9%, MCB 4.2%; under
   two cells: 27.9% and 15.4%. Variants whose bounding box is one cell thick: Fusion 20.2%, MCB 6.5%; at most two cells:
   36.4% and 20.5%. By measured shape, share thinner than one cell: Fusion strips 95%, plates 24%, rods 24%, blocks and
   cubes 0; MCB strips 55%, plates 16%, rods 3%.
1. Median thickness in cells: cube 7.5 / 6.7, block 4.8 / 4.9, rod 2.7 / 3.1, plate 1.9 / 1.9, strip 0.4 / 0.9 (Fusion /
   MCB). A plate or sheet is one or two robots thick; a rod or wire is two or three robots across.

## Quality of the renders

Looked at 15 MCB and 30 Fusion single parts (two views each, near 250 cells) and 6 of the 10 whole-assembly renders.

1. MCB: about 12 of 15 still look like the part (spacer and washer rings, toothed gear and lockwasher with their teeth,
   flanged bearing, pins and screws as bars and cylinders, hook, bush with its bore, socket, collar). Not clear: wingnut,
   fan, chain drive (read as a block with two holes). At 44-65 cells (the old cap) most were unrecognisable lumps; 250
   cells is where classes separate.
1. Fusion: about 18 of 30 read as a recognisable thing (plates with holes, C-bracket, gear ring, rods, bowls, trays);
   the other 12 are blobs or unlabelled lumps whose function a person could not name without the label. A cell label
   was needed for almost all of them.
1. Whole assemblies: 2 of 6 show structure a person would call a machine (a frame with a handle and wheel parts; a
   casing with a lid). The rest are coloured blobs: parts of 1-2 cells thickness merge, the fine parts (bolts, pins,
   washers) are 1-4 cells or disappear, and 8-13 parts in 100-280 cells leave 10-30 cells per part. Assemblies are only
   useful at this grid as context, not as goals.

## Physics questions the data raises (observations, not decisions)

1. Free motion: 30.8% of joints are not rigid (revolute 20.8%, slider 6.1%, others 4.0%). 70% of jointed assemblies
   contain at least one, and 39.6% of parts that have any joint have a free one. The cubes' only relative motions are the
   pivot over an edge and the magnet attach/detach, no rotation about an axis through a shared face and no sliding along a
   face while attached. What does a revolute joint become in a swarm: a hinge made of two cubes sharing an edge, a
   rotating cube in a socket, or something taken as a requirement that the cube cannot meet?
1. Only 21% of parts (2,827 of 13,207) are in any joint; 78.6% are posed with no joint at all. Parts held together by
   contact, gravity or being merged in the design are the majority. If the swarm builds a machine from parts, how are two
   parts held in contact: by magnets only (what holding strength?), or by a placed order?
1. Thinness: 28% of Fusion variants and 15% of MCB variants are thinner than two cells; 16% and 4% are thinner than one
   cell (the grid makes them one cell thick). A plate or sheet is two robots thick at most scales; the median plate is
   1.9 cells. A one-robot-thick plate over open air carries no load by magnets alone; the existing overhang rule applies
   directly. Shafts and pins (rods) are 2.7-3.1 cells across: do they need to bear bending?
1. Holes and bores: MCB bushes, bearings, washers, nuts and gear rings are rings. At 250 cells the hole is 2-4 cells
   wide. A ring is a loop of robots; does it hold under the pivot rules, and can another part pass through the hole?
1. Part sizes span a large range: Fusion longest dimension 3.5 mm to 2.5 m (5th to 95th percentile), aspect up to 95.
   Putting a part and its mating bolt at one cell size is impossible; the bolt is 1-4 cells in an assembly where the
   frame is 16.
1. The designer's functional labels are sparse: body names 5.3% human, any name 37.9%; function words are rare
   ("plunger", "handle", "hook", "Gear", "bracket", "Lid"). MCB class labels are plentiful (68 classes) and are about
   fasteners and standard parts. A language-grounded "build a gear" is available at 88 spur-gear parts and 9 toothed
   rings, fewer than a bolt or rivet; the library is dominated by what manufacturers catalogue.
1. Mass and materials exist per Fusion body (88% default steel), but nothing about loads. No load, force or torque
   is recorded; the data gives geometry, joints and rough mass only.

## Caveats

1. Fusion statistics cover 751 of 8,251 assemblies (one archive in eleven, the first by archive order, not a random
   draw); 3,008 of its 13,207 parts have meshes (the first 152 assemblies within a 400 MB budget); part sizes and
   thinness are over those. The joint histogram is over all 751.
1. MCB voxelised parts are the first 120 per class in tar order (test split first), not a random sample; the histogram of
   classes is over all 24,215.
1. The Fusion `label` was recomputed from the names after the run (`machine_parts_relabel.py`) because the first rule
   counted the root component's name "root" as human; `data/machine_parts` files carry the final labels. The earlier
   stats line in `logs/machine-parts.out` with 80% human names was removed for that reason.
1. Renders and `cards-parts.json` are in the session browse directory (`browse/parts-NN.png`, `asm-NN.png`); the
   "Machine parts" section is in the session `browse_page.py`.
