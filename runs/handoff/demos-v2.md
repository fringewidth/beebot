# demos-v2: redrawn voxel demonstration stores

Written 2026-11-12. Old stores (`data/vla/voxel_build`, `voxel_full`, `haiku`) are untouched; the running
trainer reads them.

## What changed in the draw

1. Moves: `--outcomes mujoco_hinge_v1` (the hinge cube, `demos.DEMO_MOVES`) in place of the old per-class
   `--failure` rates, with the overhang rule (`world.overhangs`). Everything else is as the old stores:
   `--cube --rule switch_cube --robots 6 115 --counts log --fit --compact`, seeds 0-7 (full: 1200 scenes,
   `--coverage 0.3 --objects 1.0`) and 0-1 (build: 1600 scenes, `--steps 144 --build 1.0 --only build`).
1. Intended actions: each store now also has `intended.bin` (same layout as `codes.bin`): the expert's
   action before any jitter (`controller.act` jitter, `build.JITTER` crew jams). `codes.bin` still holds the
   move that was executed, jitter included. `play_store.load` returns it as `store["intended_actions"]`.
   The trainer reads `store["actions"]` (`think_data.py`, `ThinkData.example`) today. To train on the intended
   moves pass `--data_kwargs '{"intended": true}'` (`ThinkData(intended=True)` reads `"intended_actions"`);
   only the v2 stores have the field, so use it with v2 stores only.
1. Width-1 towers: `--thin_towers` (`suite.THIN_TOWERS`, `suite.column_side`), default OFF. With it on, a
   tower the swarm cannot fill as a 2 x 2 column is a single column and fits. It changes the draw (towers
   refused before are accepted, so the random stream after them shifts and Haiku labels joined by draw
   index no longer line up), so the v2 stores here do NOT use it. Use it for the separate demo store
   (about 10 robots, fresh Haiku labels): `demos ... --thin_towers`. The config records `"thin_towers": true`.

## Label join (Haiku labels by (seed, index), ThinkData mismatch="drop")

1. Old full-s0: 3892 episodes, 1355 labelled, 229 dropped (17%), 1126 kept.
1. v2 with width-1 towers always on (first attempt): 1132 of 1135 labelled dropped (99.7%); the stream
   diverged at attempt 1. Cause: the draw loop refuses a task `fits` rejects and redraws, so letting
   towers fit moves everything after. The hinge outcomes alone do not move the draw (48 scenes, old tower
   rule plus hinge: all 198 attempts match). Hence `--thin_towers` is opt-in.
1. v2 subset check (default, 168 scenes of seed 0): 468 episodes, 430 labelled, 0 dropped, 430 kept
   (0 tower, 0 other); 715 of the first 728 attempts match the old store's family/kind/robots (the old
   store itself departs from the Haiku export at attempt 715, the known draw_ref issue), so the v2 draw is
   in fact more aligned than the old one past that point. Drops on the full stores are expected to be
   fewer than the old 17%.

## Progress

1. Runner: `scripts/draw_demos_v2_all.sh` (detached, three stores at a time, `nice -n 10`, waits when free
   memory is under 10%, skips a store whose directory has a `done` file, so it can be started again):
   `OPS_NODE=demos-v2 nohup scripts/draw_demos_v2_all.sh > logs/demos-v2-all.out 2>&1 &`
1. Per store: `logs/demos-v2-<full|build>-s<N>.out` (scenes done and kept/tried per family, a line per 24
   scenes) and `logs/demos-v2-join-<kind>-s<N>.out`; stores in `data/vla/voxel_full_v2/sN` and
   `data/vla/voxel_build_v2/sN`; finished when `done` exists.
1. `logs/demos-v2.out` gets one JSON line per finished store: `store`, `episodes`, `labels_joined`,
   `labels_dropped`, `kept` (labelled episodes surviving the join), `demos_v2_kept` (kept demos), and
   drop/keep counts by family/kind. `logs/demos-v2-all.out` says when each store finished.
1. Check: `pgrep -fl voxel.demos`, `tail -n1 logs/demos-v2-*.out`, `ls data/vla/voxel_*_v2/*/done`.
1. Speed: one full seed took 6400 s alone under nice with the trainer running; expect 8+ h for all ten.

## Use

Train on the v2 stores by listing `data/vla/voxel_full_v2/s0..s7` and `data/vla/voxel_build_v2/s0,s1` as
`--stores` with the same `--haiku_dirs`, `mismatch="drop"`, and `{"intended": true}` in `--data_kwargs`.
Scripts: `scripts/draw_demos_v2.sh` (one store), `scripts/demos_v2_join.py` (join counts for a store).
