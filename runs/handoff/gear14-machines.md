# Geared-joint cube at 14 mm: the four machines

Written 2026-11-16. Simulation only, nothing built. Code: `src/beebot/hardware/geared_joint.py`
(`nominal_design`, unchanged) driven by `scripts/small_joint_machines.py` with `JOINT_ROUTE=gear`.
Results: `runs/module_phys/joint_gear14/`. All numbers are E (simulation on stated inputs).

Invocation: `JOINT_ROUTE=gear SJ_OUT=<dir> .venv/bin/python scripts/small_joint_machines.py {run|mc|summary} [names] --L 14 [--n 100 --workers 6]`.

## 1. Answer

1. As specified (the same machines, controller and tasks as the SMA run, gear route, 14 mm), the
   geared cube does not do the arm or the walker. Crane and crank pass at 1.5x.
1. The binding failure is not the bonds alone, as it was for SMA. **Arm: the joint breaks in every trial**
   (joint peak 1.93 of the tooth limit, median; 100 of 100). **Walker: the bonds open in every trial**
   (bond load 1.43 of strength, median), because the gait runs at the geared joint's speed (period 7.8 s
   against 65 s for SMA), so the teeter loads are dynamic.
1. Both are fixable in the model by slowing down or softening, with the joint left alone:
   arm with a flexure finger (50 N/m at the tip, as in the SMA arm) passes 24 of 24 with joint 0.43
   of the limit, but bond opening at strength / 1.5 still fails 17 of 24; walker at a 30 s period passes
   nominally and 3 of 4 at 1.0x, 1 of 4 at 1.5x. These two variants are partial runs (section 3).

## 2. Monte Carlo, 14 mm, 100 trials per machine (`mc_14.00mm.jsonl`, `mc_summary_14.json`)

Parameters varied together in every trial (uniform): worm friction 0.15-0.35, split-ring loss
0.015-0.040 per unit ratio (gear efficiency 0.40-0.72 of the train, nominal-centred), spur efficiency
0.80-0.93, backlash 0.5-2x, bond strength 0.75-1.10x, table friction 0.5-1.0 (crank sheet 0.15-0.35),
cube friction 0.4-0.8. These are the existing `RANGES` of the gear route; the task's "0.6-1.0 of
nominal efficiency" is covered by mu, eps and eta_spur, which were not changed.

| machine | n | pass (95 % CI) | pass at 1.5x (95 % CI) | joint / tooth limit, median / p95 / max | bond load / strength, median / p95 | failures |
| --- | --- | --- | --- | --- | --- | --- |
| arm | 100 | 0.00 (0.00-0.04) | 0.00 (0.00-0.04) | 1.93 / 2.33 / 2.45 | 2.74 / 3.74 | joint broken 100, bond opened 100, task 0 |
| walker | 100 | 0.00 (0.00-0.04) | 0.00 (0.00-0.04) | 0.93 / 0.98 / 1.11 | 1.43 / 1.68 | bond opened 100, joint broken 2, task 0 |
| crane | 100 | 1.00 (0.96-1.00) | 1.00 (0.96-1.00) | 0.18 / 0.18 / 0.45 | 0.27 / 0.32 | none |
| crank | 100 | 1.00 (0.96-1.00) | 1.00 (0.96-1.00) | 0.30 / 0.38 / 0.39 | 0.23 / 0.28 | none |

Compare SMA 8.25 mm (small-joint.md section 4): arm 0.84 / 0.46, walker 0.72 / 0.15, crane 1.00 / 0.99, crank 1.00 / 1.00; joint
never above 0.66 of the limit. The geared cube is better only on crane and crank (1.5x pass 1.00 and 1.00 against 0.99 and 1.00).

The arm and walker task criteria (lift, carry, walk 15 cm x L / 50 mm) were met in all 100 trials; they
fail on loads, not on motion. The arm's joint load is at "5p", the jaw wrist: the geared joint closes
at speed on the block with a rigid finger and dumps the reflected rotor inertia (about 0.01 kg m2) into
the output tooth (peak 38 mN m against a 20 mN m tooth limit), at t about 21.5 s, the grip. Because the
joint breaks, the bond-opening figures there are a consequence, not a separate cause. Nominal run
(`machines_14.00mm.jsonl`): arm joint 1.92, bond 2.60; walker joint 0.93, bond 1.34; crane 0.18, 0.24; crank 0.26, 0.26.

## 3. Variants that isolate the cause (partial, CPU contention)

The machine was at load 50-150 from other jobs; these runs were stopped at the n shown.

| variant | n | pass | pass at 1.5x | joint / limit median, max | bond / strength median, p95 |
| --- | --- | --- | --- | --- | --- |
| arm, flexure finger 50 N/m (`SJ_GEAR_FINGER_K=50`, `finger50/`) | 24 | 1.00 (0.86-1.00) | 0.29 (0.15-0.49) | 0.43, 0.45 | 0.69, 0.92 |
| walker, 30 s period (`SJ_WALKER_PERIOD=30`, `walker_p30/`) | 4 | 0.75 (0.30-0.95) | 0.25 (0.05-0.70) | 0.48, 0.51 | 0.92, 1.06 |

With the finger softened the arm's joint load drops from 1.93 to 0.43 of the limit and what is left is
the same magnet-bond opening that limited the SMA arm. The walker at 30 s behaves like the SMA walker
(nominal bond 0.88 against 0.92; joint 0.48 against 0.45). The walker n of 4 is far too small to rate.

## 4. Trend in L (nominal only, one run each, `machines_12.00mm.jsonl`, `machines_18.00mm.jsonl`; no Monte Carlo)

| L | arm | walker | crane | crank |
| --- | --- | --- | --- | --- |
| 12 mm | fail: joint 1.67, bond 2.46 | fail: joint 0.76, bond 1.36 | pass, 0.16 / 0.23 | pass, 0.23 / 0.22 |
| 14 mm | fail: joint 1.92, bond 2.60 | fail: joint 0.93, bond 1.34 | pass, 0.18 / 0.24 | pass, 0.26 / 0.26 |
| 18 mm | fail: joint 2.52, bond 2.89 | fail: joint 2.61 (broken), bond 2.34, tracking error 5.3 rad | pass, 0.23 / 0.27 | fail task: 2.96 turns (needs 3), joint 0.32 |

(joint peak / tooth limit, bond / strength.) Larger L is worse, as the L^4 load scaling predicts: the
walker's gait speed is fixed by the joint (7.8 s), so at 18 mm it outruns the joint's torque.
Nothing here ranks 12 against 14 mm for the arm or walker, since all variants fail as specified.

## 5. Idealisations and caveats

1. The bonds are the same magnet bonds as the SMA run (nominal_bonds(L), rigid weld, 5 ms averaged
   load, Newmark opening model); bond strength scales with L, so at 14 mm they are stronger than at 8.25 mm.
   Bond failures in the walker (1.43) and in the arm after the joint breaks are partly real bond limits,
   partly the dynamic gait; the finger50 and p30 variants show bonds still bind at 1.5x for the arm.
1. Cube mass uses the 8.25 mm mass fit extrapolated (`joint/mass_fit.json`, m = aL^3 + bL^2).
1. Gear route uses TIME = 2.5 (move durations 2.5x layout B), against 1.5 for SMA.
1. The arm's gear grip is a set motor current with a rigid finger; the SMA arm used a flexure finger.
   The finger flexure is not a designed part in the gear cube (A).
1. Tooth limit is the Lewis strength of the weaker member of the output stage (peak 19.98 mN m at 14 mm).
   A printed split-ring stage has not been printed; no wear, backlash growth or temperature.
1. Earlier Pareto note (arm margin 2.2, walker 2.3 in small-joint.md section 7) were static/steady margins;
   the dynamic simulation does not support them for the arm (rigid finger) or the walker at the gear's top speed.

## 6. Code changes (defaults and SMA results untouched)

All in `scripts/small_joint_machines.py`; no change to `geared_joint.py`.

1. `SJ_OUT` environment variable: results directory (default unchanged). `MASS_FIT` now points at the
   original `runs/module_phys/joint/mass_fit.json` so a redirected run reads the same mass fit.
1. `SJ_GEAR_FINGER_K` (default 0, off): gear route only, applies the SMA flexure finger to the arm's jaw.
1. `SJ_WALKER_PERIOD` (default unset): overrides the walker period in seconds.

The 14 mm MC was interrupted once by a machine stall at 301 rows and resumed (trial seeds are fixed per trial, so the rows are the same set).

## 7. Files

1. `runs/module_phys/joint_gear14/mc_14.00mm.jsonl`, `mc_summary_14.json`, `machines_{12,14,18}.00mm.jsonl`.
1. `runs/module_phys/joint_gear14/finger50/`, `walker_p30/` (partial variants, own jsonl and summary).
1. Logs: `mc_log_14.txt`, `mc_log_14b.txt`, `run_12.txt`, `run_18.txt`.
