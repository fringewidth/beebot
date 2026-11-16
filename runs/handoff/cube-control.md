# cube-control handoff (worktree .claude/worktrees/cube-control, branch cube-control from harm-down 840c3e5)

Goal: how far in-move control + firmware (+ magnet strength) get toward the bar: per commanded move
reached >= 0.80, harmful <= 0.03 on 10-cube walks (3 seeds, no planner help). ops node cube-control-v2
(prefix launches with OPS_NODE=cube-control-v2; do not write to ops).

Code (uncommitted until noted):
- src/beebot/hardware/cube_firmware.py: Firmware (flags trim, meter, hinge/catch, ease; sensing noise, motor torque
  0.02 N m, brake latency 2 ms), Flight (per-mover controller: w_ref(theta) from geometry, metered brake, motor trim,
  catch, flywheel spin-down).
- cube_swarm.py: Pivot.start/edge/quarters; SwarmMagnets.ground (floor mask); Swarm(firmware=); Swarm.fly (loop),
  catch, quiet_mask; step(drive=).
- cube_outcomes.py: --firmware JSON (job["firmware"]); TEMPLATES floor_roll_row, floor_roll_patch (held rolls).
- runs/cube_control/: smoke.py, summ.py (per setting reached/harm by template or primitive), q*.sh queues, rows.
  Baseline m14-sel rows copied: runs/cube_control/{layouts-m14-sel*,stores-m14-sel}.jsonl

Progress:
- 2026-11-12 ~15:00: smoke (4 seeds, mag x1.4): hinge release frees held floor rolls (4/4 vs 0/4) and concave up 4/4;
  hinge alone makes convex down tumble (kick too big without the floor hold) -> meter fixes it (4/4).
- q1 (isolated templates, 30 trials x 17 templates per variant): m14-sel, trim, hinge-trim, hinge-trim-meter,
  htm-max (all kicks at 20000 rpm, metered), htm-max-b25 (meter brake 2.5 N m), htm-max-m10 (as-built magnets).
  Log runs/cube_control/q1.log. Summarise: PYTHONPATH=src python runs/cube_control/summ.py runs/cube_control/iso-*.jsonl

Next: rank variants on isolated templates -> run top 2-3 on 10-cube walks (seed 0) -> 3 seeds + stores for the best.
- 13:50 (clock times are local; the earlier "15:00" was wrong, it was 13:30). Committed 9ce593b (mechanism + tests).
  q1/q2 isolated findings (30 trials/template, mag x1.4, m14-sel kicks):
  - trim (motor 0.02 N m, margin 0.1) + m14-sel escalation: convex up lone .83/.17 -> .86-.97/.03-.14, convex up patch .30 -> .67,
    held floor roll out of a row .63-.77 -> 1.00; but convex down lone .53-.63/.27-.30 -> .27-.30/.70 (motor adds energy, drags the lone support).
  - hinge + meter (soft 1 N m brake cut on gyro) on convex down: 1.00/0.00 for lone, row and patch templates. Meter WITHOUT hinge: every move stays
    (soft brake cannot break the full magnet hold). Hinge on convex up / slot rolls: overshoot, 'off' at attempt 1.
  - floor roll out of a 2x3 patch front row: nothing > 0.23 reached yet; floor roll into a slot ~.5-.67 reached, .07-.2 harm.
  - concave up a lone 2-stack: topples 0.83-1.00 under every scheme (mechanical).
  Composite C = trim all + hinge&meter on convex_down/concave_down. q3: held-roll kicks (escalate to 2.5x, max+meter) w/ hinge/ease.
- 14:00 KEY: squeeze on a crowd mover is the neighbours' own diagonal pairs (15-20 N per side after the mover is released; floor ease
  barely changes it; switching the neighbours' pairs off removes it): runs/cube_control/squeeze.py. New firmware `loosen` (neighbours'
  floor-level pairs off during the move). Isolated (q4): floor roll out of a patch 0 -> .87-1.00, convex up from a patch .70 -> 1.00, no harm.
  Committed b2d679d. Composite C-loose = {"trim": true, "meter": ["convex_down","concave_down"], "hinge": ["convex_down","concave_down"], "loosen": true}
  q5: 10-cube walks seed 0 for C-loose (m14-sel kicks) and C-loose-esc2 (held rolls escalate 1.2,1.6,2.0) -> runs/cube_control/layouts-*.jsonl
- 14:25 walks seed 0 (harm/reached per move): m14-sel .053/.48 -> C-loose .082/.735 -> C-loose-esc2 (held rolls escalate 1.2,1.6,2.0) .077/.78.
  Reach nearly at the bar; harm 2.5x over. Harm sources (C-loose s0): convex_down crowd (.27 harm, 'off' = mover stalls at ~105 deg,
  far corner on the floor, hinge holding it), lone-support convex up (.77), convex_up/hinged, convex_down/hinged.
  Walk-scene bench (runs/cube_control/bench-hard-s.jsonl, 390 replayed walk scenes; benchsum.py): convex down harm m14-sel .15/reach .18,
  any firmware variant .22-.31/.62-.65 (drop on stall, margin .03, meter brake 2.5, no trim, catch 45 all tried) -> unsolved by control.
  Commits 9ce593b, b2d679d, b6f8a5d. q9 running: C-loose-esc2 walks seeds 1000, 2000, stores, and as-built magnets (10 walks/layout).
  Next: compare + per-primitive tables, planner-assisted (ban lone), stranded estimate, HARDWARE.md section, commit, report.
- 14:45 DONE. Bar not met. Final (HARDWARE.md "Control within a move", commit 9f56059 + 6144d61 on cube-control):
  C-esc2 3 seeds: 0.088 harm / 0.77 reached per move (seeds .077/.78, .100/.77, .086/.76); lone banned .065/.79; lone + crowd convex
  down banned .037/.81; stores .083/.60 (m14-sel .032/.47); as-built magnets .123/.79 (half seed). Composed C-esc2 with open-loop convex
  down: .060/.70 (.036/.72 lone banned). Binding: half turns near cubes (convex down crowd .33 harm, 0.029 of the 0.088; convex up
  onto a lone support .43, 0.020). ~6 CPU-h used. No outcome table fitted (nothing met the bar). Nothing made default.
