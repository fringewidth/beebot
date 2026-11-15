#!/bin/sh
# Monte Carlo chain for the reaction-wheel cube (scripts/module_phys/rw_sim.py), 7 mm design point,
# bare floor and steel-sheet floor, single-try and escalating-retry policies. CPU only, 6 workers.
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
LOG=runs/module_phys/rw_sim/mc_log.txt
BARE='{"catch_T": 0.001, "k_roll": 1.1, "k_top": 1.5, "k_climb": 1.35, "k_descend": 1.35, "k_split": 1.2}'
STEEL='{"steel_floor": true, "catch_T": 5e-05, "k_roll": 1.0, "k_top": 1.1, "k_climb": 1.35, "k_descend": 1.0, "k_split": 1.2}'
N=${N:-200}
for D in "$BARE" "$STEEL"; do
  nice -n 5 $PY scripts/module_phys/rw_sim.py mc --moves all --n $N --design "$D" --policy single --retries 0 --kscale 1.0 2>&1 | grep -v WARN >> $LOG
  nice -n 5 $PY scripts/module_phys/rw_sim.py mc --moves roll_floor,roll_top,climb_row,climb_lone,descend_row,descend_lone,split_pair,split_row,dock,yaw --n $N --design "$D" --policy escalate --retries 5 --escalate 1.06 --kscale 0.893 2>&1 | grep -v WARN >> $LOG
done
echo CHAIN_DONE >> $LOG
