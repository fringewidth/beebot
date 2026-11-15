#!/bin/sh
# After the MC chain: a small NSGA-II on the steel floor (5 workers) and the films (1 process), at most 6 processes.
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
until grep -q CHAIN_DONE runs/module_phys/rw_sim/mc_log.txt; do sleep 10; done
BASE='{"steel_floor": true, "k_roll": 1.0, "k_top": 1.1, "k_climb": 1.35, "k_descend": 1.0, "k_split": 1.2}'
RW_OPT_N=8 nice -n 5 $PY scripts/module_phys/rw_sim.py opt --pop 12 --gens 2 --workers 5 --base "$BASE" > runs/module_phys/rw_sim/opt_log.txt 2>&1 &
BARE='{"catch_T": 0.001, "k_roll": 1.1, "k_top": 1.5, "k_climb": 1.35, "k_descend": 1.35, "k_split": 1.2}'
nice -n 5 $PY scripts/module_phys/rw_sim.py render --moves roll_floor,dock,roll_top,climb_row,climb_lone,descend_row,split_row,split_pair,yaw,bond,fall_pair --design "$BARE" > runs/module_phys/rw_sim/render_log.txt 2>&1
wait
echo AFTER_DONE >> runs/module_phys/rw_sim/render_log.txt
