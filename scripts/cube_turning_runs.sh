#!/bin/sh
# The measured runs of runs/handoff/rotating-cube-poc.md. CPU only, 5 workers, about 25 minutes.
# Summary lines go to stdout: sh scripts/cube_turning_runs.sh > runs/cube_turning/run_all.out
cd "$(dirname "$0")/.." || exit 1
P=".venv/bin/python -m beebot.hardware.cube_turning"
O=runs/cube_turning
rm -f $O/*.jsonl
run() { name=$1; shift; echo "== $name"; $P "$@" --seeds 20 --workers 5 --out $O/$name.jsonl | tail -1; }
run crank-turning crank
run crank-hinge crank --design hinge
run crank-hinge-free crank --design hinge-free
run crank-hinge-free-207g crank --design hinge-free --mass 0.207
run crank-turning-n20 crank --face '{"torque": 0.2}'
run crank-turning-share33 crank --face '{"share": 0.3333}'
run crank-turning-150g crank --mass 0.15
run crank-turning-nodetent crank --face '{"detent": 0.0}'
run cart-turning cart
run cart-turning-share33 cart --face '{"share": 0.3333}'
run cart-turning-n20 cart --face '{"torque": 0.2}'
run crank-turning-seeds1000 crank --seed 1000
run cart-turning-seeds1000 cart --seed 1000
for shape in "1 0" "2 0" "3 0" "1 1" "1 2" "2 1"; do
  set -- $shape
  echo "== carry-$1-$2"; $P carry --along $1 --side $2 --seeds 10 --workers 5 --out $O/carry-$1-$2.jsonl | tail -1
done
run crank2-turning crank --reach 2
run crank2-hinge-free crank --reach 2 --design hinge-free
run march-hinge march
