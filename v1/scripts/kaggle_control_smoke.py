"""CPU-only Kaggle kernel for testing the kill switch and the watcher end to end (no GPU or TPU quota).
Two dummy trainers print a JSON step line every 20 s for up to 25 minutes, relayed as the real templates
relay them; the kill switch (v1/scripts/kernel_control.py) is inlined and its topic filled in at push time:

    BEEBOT_CONTROL_RUN=SLUG .venv/bin/python v1/scripts/kaggle_rest.py cpu-push OWNER SLUG v1/scripts/kaggle_control_smoke.py
    .venv/bin/python v1/scripts/kaggle_watch.py OWNER SLUG --every 30 --ntfy ''      # in the background
    .venv/bin/python v1/scripts/run_control.py kill SLUG smoke-b                       # one trainer
    .venv/bin/python v1/scripts/run_control.py kill SLUG                               # the session ends
    .venv/bin/python v1/scripts/kaggle_rest.py delete OWNER SLUG                       # afterwards
"""

import os
import subprocess
import sys
import time

CONTROL_URL = "__CONTROL_URL__"  # filled in at push time (v1/scripts/run_control.py); never printed

# __KERNEL_CONTROL__
CONTROL = KernelControl(CONTROL_URL).start()

os.makedirs("/kaggle/working/logs", exist_ok=True)
with open("/kaggle/working/smoke_started.txt", "w") as f:
    f.write(f"{time.time()}\n")  # an output written before the kill: it must survive
TRAINER = """import json, time
t0 = time.time()
for step in range(1, 100000):
    time.sleep(20)
    print(json.dumps({"step": step, "minutes": round((time.time() - t0) / 60, 2), "loss": round(1 / step, 4)}), flush=True)
    if time.time() - t0 > 25 * 60:
        break
"""
started = []
for name in ("smoke-a", "smoke-b"):
    log = open(f"/kaggle/working/logs/{name}.out", "w")
    started.append((name, CONTROL.track(name, subprocess.Popen([sys.executable, "-u", "-c", TRAINER], stdout=log, stderr=subprocess.STDOUT))))
relay(started, every=15)
print("smoke kernel done", flush=True)
