"""CPU-only Kaggle kernel for testing remote control and log following (no GPU or TPU quota).
Prints a numbered JSON line every 30 s and, with a control topic filled in at push time
(BEEBOT_CONTROL_RUN=<run> kaggle_rest.py cpu-push ...), polls it through beebot.control's logic
(copied here: the kernel has no code dataset) and exits cleanly on "stop".
"""

import json
import time
import urllib.request

CONTROL_URL = "__CONTROL_URL__"
EVERY = 60  # seconds between control polls
LIMIT = 20 * 60  # never run longer than this
t0 = time.time()
since = int(t0)
last = t0
step = 0
print("smoke kernel started, control", "on" if not CONTROL_URL.startswith("__") else "off", flush=True)
while time.time() - t0 < LIMIT:
    step += 1
    print(json.dumps({"step": step, "minutes": round((time.time() - t0) / 60, 2), "loss": round(1 / step, 4)}), flush=True)
    if not CONTROL_URL.startswith("__"):  # a heartbeat on the topic (tells from outside whether the session is alive)
        try:
            urllib.request.urlopen(urllib.request.Request(CONTROL_URL, data=f"heartbeat {step}".encode(), method="POST", headers={"User-Agent": "beebot-control"}), timeout=10).read()
        except Exception:
            pass
    time.sleep(30)
    if not CONTROL_URL.startswith("__") and time.time() - last >= EVERY:
        last = time.time()
        try:
            request = urllib.request.Request(f"{CONTROL_URL}/json?poll=1&since={since}", headers={"User-Agent": "beebot-control"})
            text = urllib.request.urlopen(request, timeout=10).read().decode()
        except Exception as error:
            print("control: poll failed", type(error).__name__, flush=True)
            continue
        words = []
        for line in text.splitlines():
            row = json.loads(line)
            if row.get("event") == "message":
                since = row["id"]
                words.append(row.get("message", "").strip().lower())
        if "stop" in words:
            print(json.dumps({"event": "stopped_by_control", "step": step}), flush=True)
            break
print("smoke kernel done", flush=True)
