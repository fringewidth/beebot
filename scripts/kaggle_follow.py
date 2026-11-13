"""Follow a Kaggle session's log into a local file while it runs (network only, light work on the Mac).

    python scripts/kaggle_follow.py OWNER SLUG --out logs/RUN.out [--every 180] [--window 25]

Every --every seconds it reads the session's log so far (kaggle_rest.logs; a running session's is an
open event stream read for --window seconds), appends the lines not yet written to --out, and stops
when the session is no longer queued or running (after one last read). The log is append-only, so
lines are de-duplicated by count; the count is kept in --out + ".follow" so a restarted follower
continues where it stopped. A last line without its newline is held back until it is complete (or
the session has ended). Point `ops tail logs/RUN.out --label run=RUN` or the node's log link at --out.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import kaggle_rest

LIVE = ("QUEUED", "RUNNING", "NEW")


def new_lines(text, seen, final):
    """(lines to write, new count) from the whole log text so far."""
    lines = text.split("\n")
    complete = lines if final else lines[:-1]  # the last piece has no newline yet: wait for the rest
    if final and lines and lines[-1] == "":
        complete = lines[:-1]
    return complete[seen:], max(seen, len(complete))


def status(owner, slug):
    out = kaggle_rest.call(f"{kaggle_rest.RPC}/GetKernelSessionStatus", {"userName": owner, "kernelSlug": slug})
    return "GONE" if out.get("http_error") == 403 else "UNKNOWN" if "http_error" in out else out.get("status", "QUEUED")


def poll(owner, slug, out, state, window, final):
    seen = int(state.read_text()) if state.exists() else 0
    lines, seen = new_lines(kaggle_rest.logs(owner, slug, window), seen, final)
    if lines:
        with open(out, "a") as f:
            f.write("\n".join(lines) + "\n")
    state.write_text(str(seen))
    return len(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("owner")
    parser.add_argument("slug")
    parser.add_argument("--out", required=True)
    parser.add_argument("--every", type=float, default=180)
    parser.add_argument("--window", type=float, default=25)
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    state = Path(str(out) + ".follow")
    while True:
        now = status(args.owner, args.slug)
        ended = now not in LIVE and now != "UNKNOWN"  # an unreadable status (network) is not the end
        if now != "GONE":
            count = poll(args.owner, args.slug, out, state, args.window, ended)
            print(f"{time.strftime('%H:%M:%S')} {now}: {count} new lines", flush=True)
        if ended:
            print(f"session ended ({now}); log in {out}", flush=True)
            return
        time.sleep(args.every)


if __name__ == "__main__":
    main()
