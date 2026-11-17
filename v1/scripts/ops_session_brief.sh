#!/bin/zsh
# SessionStart hook (.claude/settings.json): a short brief on the shared state before a session starts work.
#
#   ops board (running, queue, ready, messages), the Mac's uptime and whether it rebooted since the previous
#   session, running claims on this host whose PID is gone, whether the `ops agent` collector loop is alive
#   (and the command that restarts it), and the Claude weekly usage from claude_limit_percent.
#
# Reads only ~/ops and the public metrics Worker; never reads credentials. Always exits 0 within a few seconds.

OPS=$HOME/ops
STATE=$HOME/.cache/ops
mkdir -p $STATE 2>/dev/null

print "# ops brief ($(date '+%Y-%m-%d %H:%M'))"

# The board, without its git log; long sections are cut.
python3 $OPS/ops.py -p beebot board 2>/dev/null | awk '
  /^## Log/ { exit }
  /^## / { shown = 0; print; next }
  /^- / { shown++; if (shown == 13) print "- ..."; if (shown > 12) next }
  { print }
' | sed -e '/^$/N;/^\n$/D'

# Uptime, and a reboot since the previous session.
boot=$(sysctl -n kern.boottime 2>/dev/null | sed -E 's/^[{] sec = ([0-9]+),.*/\1/')
now=$(date +%s)
if [[ -n $boot ]]; then
  print "\n## Machine"
  print "booted $(date -r $boot '+%m-%d %H:%M'), up $(( (now - boot) / 3600 )) h $(( (now - boot) % 3600 / 60 )) min"
  previous=$(cat $STATE/last_session 2>/dev/null)
  if [[ -n $previous ]]; then
    print "previous session started $(date -r $previous '+%m-%d %H:%M')"
    if (( boot > previous )); then
      print "REBOOTED since the previous session: training, the ops agent and anything under /private/tmp are gone; check the board's running nodes"
    fi
  fi
  print $now > $STATE/last_session
fi

# Running claims on this host whose PID is gone (the claims collector, run once).
dead=$(OPS_HOME=$OPS python3 $OPS/collectors/claims 2>/dev/null | grep 'project="beebot"' | grep ' 0$' | sed -E 's/.*node="([^"]+)".*/\1/' | tr '\n' ' ')
if [[ -n $dead ]]; then
  print "\n## DEAD claims (running in ops, process gone): $dead"
  print "finish each: python3 ~/ops/ops.py done <id> --failed --result \"died, found by <you>\" (then message its owner)"
fi

# The collector loop.
print "\n## ops agent"
if pgrep -f "ops.py agent" >/dev/null 2>&1; then
  print "alive (pid $(pgrep -f 'ops.py agent' | head -1))"
else
  print "NOT RUNNING: no numbers reach the dashboard. Restart it (never as a LaunchAgent):"
  print "  cd ~ && SSL_CERT_FILE=/etc/ssl/cert.pem nohup python3 ~/ops/ops.py agent > ~/Library/Logs/ops-agent.log 2>&1 &"
fi

# Claude usage, from the claude_usage collector's series on the Worker.
curl -s --max-time 4 "https://ops.fringewidth.workers.dev/series?q=claude_limit_percent" 2>/dev/null | python3 -c '
import json, sys, time
try:
    rows = [r for r in json.load(sys.stdin) if r["labels"].get("kind") in ("weekly_all", "session") and "source" not in r["labels"] and "account" not in r["labels"]]
except Exception:
    sys.exit()
latest = {}
for r in rows:
    kind = r["labels"]["kind"]
    if kind not in latest or r["last_t"] > latest[kind]["last_t"]:
        latest[kind] = r
parts = ["%s %.0f%% (%.1f h old)" % (k, r["last_v"], (time.time() * 1000 - r["last_t"]) / 3600000) for k, r in sorted(latest.items(), reverse=True)]
if parts:
    print("\n## Claude usage: " + ", ".join(parts))
' 2>/dev/null

exit 0
