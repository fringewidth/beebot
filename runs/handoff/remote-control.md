# Remote monitoring and control of Kaggle and rented runs

Two pieces: a local follower that copies a running session's log to a file the ops collector can
follow, and a kill switch the trainer polls over a private ntfy.sh topic. Verified 2026-11-13 on
a CPU-only Kaggle kernel of the fringewidth account (no GPU or TPU quota used).

## Can a running Kaggle session be cancelled through the API?

1. `kaggle_rest.py delete OWNER SLUG` (RPC `DeleteKernel`) stops a running session. Verified: a
   kernel that posted a heartbeat every 30 s to a topic went silent within one interval after the
   delete. Cost: the kernel and its output are gone, so the checkpoints in `/kaggle/working` are
   lost. It is a last resort for a run that has nothing worth keeping.
1. `CancelKernelSession` exists on the same service but the token is refused with `Permission
   'kernelSessions.cancel' was denied` (the access token's scope lacks it), whatever the body.
   Other guessed names return 404. So there is no way to cancel and keep the output through the
   API with this token. The kill switch below is the way to stop and keep checkpoints.

## Monitor

```sh
.venv/bin/python scripts/kaggle_follow.py OWNER SLUG --out logs/RUN.out --every 180 &
ops tail logs/RUN.out --label run=RUN     # or the node's --link logs/RUN.out
```

Use `.venv/bin/python` (the python.org python on this Mac lacks CA certificates). The follower
polls `kaggle_rest.logs` (a running session's log is an open event stream: the whole backlog
arrives, then it is cut after `--window` seconds), appends lines not yet written (de-duplicated by
count, kept in `logs/RUN.out.follow`, so a restarted follower continues), holds back a last line
without a newline, reads once more when the session ends, and exits. It reports QUEUED, RUNNING,
COMPLETE, ERROR. Network only. Verified: no duplicates over several polls, ends on COMPLETE.

## Kill switch

`src/beebot/control.py`; opt-in in `think_train` (`--control_url`, or `BEEBOT_CONTROL_URL`;
`--control_every_minutes`, default 5). Off by default, so the Mac chain is unchanged.

1. The control channel is a ntfy.sh topic with a random name (`https://ntfy.sh/beebot-<32 random
   bytes>`, no account). The name is the secret. It lives in `~/.config/beebot/runs/<run>.json`
   (mode 600), made by `scripts/run_control.py new RUN`. It is never committed, never an
   argument (arguments are printed to logs and config.json; the trainer's config.json records
   only "set"). It reaches the trainer as the environment variable `BEEBOT_CONTROL_URL`.
1. Injection: the kernel templates (`scripts/kaggle_tpu/think_kernel.py`,
   `scripts/kaggle_voxel/kernel_template.py`) carry the placeholder `__CONTROL_URL__` and export
   it as `BEEBOT_CONTROL_URL`. At push time the pushed (private) copy gets the real URL:
   `BEEBOT_CONTROL_RUN=RUN python scripts/kaggle_rest.py tpu-push ...` (also used by `handoff`) or
   `python scripts/vla_kaggle.py voxel-run SLUG --control RUN ...`. An unfilled placeholder means
   off. Only `think_train` has the poll; `voxel.lm` runs ignore the variable.
1. Poll: every `control_every_minutes` the lead process fetches `<url>/json?poll=1&since=<last id>`.
   Only messages after the process started count, so a run resumed after a stop is not stopped
   again. A message whose text is `stop` makes the loop leave, run the final validation and save
   `policy_step<N>.pt` plus `resume.pt`, then exit normally. `save` makes the next iteration take a
   checkpoint and continue. `stop` wins over `save`. With several processes the poll happens at
   log steps, where they agree on stop flags.
1. Network failures: 10 s timeout, one log line, training continues.
1. Rented machines: export `BEEBOT_CONTROL_URL=$(python scripts/run_control.py url RUN)` before
   starting the trainer; same behaviour. Monitor with the log file on the machine, or push to ops.

Send commands from the Mac:

```sh
python scripts/run_control.py new RUN      # once, before the push
python scripts/run_control.py save RUN
python scripts/run_control.py stop RUN     # arrives within control_every_minutes
```

After a stop on Kaggle the session completes and its output (the checkpoints) can be pulled as
usual; resume from `resume.pt`.

## Verified

1. `tests/test_control.py` (6): off by default, environment fallback, commands and `since`,
   network failure never raises, a fake training loop against a local HTTP server (save, then
   stop, loop ends early), an unreachable server never stops the loop. `tests/test_kaggle_follow.py`
   (3): de-duplication. No model is loaded in any of them.
1. CPU Kaggle kernel `scripts/kaggle_voxel/control_smoke_kernel.py` (prints a JSON line every 30 s,
   polls its topic every 60 s): pushed with `BEEBOT_CONTROL_RUN`, followed to `logs/` (QUEUED,
   then RUNNING lines, none repeated), `run_control.py stop` sent, the kernel printed
   `stopped_by_control` and the session went COMPLETE within about 100 s; the follower ended.
1. Not run: the real `think_train` poll on a GPU or TPU (no GPU was used). The trainer wiring is
   a few lines in the training loop; test it at the start of the next real run by sending `save`
   once and checking that a `policy_step` appears.
