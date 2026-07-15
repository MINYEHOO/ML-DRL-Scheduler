#!/bin/bash
# _autorecover.sh -- single-source container-restart recovery (2026-07-15).
#
# WHAT: after a container restart (overlay / wiped; /home persistent), this
# reinstalls tmux (+gh) and relaunches the live training sessions and the
# hang-watchdog, resuming every run from its checkpoint via the wrappers.
#
# WHO CALLS IT (all safe -- idempotent + lock-guarded):
#   1. ~/.ssh/rc          -> fires on EVERY ssh/Cursor/scp connection
#   2. ~/.bashrc          -> fires on interactive login shells
#   3. (future) admin boot hook /home/MYH/.on_boot.sh may point here
#
# IDEMPOTENCE: tmux binary present == container not freshly restarted ->
# exit immediately (dead-session recovery within a healthy container is the
# WATCHDOG's job, not ours). A mkdir lock serializes concurrent callers
# (Cursor opens many ssh sessions at once).
#
# RUN LIST: keep in sync with Run3/_watchdog.sh RUNS (same-breath rule:
# retiring a run means editing BOTH files and restarting the watchdog).
cd /home/MYH/ML_DRL_Scheduler || exit 0
LOG=Run3/_autorecover.log
say(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }

# healthy container -> nothing to do (cheap guard, runs on every ssh session)
command -v tmux >/dev/null 2>&1 && exit 0

# freshly restarted container: serialize concurrent recovery attempts
LOCK=/tmp/mldrl_autorecover.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  say "another recovery in progress -- skip"; exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

say "container restart detected (no tmux) -- recovering"
sudo -n apt-get update -qq >>"$LOG" 2>&1
sudo -n apt-get install -y -qq tmux gh >>"$LOG" 2>&1
command -v tmux >/dev/null 2>&1 || { say "tmux install FAILED (network?)"; exit 1; }

RECOVER_RUNS=(
  "Run4/GenieFineTune:genie_finetune:Run4/_wrap_genie_finetune.sh"
  "Run4/QueueMixedFairLA:queuemixedfairla:Run4/_wrap_queuemixedfairla.sh"
  "Run4/QueuePostRZF:queuepostrzf:Run4/_wrap_queuepostrzf.sh"
)
for spec in "${RECOVER_RUNS[@]}"; do
  IFS=: read -r run sess wrap <<< "$spec"
  # marker BEFORE launch: gives the watchdog its HANG_MIN grace and stops it
  # from killing a just-revived session whose CSV is hours stale (the
  # 2026-07-14/15 double-fire churn)
  [ -d "$run" ] && touch "$run/.wd_marker"
  tmux has-session -t "$sess" 2>/dev/null || {
    tmux new-session -d -s "$sess" "bash /home/MYH/ML_DRL_Scheduler/$wrap"
    say "relaunched $sess ($run)"
  }
done
tmux has-session -t watchdog 2>/dev/null || {
  tmux new-session -d -s watchdog "bash /home/MYH/ML_DRL_Scheduler/Run3/_watchdog.sh"
  say "relaunched watchdog"
}
say "recovery complete (${#RECOVER_RUNS[@]} runs + watchdog)"
