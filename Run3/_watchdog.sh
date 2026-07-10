#!/bin/bash
# Run3/_watchdog.sh -- keep all Run3 trainings alive without manual intervention.
#
# Two failure modes it covers:
#   (1) a run's tmux session is GONE (died / reaped)          -> relaunch it
#   (2) a run is HUNG (frozen, no progress) while tmux alive  -> kill + resume
#
# Progress signal = env_metrics.csv mtime. train_phase2.py flushes that file
# EVERY update (see train_phase2.py:462-463), so its mtime == "last completed
# update time". If it has not advanced for HANG_MIN minutes the run is frozen.
#
# A per-run marker (.wd_marker) is touched on every (re)launch so a freshly
# restarted run gets HANG_MIN of grace before its first flush -- K=32's first
# update after resume (TF init + uncached channel gen) is slow.
#
# HANG_MIN is generous on purpose: measured worst-case healthy gap is one
# eval-update under peak contention ~25-30 min, so 60 min never false-kills a
# slow-but-healthy run, yet catches a real freeze within ~HANG_MIN+CHECK_SEC.
#
# Launched in its own tmux session ("watchdog"); relaunched by ~/.bashrc after a
# container restart. Idempotent: uses `tmux has-session` before every launch.

cd /home/MYH/ML_DRL_Scheduler || exit 1
HANG_MIN=${HANG_MIN:-60}      # minutes without progress -> hung -> restart
CHECK_SEC=${CHECK_SEC:-300}   # check cadence (5 min)
LOG=Run3/_watchdog.log

# run_dir : tmux_session : wrapper_script -- ALL repo-relative paths
# (2026-07-06 generalized from Run3-prefixed names so Run4+ runs fit too;
# wrappers self-set CUDA/OMP env). KEEP IN SYNC with ~/.bashrc RECOVER_RUNS:
# stopping a run means removing it from BOTH lists in the same breath
# (lesson: the 2026-07-06 02:55 restart resurrected 3 stopped runs from a
# stale hardcoded bashrc list).
RUNS=(
  # Run3 training phase CLOSED 2026-07-02/06 (all best.pt preserved).
  "Run4/QueueHighLoad:queuehighload:Run4/_wrap_queuehighload.sh"  # GPU4, launched 2026-07-07
  "Run4/QueueMixedArrival:queuemixedarrival:Run4/_wrap_queuemixedarrival.sh"  # GPU2, launched 2026-07-07
  "Run4/GenieL2b:genie_l2b:Run4/_wrap_genie_l2b.sh"  # CPU, perfect-CSI L2b 2026-07-08
  "Run4/GenieFineTune:genie_finetune:Run4/_wrap_genie_finetune.sh"  # GPU3, genie warm-start 2026-07-09
)

say(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }

launch(){  # session wrapper run_dir
  tmux new-session -d -s "$1" "bash /home/MYH/ML_DRL_Scheduler/$2"
  touch "$3/.wd_marker"
}

command -v tmux >/dev/null 2>&1 || { say "tmux missing -> exit (bashrc will relaunch)"; exit 0; }
say "watchdog up (HANG_MIN=$HANG_MIN, CHECK_SEC=$CHECK_SEC, runs=${#RUNS[@]})"

while true; do
  now=$(date +%s)
  for spec in "${RUNS[@]}"; do
    IFS=: read -r run sess wrap <<< "$spec"
    em="$run/csv_logs/env_metrics.csv"
    mk="$run/.wd_marker"

    # (1) dead session -> relaunch (wrapper resumes from latest.pt)
    if ! tmux has-session -t "$sess" 2>/dev/null; then
      say "$run: tmux '$sess' GONE -> launch"
      launch "$sess" "$wrap" "$run"
      continue
    fi

    # progress reference = latest of (env_metrics flush, last watchdog action)
    ref=0
    [ -f "$em" ] && ref=$(stat -c %Y "$em")
    if [ -f "$mk" ]; then m=$(stat -c %Y "$mk"); [ "$m" -gt "$ref" ] && ref=$m; fi
    if [ "$ref" -eq 0 ]; then touch "$mk"; continue; fi   # no data yet: start grace

    age=$(( (now - ref) / 60 ))
    # (2) hung -> kill whole session + orphan python, then resume
    if [ "$age" -ge "$HANG_MIN" ]; then
      say "$run: no progress ${age}min (>=$HANG_MIN) -> HUNG, kill+resume"
      tmux kill-session -t "$sess" 2>/dev/null
      pkill -f -- "--resume $run/ckpt" 2>/dev/null
      sleep 3
      launch "$sess" "$wrap" "$run"
    fi
  done
  sleep "$CHECK_SEC"
done
