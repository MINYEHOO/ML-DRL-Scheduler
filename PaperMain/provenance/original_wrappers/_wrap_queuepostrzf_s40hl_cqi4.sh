#!/bin/bash
# QueuePostRZF_S40HL_CQI4 -- HighLoad world + 4-bit NR CQI feedback, 2026-08-01.
# World = S40HighLoad (queue, p U(0.15,0.50), speed U(5,40), K=32, p_csi 0.6,
# type2 56-bit PMI) + cqi_mode=nr4bit (TS 38.214 Table 5.2.2.1-3, floor snap)
# + recalibrated depth-wise calibration factors beta_m (10th-pct recipe,
# first-ACK ~0.88): 1.0018,0.7499,0.6592,0.6058.
#
# Runs from the _cqi4dev WORKTREE (branch cqi4) -- main tree stays on pin
# 71c0bd4 for the live genie pair; merge to main after their retirement.
# Pin d3efef9. HISTORY: updates 0-11 are the promoted smoke run (executed on
# 49550e2; diff to pin is comments+test only, verified). First resume needs
# ALLOW_HASH_MISMATCH=1 once (ckpt hash 49550e2 != pin); after the first
# save the ckpt records d3efef9 and strict pin_check resumes.
cd /home/MYH/ML_DRL_Scheduler/_cqi4dev || exit 1
export CUDA_VISIBLE_DEVICES=4 \
       OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
       NUMEXPR_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8 \
       TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=2
RUN_DIR=/home/MYH/ML_DRL_Scheduler/Run4/QueuePostRZF_S40HL_CQI4
LOG=/home/MYH/ML_DRL_Scheduler/Run4/QueuePostRZF_S40HL_CQI4_run.log
PIN_BASE=d3efef9cf39643baaf72fb3620ee7c23cc1fbfd6

pin_check() {
  git cat-file -e "$PIN_BASE" 2>/dev/null || { echo "pin commit missing"; return 1; }
  if ! git diff --quiet "$PIN_BASE" HEAD -- ':(glob)*.py'; then
    echo "root *.py differs from pinned $PIN_BASE"; return 1
  fi
  if [ -n "$(git status --porcelain -- ':(glob)*.py')" ]; then
    echo "root *.py working tree dirty"; return 1
  fi
  if [ -f "$RUN_DIR/ckpt/latest.pt" ]; then
    CK=$(python3 -c "import torch;print(torch.load('$RUN_DIR/ckpt/latest.pt',map_location='cpu')['cfg'].get('git_hash','none'))" 2>/dev/null)
    [ -n "$CK" ] && [ "$CK" != "none" ] || { echo "ckpt has no git_hash"; return 1; }
    git cat-file -e "$CK" 2>/dev/null || { echo "ckpt commit $CK unknown"; return 1; }
    git diff --quiet "$CK" "$PIN_BASE" -- ':(glob)*.py' || { echo "ckpt code differs from pin"; return 1; }
  fi
  return 0
}

for i in $(seq 1 1000); do
  if [ "${ALLOW_HASH_MISMATCH:-0}" != "1" ]; then
    REASON=$(pin_check) || {
      echo "===== [wrap] PIN FAIL attempt $i $(date '+%F %T') :: $REASON -- HOLDING" >> "$LOG"
      hold_n=0
      while [ "${ALLOW_HASH_MISMATCH:-0}" != "1" ]; do
        if [ $((hold_n % 10)) -eq 0 ] && pin_check >/dev/null; then break; fi
        sleep 30; hold_n=$((hold_n + 1))
        [ -d "$RUN_DIR" ] && touch "$RUN_DIR/.wd_marker"
      done
    }
  fi
  if [ -f "$RUN_DIR/ckpt/latest.pt" ]; then
    ARGS="--resume $RUN_DIR/ckpt/latest.pt"
  else
    ARGS="--run_root /home/MYH/ML_DRL_Scheduler/Run4 --run_name QueuePostRZF_S40HL_CQI4"
  fi
  echo "===== [wrap] attempt $i $(date '+%F %T') :: HEAD=$(git rev-parse --short HEAD) :: $ARGS =====" >> "$LOG"
  python3 train_phase2.py --mode queue \
    --p_arrival_min 0.15 --p_arrival_max 0.50 \
    --ue_speed_min 5 --ue_speed_max 40 \
    --la_mode post_rzf --decode_order rbg_major \
    --la_beta_by_depth 1.0018,0.7499,0.6592,0.6058 \
    --cqi_mode nr4bit \
    --entropy_coef 0.02 --ppo_save_every 1 \
    --patience_evals 100000 --seed 2024 $ARGS >> "$LOG" 2>&1
  grep -q "^Done\. " "$LOG" && { echo "===== [wrap] DONE =====" >> "$LOG"; break; }
  echo "===== [wrap] exit -> relaunch 20s =====" >> "$LOG"; sleep 20
done
