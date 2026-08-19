#!/bin/bash
# QueuePostRZF_S40HL_CQI4_s4024 -- TRAINING-SEED REPLICATE of the paper's main
# run (Run4/QueuePostRZF_S40HL_CQI4, seed 2024), created 2026-08-19.
#
# PURPOSE: the main result rests on a single training seed. This run is an
# independent replicate of the SAME training procedure with ONLY --seed changed,
# so the paper can report the across-training-seed spread instead of n=1.
#
# EVERYTHING ELSE IS BYTE-IDENTICAL to the 2024 run:
#   world  = S40HighLoad + 4-bit NR CQI  (queue, p U(0.15,0.50), speed U(5,40),
#            K=32 / K_act U{16..32}, p_csi 0.6, type2 56-bit PMI, queue_size 8)
#   LA     = post_rzf, rbg_major, beta_m = 1.0018,0.7499,0.6592,0.6058
#            (NOT recalibrated -- beta_m is scheduler-independent physics
#             calibration; recalibrating per seed would move the comparison)
#   PPO    = entropy_coef 0.02, save_every 1, patience disabled
#   budget = 866 updates, matching the realized budget of the 2024 run
#            (which ran 0..865 of a 1500 default before manual stop; best@409)
#
# SEED SPACING: env.py:90 uses episode_seed = cfg.seed + episode_idx (ADDITIVE),
# so seeds 1 apart would replay the same channel sequence shifted by one update.
# Replicate seeds are spaced 1000 apart (3024/4024/5024) > the 866-update
# horizon, so no channel realization is shared with the 2024 run or each other.
#
# Code pin: 3cc31ac (root *.py verified IDENTICAL to the 2024 run's pin d3efef9,
# so the training code is the same; the _cqi4dev worktree was merged and removed).
cd /home/MYH/ML_DRL_Scheduler || exit 1
export CUDA_VISIBLE_DEVICES=4 \
       OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
       NUMEXPR_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8 \
       TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=2
RUN_DIR=/home/MYH/ML_DRL_Scheduler/Run4/QueuePostRZF_S40HL_CQI4_s4024
LOG=/home/MYH/ML_DRL_Scheduler/Run4/QueuePostRZF_S40HL_CQI4_s4024_run.log
PIN_BASE=3cc31ac59aebf6e250d76e5789bafcbbf53ee4a3

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
    ARGS="--run_root /home/MYH/ML_DRL_Scheduler/Run4 --run_name QueuePostRZF_S40HL_CQI4_s4024"
  fi
  echo "===== [wrap] attempt $i $(date '+%F %T') :: HEAD=$(git rev-parse --short HEAD) :: $ARGS =====" >> "$LOG"
  python3 train_phase2.py --mode queue \
    --p_arrival_min 0.15 --p_arrival_max 0.50 \
    --ue_speed_min 5 --ue_speed_max 40 \
    --la_mode post_rzf --decode_order rbg_major \
    --la_beta_by_depth 1.0018,0.7499,0.6592,0.6058 \
    --cqi_mode nr4bit \
    --entropy_coef 0.02 --ppo_save_every 1 \
    --num_updates 866 \
    --patience_evals 100000 --seed 4024 $ARGS >> "$LOG" 2>&1
  grep -q "^Done\. " "$LOG" && { echo "===== [wrap] DONE =====" >> "$LOG"; break; }
  echo "===== [wrap] exit -> relaunch 20s =====" >> "$LOG"; sleep 20
done
