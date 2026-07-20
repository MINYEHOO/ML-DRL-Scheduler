#!/bin/bash
# MixedSpeed_L2c -- L2b-lineage universal-donor retrain in the corrected
# post-RZF world (user decision 2026-07-16). Env = MixedSpeed_L2b (hetero
# K=32, n_active U{16..32}) with speed widened to U(5,40) km/h.
# NO-BETA WORLD (owner decision 2026-07-16): la_beta stays at default 1.0,
# no la_beta_by_depth -- the 'no-OLLA' honest world. Prediction is unbiased
# (post-RZF) but unmargined: expect stochastic first-NACKs (~25-60% by
# depth) recovered by IR; no per-world calibration procedure exists. GPU2.
#
# CODE PINNING (audit round 7): every launch AND resume verifies that the
# EXECUTABLE TRAINING CODE (root-level *.py) is identical to the pinned
# baseline commit, the working tree is clean, and -- when resuming -- the
# checkpoint was produced by that same code. Docs/wrapper/analysis commits
# are allowed (they do not touch root *.py); any training-code drift halts
# the run in an idle HOLD loop (tmux session stays alive so the watchdog
# does not zombie-churn) until a human intervenes or sets
# ALLOW_HASH_MISMATCH=1 for one deliberate override.
cd /home/MYH/ML_DRL_Scheduler || exit 1
export CUDA_VISIBLE_DEVICES=5 \
       OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
       NUMEXPR_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8 \
       TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=2
RUN_DIR=Run4/MixedSpeed_L2c
LOG=Run4/MixedSpeed_L2c_run.log
PIN_BASE=9ed28d0706b979bbede8fa1b1b57eb8a9cfe1c9c   # round-8 commit (snr_m fix; current root-py baseline)

pin_check() {  # returns non-zero with a reason on stdout if the pin fails
  git cat-file -e "$PIN_BASE" 2>/dev/null || { echo "pin commit missing"; return 1; }
  if ! git diff --quiet "$PIN_BASE" HEAD -- ':(glob)*.py'; then
    echo "root *.py differs from pinned $PIN_BASE (HEAD=$(git rev-parse --short HEAD))"; return 1
  fi
  if [ -n "$(git status --porcelain -- ':(glob)*.py')" ]; then
    echo "root *.py working tree dirty"; return 1
  fi
  if [ -f "$RUN_DIR/ckpt/latest.pt" ]; then
    CK=$(python3 -c "import torch;print(torch.load('$RUN_DIR/ckpt/latest.pt',map_location='cpu')['cfg'].get('git_hash','none'))" 2>/dev/null)
    if [ -z "$CK" ] || [ "$CK" = "none" ]; then
      echo "checkpoint has no git_hash stamp"; return 1
    fi
    git cat-file -e "$CK" 2>/dev/null || { echo "ckpt commit $CK unknown"; return 1; }
    if ! git diff --quiet "$CK" "$PIN_BASE" -- ':(glob)*.py'; then
      echo "ckpt code ($CK) differs from pinned $PIN_BASE"; return 1
    fi
  fi
  return 0
}

for i in $(seq 1 1000); do
  if [ "${ALLOW_HASH_MISMATCH:-0}" != "1" ]; then
    REASON=$(pin_check) || {
      echo "===== [wrap] PIN FAIL attempt $i $(date '+%F %T') :: $REASON -- HOLDING (no relaunch)" >> "$LOG"
      # HOLD: keep the watchdog's progress marker fresh every 30s (else its
      # HANG_MIN no-progress rule kills the session and churns the HOLD);
      # re-evaluate the pin only every ~5min (pin_check torch-loads the ckpt).
      hold_n=0
      while [ "${ALLOW_HASH_MISMATCH:-0}" != "1" ]; do
        if [ $((hold_n % 10)) -eq 0 ] && pin_check >/dev/null; then break; fi
        sleep 30
        hold_n=$((hold_n + 1))
        [ -d "$RUN_DIR" ] && touch "$RUN_DIR/.wd_marker"
      done
      echo "===== [wrap] pin restored $(date '+%F %T') -- resuming loop" >> "$LOG"
    }
  fi
  if [ -f "$RUN_DIR/ckpt/latest.pt" ]; then
    ARGS="--resume $RUN_DIR/ckpt/latest.pt"
  else
    [ -n "$RUN_DIR" ] && [ -e "$RUN_DIR" ] && mv "$RUN_DIR" "${RUN_DIR}.stale.$(date +%s)"
    ARGS="--run_root Run4 --run_name MixedSpeed_L2c"
  fi
  echo "===== [wrap] attempt $i $(date '+%F %T') :: HEAD=$(git rev-parse --short HEAD) :: $ARGS =====" >> "$LOG"
  python3 train_phase2.py --mode hetero --num_ue 32 \
    --n_active_min 16 --n_active_max 32 \
    --ue_speed_min 5 --ue_speed_max 40 \
    --la_mode post_rzf --decode_order rbg_major \
    --entropy_coef 0.02 --patience_evals 100000 --seed 2024 $ARGS >> "$LOG" 2>&1
  grep -q "^Done\. " "$LOG" && { echo "===== [wrap] DONE =====" >> "$LOG"; break; }
  echo "===== [wrap] exit -> relaunch 20s =====" >> "$LOG"; sleep 20
done
