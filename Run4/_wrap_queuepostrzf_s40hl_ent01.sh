#!/bin/bash
# QueuePostRZF_S40HighLoad_Ent01 -- S40-generation HIGH-LOAD arm (user directive 2026-07-20).
# Same S40 world as QueuePostRZF_S40Ent02 with ONE delta: p_arrival per-episode
# U(0.15, 0.50) instead of U(0.15, 0.40) (min kept, max raised -- user choice).
# Load-headroom probe: scheduler spread grows 2697->7311 over p 0.3->0.5.
export CUDA_VISIBLE_DEVICES=4 \
       OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
       NUMEXPR_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8 \
       TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=2
RUN_DIR=Run4/QueuePostRZF_S40HighLoad_Ent01
LOG=Run4/QueuePostRZF_S40HighLoad_Ent01_run.log
PIN_BASE=71c0bd4085f6ae0ce7ee7460cffe958046a63543   # S40-generation pin (eval CSV v2 + save_every CLI)

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
    ARGS="--run_root Run4 --run_name QueuePostRZF_S40HighLoad_Ent01"
  fi
  echo "===== [wrap] attempt $i $(date '+%F %T') :: HEAD=$(git rev-parse --short HEAD) :: $ARGS =====" >> "$LOG"
  python3 train_phase2.py --mode queue \
    --p_arrival_min 0.15 --p_arrival_max 0.50 \
    --ue_speed_min 5 --ue_speed_max 40 \
    --la_mode post_rzf --decode_order rbg_major \
    --la_beta_by_depth 0.9757,0.7180,0.6355,0.5837 \
    --entropy_coef 0.01 --ppo_save_every 1 \
    --patience_evals 100000 --seed 2024 $ARGS >> "$LOG" 2>&1
  grep -q "^Done\. " "$LOG" && { echo "===== [wrap] DONE =====" >> "$LOG"; break; }
  echo "===== [wrap] exit -> relaunch 20s =====" >> "$LOG"; sleep 20
done
