#!/bin/bash
# Durable auto-resume wrapper for Run4/QueueHighLoad (GPU 4, user-designated 2026-07-07).
# Companion to QueueMain: ONE change (p_arrival 0.22 -> 0.33, nominal rho 0.81~1.62)
# for clean load-axis attribution. Probe basis: Run4/_calib_20260706 + rho_calib_033.
cd /home/MYH/ML_DRL_Scheduler || exit 1
export CUDA_VISIBLE_DEVICES=4
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
       NUMEXPR_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8 \
       TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=2
RUN_DIR=Run4/QueueHighLoad
LOG=Run4/QueueHighLoad_run.log
for i in $(seq 1 1000); do
  if [ -f "$RUN_DIR/ckpt/latest.pt" ]; then
    ARGS="--resume $RUN_DIR/ckpt/latest.pt"
  else
    [ -n "$RUN_DIR" ] && [ -e "$RUN_DIR" ] && mv "$RUN_DIR" "${RUN_DIR}.stale.$(date +%s)"
    ARGS="--run_root Run4 --run_name QueueHighLoad"
  fi
  echo "===== [wrap] attempt $i $(date '+%F %T') :: $ARGS =====" >> "$LOG"
  python3 train_phase2.py --mode queue --p_arrival 0.33 --seed 2024 $ARGS >> "$LOG" 2>&1
  grep -q "^Done\. " "$LOG" && { echo "===== [wrap] DONE =====" >> "$LOG"; break; }
  echo "===== [wrap] exit -> relaunch 20s =====" >> "$LOG"; sleep 20
done
