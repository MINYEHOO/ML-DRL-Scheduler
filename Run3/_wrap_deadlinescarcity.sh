#!/bin/bash
# Durable CPU-resume wrapper for DeadlineScarcity (deadline [2,6], 12 threads).
cd /home/MYH/ML_DRL_Scheduler || exit 1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 \
       NUMEXPR_NUM_THREADS=12 VECLIB_MAXIMUM_THREADS=12 \
       TF_NUM_INTRAOP_THREADS=12 TF_NUM_INTEROP_THREADS=2
RUN_DIR=Run3/DeadlineScarcity
LOG=Run3/DeadlineScarcity_run.log
for i in $(seq 1 1000); do
  if [ -f "$RUN_DIR/ckpt/latest.pt" ]; then
    ARGS="--resume $RUN_DIR/ckpt/latest.pt"
  else
    [ -n "$RUN_DIR" ] && rm -rf "$RUN_DIR"
    ARGS="--run_root Run3 --run_name DeadlineScarcity"
  fi
  echo "===== [wrap] attempt $i $(date '+%F %T') :: $ARGS =====" >> "$LOG"
  python3 train_phase2.py --mode hetero --ue_speed_kmh 10 --entropy_coef 0.02 \
    --deadline_min 2 --deadline_max 6 --patience_evals 100000 --seed 2024 $ARGS >> "$LOG" 2>&1
  grep -q "^Done\. " "$LOG" && { echo "===== [wrap] DONE =====" >> "$LOG"; break; }
  echo "===== [wrap] exit -> relaunch 20s =====" >> "$LOG"; sleep 20
done
