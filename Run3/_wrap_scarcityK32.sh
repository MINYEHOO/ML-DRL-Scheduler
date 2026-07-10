#!/bin/bash
# Durable CPU-resume wrapper for ScarcityK32 (num_ue 32).
cd /home/MYH/ML_DRL_Scheduler || exit 1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 \
       NUMEXPR_NUM_THREADS=24 VECLIB_MAXIMUM_THREADS=24 \
       TF_NUM_INTRAOP_THREADS=24 TF_NUM_INTEROP_THREADS=2
RUN_DIR=Run3/ScarcityK32
LOG=Run3/ScarcityK32_run.log
for i in $(seq 1 1000); do
  if [ -f "$RUN_DIR/ckpt/latest.pt" ]; then
    ARGS="--resume $RUN_DIR/ckpt/latest.pt"
  else
    [ -n "$RUN_DIR" ] && rm -rf "$RUN_DIR"
    ARGS="--run_root Run3 --run_name ScarcityK32"
  fi
  echo "===== [wrap] attempt $i $(date '+%F %T') :: $ARGS =====" >> "$LOG"
  python3 train_phase2.py --mode hetero --ue_speed_kmh 10 --entropy_coef 0.02 \
    --num_ue 32 --patience_evals 100000 --seed 2024 $ARGS >> "$LOG" 2>&1
  grep -q "^Done\. " "$LOG" && { echo "===== [wrap] DONE =====" >> "$LOG"; break; }
  echo "===== [wrap] exit -> relaunch 20s =====" >> "$LOG"; sleep 20
done
