#!/bin/bash
# Durable CPU-resume wrapper for Uniform10_Ent002 (kept in Run3/ so /tmp cleanup can't delete it).
cd /home/MYH/ML_DRL_Scheduler || exit 1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 \
       NUMEXPR_NUM_THREADS=24 VECLIB_MAXIMUM_THREADS=24 \
       TF_NUM_INTRAOP_THREADS=24 TF_NUM_INTEROP_THREADS=2
RUN_DIR=Run3/Uniform10_Ent002
LOG=Run3/Uniform10_Ent002_run.log
for i in $(seq 1 1000); do
  if [ -f "$RUN_DIR/ckpt/latest.pt" ]; then
    ARGS="--resume $RUN_DIR/ckpt/latest.pt"
  else
    [ -n "$RUN_DIR" ] && [ -e "$RUN_DIR" ] && mv "$RUN_DIR" "${RUN_DIR}.stale.$(date +%s)"
    ARGS="--run_root Run3 --run_name Uniform10_Ent002"
  fi
  echo "===== [wrap] attempt $i $(date '+%F %T') :: $ARGS =====" >> "$LOG"
  python3 train_phase2.py --mode hetero --ue_speed_kmh 10 --entropy_coef 0.02 \
    --patience_evals 100000 --seed 2024 $ARGS >> "$LOG" 2>&1
  grep -q "^Done\. " "$LOG" && { echo "===== [wrap] DONE =====" >> "$LOG"; break; }
  echo "===== [wrap] exit -> relaunch 20s =====" >> "$LOG"; sleep 20
done
