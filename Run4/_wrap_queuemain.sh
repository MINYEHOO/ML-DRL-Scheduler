#!/bin/bash
# Durable auto-resume wrapper for Run4/QueueMain (GPU 5, user-designated 2026-07-06).
# Kept in the repo so /tmp cleanup can't delete it. Relaunched by tmux session
# 'queuemain'; watched by Run3/_watchdog.sh; resurrected by ~/.bashrc after a
# container restart. Always passes --mode queue so a resume rebuilds the exact
# phase4 queue config (resume cfg-mismatch warnings stay silent).
cd /home/MYH/ML_DRL_Scheduler || exit 1
export CUDA_VISIBLE_DEVICES=5
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
       NUMEXPR_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8 \
       TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=2
RUN_DIR=Run4/QueueMain
LOG=Run4/QueueMain_run.log
for i in $(seq 1 1000); do
  if [ -f "$RUN_DIR/ckpt/latest.pt" ]; then
    ARGS="--resume $RUN_DIR/ckpt/latest.pt"
  else
    [ -n "$RUN_DIR" ] && rm -rf "$RUN_DIR"
    ARGS="--run_root Run4 --run_name QueueMain"
  fi
  echo "===== [wrap] attempt $i $(date '+%F %T') :: $ARGS =====" >> "$LOG"
  python3 train_phase2.py --mode queue --seed 2024 $ARGS >> "$LOG" 2>&1
  grep -q "^Done\. " "$LOG" && { echo "===== [wrap] DONE =====" >> "$LOG"; break; }
  echo "===== [wrap] exit -> relaunch 20s =====" >> "$LOG"; sleep 20
done
