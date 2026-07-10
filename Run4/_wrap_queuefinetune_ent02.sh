#!/bin/bash
# Durable auto-resume wrapper for Run4/QueueFineTuneEnt02 (GPU 1, user-designated 2026-07-08).
# FORK of QueueFineTune at update-199 boundary (cp -a snapshot, best 5793 +
# patience 4/10 carried): controlled A/B on the entropy coefficient --
#   original (GPU 3): 0.01 flat (cfg default; the brief anneal detour on the
#   original was reverted -- see its wrapper's ENTROPY HISTORY note)
#   this fork (GPU 1): FLAT 0.02 (the Run3-recipe value; user design)
cd /home/MYH/ML_DRL_Scheduler || exit 1
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
       NUMEXPR_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8 \
       TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=2
RUN_DIR=Run4/QueueFineTuneEnt02
LOG=Run4/QueueFineTuneEnt02_run.log
for i in $(seq 1 1000); do
  if [ -f "$RUN_DIR/ckpt/latest.pt" ]; then
    ARGS="--resume $RUN_DIR/ckpt/latest.pt"
  else
    # fork lost its ckpt (should not happen) -> rebuild from the L2b init
    [ -n "$RUN_DIR" ] && [ -e "$RUN_DIR" ] && mv "$RUN_DIR" "${RUN_DIR}.stale.$(date +%s)"
    ARGS="--run_root Run4 --run_name QueueFineTuneEnt02 --init_from Run4/_init_from_L2b/l2b_padded_init.pt"
  fi
  echo "===== [wrap] attempt $i $(date '+%F %T') :: $ARGS =====" >> "$LOG"
  python3 train_phase2.py --mode queue --seed 2024 --patience_evals 10 \
    --entropy_coef 0.02 \
    $ARGS >> "$LOG" 2>&1
  grep -q "^Done\. " "$LOG" && { echo "===== [wrap] DONE =====" >> "$LOG"; break; }
  echo "===== [wrap] exit -> relaunch 20s =====" >> "$LOG"; sleep 20
done
