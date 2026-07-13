#!/bin/bash
# Durable auto-resume wrapper for Run4/QueueMixedFairLA (GPU 5, user-designated 2026-07-10).
# EXACT QueueMixedArrival config (queue mode, p ~ U(0.15,0.40)/episode, seed 2024)
# + mu_aware_la=True: the ONLY change is fair link adaptation (B_tx de-rated by
# the planned stream count -- the post-audit fix). Clean A/B vs QueueMixedArrival
# (GPU 2, old rule): does PPO trained IN the fair-LA world learn deeper MU and
# beat the old-rule policy's 6020 (+21.4%)? Baselines inside run_eval share the
# same m-aware world (env-level flag), so within-run comparisons stay consistent.
cd /home/MYH/ML_DRL_Scheduler || exit 1
export CUDA_VISIBLE_DEVICES=5
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
       NUMEXPR_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8 \
       TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=2
RUN_DIR=Run4/QueueMixedFairLA
LOG=Run4/QueueMixedFairLA_run.log
for i in $(seq 1 1000); do
  if [ -f "$RUN_DIR/ckpt/latest.pt" ]; then
    ARGS="--resume $RUN_DIR/ckpt/latest.pt"
  else
    [ -n "$RUN_DIR" ] && [ -e "$RUN_DIR" ] && mv "$RUN_DIR" "${RUN_DIR}.stale.$(date +%s)"
    ARGS="--run_root Run4 --run_name QueueMixedFairLA"
  fi
  echo "===== [wrap] attempt $i $(date '+%F %T') :: $ARGS =====" >> "$LOG"
  python3 train_phase2.py --mode queue --p_arrival_min 0.15 --p_arrival_max 0.40 \
    --mu_aware_la --seed 2024 $ARGS >> "$LOG" 2>&1
  grep -q "^Done\. " "$LOG" && { echo "===== [wrap] DONE =====" >> "$LOG"; break; }
  echo "===== [wrap] exit -> relaunch 20s =====" >> "$LOG"; sleep 20
done
