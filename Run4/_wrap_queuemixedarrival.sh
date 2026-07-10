#!/bin/bash
# Durable auto-resume wrapper for Run4/QueueMixedArrival (GPU 2, user-designated 2026-07-07).
# Level-2 mixed-arrival: env = QueueMain except p_arrival ~ U(0.15, 0.40) drawn
# per episode (dedicated RNG stream; nominal rho spans ~0.37-1.96 combined with
# n_active 16-32). Load is decorrelated from n_active, so reading the queue
# observations is the ONLY way to know the traffic intensity -- direct attack
# on the queue-blind 5375 plateau measured by the L2b zero-shot probe.
cd /home/MYH/ML_DRL_Scheduler || exit 1
export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
       NUMEXPR_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8 \
       TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=2
RUN_DIR=Run4/QueueMixedArrival
LOG=Run4/QueueMixedArrival_run.log
for i in $(seq 1 1000); do
  if [ -f "$RUN_DIR/ckpt/latest.pt" ]; then
    ARGS="--resume $RUN_DIR/ckpt/latest.pt"
  else
    [ -n "$RUN_DIR" ] && rm -rf "$RUN_DIR"
    ARGS="--run_root Run4 --run_name QueueMixedArrival"
  fi
  echo "===== [wrap] attempt $i $(date '+%F %T') :: $ARGS =====" >> "$LOG"
  python3 train_phase2.py --mode queue --p_arrival_min 0.15 --p_arrival_max 0.40 \
    --seed 2024 $ARGS >> "$LOG" 2>&1
  grep -q "^Done\. " "$LOG" && { echo "===== [wrap] DONE =====" >> "$LOG"; break; }
  echo "===== [wrap] exit -> relaunch 20s =====" >> "$LOG"; sleep 20
done
