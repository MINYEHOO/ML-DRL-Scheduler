#!/bin/bash
# Durable auto-resume wrapper for Run4/QueueFineTune (GPU 3, user-designated 2026-07-07).
# Fine-tuning experiment: env identical to QueueMain (p_arrival 0.22); ONLY the
# initialization differs -- actor warm-started from MixedSpeed_L2b best.pt
# (zero-padded 70->73, queue-blind; value_head/ret_ fresh). Zero-shot basis:
# L2b-padded already scores 5375 (+11.4% vs SUS+CQI, 10/10) on this env, so
# whatever fine-tuning adds ABOVE that plateau is the value of queue information.
# patience 10 (user: fine-tune can harvest early once stable) + arming at upd 100.
# ENTROPY HISTORY (messy, disclosed for the A/B record):
#   upd 0-199: 0.01 (default). An anneal (0.02->0.002@600) was added to this
#   wrapper 07-07 08:31 but never applied (running bash keeps its parsed loop);
#   the 07-08 00:05 container restart DID apply it, so upd 200~219 ran at
#   ~0.0138 unintentionally. 07-08 (user directive): this ORIGINAL arm reverts
#   to the plain 0.01 default (flags removed below) = the "as-is" control;
#   the GPU-1 fork QueueFineTuneEnt02 (branched at upd 199) is the 0.02-flat
#   arm. A/B caveat: control arm has a ~20-update 0.0138 segment (200-219).
cd /home/MYH/ML_DRL_Scheduler || exit 1
export CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
       NUMEXPR_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8 \
       TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=2
RUN_DIR=Run4/QueueFineTune
LOG=Run4/QueueFineTune_run.log
for i in $(seq 1 1000); do
  if [ -f "$RUN_DIR/ckpt/latest.pt" ]; then
    ARGS="--resume $RUN_DIR/ckpt/latest.pt"
  else
    [ -n "$RUN_DIR" ] && [ -e "$RUN_DIR" ] && mv "$RUN_DIR" "${RUN_DIR}.stale.$(date +%s)"
    ARGS="--run_root Run4 --run_name QueueFineTune --init_from Run4/_init_from_L2b/l2b_padded_init.pt"
  fi
  echo "===== [wrap] attempt $i $(date '+%F %T') :: $ARGS =====" >> "$LOG"
  python3 train_phase2.py --mode queue --seed 2024 --patience_evals 10 \
    $ARGS >> "$LOG" 2>&1
  grep -q "^Done\. " "$LOG" && { echo "===== [wrap] DONE =====" >> "$LOG"; break; }
  echo "===== [wrap] exit -> relaunch 20s =====" >> "$LOG"; sleep 20
done
