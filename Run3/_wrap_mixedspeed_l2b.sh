#!/bin/bash
# MixedSpeed_L2b: restart of MixedSpeed_L2 (stopped 2026-07-02 at ~update 141)
# from its best.pt (update 29, eval 9179 -- pre-collapse peak). Environment is
# IDENTICAL to MixedSpeed_L2 (K=32, n_active U[16,32], per-UE speed U(5,30),
# p_csi 0.6, hard load/deadline); the changes are training-recipe only:
#   (1) --target_kl 0.02  : KL early-stop -- guards against the destructive
#       updates that collapsed L2 at update 39/52 (KL 0.054/0.066, clip 43/47%)
#   (2) --critic_v2       : +27 structured value features (deadline histogram,
#       backlog totals, retx-grid, CQI/age, slot phase); offline probe:
#       held-out R^2 -0.99 -> +0.52. ValueHead input 134 -> 161.
#   (3) --init_from       : actor-only warm-start from L2 best.pt; value head +
#       return normalizer + optimizer start fresh (old critic probe-confirmed
#       worthless, ev ~0.04).
# Flags (1)+(2) are passed on EVERY launch incl. resumes -- cfg is rebuilt from
# CLI each time; forgetting --critic_v2 on resume would crash on value_head
# shape mismatch. On GPU4 (the slot the old L2 session used).
cd /home/MYH/ML_DRL_Scheduler || exit 1
export CUDA_VISIBLE_DEVICES=4
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 \
       NUMEXPR_NUM_THREADS=6 VECLIB_MAXIMUM_THREADS=6 \
       TF_NUM_INTRAOP_THREADS=6 TF_NUM_INTEROP_THREADS=2
RUN_DIR=Run3/MixedSpeed_L2b
LOG=Run3/MixedSpeed_L2b_run.log
for i in $(seq 1 1000); do
  if [ -f "$RUN_DIR/ckpt/latest.pt" ]; then
    ARGS="--resume $RUN_DIR/ckpt/latest.pt"
  else
    [ -n "$RUN_DIR" ] && [ -e "$RUN_DIR" ] && mv "$RUN_DIR" "${RUN_DIR}.stale.$(date +%s)"
    ARGS="--run_root Run3 --run_name MixedSpeed_L2b --init_from Run3/MixedSpeed_L2/ckpt/best.pt"
  fi
  echo "===== [wrap] attempt $i $(date '+%F %T') :: $ARGS =====" >> "$LOG"
  python3 train_phase2.py --mode hetero --entropy_coef 0.02 \
    --num_ue 32 --n_active_min 16 --n_active_max 32 \
    --ue_speed_min 5 --ue_speed_max 30 \
    --patience_evals 100000 --seed 2024 \
    --target_kl 0.02 --critic_v2 $ARGS >> "$LOG" 2>&1
  grep -q "^Done\. " "$LOG" && { echo "===== [wrap] DONE =====" >> "$LOG"; break; }
  echo "===== [wrap] exit -> relaunch 20s =====" >> "$LOG"; sleep 20
done
