#!/bin/bash
# Durable auto-resume wrapper for Run4/GenieFineTune (GPU 3, user-designated 2026-07-09).
# WARM-START twin of GenieL2b: identical genie environment (pmi_mode=genie +
# p_csi=1.0 = perfect CSI, exact MixedSpeed_L2b config) but the actor is
# warm-started from MixedSpeed_L2b best.pt (update 859, eval 10030) instead of
# fresh. Controlled A/B against GenieL2b (fresh, CPU): does the Run3-trained
# policy transfer faster / higher in the perfect-CSI world, or does its
# imperfect-CSI SU-bias hold it back when MU is now safe? No surgery needed --
# both are hetero mode (70-dim encoder, 161-dim critic_v2), so L2b loads direct.
cd /home/MYH/ML_DRL_Scheduler || exit 1
export CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
       NUMEXPR_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8 \
       TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=2
RUN_DIR=Run4/GenieFineTune
LOG=Run4/GenieFineTune_run.log
for i in $(seq 1 1000); do
  if [ -f "$RUN_DIR/ckpt/latest.pt" ]; then
    ARGS="--resume $RUN_DIR/ckpt/latest.pt"
  else
    [ -n "$RUN_DIR" ] && rm -rf "$RUN_DIR"
    ARGS="--run_root Run4 --run_name GenieFineTune --init_from Run3/MixedSpeed_L2b/ckpt/best.pt"
  fi
  echo "===== [wrap] attempt $i $(date '+%F %T') :: $ARGS =====" >> "$LOG"
  python3 train_phase2.py --mode hetero --entropy_coef 0.02 \
    --num_ue 32 --n_active_min 16 --n_active_max 32 \
    --ue_speed_min 5 --ue_speed_max 30 \
    --pmi_mode genie --p_csi 1.0 \
    --patience_evals 100000 --seed 2024 \
    --target_kl 0.02 --critic_v2 $ARGS >> "$LOG" 2>&1
  grep -q "^Done\. " "$LOG" && { echo "===== [wrap] DONE =====" >> "$LOG"; break; }
  echo "===== [wrap] exit -> relaunch 20s =====" >> "$LOG"; sleep 20
done
