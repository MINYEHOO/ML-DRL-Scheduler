#!/bin/bash
# Durable auto-resume wrapper for Run4/GenieL2b (CPU, user-designated 2026-07-08).
# EXACT MixedSpeed_L2b config + GENIE perfect CSI: pmi_mode=genie (h_hat==h_true,
# zero quantization) + p_csi=1.0 (zero staleness) => the BS/scheduler and every
# baseline act on the true channel throughout. Upper-bound reference: how much
# do PPO and the heuristics gain when the imperfect-CSI penalty (measured ~1.4 dB
# SU, larger for MU null leakage) is removed. CPU-only (all GPUs run other Run4).
cd /home/MYH/ML_DRL_Scheduler || exit 1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 \
       NUMEXPR_NUM_THREADS=12 VECLIB_MAXIMUM_THREADS=12 \
       TF_NUM_INTRAOP_THREADS=12 TF_NUM_INTEROP_THREADS=2
RUN_DIR=Run4/GenieL2b
LOG=Run4/GenieL2b_run.log
for i in $(seq 1 1000); do
  if [ -f "$RUN_DIR/ckpt/latest.pt" ]; then
    ARGS="--resume $RUN_DIR/ckpt/latest.pt"
  else
    [ -n "$RUN_DIR" ] && rm -rf "$RUN_DIR"
    ARGS="--run_root Run4 --run_name GenieL2b"
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
