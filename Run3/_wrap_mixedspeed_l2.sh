#!/bin/bash
# Level-2 mixed-load + mixed-speed: n_active random [16,32] AND per-UE speed U(5,30)
# each episode -> ONE policy sees varying load AND CSI-reliability. On GPU4 for now;
# migrate to CPU after live runs finish (set CUDA_VISIBLE_DEVICES="" + relaunch).
cd /home/MYH/ML_DRL_Scheduler || exit 1
export CUDA_VISIBLE_DEVICES=4
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 \
       NUMEXPR_NUM_THREADS=6 VECLIB_MAXIMUM_THREADS=6 \
       TF_NUM_INTRAOP_THREADS=6 TF_NUM_INTEROP_THREADS=2
RUN_DIR=Run3/MixedSpeed_L2
LOG=Run3/MixedSpeed_L2_run.log
for i in $(seq 1 1000); do
  if [ -f "$RUN_DIR/ckpt/latest.pt" ]; then
    ARGS="--resume $RUN_DIR/ckpt/latest.pt"
  else
    [ -n "$RUN_DIR" ] && [ -e "$RUN_DIR" ] && mv "$RUN_DIR" "${RUN_DIR}.stale.$(date +%s)"
    ARGS="--run_root Run3 --run_name MixedSpeed_L2"
  fi
  echo "===== [wrap] attempt $i $(date '+%F %T') :: $ARGS =====" >> "$LOG"
  python3 train_phase2.py --mode hetero --entropy_coef 0.02 \
    --num_ue 32 --n_active_min 16 --n_active_max 32 \
    --ue_speed_min 5 --ue_speed_max 30 \
    --patience_evals 100000 --seed 2024 $ARGS >> "$LOG" 2>&1
  grep -q "^Done\. " "$LOG" && { echo "===== [wrap] DONE =====" >> "$LOG"; break; }
  echo "===== [wrap] exit -> relaunch 20s =====" >> "$LOG"; sleep 20
done
