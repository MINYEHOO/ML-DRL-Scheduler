#!/bin/bash
# Level-2 mixed-load run: ONE policy, n_active random [16,32] per episode.
# On GPU5 for now (CPU busy with live 3 runs); migrate to CPU after they finish
# by changing CUDA_VISIBLE_DEVICES="" and relaunching (--resume from latest.pt).
cd /home/MYH/ML_DRL_Scheduler || exit 1
export CUDA_VISIBLE_DEVICES=5
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 \
       NUMEXPR_NUM_THREADS=6 VECLIB_MAXIMUM_THREADS=6 \
       TF_NUM_INTRAOP_THREADS=6 TF_NUM_INTEROP_THREADS=2
RUN_DIR=Run3/MixedLoad_L2
LOG=Run3/MixedLoad_L2_run.log
for i in $(seq 1 1000); do
  if [ -f "$RUN_DIR/ckpt/latest.pt" ]; then
    ARGS="--resume $RUN_DIR/ckpt/latest.pt"
  else
    [ -n "$RUN_DIR" ] && rm -rf "$RUN_DIR"
    ARGS="--run_root Run3 --run_name MixedLoad_L2"
  fi
  echo "===== [wrap] attempt $i $(date '+%F %T') :: $ARGS =====" >> "$LOG"
  python3 train_phase2.py --mode hetero --ue_speed_kmh 15 --entropy_coef 0.02 \
    --num_ue 32 --n_active_min 16 --n_active_max 32 \
    --patience_evals 100000 --seed 2024 $ARGS >> "$LOG" 2>&1
  grep -q "^Done\. " "$LOG" && { echo "===== [wrap] DONE =====" >> "$LOG"; break; }
  echo "===== [wrap] exit -> relaunch 20s =====" >> "$LOG"; sleep 20
done
