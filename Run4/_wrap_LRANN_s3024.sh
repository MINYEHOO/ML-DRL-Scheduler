#!/bin/bash
cd /home/MYH/ML_DRL_Scheduler || exit 1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6
RUN_DIR=Run4/QueuePostRZF_S40HL_CQI4_LRANN_s3024
LOG=Run4/QueuePostRZF_S40HL_CQI4_LRANN_s3024_run.log
for i in $(seq 1 1000); do
  if [ -f "$RUN_DIR/ckpt/latest.pt" ]; then ARGS="--resume $RUN_DIR/ckpt/latest.pt"
  else ARGS="--run_root Run4 --run_name QueuePostRZF_S40HL_CQI4_LRANN_s3024"; fi
  echo "===== [wrap] attempt $i $(date '+%F %T') :: $ARGS =====" >> "$LOG"
  python3 train_phase2.py --mode queue --la_mode post_rzf --decode_order rbg_major \
    --la_beta_by_depth 1.0018,0.7499,0.6592,0.6058 --cqi_mode nr4bit \
    --entropy_coef 0.02 --ppo_save_every 1 --patience_evals 100000 --seed 3024 \
    --num_updates 866 --eval_every 10 --batched_replay \
    --lr_final 0.0 --lr_decay_updates 866 --p_arrival_min 0.15 --p_arrival_max 0.50 --ue_speed_min 5 --ue_speed_max 40 $ARGS >> "$LOG" 2>&1
  grep -q "^Done\. " "$LOG" && { echo "===== [wrap] DONE =====" >> "$LOG"; break; }
  echo "===== [wrap] exit -> relaunch 20s =====" >> "$LOG"; sleep 20
done
