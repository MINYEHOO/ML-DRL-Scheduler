#!/bin/bash
# NARROW world + LR annealing, 1500 updates, GPU 5.
#
#   world  : speed 20 / p_arrival 0.30 / K_act 24 all FIXED -- the wide world's
#            MEANS, so difficulty is held and only diversity is removed. NARROW
#            at 866 already tied base on the wide holdout (4797 vs 4787) and
#            held across all nine OOD worlds, so the randomisation control is
#            re-run here under the final recipe.
#   optim  : lr 3e-4 -> 0 linear over the FULL 1500 updates. Decaying over 866
#            instead would pin lr at 0 from update 866 and freeze the last 634
#            updates, so the horizon must match --num_updates.
#   why 1500: every run so far was cut at 866 while still moving, so none of
#            them shows where the recipe CONVERGES -- only where it was
#            stopped. With lr annealed to zero the optimizer actually settles,
#            which makes this the first run that can answer that.
cd /home/MYH/ML_DRL_Scheduler || exit 1
export CUDA_VISIBLE_DEVICES=5
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6
RUN_DIR=Run4/QueuePostRZF_S40HL_CQI4_NARROW_LRANN
LOG=Run4/QueuePostRZF_S40HL_CQI4_NARROW_LRANN_run.log
for i in $(seq 1 1000); do
  if [ -f "$RUN_DIR/ckpt/latest.pt" ]; then
    ARGS="--resume $RUN_DIR/ckpt/latest.pt"
  else
    ARGS="--run_root Run4 --run_name QueuePostRZF_S40HL_CQI4_NARROW_LRANN"
  fi
  echo "===== [wrap] attempt $i $(date '+%F %T') :: $ARGS =====" >> "$LOG"
  python3 train_phase2.py --mode queue --la_mode post_rzf --decode_order rbg_major \
    --la_beta_by_depth 1.0018,0.7499,0.6592,0.6058 --cqi_mode nr4bit \
    --entropy_coef 0.02 --ppo_save_every 1 --patience_evals 100000 --seed 2024 \
    --num_updates 1500 --eval_every 10 --batched_replay \
    --lr_final 0.0 --lr_decay_updates 1500 \
    --p_arrival_min 0.30 --p_arrival_max 0.30 \
    --ue_speed_min 20 --ue_speed_max 20 \
    --n_active_min 24 --n_active_max 24 $ARGS >> "$LOG" 2>&1
  grep -q "^Done\. " "$LOG" && { echo "===== [wrap] DONE =====" >> "$LOG"; break; }
  echo "===== [wrap] exit -> relaunch 20s =====" >> "$LOG"; sleep 20
done
