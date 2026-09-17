#!/bin/bash
# Durable native-resume loop for a PaperMain run (persistent disk, so a container
# restart cannot delete it; relaunch the same command after a restart).
#
#   bash review/runners/resume_loop.sh <entry.py> <recipe> <run_name> <gpu>
#
# Each attempt is the protected trainer's own --resume from runs/<run>/ckpt/latest.pt,
# which re-validates sources, manifest, schedule and checkpoint before writing
# anything, so repeating it is safe. Stops when the trainer prints "Done." or
# after 200 attempts. Console goes to review/logs/<run>.log (appended) so the
# run folder's console.log alias keeps pointing at the live output.
set -u
ENTRY=$1; RECIPE=$2; RUN=$3; GPU=$4
cd /home/MYH/ML_DRL_Scheduler/PaperMain || exit 1
LOG=review/logs/$RUN.log
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
for i in $(seq 1 200); do
  echo "===== [resume_loop] attempt $i $(date '+%F %T') :: $ENTRY --recipe $RECIPE --resume runs/$RUN/ckpt/latest.pt (gpu $GPU) =====" >> "$LOG"
  python3 -u "$ENTRY" --recipe "$RECIPE" --resume "runs/$RUN/ckpt/latest.pt" >> "$LOG" 2>&1
  code=$?
  if grep -q "^Done\. " "$LOG"; then echo "===== [resume_loop] DONE =====" >> "$LOG"; exit 0; fi
  if grep -q "already reached --num-updates" "$LOG"; then echo "===== [resume_loop] target already reached =====" >> "$LOG"; exit 0; fi
  echo "===== [resume_loop] exit $code -> retry in 30s =====" >> "$LOG"; sleep 30
done
