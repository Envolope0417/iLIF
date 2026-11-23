#!/usr/bin/env bash
# iLIF reproduction runner — run inside `tmux` (see usage below)
set -e

source /home/guanhaochen/miniconda3/bin/activate ilif
cd /home/guanhaochen/iLIF

export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_CO_RESOLVE_ENDPOINT=https://hf-mirror.com

LOG=/home/guanhaochen/iLIF/repro.log
echo "=== iLIF reproduction started: $(date '+%F %T') ===" | tee "$LOG"

python main.py \
  --model_name /home/guanhaochen/iLIF/pretrained/bert-base-uncased \
  --t_lr 2e-5 --lr 1e-4 --w 0.6 --threshold 2 --epoch 30 --batch_size 1 \
  --beta_intra 0.3 --beta_inter 0.7 --max_iteration 9 --min_iteration 2 \
  --num_heads 4 --mlp_drop 0.4 --seed 209 --gamma 2.0 \
  2>&1 | tee -a "$LOG"

echo "=== iLIF reproduction finished: $(date '+%F %T') ===" | tee -a "$LOG"
