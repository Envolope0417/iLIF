#!/usr/bin/env bash
# Experiment: inference-only REVERSE anneal (loose -> strict): 0.55 -> 0.60 -> 0.65.
# Training stays argmax (rate=0.0); only EVAL uses the increasing threshold schedule.
# 4 folds (0-3), 4 GPUs, 1 process per GPU (single wave, no queue).
# Trailing `wait` keeps this script (and its tmux session) alive until all folds finish,
# so the launched processes are not killed by the script exiting.

source /home/guanhaochen/miniconda3/bin/activate ilif
cd /home/guanhaochen/iLIF

export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_CO_RESOLVE_ENDPOINT=https://hf-mirror.com

COMMON="--model_name /home/guanhaochen/iLIF/pretrained/bert-base-uncased --t_lr 2e-5 --lr 1e-4 --w 0.6 --threshold 2 --epoch 30 --batch_size 1 --beta_intra 0.3 --beta_inter 0.7 --max_iteration 9 --min_iteration 2 --num_heads 4 --mlp_drop 0.4 --seed 209 --gamma 2.0 --use_infer_anneal --anneal_direction up --anneal_start 0.55 --anneal_step 0.05 --anneal_ceiling 0.65"

run() {  # $1=gpu  $2=fold
  nohup env CUDA_VISIBLE_DEVICES=$1 python main.py $COMMON --fold_idx $2 > anneal_up_fold$2.out 2>&1 &
  echo "anneal_up fold $2 -> GPU $1 (pid $!)"
}

run 0 0
run 1 1
run 2 2
run 3 3

echo "All 4 folds dispatched; waiting for completion..."
wait
echo "All 4 folds finished."
