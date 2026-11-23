#!/usr/bin/env bash
# Experiment: inference-only annealed threshold (0.65 -> 0.45, -0.05/iteration, floor 0.45).
# Training stays argmax (rate=0.0); ONLY eval uses the annealed threshold.
# 5 folds, 4 GPUs, 1 process per GPU (per-process peak ~9 GiB on 11 GB; 2/GPU OOMs).
# 2 waves: folds 0-3 (1 per GPU) -> fold 4 on GPU 0 after fold 0 finishes.

source /home/guanhaochen/miniconda3/bin/activate ilif
cd /home/guanhaochen/iLIF

export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_CO_RESOLVE_ENDPOINT=https://hf-mirror.com

COMMON="--model_name /home/guanhaochen/iLIF/pretrained/bert-base-uncased --t_lr 2e-5 --lr 1e-4 --w 0.6 --threshold 2 --epoch 30 --batch_size 1 --beta_intra 0.3 --beta_inter 0.7 --max_iteration 9 --min_iteration 2 --num_heads 4 --mlp_drop 0.4 --seed 209 --gamma 2.0 --use_infer_anneal --anneal_start 0.65 --anneal_step 0.05 --anneal_floor 0.45"

run() {  # $1=gpu  $2=fold
  nohup env CUDA_VISIBLE_DEVICES=$1 python main.py $COMMON --fold_idx $2 > anneal_fold$2.out 2>&1 &
  echo "anneal fold $2 -> GPU $1 (pid $!)"
}

# ---- wave 1: folds 0-3, 1 per GPU ----
run 0 0; pid0=$!
run 1 1
run 2 2
run 3 3

# ---- wave 2: fold 4 on GPU 0 (after fold 0) ----
wait $pid0 || true; run 0 4

echo "All 5 folds dispatched (anneal experiment)."
