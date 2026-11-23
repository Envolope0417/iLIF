#!/usr/bin/env bash
# Launch both iLIF improvement experiments.
# Per-process peak is ~4.4-6.3 GiB (long doc + 9 iterations), so 2/GPU OOMs on 11GB.
#   Experiment ② (dynamic-only):        folds on GPU 0,1
#   Experiment ③ (dynamic+persistence): folds on GPU 2,3
# 1 process per GPU; 3 waves: folds (0,1) -> (2,3) -> (4), gated on the prior fold
# on the same GPU finishing.

source /home/guanhaochen/miniconda3/bin/activate ilif
cd /home/guanhaochen/iLIF

export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_CO_RESOLVE_ENDPOINT=https://hf-mirror.com

COMMON="--model_name /home/guanhaochen/iLIF/pretrained/bert-base-uncased --t_lr 2e-5 --lr 1e-4 --w 0.6 --threshold 2 --epoch 30 --batch_size 1 --beta_intra 0.3 --beta_inter 0.7 --max_iteration 9 --min_iteration 2 --num_heads 4 --mlp_drop 0.4 --seed 209 --gamma 2.0"

run() {  # $1=gpu  $2=flags  $3=fold  $4=tag
  nohup env CUDA_VISIBLE_DEVICES=$1 python main.py $COMMON $2 --fold_idx $3 > ${4}_fold${3}.out 2>&1 &
  echo "${4} fold $3 -> GPU $1 (pid $!)"
}

# ---- wave 1 (1 per GPU) ----
run 0 "--use_dynamic" 0 dynamic;                 pid_d0=$!
run 1 "--use_dynamic" 1 dynamic;                 pid_d1=$!
run 2 "--use_dynamic --use_persistence" 0 full;  pid_f0=$!
run 3 "--use_dynamic --use_persistence" 1 full;  pid_f1=$!

# ---- wave 2 ----
wait $pid_d0 || true; run 0 "--use_dynamic" 2 dynamic;                 pid_d2=$!
wait $pid_d1 || true; run 1 "--use_dynamic" 3 dynamic
wait $pid_f0 || true; run 2 "--use_dynamic --use_persistence" 2 full;  pid_f2=$!
wait $pid_f1 || true; run 3 "--use_dynamic --use_persistence" 3 full

# ---- wave 3 ----
wait $pid_d2 || true; run 0 "--use_dynamic" 4 dynamic
wait $pid_f2 || true; run 2 "--use_dynamic --use_persistence" 4 full

echo "All 10 folds dispatched (1 per GPU, 3 waves)."
