# -*- coding: utf-8 -*-
"""Measure baseline eval-time graph size: argmax (w=0.0) vs threshold (w=0.6).
Trains baseline (use_dynamic=False) on fold 0 for a few epochs and reports,
after each epoch, the number of graph edges admitted at eval time on the dev set.
"""
import sys, os
sys.path.insert(0, '/home/guanhaochen/iLIF')
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
from data.process_esc_csr import ESC_features
import __main__
__main__.ESC_features = ESC_features
from utils import setup_seed
from sklearn.model_selection import KFold
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer

sys.argv = ['measure', '--model_name', '/home/guanhaochen/iLIF/pretrained/bert-base-uncased',
            '--seed', '209', '--epoch', '5']
from parameter import parse_args
args = parse_args()
args.device = torch.device('cuda')

setup_seed(args.seed)
tokenizer = BertTokenizer.from_pretrained(args.model_name, add_special_tokens=True)
from data import ESC_processor
processor = ESC_processor(args, tokenizer, print)
_, train_dataset = processor.generate_dataloader('train')
_, dev_dataset = processor.generate_dataloader('dev')

splits = KFold(n_splits=args.k_fold, shuffle=True, random_state=args.seed)
train_idx, test_idx = list(splits.split(train_dataset))[0]
train_ds = Subset(train_dataset, train_idx)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=RandomSampler(train_ds),
                          collate_fn=train_dataset.collate_fn)
valid_loader = DataLoader(dev_dataset, batch_size=1, sampler=SequentialSampler(dev_dataset),
                          collate_fn=dev_dataset.collate_fn)

from model import bertCSRModel
model = bertCSRModel(args).to(args.device)

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
opt_group = [
    {'params': [p for n, p in model.pretrained_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.t_lr},
    {'params': [p for n, p in model.pretrained_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.t_lr},
    {'params': [p for n, p in model.mlp.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
    {'params': [p for n, p in model.mlp.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    {'params': [p for n, p in model.CGE.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
    {'params': [p for n, p in model.CGE.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
]
optimizer = AdamW(opt_group, lr=args.lr, eps=args.adam_epsilon)
total_steps = len(train_loader) * args.epoch
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * args.warm_up_ratio),
                                            num_training_steps=total_steps)


def eval_graph(w_val):
    model.eval()
    old_w = model.w
    model.w = w_val
    tot_edges = 0
    tot_pairs = 0
    with torch.no_grad():
        for batch in valid_loader:
            inputs = {'enc_input_ids': batch[0].to(args.device), 'enc_mask_ids': batch[1].to(args.device),
                      'node_event': batch[2], 't1_pos': batch[3], 't2_pos': batch[4], 'target': batch[5],
                      'rel_type': batch[6], 'event_pairs': batch[7]}
            _, opt = model(**inputs)
            tot_edges += model._debug_nedge[0]
            tot_pairs += len(batch[7])
    model.w = old_w
    return tot_edges, tot_pairs


print("epoch  argmax_edges/pairs  w0.6_edges/pairs", flush=True)
for epoch in range(1, args.epoch + 1):
    model.train()
    for batch in train_loader:
        inputs = {'enc_input_ids': batch[0].to(args.device), 'enc_mask_ids': batch[1].to(args.device),
                  'node_event': batch[2], 't1_pos': batch[3], 't2_pos': batch[4], 'target': batch[5],
                  'rel_type': batch[6], 'event_pairs': batch[7]}
        loss, _ = model(**inputs)
        loss = loss / args.gradient_accumulation_steps
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        model.zero_grad()

    e_arg, p_arg = eval_graph(0.0)
    e_w, p_w = eval_graph(0.6)
    print(f"{epoch:5d}  {e_arg:6d}/{p_arg} ({e_arg/p_arg:.3f})   {e_w:6d}/{p_w} ({e_w/p_w:.3f})", flush=True)
