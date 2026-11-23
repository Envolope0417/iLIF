# -*- coding: utf-8 -*-
"""Smoke test for the inference-only annealed threshold.

Checks:
  1. no crash, loss finite, gradients flow;
  2. TRAIN stays argmax (omega trace is all 0.0 == rate);
  3. EVAL omega trace = [0.65, 0.60, 0.55, 0.50, 0.45, 0.45, ...]
     (start 0.65, -0.05 per iteration, floor 0.45).
"""
import sys, os, math

sys.path.insert(0, '/home/guanhaochen/iLIF')

import torch
from torch.utils.data import DataLoader, SequentialSampler

from data.process_esc_csr import ESC_features
import __main__
__main__.ESC_features = ESC_features

from utils import setup_seed


def build_args():
    sys.argv = [
        'smoke_anneal.py',
        '--model_name', '/home/guanhaochen/iLIF/pretrained/bert-base-uncased',
        '--seed', '209',
        '--epoch', '1',
        '--use_infer_anneal',
        '--anneal_start', '0.65',
        '--anneal_step', '0.05',
        '--anneal_floor', '0.45',
    ]
    from parameter import parse_args
    args = parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return args


def get_loader(args):
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_name, add_special_tokens=True)
    from data import ESC_processor
    processor = ESC_processor(args, tokenizer, print)
    _, dataset = processor.generate_dataloader('train')
    loader = DataLoader(dataset, batch_size=1, sampler=SequentialSampler(dataset),
                        collate_fn=dataset.collate_fn)
    return loader


def make_inputs(batch, args):
    return {'enc_input_ids': batch[0].to(args.device),
            'enc_mask_ids': batch[1].to(args.device),
            'node_event': batch[2], 't1_pos': batch[3], 't2_pos': batch[4],
            'target': batch[5], 'rel_type': batch[6], 'event_pairs': batch[7]}


def main():
    args = build_args()
    setup_seed(args.seed)
    loader = get_loader(args)
    from model import bertCSRModel
    model = bertCSRModel(args).to(args.device)
    it = iter(loader)

    # ---- TRAIN: must stay argmax (omega == 0.0) ----
    model.train()
    batch = next(it)
    inputs = make_inputs(batch, args)
    loss, _ = model(**inputs)
    assert math.isfinite(loss.item()), f"non-finite train loss: {loss.item()}"
    loss.backward()
    model.zero_grad()
    omega_train = [round(w, 3) for w in model._debug_omega]
    print(f"[train] loss={loss.item():.4f} omega={omega_train} nedge={model._debug_nedge}")
    assert all(abs(w) < 1e-6 for w in model._debug_omega), \
        f"TRAIN must stay argmax (omega==0.0), got {omega_train}"

    # ---- EVAL: anneal 0.65 -> 0.45 ----
    model.eval()
    with torch.no_grad():
        batch = next(it)
        inputs = make_inputs(batch, args)
        loss, _ = model(**inputs)
        assert math.isfinite(loss.item()), f"non-finite eval loss: {loss.item()}"
    omega_eval = model._debug_omega
    nedge_eval = model._debug_nedge
    print(f"[eval] loss={loss.item():.4f} omega={[round(w, 3) for w in omega_eval]} nedge={nedge_eval}")

    assert omega_eval, "empty eval omega trace"
    assert abs(omega_eval[0] - 0.65) < 1e-6, f"eval omega should start 0.65, got {omega_eval[0]}"
    # monotonic non-increasing
    assert all(omega_eval[i] <= omega_eval[i - 1] + 1e-9 for i in range(1, len(omega_eval))), \
        f"eval omega not non-increasing: {omega_eval}"
    # step of -0.05 until floor 0.45
    for i in range(1, len(omega_eval)):
        prev = omega_eval[i - 1]
        cur = omega_eval[i]
        if prev > 0.45 + 1e-9:
            assert abs(prev - cur - 0.05) < 1e-6, \
                f"expected -0.05 step at iter {i}: {prev} -> {cur}"
        else:
            assert abs(cur - 0.45) < 1e-6, f"floor should hold at 0.45: {cur}"
    assert all(w >= 0.45 - 1e-9 for w in omega_eval), f"eval omega below floor: {omega_eval}"

    print("=== ANNEAL SMOKE TEST PASSED ===")


if __name__ == '__main__':
    main()
