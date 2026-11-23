# -*- coding: utf-8 -*-
"""Smoke test for the adaptive-threshold modification of iLIF.

Runs a few forward+backward passes (train + eval) under three configs:
  baseline (rate=0 train / w=0.6 eval),
  dynamic-only,
  dynamic+persistence.

Checks:
  1. no crash, loss finite, gradients flow;
  2. omega trace starts at w_max=0.70 and stays within [w_min, w_max];
  3. train-time graph actually uses the threshold (iteration-0 edge count <= argmax).
"""
import sys, os, math

sys.path.insert(0, '/home/guanhaochen/iLIF')

import torch
from torch.utils.data import DataLoader, SequentialSampler

from data.process_esc_csr import ESC_features
import __main__
__main__.ESC_features = ESC_features

from utils import setup_seed


def build_args(dynamic, persistence):
    sys.argv = [
        'smoke_test.py',
        '--model_name', '/home/guanhaochen/iLIF/pretrained/bert-base-uncased',
        '--seed', '209',
        '--epoch', '1',
    ]
    if dynamic:
        sys.argv.append('--use_dynamic')
    if persistence:
        sys.argv.append('--use_persistence')
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


def run_one(args, tag, n_batches=3):
    from model import bertCSRModel
    setup_seed(args.seed)
    loader = get_loader(args)
    model = bertCSRModel(args).to(args.device)
    model.train()
    it = iter(loader)
    first_nedge = None
    for i in range(n_batches):
        batch = next(it)
        inputs = make_inputs(batch, args)
        loss, opt = model(**inputs)
        assert math.isfinite(loss.item()), f"non-finite loss in {tag} batch {i}: {loss.item()}"
        loss.backward()
        model.zero_grad()
        if i == 0:
            first_nedge = model._debug_nedge[0]
        print(f"[{tag}] batch {i}: loss={loss.item():.4f} "
              f"omega={[round(w, 3) for w in model._debug_omega]} "
              f"nedge={model._debug_nedge}")

    # eval mode forward
    model.eval()
    with torch.no_grad():
        batch = next(it)
        inputs = make_inputs(batch, args)
        loss, opt = model(**inputs)
        assert math.isfinite(loss.item()), f"non-finite loss in {tag} eval: {loss.item()}"
    print(f"[{tag}] eval: loss={loss.item():.4f} "
          f"omega={[round(w, 3) for w in model._debug_omega]} nedge={model._debug_nedge}")

    # omega-range checks (dynamic configs only)
    if model.use_dynamic:
        w = model._debug_omega
        assert w, f"{tag}: empty omega trace"
        assert abs(w[0] - 0.70) < 1e-6, f"{tag}: omega should start at 0.70, got {w[0]}"
        assert all(model.w_min - 1e-6 <= x <= model.w_max + 1e-6 for x in w), \
            f"{tag}: omega out of [{model.w_min},{model.w_max}]: {w}"
        assert all(w[i] <= w[i - 1] + 1e-9 for i in range(1, len(w))), \
            f"{tag}: omega not monotonically non-increasing: {w}"
    return model, first_nedge


if __name__ == '__main__':
    print("=== smoke: baseline (use_dynamic=False) ===")
    _, nedge_base = run_one(build_args(False, False), 'baseline')
    print("=== smoke: dynamic-only ===")
    _, nedge_dyn = run_one(build_args(True, False), 'dynamic')
    print("=== smoke: dynamic+persistence ===")
    _, nedge_full = run_one(build_args(True, True), 'full')

    # iteration-0 edge count under w_max=0.70 must not exceed the argmax baseline
    assert nedge_dyn <= nedge_base, f"threshold should not add edges vs argmax: dyn={nedge_dyn} base={nedge_base}"
    print(f"[check] iter0 nedge: baseline(argmax)={nedge_base}, dynamic(w=0.70)={nedge_dyn}, full={nedge_full}")
    print("=== ALL SMOKE TESTS PASSED ===")
