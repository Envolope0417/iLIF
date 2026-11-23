# -*- coding: utf-8 -*-

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='ECI')

    # dataset
    parser.add_argument("--dataset_type", default='EventStoryLine', type=str, help='dataset name')
    parser.add_argument("--cache_path", default='./data/database', type=str)
    parser.add_argument("--model_type", default='bert', type=str)
    parser.add_argument("--inter_or_intra", default='intra_and_inter', type=str)
    parser.add_argument("--k_fold", default=5, type=int)
    parser.add_argument('--fold_idx', default=-1, type=int, help='run only this fold (0-indexed); -1 = all folds')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--use_cache', default=True, action="store_true")
    parser.add_argument('--in_channels', default={'event': 768}, type=dict)
    parser.add_argument('--metadata', default=(['event'], [('event','intra','event'),('event','inter','event')]), type=tuple)
    parser.add_argument('--beta_inter', default=0.3, type=float)
    parser.add_argument('--beta_intra', default=0.7, type=float)

    # # model arguments
    parser.add_argument('--model_name', default='bert-base-uncased', type=str, help='Log model name')
    parser.add_argument('--mlp_size', default=2304, type=int, help='mlp layer_size')
    parser.add_argument('--mlp_drop', default=0.4, type=float, help='mlp dropout layer')
    parser.add_argument('--n_last', default=768, type=int, help='CGE out layer')
    parser.add_argument('--threshold', default=2, type=int, help='structural difference threshold δ')
    parser.add_argument('--w', default=0.6, type=float, help='The relation confidence threshold ω')

    # # loss arguments
    parser.add_argument('--gamma', default=2.0, type=float, help='loss gamma argument')
    parser.add_argument('--no_of_classes', default=3, type=int, help='loss classes argument')
    parser.add_argument('--loss_type', default="focal", type=str, help='loss type')
    parser.add_argument('--max_iteration', default=9, type=int, help='Maximum iteration number L')
    parser.add_argument('--min_iteration', default=2, type=int, help='Minimum iteration number')
    parser.add_argument('--num_heads', default=4, type=int, help='The number of attention head K')

    # # training arguments
    parser.add_argument('--seed', default=209, type=int, help='seed for reproducibility')
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument('--epoch', default=30, type=int, help='number of total epochs to run')
    parser.add_argument('--t_lr', default=2e-5, type=float, help='initial transformer learning rate')
    parser.add_argument('--lr', default=1e-4, type=float, help='initial other module learning rate')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument('--rate', default=0.0, type=float)

    # --- adaptive confidence threshold (iLIF improvement) ---
    parser.add_argument('--use_dynamic', action='store_true', default=False,
                        help='enable structure-difference-driven adaptive threshold')
    parser.add_argument('--use_persistence', action='store_true', default=False,
                        help='enable temporal-persistence edge-level threshold discount')
    parser.add_argument('--w_max', default=0.70, type=float, help='initial (max) confidence threshold omega_max')
    parser.add_argument('--w_min', default=0.40, type=float, help='minimum confidence threshold omega_min')
    parser.add_argument('--ema_alpha', default=0.5, type=float, help='EMA smoothing for structural difference')
    parser.add_argument('--persist_lambda', default=0.05, type=float, help='per-iteration persistence discount')
    parser.add_argument('--persist_base', default=0.5, type=float, help='confidence floor to accumulate persistence')
    parser.add_argument('--w_floor', default=0.35, type=float, help='absolute edge-level threshold floor')

    # --- inference-only annealed threshold ---
    parser.add_argument('--use_infer_anneal', action='store_true', default=False,
                        help='anneal the confidence threshold during inference only (training stays argmax)')
    parser.add_argument('--anneal_start', default=0.65, type=float, help='inference threshold at iteration 0')
    parser.add_argument('--anneal_step', default=0.05, type=float, help='per-iteration threshold step (magnitude)')
    parser.add_argument('--anneal_floor', default=0.45, type=float, help='inference threshold floor (for direction=down)')
    parser.add_argument('--anneal_direction', default='down', type=str, choices=['down', 'up'],
                        help='anneal direction: down (strict->loose) or up (loose->strict)')
    parser.add_argument('--anneal_ceiling', default=0.65, type=float, help='inference threshold ceiling (for direction=up)')
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--warm_up_ratio", default=0.1, type=float)
    parser.add_argument("--class_weight", default=0.75, type=float)
    parser.add_argument('--logging_steps', default=50, type=int)

    parser.add_argument('--valid_epoch', default=1, type=int)
    parser.add_argument('--test_epoch', default=1, type=int)

    parser.add_argument('--log', default='', type=str, help='Log result file name')

    args = parser.parse_args()
    return args
