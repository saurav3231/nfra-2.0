"""
Recall-probe root-cause diagnostic (H3 follow-up).

The H3 probe found NFRA Brain FLAT at the floor (ln 16) for every k — it did
not even fit the training set (dim 224, 600 steps). This script reproduces
that exact regime across three NFRA variants, run CONCURRENTLY on separate
CUDA streams (wall time ~= one model's), so ONE run pins the mechanism:

  fix     : current Brain block (residual = x, not the self-prediction)
            at k=4  — the H3 failing case, post-audit-fix.
  k1      : same model at k=1 (per-token map, ZERO memory needed).
  noshare : 12 distinct blocks in a single pass (no depth weight-sharing)
            at k=4 — isolates "shared-depth gradient" as the culprit.

Reports per-variant train first->last, held-out span CE, and the block
predictor's deviation from identity (mean |W_p - I|) — watching the collapse.

Reading the verdict:
  - `fix` k=4 LEARNS (drops well below floor)     -> self-prediction residual
    was the root cause; the fix is confirmed.
  - `fix` k=4 still floors but `noshare` learns    -> depth weight-sharing is
    the problem (12 distinct blocks needed), not the predictor.
  - `k1` ALSO floors                              -> cannot learn even a per-token
    map at dim 224 -> capacity/optimization -> run the dim-512 probe instead.
  - `fix` k=4 floors AND pred|W-I| near 0         -> the collapse developed
    during training; residual=x fix wasn't enough on its own.

Env (all optional):
  NFRA_DIAG_DIM     dim        (default 224)
  NFRA_DIAG_STEPS   steps      (default 600)
  NFRA_DIAG_BATCH   batch      (default 8)
  NFRA_DIAG_SEQ     seq len    (default 256)

Usage: python -m nfra.benchmark.recall_diag     (Kaggle T4 recommended)
"""

import os
import math
import time

os.environ.setdefault('NFRA_SEEDS', '1')
os.environ.setdefault('NFRA_SIZES', '5')

import functools
print = functools.partial(print, flush=True)

import torch

from nfra import NFRAConfig, NFRAForCausalLM
from nfra.benchmark.compare import (
    rescale_embed, count_params, DEVICE,
)
from nfra.benchmark.recall_probe import (
    make_loader, metric_by_span, _train_concurrent, V,
)


def build(dim, unique, depth, seed=0):
    torch.manual_seed(seed)
    cfg = NFRAConfig(mode='brain', vocab_size=V, hidden_size=dim,
                     num_layers=depth, n_bands=16, dropout=0.1,
                     depth_shared=True, unique_blocks=unique,
                     gradient_checkpointing=True)
    m = NFRAForCausalLM(cfg).to(DEVICE)
    rescale_embed(m)
    return m


@torch.no_grad()
def predictor_identity_dev(model):
    eye = torch.eye(model.config.hidden_size, device=DEVICE)
    devs = []
    for lay in model.layers:
        devs.append((lay.predictor.weight.data - eye).abs().mean().item())
    return sum(devs) / max(len(devs), 1)


def run(dim, steps, seq, batch):
    """Build + concurrently train the 3 diagnostic variants, return results."""
    floor = math.log(V)
    torch.manual_seed(0)
    variants = [
        ('fix',     'k=4  residual=x (current)', dict(k=4, unique=4, depth=12)),
        ('k1',      'k=1  no memory needed',      dict(k=1, unique=4, depth=12)),
        ('noshare', 'k=4  12 distinct blocks',    dict(k=4, unique=12, depth=12)),
    ]
    tasks = []
    for sid, label, kw in variants:
        m = build(dim, kw['unique'], kw['depth'])
        train_loader = make_loader(kw['k'], seq, batch, seed=42)
        eval_loader = make_loader(kw['k'], seq, batch, seed=7)
        tasks.append((sid, label, kw['k'], m, train_loader, eval_loader))
    recs, wall = _train_concurrent([t[3] for t in tasks], steps,
                                   [t[4] for t in tasks])
    results = {}
    for (sid, label, k, m, _tr, ev), rec in zip(tasks, recs):
        ce_span, acc_span, ce_pad = metric_by_span(m, ev, k)
        results[sid] = {
            'label': label, 'k': k, 'params': count_params(m),
            'train_first': round(rec['loss_hist'][0], 4),
            'train_last': round(rec['loss_hist'][-1], 4),
            'span_ce': round(ce_span, 4),
            'span_acc': round(acc_span, 4),
            'pad_ce': round(ce_pad, 4),
            'pred_identity_dev': round(predictor_identity_dev(m), 4),
        }
        print('[%-8s] %s train %.4f -> %.4f | span_ce %.4f span_acc %.4f '
              'pad_ce %.4f | pred|W-I| %.4f'
              % (sid, label, results[sid]['train_first'],
                 results[sid]['train_last'], ce_span, acc_span, ce_pad,
                 results[sid]['pred_identity_dev']))
    print('[concurrent] 3 diagnostic trainings in %.1fs' % wall)
    return results, wall, floor


def main():
    dim = int(os.environ.get('NFRA_DIAG_DIM', '224'))
    steps = int(os.environ.get('NFRA_DIAG_STEPS', '600'))
    batch = int(os.environ.get('NFRA_DIAG_BATCH', '8'))
    seq = int(os.environ.get('NFRA_DIAG_SEQ', '256'))
    floor = math.log(V)

    print('=' * 78)
    print('recall root-cause diagnostic  |  dim=%d  steps=%d  seq=%d  batch=%d'
          % (dim, steps, seq, batch))
    print('V=%d  floor ln(V)=%.3f  (flat at floor == model learned NOTHING)'
          % (V, floor))
    print('=' * 78)

    results, wall, floor = run(dim, steps, seq, batch)

    print('\n' + '=' * 78)
    print('summary  (train last | span_ce | floor %.3f)' % floor)
    for label, r in results.items():
        print('  %-32s train %s -> %s   span_ce %s'
              % (label, r['train_first'], r['train_last'], r['span_ce']))
    return results


if __name__ == '__main__':
    main()
