"""
Recall-probe root-cause diagnostic (H3 follow-up).

The H3 probe found NFRA Brain FLAT at the floor (ln 16) for every k — it did
not even fit the training set (dim 224, 600 steps). This script reproduces
that exact regime across a few variants so ONE Kaggle run pins the mechanism:

  fix     : current Brain block (residual = x, not the self-prediction)
            at k=4  — the H3 failing case, post-audit-fix.
  k1      : same model at k=1 (per-token map, ZERO memory needed).
  noshare : 12 distinct blocks in a single pass (no depth weight-sharing)
            at k=4 — isolates "shared-depth gradient" as the culprit.

Every 50 steps it prints train loss + held-out span CE + the block predictor's
deviation from identity (mean |W_p - I|), to watch the collapse develop.

Reading the verdict:
  - `fix` k=4 LEARNS (drops well below floor)     -> self-prediction residual
    was the root cause; the fix is confirmed.
  - `fix` k=4 still floors but `noshare` learns    -> depth weight-sharing is
    the problem (12 distinct blocks needed), not the predictor.
  - `k1` ALSO floors                              -> cannot learn even a per-token
    map at dim 224 -> capacity/optimization -> run the dim-512 probe instead.
  - `fix` k=4 floors AND pred|W-I| grows toward 0 over steps -> the collapse
    develops with training length; residual=x fix wasn't enough on its own.

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
    rescale_embed, compute_loss, make_optimizer, count_params, DEVICE,
)
from nfra.benchmark.recall_probe import make_loader, metric_by_span, V


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


def train(model, k, steps, seq, batch, gap):
    train_loader = make_loader(k, seq, batch, seed=42)
    eval_loader = make_loader(k, seq, batch, seed=7)
    opt, sched = make_optimizer(model, lr=3e-4,
                                warmup=min(50, max(steps // 10, 1)), total=steps)
    it = iter(train_loader)
    t0 = time.perf_counter()
    first = last = None
    rows = []
    for step in range(1, steps + 1):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y = next(it)
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        opt.zero_grad()
        loss = compute_loss(model, x, y, surprise=False)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if math.isfinite(gnorm):
            opt.step()
        else:
            opt.zero_grad(set_to_none=True)
        sched.step()
        if step == 1:
            first = loss.item()
        last = loss.item()
        if step % gap == 0 or step == steps:
            ce_span, acc_span, ce_pad = metric_by_span(model, eval_loader, k)
            rows.append((step, round(last, 4), round(ce_span, 4),
                         round(acc_span, 4), round(predictor_identity_dev(model), 4)))
            print('  step %4d/%d  train %.3f  span_ce %.3f  span_acc %.3f  '
                  'pred|W-I| %.3f'
                  % (step, steps, last, ce_span, acc_span,
                     predictor_identity_dev(model)))
    return {
        'train_first': round(first, 4), 'train_last': round(last, 4),
        'rows': rows, 'params': count_params(model),
        'wall_s': round(time.perf_counter() - t0, 1),
    }


def main():
    dim = int(os.environ.get('NFRA_DIAG_DIM', '224'))
    steps = int(os.environ.get('NFRA_DIAG_STEPS', '600'))
    batch = int(os.environ.get('NFRA_DIAG_BATCH', '8'))
    seq = int(os.environ.get('NFRA_DIAG_SEQ', '256'))
    gap = max(50, steps // 6)
    floor = math.log(V)

    print('=' * 78)
    print('recall root-cause diagnostic  |  dim=%d  steps=%d  seq=%d  batch=%d'
          % (dim, steps, seq, batch))
    print('V=%d  floor ln(V)=%.3f  (flat at floor == model learned NOTHING)'
          % (V, floor))
    print('=' * 78)

    variants = [
        ('fix      k=4  residual=x (current)', dict(k=4, unique=4, depth=12)),
        ('k1       k=1  no memory needed',      dict(k=1, unique=4, depth=12)),
        ('noshare  k=4  12 distinct blocks',    dict(k=4, unique=12, depth=12)),
    ]

    results = {}
    for label, kw in variants:
        print('\n== %s' % label)
        m = build(dim, kw['unique'], kw['depth'])
        rec = train(m, kw['k'], steps, seq, batch, gap)
        results[label] = rec
        print('   train %.3f -> %.3f | %.1fM params | %.0fs'
              % (rec['train_first'], rec['train_last'],
                 rec['params'] / 1e6, rec['wall_s']))

    print('\n' + '=' * 78)
    print('summary  (train last | span_ce | floor %.3f)' % floor)
    for label, rec in results.items():
        print('  %-32s train %s -> %s   span_ce %s'
              % (label, rec['train_first'], rec['train_last'],
                 '? (see rows)' if not rec['rows'] else
                 '%.3f' % rec['rows'][-1][2]))


if __name__ == '__main__':
    main()
