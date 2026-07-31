"""H3 — memory-horizon probe (FUTURE_PLAN Part 2).

Diagnostic that tells us WHICH lever to bet on:

  * If NFRA recall collapses as the key->query distance grows, the loss gap
    to Mamba is a MEMORY problem  -> bet on AFC-alpha (adaptive band decay).
  * If recall is flat across distance but loss lags anyway, the gap is a
    CAPACITY problem             -> bet on AFC-LoRA / MoE / band-drop.

Method: associative-recall sequences. Token at position t is the value of the
key that appeared k steps earlier (value = key+1 mod V). Predicting position t
therefore requires (a) retrieving the key from t-k and (b) applying the
key->value map. The first k positions are unlearnable padding (floor = ln V).

We train NFRA Brain and Mamba on fixed-k datasets and report per-k loss and
accuracy on the span positions (t >= k). A rising curve in k = collapsing
recall (memory); a flat high curve = weak extraction (capacity).

Usage:
  python -m nfra.benchmark.recall_probe

Env:
  NFRA_RECALL_KS      comma list of spans      (default 4,16,64,128)
  NFRA_RECALL_STEPS   train steps per (k, model) (default 400)
  NFRA_RECALL_DIM     model width              (default 128)
  NFRA_RECALL_DEPTH   NFRA depth               (default 12)
  NFRA_RECALL_UNIQUE  unique fractal blocks    (default 2)
  NFRA_RECALL_BATCH   batch size               (default 8)
  NFRA_RECALL_SEQ     sequence length          (default 256)
"""

import os
import math
import json
import time
import contextlib

os.environ.setdefault('NFRA_SEEDS', '1')
os.environ.setdefault('NFRA_SIZES', '5')

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .arena import build_nfra, build_mamba, train_one, SEED_LIST
from .compare import (
    count_params, DEVICE, rescale_embed, make_optimizer, compute_loss,
    USE_AMP, HAS_CUDA, SEQ_LEN,
)

V = 16  # symbol vocab


class RecallDataset(Dataset):
    """Token at t = value(key_{t-k}); span positions require k-step recall."""

    def __init__(self, num_seqs, seq_len, k, seed=0):
        super().__init__()
        self.seq_len = seq_len
        rng = np.random.RandomState(seed)
        keys = rng.randint(0, V, size=(num_seqs, seq_len))
        values = (keys + 1) % V
        toks = np.empty((num_seqs, seq_len), dtype=np.int64)
        for t in range(seq_len):
            src = t - k
            if src >= 0:
                toks[:, t] = values[:, src]
            else:
                toks[:, t] = rng.randint(0, V, size=num_seqs)
        self.toks = toks

    def __len__(self):
        return self.toks.shape[0]

    def __getitem__(self, idx):
        x = self.toks[idx, :-1]
        y = self.toks[idx, 1:]
        return (torch.from_numpy(x), torch.from_numpy(y))


def make_loader(k, seq_len, batch, seed=0, num_seqs=512):
    ds = RecallDataset(num_seqs, seq_len + 1, k, seed=seed)
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0,
                      pin_memory=HAS_CUDA)


@torch.no_grad()
def metric_by_span(model, loader, k, V=V):
    """CE + accuracy on span positions (t>=k) vs padding floor (t<k)."""
    model.eval()
    ce_span, acc_span, ce_pad = [], [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)['logits']
        logp = logits.log_softmax(-1)
        pred = logits.argmax(-1)
        for b in range(x.size(0)):
            yy = y[b]
            span = torch.arange(yy.numel(), device=DEVICE) >= k
            ce = -logp[b].gather(-1, yy.unsqueeze(-1)).squeeze(-1)
            ce_span.append(ce[span].mean().item())
            ce_pad.append(ce[~span].mean().item())
            acc_span.append((pred[b][span] == yy[span]).float().mean().item())
    model.train()
    return (float(np.mean(ce_span)), float(np.mean(acc_span)),
            float(np.mean(ce_pad)))


def _train_concurrent(models, steps, loaders):
    """Train several models in parallel, one CUDA stream each.

    A 5M model's step is a stream of ~1000 tiny kernels with launch gaps the
    GPU idles through. Running several models on separate streams fills those
    gaps with each other's work, so aggregate GPU utilization approaches 100%
    and the whole batch of tiny trainings finishes in ~one model's wall time.
    Each model keeps its own optimizer/scheduler/loader; results are the same
    as running them sequentially (init RNG is seeded to match, see _run_all)."""
    n = len(models)
    streams = ([torch.cuda.Stream() for _ in range(n)] if HAS_CUDA
               else [None] * n)
    opts, scheds, scalers = [], [], []
    for m in models:
        opt, sched = make_optimizer(m, lr=3e-4,
                                    warmup=min(50, max(steps // 10, 1)),
                                    total=steps)
        opts.append(opt)
        scheds.append(sched)
        scalers.append(torch.amp.GradScaler(str(DEVICE)) if USE_AMP else None)
    iters = [iter(l) for l in loaders]
    hist = [[] for _ in range(n)]
    nan = [0] * n
    nullctx = contextlib.nullcontext()
    t0 = time.perf_counter()
    for _ in range(steps):
        for i in range(n):
            with (torch.cuda.stream(streams[i]) if HAS_CUDA else nullctx):
                try:
                    x, y = next(iters[i])
                except StopIteration:
                    iters[i] = iter(loaders[i])
                    x, y = next(iters[i])
                x = x.to(DEVICE, non_blocking=HAS_CUDA)
                y = y.to(DEVICE, non_blocking=HAS_CUDA)
                opts[i].zero_grad()
                with torch.amp.autocast(device_type=DEVICE.type,
                                        enabled=USE_AMP):
                    loss = compute_loss(models[i], x, y, surprise=False)
                if scalers[i]:
                    scalers[i].scale(loss).backward()
                    scalers[i].unscale_(opts[i])
                    gnorm = torch.nn.utils.clip_grad_norm_(
                        models[i].parameters(), 1.0)
                    if not math.isfinite(gnorm):
                        opts[i].zero_grad(set_to_none=True)
                        nan[i] += 1
                    scalers[i].step(opts[i])
                    scalers[i].update()
                else:
                    loss.backward()
                    gnorm = torch.nn.utils.clip_grad_norm_(
                        models[i].parameters(), 1.0)
                    if math.isfinite(gnorm):
                        opts[i].step()
                    else:
                        opts[i].zero_grad(set_to_none=True)
                        nan[i] += 1
                scheds[i].step()
                hist[i].append(loss.detach())
    if HAS_CUDA:
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    recs = []
    for i in range(n):
        bs = getattr(loaders[i], 'batch_size', 1)
        recs.append({
            'loss_hist': [float(v) for v in hist[i]],
            'nan_steps': nan[i],
            'wall_s': wall,
            'ms_per_step': wall * 1000.0 / steps,
            'tok_s': bs * SEQ_LEN * steps / max(wall, 1e-6),
        })
    return recs, wall


def _run_all(ks, steps, dim, seq_len, batch, unique, depth, concurrent=False):
    """Run NFRA + Mamba across all k, sequentially or concurrently.

    Returns {'nfra': {k: row, ...}, 'mamba': {k: row, ...}}. In concurrent
    mode models are BUILT in the same order sequential runs would build them
    (nfra for every k, then mamba for every k) after one torch.manual_seed(0),
    so parameter init matches the sequential run exactly."""
    def nfra_b(v, d):
        return build_nfra(v, d, unique, depth=depth)

    def mamba_b(v, d):
        return build_mamba(v, d, 8)

    rows = {'nfra': {}, 'mamba': {}}
    if concurrent:
        torch.manual_seed(0)
        tasks = []
        for fam, builder in [('nfra', nfra_b), ('mamba', mamba_b)]:
            for k in ks:
                train_loader = make_loader(k, seq_len, batch, seed=42)
                eval_loader = make_loader(k, seq_len, batch, seed=7)
                m = builder(V, dim).to(DEVICE)
                rescale_embed(m)
                tasks.append((fam, k, m, train_loader, eval_loader))
        recs, wall = _train_concurrent([t[2] for t in tasks], steps,
                                       [t[3] for t in tasks])
        agg_tok_s = (sum(getattr(t[3], 'batch_size', 1) for t in tasks)
                     * SEQ_LEN * steps / max(wall, 1e-6))
        print('[concurrent] %d trainings in %.1fs -> %.0f tok/s aggregate'
              % (len(tasks), wall, agg_tok_s))
        for (fam, k, m, _tr, ev), rec in zip(tasks, recs):
            ce_span, acc_span, ce_pad = metric_by_span(m, ev, k)
            rows[fam][k] = {
                'span_ce': round(ce_span, 4),
                'span_acc': round(acc_span, 4),
                'pad_ce': round(ce_pad, 4),
                'train_first': round(rec['loss_hist'][0], 4),
                'train_last': round(rec['loss_hist'][-1], 4),
                'params': count_params(m),
            }
            print('[%s] k=%-4d train %.4f -> %.4f | span_ce=%.4f span_acc=%.4f '
                  'pad_ce=%.4f (floor %.2f)'
                  % (fam, k, rows[fam][k]['train_first'],
                     rows[fam][k]['train_last'], ce_span, acc_span, ce_pad,
                     math.log(V)))
    else:
        rows['nfra'] = _run_model('nfra', nfra_b, ks, steps, dim, seq_len,
                                  batch)
        rows['mamba'] = _run_model('mamba', mamba_b, ks, steps, dim, seq_len,
                                   batch)
    return rows


def _run_model(name, builder, ks, steps, dim, seq_len, batch):
    torch.manual_seed(0)
    rows = {}
    for k in ks:
        train_loader = make_loader(k, seq_len, batch, seed=42)
        eval_loader = make_loader(k, seq_len, batch, seed=7)  # held-out seqs
        model = builder(V, dim).to(DEVICE)
        rescale_embed(model)
        params = count_params(model)
        rec = train_one(model, V, steps, train_loader, eval_loader,
                        max(25, steps // 6), ema_decay=0.0, surprise=False)
        ce_span, acc_span, ce_pad = metric_by_span(model, eval_loader, k)
        rows[k] = {
            'span_ce': round(ce_span, 4),
            'span_acc': round(acc_span, 4),
            'pad_ce': round(ce_pad, 4),
            'train_first': round(rec['loss_hist'][0], 4),
            'train_last': round(rec['loss_hist'][-1], 4),
            'params': params,
        }
        print('[%s] k=%-4d train %.4f -> %.4f | span_ce=%.4f span_acc=%.4f '
              'pad_ce=%.4f (floor %.2f)'
              % (name, k, rows[k]['train_first'], rows[k]['train_last'],
                 ce_span, acc_span, ce_pad, math.log(V)))
    return rows


def main():
    ks = [int(x) for x in
          os.environ.get('NFRA_RECALL_KS', '4,16,64,128').split(',') if x.strip()]
    steps = int(os.environ.get('NFRA_RECALL_STEPS', '600'))
    dim = int(os.environ.get('NFRA_RECALL_DIM', '224'))
    depth = int(os.environ.get('NFRA_RECALL_DEPTH', '12'))
    unique = int(os.environ.get('NFRA_RECALL_UNIQUE', '4'))
    batch = int(os.environ.get('NFRA_RECALL_BATCH', '8'))
    seq_len = int(os.environ.get('NFRA_RECALL_SEQ', '256'))
    concurrent = os.environ.get('NFRA_RECALL_CONCURRENT', '0') == '1'
    out_json = os.environ.get('NFRA_RECALL_OUT', 'recall_probe.json')

    print('H3 recall probe  |  V=%d ks=%s steps=%d dim=%d seq=%d%s'
          % (V, ks, steps, dim, seq_len,
             '  [concurrent streams]' if concurrent else ''))
    print('random loss floor ln(%d) = %.3f' % (V, math.log(V)))

    rows = _run_all(ks, steps, dim, seq_len, batch, unique, depth,
                    concurrent=concurrent)
    nfra_rows = rows['nfra']
    mamba_rows = rows['mamba']

    print('\nresults (span CE, lower is better; floor %.3f):' % math.log(V))
    print(' k      NFRA      Mamba     | verdict')
    for k in ks:
        a = nfra_rows[k]['span_ce']
        b = mamba_rows[k]['span_ce']
        verdict = ('nfra worse' if a - b > 0.1 else
                   'nfra better' if b - a > 0.1 else 'wash')
        print(' %-4d   %-8.4f  %-8.4f  | %s' % (k, a, b, verdict))

    out = {'vocab': V, 'ks': ks, 'steps': steps, 'dim': dim,
           'floor': round(math.log(V), 4), 'nfra': nfra_rows,
           'mamba': mamba_rows}
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=True, indent=2)
    print('Wrote %s' % out_json)


if __name__ == '__main__':
    main()
