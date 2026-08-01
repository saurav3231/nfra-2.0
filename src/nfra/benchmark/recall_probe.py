"""H3 — memory-horizon probe (FUTURE_PLAN Part 2).

Diagnostic that tells us WHICH lever to bet on:

  * If NFRA recall collapses as the key->query distance grows, the loss gap
    to Mamba is a MEMORY problem  -> bet on AFC-alpha (adaptive band decay).
  * If recall is flat across distance but loss lags anyway, the gap is a
    CAPACITY problem             -> bet on AFC-LoRA / MoE / band-drop.

Method: associative-recall sequences with OBSERVABLE keys. The observed
stream IS the keys; the target at position t is the value (key+1 mod V) of
the key that appeared k positions earlier. Predicting position t therefore
requires (a) retrieving the key from t-k (always visible in the prefix) and
(b) applying the key->value map. The first k positions are unlearnable
padding (floor = ln V). k=1 is a memory-free per-token map (baseline).

We train NFRA Brain and Mamba on fixed-k datasets and report per-k loss and
accuracy on the span positions (t >= k). A rising curve in k = collapsing
recall (memory); a flat high curve = weak extraction (capacity).

Usage:
  python -m nfra.benchmark.recall_probe

Env:
  NFRA_RECALL_KS         comma list of spans      (default 4,16,64,128)
  NFRA_RECALL_STEPS      train steps per (k, model) (default 600)
  NFRA_RECALL_DIM        model width              (default 224)
  NFRA_RECALL_DEPTH      NFRA depth               (default 12)
  NFRA_RECALL_UNIQUE     unique fractal blocks    (default 4)
  NFRA_RECALL_BATCH      batch size               (default 8)
  NFRA_RECALL_SEQ        sequence length          (default 256)
  NFRA_RECALL_CONCURRENT 1 = train all configs on separate CUDA streams
  NFRA_RECALL_FAMILIES   comma list: nfra,mamba,rwkv,retnet,gpt2 (default nfra,mamba)
"""

import os
import math
import json
import time
import contextlib
import functools

os.environ.setdefault('NFRA_SEEDS', '1')
os.environ.setdefault('NFRA_SIZES', '5')

print = functools.partial(print, flush=True)

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .arena import build_nfra, build_mamba, build_rwkv, build_retnet, \
    build_gpt2, train_one, SEED_LIST
from .compare import (
    count_params, DEVICE, rescale_embed, make_optimizer, compute_loss,
    USE_AMP, HAS_CUDA, SEQ_LEN,
)

V = 16  # symbol vocab


class RecallDataset(Dataset):
    """Associative recall with OBSERVABLE keys — a real memory test.

    The observed stream IS the key stream (every key is visible in the input),
    and the target at output position t is the VALUE (key+1 mod V) of the key
    k positions back:

        x[t]  = keys[t]                       (visible)
        y[t]  = (keys[t-k] + 1) % V   for t>=k      else random (padding)

    Predicting y[t] therefore requires recalling keys[t-k] from the visible
    prefix. k=1 is a memory-free per-token map (trivial baseline); k=4 is a
    4-step recall. A model that learns the rule generalizes to the eval loader
    (different keys); one that can't do k-step recall floors on the span.

    NOTE: the earlier version placed only VALUES (keys+1) in the stream, so
    the keys were never observable — H(y|prefix) = ln(V) for every position
    and NO model (causal or not) could learn the span. Both NFRA and Mamba
    "floored" for that reason (dim-224 and dim-512 runs). The keys must be
    in the stream for the probe to measure memory at all."""

    def __init__(self, num_seqs, seq_len, k, seed=0):
        super().__init__()
        self.seq_len = seq_len
        self.k = k
        rng = np.random.RandomState(seed)
        keys = rng.randint(0, V, size=(num_seqs, seq_len))
        vals = (keys + 1) % V
        self.toks = keys.copy()
        targets = np.empty((num_seqs, seq_len - 1), dtype=np.int64)
        for t in range(seq_len - 1):
            src = t - k
            if src >= 0:
                targets[:, t] = vals[:, src]
            else:
                targets[:, t] = rng.randint(0, V, size=num_seqs)
        self.targets = targets

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.toks[idx, :-1])
        y = torch.from_numpy(self.targets[idx])
        return x, y


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


def _train_concurrent(models, steps, loaders, seq_len=256):
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
    prog = max(50, steps // 12)
    for step in range(1, steps + 1):
        if step % prog == 0 or step == steps:
            print('[concurrent] step %d/%d  %.0fs elapsed'
                  % (step, steps, time.perf_counter() - t0))
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
            'tok_s': bs * seq_len * steps / max(wall, 1e-6),
        })
    return recs, wall


def _run_all(ks, steps, dim, seq_len, batch, unique, depth, concurrent=False,
             families=('nfra', 'mamba')):
    """Run the given families across all k, sequentially or concurrently.

    Returns {fam: {k: row, ...}} for each family in `families`. In BOTH modes
    every model is BUILT under a single torch.manual_seed(0) in the same
    ((family, k) for every k then family) order, so parameter init matches
    across sequential and concurrent runs exactly. (Dropout order during
    training still differs between the two modes — that is honest
    stochasticity, not an init mismatch.)"""
    builders = {
        'nfra': lambda v, d: build_nfra(v, d, unique, depth=depth),
        'mamba': lambda v, d: build_mamba(v, d, 8),
        'rwkv': lambda v, d: build_rwkv(v, d, 8),
        'retnet': lambda v, d: build_retnet(v, d, 8, n_heads=8),
        'gpt2': lambda v, d: build_gpt2(v, d, 8, n_heads=8),
    }
    fams = [f for f in families if f in builders]
    rows = {f: {} for f in fams}
    if concurrent:
        torch.manual_seed(0)
        tasks = []
        for fam in fams:
            builder = builders[fam]
            for k in ks:
                train_loader = make_loader(k, seq_len, batch, seed=42)
                eval_loader = make_loader(k, seq_len, batch, seed=7)
                m = builder(V, dim).to(DEVICE)
                rescale_embed(m)
                tasks.append((fam, k, m, train_loader, eval_loader))
        recs, wall = _train_concurrent([t[2] for t in tasks], steps,
                                       [t[3] for t in tasks], seq_len)
        agg_tok_s = (sum(getattr(t[3], 'batch_size', 1) for t in tasks)
                     * seq_len * steps / max(wall, 1e-6))
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
        # Build ALL models first under one RNG stream, in the same (family, k)
        # order concurrent mode builds them, so init matches exactly.
        torch.manual_seed(0)
        models = []
        for fam in fams:
            builder = builders[fam]
            for k in ks:
                models.append((fam, k, builder(V, dim).to(DEVICE)))
        for fam, k, model in models:
            rescale_embed(model)
            params = count_params(model)
            train_loader = make_loader(k, seq_len, batch, seed=42)
            eval_loader = make_loader(k, seq_len, batch, seed=7)
            rec = train_one(model, V, steps, train_loader, eval_loader,
                            max(25, steps // 6), ema_decay=0.0, surprise=False)
            ce_span, acc_span, ce_pad = metric_by_span(model, eval_loader, k)
            rows[fam][k] = {
                'span_ce': round(ce_span, 4),
                'span_acc': round(acc_span, 4),
                'pad_ce': round(ce_pad, 4),
                'train_first': round(rec['loss_hist'][0], 4),
                'train_last': round(rec['loss_hist'][-1], 4),
                'params': params,
            }
            print('[%s] k=%-4d train %.4f -> %.4f | span_ce=%.4f span_acc=%.4f '
                  'pad_ce=%.4f (floor %.2f)'
                  % (fam, k, rows[fam][k]['train_first'],
                     rows[fam][k]['train_last'], ce_span, acc_span, ce_pad,
                     math.log(V)))
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
    families = tuple(f.strip().lower() for f in
                     os.environ.get('NFRA_RECALL_FAMILIES', 'nfra,mamba').split(',')
                     if f.strip())

    print('H3 recall probe  |  V=%d ks=%s steps=%d dim=%d seq=%d fam=%s%s'
          % (V, ks, steps, dim, seq_len, ','.join(families),
             '  [concurrent streams]' if concurrent else ''))
    print('random loss floor ln(%d) = %.3f' % (V, math.log(V)))

    rows = _run_all(ks, steps, dim, seq_len, batch, unique, depth,
                    concurrent=concurrent, families=families)

    print('\nresults (span CE, lower is better; floor %.3f):' % math.log(V))
    hdr = ' k     ' + '  '.join('%-8s' % f for f in families)
    print(hdr)
    for k in ks:
        cells = '  '.join('%-8.4f' % rows[f][k]['span_ce'] for f in families)
        print(' %-4d  %s' % (k, cells))

    out = {'vocab': V, 'ks': ks, 'steps': steps, 'dim': dim,
           'floor': round(math.log(V), 4), 'families': families}
    for f in families:
        out[f] = rows[f]
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=True, indent=2)
    print('Wrote %s' % out_json)


if __name__ == '__main__':
    main()
