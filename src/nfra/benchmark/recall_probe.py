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

os.environ.setdefault('NFRA_SEEDS', '1')
os.environ.setdefault('NFRA_SIZES', '5')

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .arena import build_nfra, build_mamba, train_one, SEED_LIST
from .compare import count_params, DEVICE, rescale_embed

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
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)


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


def _run_model(name, builder, ks, steps, dim, seq_len, batch):
    torch.manual_seed(0)
    rows = {}
    for k in ks:
        loader = make_loader(k, seq_len, batch, seed=42)
        model = builder(V, dim).to(DEVICE)
        rescale_embed(model)
        params = count_params(model)
        train_one(model, V, steps, loader, loader, max(25, steps // 6),
                  ema_decay=0.0, surprise=False)
        ce_span, acc_span, ce_pad = metric_by_span(model, loader, k)
        rows[k] = {
            'span_ce': round(ce_span, 4),
            'span_acc': round(acc_span, 4),
            'pad_ce': round(ce_pad, 4),
            'params': params,
        }
        print('[%s] k=%-4d span_ce=%.4f span_acc=%.4f pad_ce=%.4f (floor %.2f)'
              % (name, k, ce_span, acc_span, ce_pad, math.log(V)))
    return rows


def main():
    ks = [int(x) for x in
          os.environ.get('NFRA_RECALL_KS', '4,16,64,128').split(',') if x.strip()]
    steps = int(os.environ.get('NFRA_RECALL_STEPS', '400'))
    dim = int(os.environ.get('NFRA_RECALL_DIM', '128'))
    depth = int(os.environ.get('NFRA_RECALL_DEPTH', '12'))
    unique = int(os.environ.get('NFRA_RECALL_UNIQUE', '2'))
    batch = int(os.environ.get('NFRA_RECALL_BATCH', '8'))
    seq_len = int(os.environ.get('NFRA_RECALL_SEQ', '256'))
    out_json = os.environ.get('NFRA_RECALL_OUT', 'recall_probe.json')

    print('H3 recall probe  |  V=%d ks=%s steps=%d dim=%d seq=%d'
          % (V, ks, steps, dim, seq_len))
    print('random loss floor ln(%d) = %.3f' % (V, math.log(V)))

    nfra_rows = _run_model(
        'nfra', lambda v, d: build_nfra(v, d, unique, depth=depth),
        ks, steps, dim, seq_len, batch)
    mamba_rows = _run_model(
        'mamba', lambda v, d: build_mamba(v, d, 8),
        ks, steps, dim, seq_len, batch)

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
