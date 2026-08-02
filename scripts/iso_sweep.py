"""
ISOLATION SWEEP (FUTURE_PLAN Part 11) -- attribute the 3.3b quality win.

For each 3.3b mechanism we turn ONLY that mechanism OFF (everything else stays
the exact verified architecture) and train at the SAME 5M/600-step protocol as
the verified board. If disabling a mechanism costs >= 0.02 eval loss, that
mechanism is carrying part of the win and must stay. If it changes nothing
(within run-to-run noise ~0.005), the mechanism is decoration and gets pruned.

Configs (all others keep the default verified settings):
  baseline  - nothing disabled (reproduces nfra@5M ~1.961/1.945)
  gland     - neuromodulator removed (no ACh/NE modulation anywhere)
  vgate     - input-dependent value gate removed (pure retention, no gate)
  rgate     - output receptance gate removed
  phase     - resonance phase modulation removed
  exit      - adaptive-compute exit gate removed (plain full-depth forward)

Env to match the verified board exactly:
  NFRA_ISO=<names>  sets which mechanisms are OFF (empty = baseline)
  Usage (Kaggle):   python scripts/iso_sweep.py
"""

import os
import sys
import time
import functools
import warnings

print = functools.partial(print, flush=True)
warnings.filterwarnings('ignore')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Environment: bit-identical to the verified overnight board ──────────────
os.environ['NFRA_MODE'] = 'standard'
os.environ['NFRA_STEPS'] = '600'
os.environ['NFRA_SIZES'] = '5'
os.environ['NFRA_SEEDS'] = '1'
os.environ['NFRA_DATA'] = 'wikitext2'
os.environ['NFRA_EMA'] = '0.99'
# Eager (no torch.compile): fresh Kaggle images carry a torch whose Inductor
# crashes in codegen on the neuromodulator's cumsum/arange division
# (TypeError in TritonSymbols.get_block_shape) -- dies on FIRST forward, which
# train_one's compile fallback cannot catch. torch.compile is exact-math, so
# eager gives the SAME loss as the compiled board; only tok/s is lower (still
# comparable across configs, which is what the cost side of the rule needs).
os.environ['NFRA_COMPILE'] = '0'
os.environ['NFRA_SCAN_KERNEL'] = '0'
os.environ['NFRA_CHECKPOINT'] = '0'
os.environ['NFRA_CORTEX'] = '1'

# WikiText-2 MUST exist BEFORE compare/arena are imported: DATA_SOURCE is decided
# at import time from the files' presence, and a missing dataset silently falls
# back to synthetic (different loader shape/quality than the verified board).
# Mirror overnight.ensure_wikitext (streaming, mirror fallback, hard timeout) so
# the sweep is self-sufficient on a fresh clone.
_WIKI_FILES = ['wikitext-train-raw-v1.txt', 'wikitext-valid-raw-v1.txt']
_WIKI_URLS = [
    'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip',
    'https://huggingface.co/datasets/ggml-org/ci/resolve/927b3642933080f1b0e811e2f916e14c292992f9/wikitext-2-raw-v1.zip',
]


def _ensure_wikitext():
    import urllib.request
    import zipfile
    if all(os.path.exists(f) for f in _WIKI_FILES):
        return
    print('  [iso] WikiText-2 missing, downloading...')
    tmp = 'wikitext-2-raw-v1.zip'
    for url in _WIKI_URLS:
        try:
            with urllib.request.urlopen(url, timeout=120) as resp, \
                    open(tmp, 'wb') as out:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            if os.path.getsize(tmp) < 1024 * 1024:
                raise ValueError('download too small')
            with zipfile.ZipFile(tmp) as z:
                for src, dst in [('wikitext-2-raw/wiki.train.raw', _WIKI_FILES[0]),
                                 ('wikitext-2-raw/wiki.valid.raw', _WIKI_FILES[1])]:
                    with z.open(src) as f, open(dst, 'wb') as o:
                        o.write(f.read())
            os.remove(tmp)
            print('  [iso] WikiText-2 ready')
            return
        except Exception as e:
            print('  [warn] mirror failed (%r), trying next...' % e)
            try:
                os.remove(tmp)
            except OSError:
                pass
    raise SystemExit('[iso] could not download WikiText-2')


_ensure_wikitext()

import numpy as np
import torch

from nfra.benchmark import arena
from nfra.benchmark.arena import build_nfra, build_family_spec, train_one, \
    make_loaders, EVAL_GAP
from nfra.benchmark.compare import count_params, rescale_embed, \
    DEVICE, HAS_CUDA, CHAR_VOCAB

SEED = 42
STEPS = arena.STEPS
VOCAB = len(CHAR_VOCAB)          # 96, the WikiText-2 char set (overnight.VOCAB)

# Each config: name -> iso flags set to True (everything else stays ON).
CONFIGS = [
    ('baseline', {}),
    ('gland',    dict(iso_gland=True)),
    ('vgate',    dict(iso_vgate=True)),
    ('rgate',    dict(iso_rgate=True)),
    ('phase',    dict(iso_phase=True)),
    ('exit',     dict(iso_exit=True)),
]


def main():
    vocab = VOCAB
    spec = build_family_spec('nfra', 5, vocab)
    dim, U, depth = spec['dim'], spec['extra']['unique_blocks'], spec['extra']['depth']
    print('nfra 5M cortex geometry: dim %d, depth %d, unique %d, %.2fM params'
          % (dim, depth, U, spec['params'] / 1e6))
    print('protocol: %d steps, batch 8, seq 256, fp16 AMP, EMA 0.99, '
          'eager (compile exact-math)' % STEPS)

    train_loaders, ev, _ = make_loaders(0)
    tr = train_loaders[SEED]
    print('train %d/256  eval %d/256' % (len(tr.dataset), len(ev.dataset)))

    rows = []
    for name, off in CONFIGS:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        t0 = time.perf_counter()
        m = build_nfra(vocab, dim, U, depth=depth, **off).to(DEVICE)
        rescale_embed(m)
        n_params = count_params(m) / 1e6
        rec = train_one(m, vocab, STEPS, tr, ev, EVAL_GAP,
                        ema_decay=arena.EMA_DECAY, seed=SEED)
        final = rec['eval_hist'][-1][1] if rec['eval_hist'] else float('nan')
        rows.append((name, n_params, final, rec['tok_s'], rec['peak_mem'],
                     rec['wall_s']))
        print('  [%s] final %.3f   %.0f tok/s   %.2f GB   %.2fM   (%.0fs)'
              % (name, final, rec['tok_s'], rec['peak_mem'], n_params,
                 rec['wall_s']))
        del m
        if HAS_CUDA:
            torch.cuda.empty_cache()

    base = rows[0][2]
    print('\n' + '-' * 64)
    print('%-9s %7s %8s %9s %8s %8s' % ('config', 'params', 'loss', 'tok/s',
                                         'mem/GB', 'delta'))
    for name, n_params, loss, tok, mem, _w in rows:
        delta = loss - base
        flag = 'KEEP' if delta >= 0.02 else ('(washes)' if abs(delta) < 0.02
                                             else 'keep? +%.3f' % delta)
        print('%-9s %6.2fM %8.3f %9.0f %8.2f %+7.3f  %s'
              % (name, n_params, loss, tok, mem, delta, flag))
    print('-' * 64)
    print('Rule (FUTURE_PLAN Part 11): delta >= +0.02 -> mechanism carries the')
    print('win, keep. |delta| < 0.02 -> decoration, prune if it adds kernels.')
    print('baseline reference expected ~1.95-1.97 (verified board 1.961/1.945).')


if __name__ == '__main__':
    main()
