"""
OVERNIGHT GRAND ARENA — the final, very big and broad NFRA comparison.

Designed to run for a full Kaggle GPU session (T4/P100, ~9h). It compares
NFRA Brain vs RWKV vs RetNet vs GPT-2 (Mamba-SSM optional, it's slow) on REAL
text only:

  PRIMARY DATA (mandatory):  WikiText-2, character-level (standard, learnable).
                             The synthetic "unlearnable bigram" is intentionally
                             REJECTED (NFRA_OVN_DATA != wikitext2 -> hard exit).
  CROSS-DATA   (optional):   TinyShakespeare, character-level (real text).

Phases (env NFRA_OVN_PHASES; default all; resumable via overnight_state.json):
  core        head-to-head + measured scaling at SIZES x SEEDS x families.
  context     sequence-length generalization: train @256, eval @256/512/1024.
  efficiency  NFRA energy-budget sweep (0.25/0.5/0.75/1.0): compute vs quality.
  ablate      NFRA "small but powerful" lever A/B (EMA/surprise/k-WTA/...).
  recall      memory-horizon diagnostic (associative recall, learnable; NFRA vs
              Mamba vs GPT-2 across k). Clearly a diagnostic, not language data.
  deploy      INT8 quantization of the primary-size models: size / accuracy
              preservation / CPU prefill throughput (real deployment axis).
  perf        inference battery: prefill tok/s, generation tok/s, peak memory.
  data2       cross-dataset robustness check on TinyShakespeare.

Controls (env, all optional):
  NFRA_OVN_MODE      quick | standard | big         (default standard)
                        quick    300 steps, sizes 5M,            1 seed
                        standard 600 steps, sizes 5,20M,         2 seeds
                        big      1500 steps, sizes 5,10,20,50M,  3 seeds
  NFRA_OVN_STEPS     override train steps
  NFRA_OVN_SIZES     override sizes (M)
  NFRA_OVN_SEEDS     override seed count
  NFRA_OVN_PHASES    comma list of phases
  NFRA_OVN_FAMILIES  comma list: nfra,rwkv,retnet,mamba,gpt2 (default nfra,rwkv,
                     retnet,gpt2 — mamba is slow, add it explicitly if wanted)
  NFRA_OVN_MAX_MIN   time budget in minutes (default 400)
  NFRA_OVN_OUTDIR    output directory (default CWD)
  NFRA_DATA          must be "wikitext2"; anything else exits.

Outputs (in OUTDIR):
  overnight_results.json, overnight_report.md, overnight_state.json,
  core.csv, context.csv, efficiency.csv, ablate.csv, recall.csv,
  deploy.csv, perf.csv, data2.csv.

Usage (Kaggle):  python -m nfra.benchmark.overnight
"""

import os
import sys
import time
import math
import json
import zipfile
import urllib.request
import functools
import warnings
import traceback

print = functools.partial(print, flush=True)
warnings.filterwarnings('ignore')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# Some benchmark code prints non-ASCII (e.g. compare.py's box glyph). Keep the
# console alive even on legacy cp1252 terminals (Kaggle is UTF-8 so no effect).
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ───────────────────────────── config (set BEFORE benchmark imports) ─────────
MODE = os.environ.get('NFRA_OVN_MODE', 'standard')
CFG = {
    'quick':    dict(steps=300, sizes=[5], seeds=1),
    'standard': dict(steps=600, sizes=[5, 20], seeds=2),
    'big':      dict(steps=1500, sizes=[5, 10, 20, 50], seeds=3),
}
if MODE not in CFG:
    raise SystemExit('NFRA_OVN_MODE must be quick | standard | big')

PHASES = [p.strip() for p in
          os.environ.get('NFRA_OVN_PHASES',
                         'core,context,efficiency,ablate,recall,deploy,perf,data2')
          .split(',') if p.strip()]
STEPS = int(os.environ.get('NFRA_OVN_STEPS', str(CFG[MODE]['steps'])))
SIZES = [int(x) for x in os.environ.get(
    'NFRA_OVN_SIZES', ','.join(map(str, CFG[MODE]['sizes']))).split(',') if x.strip()]
if not SIZES:
    SIZES = CFG[MODE]['sizes']
SEED_CNT = int(os.environ.get('NFRA_OVN_SEEDS', str(CFG[MODE]['seeds'])))
DATA = os.environ.get('NFRA_OVN_DATA', 'wikitext2').lower()
if DATA != 'wikitext2':
    raise SystemExit(
        'NFRA_OVN_DATA must be "wikitext2" (standard real text). Synthetic '
        'unlearnable data is intentionally disallowed for the headline run.')
MAX_MIN = float(os.environ.get('NFRA_OVN_MAX_MIN', '400'))
OUTDIR = os.environ.get('NFRA_OVN_OUTDIR', os.getcwd())
os.makedirs(OUTDIR, exist_ok=True)

FAMILIES = [f.strip().lower() for f in
            os.environ.get('NFRA_OVN_FAMILIES', 'nfra,rwkv,retnet,gpt2').split(',')
            if f.strip()]
ENERGY_BUDGETS = [0.25, 0.5, 0.75, 1.0]
CONTEXT_LENS = [256, 512, 1024]
RECALL_KS = [4, 16, 64, 128]

# The canonical S3 URL (research.metamind.io) now returns a 301 redirect loop,
# so we fall back across mirrors in order. ggml-org/ci hosts the identical file
# (same sha256 ef7edb56...) as LFS on HF. All candidates produce the SAME zip.
WIKI_URLS = [
    'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip',
    'https://huggingface.co/datasets/ggml-org/ci/resolve/927b3642933080f1b0e811e2f916e14c292992f9/wikitext-2-raw-v1.zip',
]
SHAKES_URL = ('https://raw.githubusercontent.com/karpathy/char-rnn/master/'
              'data/tinyshakespeare/input.txt')


def _fetch(url, dest, timeout=120):
    """Download with a HARD timeout. urlretrieve accepts no timeout and can
    hang forever on a stalled connection, which previously froze the whole
    benchmark with no output. Stream via urlopen (which honors timeout) and
    copy in chunks."""
    print('  [fetch] %s -> %s' % (url, dest))
    with urllib.request.urlopen(url, timeout=timeout) as resp, \
            open(dest, 'wb') as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)


# compare.py's WIKI_PATHS (hardcoded here so we do NOT import compare before
# the files exist — importing it early would silently fall back to synthetic).
WIKI_FILES = {'train': 'wikitext-train-raw-v1.txt',
              'validation': 'wikitext-valid-raw-v1.txt'}


def ensure_wikitext():
    """Download WikiText-2-raw-1 into the CWD BEFORE compare.py is imported
    (compare.py silently falls back to the unlearnable synthetic set if the
    files are missing — we must never let that happen)."""
    missing = [f for f in WIKI_FILES.values() if not os.path.exists(f)]
    if not missing:
        return
    print('[data] WikiText-2 files missing, downloading')
    tmp = 'wikitext-2-raw-v1.zip'
    last_err = None
    for url in WIKI_URLS:
        try:
            _fetch(url, tmp)
            if os.path.getsize(tmp) < 1024 * 1024:
                raise ValueError('download too small (%d bytes), likely an '
                                 'error page' % os.path.getsize(tmp))
            with zipfile.ZipFile(tmp) as z:
                for src, dst in [('wikitext-2-raw/wiki.train.raw', WIKI_FILES['train']),
                                 ('wikitext-2-raw/wiki.valid.raw', WIKI_FILES['validation'])]:
                    with z.open(src) as f, open(dst, 'wb') as o:
                        o.write(f.read())
            os.remove(tmp)
            break
        except Exception as e:
            last_err = e
            print('  [warn] mirror failed (%r), trying next...' % e)
            try:
                os.remove(tmp)
            except OSError:
                pass
    else:
        # Both zip mirrors failed. HF no longer serves the raw .txt files, but
        # the dataset still ships as parquet; huggingface_hub handles the signed
        # CDN redirects that plain urllib/wget often miss. Fall back to that.
        print('  [warn] zip mirrors failed (%r) - trying parquet via '
              'huggingface_hub' % last_err)
        try:
            from huggingface_hub import hf_hub_download
            import pandas as pd
            for split, out in [('train', WIKI_FILES['train']),
                               ('validation', WIKI_FILES['validation'])]:
                path = hf_hub_download(
                    'Salesforce/wikitext',
                    'wikitext-2-raw-v1/%s-00000-of-00001.parquet' % split,
                    repo_type='dataset')
                df = pd.read_parquet(path)
                with open(out, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(str(t) for t in df['text']))
        except Exception as pe:
            raise SystemExit(
                '[data] could not download WikiText-2 (zip: %r; parquet: %r). '
                'Place the two raw .txt files (%s) in the working directory '
                'and re-run.'
                % (last_err, pe, list(WIKI_FILES.values())))
    for f in WIKI_FILES.values():
        assert os.path.exists(f), 'missing %s' % f
        print('  [ok] %s (%d KB)' % (f, os.path.getsize(f) // 1024))


def ensure_shakespeare():
    dst = os.path.join(OUTDIR, 'tinyshakespeare.txt')
    if not os.path.exists(dst):
        print('[data] TinyShakespeare missing, downloading')
        try:
            _fetch(SHAKES_URL, dst)
        except Exception as e:
            raise SystemExit('[data] could not download TinyShakespeare (%r)' % e)
    return dst


# Print BEFORE any heavy work so a healthy run is never silently dark: the
# torch/arena/compare imports below take ~20-30s with zero output, and users
# (rightly) assume a dead cell when nothing appears.
print('  [overnight] starting - checking data, then importing torch (~20s silent)')
ensure_wikitext()
print('  [overnight] wikitext ready - importing torch/arena/compare...')

# Align the shared benchmark modules with this run's config.
os.environ['NFRA_MODE'] = 'standard'
os.environ['NFRA_SIZES'] = ','.join(map(str, SIZES))
os.environ['NFRA_SEEDS'] = str(SEED_CNT)
os.environ['NFRA_STEPS'] = str(STEPS)
os.environ['NFRA_DATA'] = 'wikitext2'
for k in ('NFRA_EMA', 'NFRA_SURPRISE', 'NFRA_KWTA', 'NFRA_LOCALROUTE',
          'NFRA_DIVNORM', 'NFRA_ASTRO', 'NFRA_THETA', 'NFRA_ACH_RETAIN',
          'NFRA_GAIN_NOV'):
    os.environ.setdefault(k, '0')
os.environ.setdefault('NFRA_LORA_RANK', '0')
# ── NFRA speed knobs (both are exact-math, zero-tradeoff, and only touch NFRA).
# 1) NFRA_CHECKPOINT=0: the NFRA 5M model needs 0.2 GB and 50M ~1 GB, so there is
#    no memory pressure on a 16 GB T4 — gradient checkpointing only ADDED recompute
#    cost in backward (~+30-50% train time). Off = same results, strictly faster.
# 2) NFRA_COMPILE=1 + NFRA_SCAN_KERNEL=0: torch.compile fuses the model's hundreds
#    of small per-block ops (neuromodulator cumsums, scan, local attention, routing
#    MLP) into a handful of kernels — the exact reason NFRA is launch-bound slow.
#    The scan must stay traceable, so force the pure-torch closed-form scan (the
#    custom ScanFunction autograd.Function would break the compiled graph). Same
#    numerics, 1.5-3x fewer launches. train_one already falls back to eager on any
#    compile error, so this is safe.
os.environ.setdefault('NFRA_CHECKPOINT', '0')
os.environ.setdefault('NFRA_COMPILE', '1')
os.environ.setdefault('NFRA_SCAN_KERNEL', '0')

# Loss-focus lever: EMA (weight averaging) applied to ALL families during
# training — a near-free loss gain (~-0.1..-0.3 nats at these sizes) for
# negligible memory. Off (0.0) reproduces the pre-EMA baseline.
EMA_DECAY = float(os.environ.get('NFRA_EMA', '0.99'))
os.environ.setdefault('NFRA_EMA', str(EMA_DECAY))

import copy

import numpy as np
import torch
from torch.utils.data import DataLoader

from nfra.benchmark import arena
from nfra.benchmark.arena import (
    build_family_spec, train_one, make_loaders, prefill_tok_s,
    generate_metrics, sample_auc, mean_std, fit_scaling, composite_score,
    winner_of, make_verdict, METRIC_SPEC, SEED_LIST,
)
from nfra.benchmark.compare import (
    count_params, rescale_embed, evaluate, compute_loss, make_optimizer,
    EMA, DEVICE, HAS_CUDA, USE_AMP, BATCH, SEQ_LEN, CHAR_VOCAB,
    WikiText2Dataset, HierarchicalDataset,
)
from nfra.benchmark.global_arena import (
    ABLATE, _build_family as _build_fam_ga, _run as _run_ga,
    _agg as _agg_ga,
)
from nfra.utils.quantization import apply_int8_to_model, Int8Linear

# nfra.benchmark/__init__ imports compare+arena eagerly, so DATA_SOURCE is
# frozen to 'synthetic' BEFORE the env var above takes effect. make_loaders and
# the build paths read the module global at call time, so force it here — this
# is the load-bearing line that keeps the whole run on real WikiText-2.
import nfra.benchmark.compare as _cmp_mod
_cmp_mod.DATA_SOURCE = 'wikitext2' if DATA == 'wikitext2' else DATA
arena.DATA_SOURCE = _cmp_mod.DATA_SOURCE

SEEDS = SEED_LIST[:SEED_CNT]
PRIMARY = max(SIZES)
# Match arena.main's convention: WikiText-2 char vocab is sized 96 (CHAR_VOCAB
# has 95 real entries + one unused row). Building at 96 is strictly safe — data
# tokens can never exceed 94 — and keeps us consistent with the rest of the
# benchmark. (A mismatched smaller vocab is what surfaces as a cryptic
# "index out of range in self" in the embedding layer.)
VOCAB = 96
RANDOM_LOSS = math.log(VOCAB)
ETA_DEF = 0.05                          # sec/step estimate on a T4, first guess


# ───────────────────────────── time budget ─────────────────────────────
class Budget:
    def __init__(self, max_min):
        self.t0 = time.perf_counter()
        self.deadline = self.t0 + max_min * 60.0

    def left(self):
        return max(0.0, self.deadline - time.perf_counter())

    def used_min(self):
        return (time.perf_counter() - self.t0) / 60.0


BUDGET = Budget(MAX_MIN)
EST_SEC_STEP = {}                       # family -> estimated seconds/step


def _note_perf(fam, params, wall, steps):
    """Adaptively learn seconds/step from completed runs for ETA planning."""
    s = wall / max(steps, 1) * max(1.0, params / 20e6)
    EST_SEC_STEP[fam] = 0.6 * EST_SEC_STEP.get(fam, ETA_DEF) + 0.4 * s


def plan_steps(label, n_runs, base_steps, min_steps=150):
    """Shrink steps so the phase fits the remaining budget. 0 = skip."""
    left = BUDGET.left()
    if left <= 180:
        print('  [budget] %.0f min left - skipping %s' % (left / 60, label))
        return 0
    est = n_runs * base_steps * ETA_DEF * 1.15
    if EST_SEC_STEP:
        est = n_runs * base_steps * max(EST_SEC_STEP.values()) * 1.15
    if est <= left * 0.75:
        return base_steps
    k = left * 0.6 / max(n_runs * base_steps * max(EST_SEC_STEP.values(), default=ETA_DEF), 1e-6)
    steps = int(base_steps * k)
    print('  [budget] %s phase: %.0f min left, scaling %d -> %d steps'
          % (label, left / 60, base_steps, max(steps, min_steps)))
    return max(steps, min_steps)


# ───────────────────────────── model cache ─────────────────────────────
_CACHE = {}                             # fam -> {'model':..., 'steps':...}


def _cached_primary(fam, vocab, steps):
    """Reuse a primary-size model trained earlier this process, else train a
    short one. Guarantees perf/deploy phases never re-run the full core."""
    if fam in _CACHE and _CACHE[fam]['steps'] >= steps:
        return _CACHE[fam]['model']
    spec = build_family_spec(fam, PRIMARY, vocab)
    torch.manual_seed(0)
    m = spec['builder'](vocab, spec['dim'], **spec['extra']).to(DEVICE)
    rescale_embed(m)
    train_loaders, eval_loader, _ext = make_loaders(SIZES.index(PRIMARY))
    train_one(m, vocab, steps, train_loaders[SEEDS[-1]], eval_loader,
              max(50, steps // 6), ema_decay=EMA_DECAY, seed=SEEDS[-1])
    _CACHE[fam] = {'model': m, 'steps': steps}
    return m


def _primary_val_loader(max_batches=None):
    _ds = WikiText2Dataset('validation', SEQ_LEN)
    return DataLoader(_ds, batch_size=BATCH, shuffle=False,
                      num_workers=0, pin_memory=HAS_CUDA)


def mb(model):
    return sum(v.numel() * v.element_size()
               for v in model.state_dict().values()) / 1e6


def cpu_prefill_tok_s(model, vocab, seq=256, batch=1, iters=5):
    m = model.cpu().eval()
    x = torch.randint(0, vocab, (batch, seq))
    with torch.no_grad():
        for _ in range(2):
            m(x)
        t0 = time.perf_counter()
        for _ in range(iters):
            m(x)
    return batch * seq * iters / max(time.perf_counter() - t0, 1e-6)


def seeded_train_loader(train_loader, seed):
    return iter(DataLoader(train_loader.dataset, batch_size=train_loader.batch_size,
                           shuffle=True,
                           generator=torch.Generator().manual_seed(seed),
                           num_workers=0, pin_memory=train_loader.pin_memory))


def assert_tokens_in_range(model, loader, vocab, label):
    """Sanity check before training: every sample's input and target tokens must
    fit the model's embedding/lm-head width. Turns any dataset-model vocab
    mismatch into a clear error instead of a cryptic IndexError. O(1): the raw
    char-id tensor is cached on the dataset, and targets are just inputs shifted
    by one, so the max token is identical."""
    data = getattr(loader.dataset, 'data', None)
    if data is None:
        return
    if data.numel() == 0 if hasattr(data, 'numel') else data.size == 0:
        return
    mx = int(data.max())
    if mx >= vocab:
        raise SystemExit(
            '[%s] token mismatch: max token %d >= model vocab %d. Check the '
            'WikiText-2 char files in the working directory.'
            % (label, mx, vocab))


# ───────────────────────────── phase: core ─────────────────────────────
def phase_core(vocab, random_loss):
    print('\n' + '=' * 72)
    print('PHASE core — head-to-head + scaling on WikiText-2 (real text)')
    print('=' * 72)
    specs, runs, battery = {}, {}, {}
    for size in SIZES:
        specs[size] = {f: build_family_spec(f, size, vocab) for f in FAMILIES}
        for f, s in specs[size].items():
            print('  [build] %-6s @ %dM: dim %-4d %.2fM depth %d'
                  % (f, size, s['dim'], s['params'] / 1e6, s['depth']))
    t_phase = time.perf_counter()
    for size in SIZES:
        train_loaders, eval_loader, ext_loader = make_loaders(SIZES.index(size))
        runs[size] = {}
        for seed in SEEDS:
            runs[size][seed] = {}
            torch.manual_seed(seed)
            np.random.seed(seed)
            for fam in FAMILIES:
                t0 = time.perf_counter()
                spec = specs[size][fam]
                m = spec['builder'](vocab, spec['dim'], **spec['extra']).to(DEVICE)
                rescale_embed(m)
                assert_tokens_in_range(m, train_loaders[seed], vocab,
                                       'core:%s@%dM' % (fam, size))
                rec = train_one(m, vocab, STEPS, train_loaders[seed], eval_loader,
                                arena.EVAL_GAP, ema_decay=EMA_DECAY, seed=seed)
                _note_perf(fam, spec['params'], rec['wall_s'], STEPS)
                runs[size][seed][fam] = rec
                if seed == SEEDS[-1]:
                    if size == PRIMARY:
                        _CACHE[fam] = {'model': m, 'steps': STEPS}
                    ext = evaluate(m, ext_loader, max_batches=6)
                    battery.setdefault(size, {})[fam] = {
                        'extrap_loss': ext,
                        'extrap_delta': ext - rec['eval_hist'][-1][1]
                        if rec['eval_hist'] else None,
                    }
                    if size == PRIMARY:
                        pre = prefill_tok_s(m, max(arena.PRE_HEAD, 8), SEQ_LEN, vocab)
                        gen = generate_metrics(m, vocab)
                        battery[size][fam].update({
                            'prefill_tok_s': pre, 'gen_tok_s': gen['gen_tok_s'],
                            'ms_per_token': gen['ms_per_token'],
                            'infer_mem': gen['infer_mem']})
                print('  [train] %-6s @ %dM seed %-4d final %s  %.0f tok/s  '
                      '%.2f GB  (%.0fs)'
                      % (fam, size, seed,
                         '%.3f' % rec['eval_hist'][-1][1] if rec['eval_hist'] else 'NA',
                         rec['tok_s'], rec['peak_mem'], time.perf_counter() - t0))
                # Only primary-size, last-seed models stay cached (perf/deploy
                # phases reuse them). Free everything else so non-primary and
                # extra-seed models don't accumulate and OOM the second size.
                if not (size == PRIMARY and seed == SEEDS[-1]):
                    del m
                    if HAS_CUDA:
                        torch.cuda.empty_cache()
    metrics = {}
    for size in SIZES:
        metrics[size] = {}
        for fam in FAMILIES:
            recs = [runs[size][s][fam] for s in SEEDS]
            m_final, sd_final = mean_std([r['eval_hist'][-1][1] for r in recs
                                          if r['eval_hist']])
            m_auc, _ = mean_std([sample_auc(r['eval_hist']) for r in recs
                                 if r['eval_hist']])
            m_tok, _ = mean_std([r['tok_s'] for r in recs])
            m_ms, _ = mean_std([r['ms_per_step'] for r in recs])
            m_mem, _ = mean_std([r['peak_mem'] for r in recs])
            params = specs[size][fam]['params']
            row = {
                'params': params, 'depth': specs[size][fam]['depth'],
                'spec': {k: v for k, v in specs[size][fam].items()
                         if k != 'builder'},
                'final_eval': m_final, 'final_eval_sd': sd_final,
                'final_eval_n': len(recs),
                'ppl': math.exp(min(m_final, 30)) if m_final else None,
                'sample_auc': m_auc, 'tok_s_train': m_tok,
                'ms_per_step': m_ms, 'peak_mem': m_mem,
                'nan_steps': sum(r['nan_steps'] for r in recs),
                'param_eff': ((random_loss - m_final) / (params / 1e6))
                if m_final else None,
            }
            if fam in battery.get(size, {}):
                row.update(battery[size][fam])
            metrics[size][fam] = row
    scaling = {f: fit_scaling([(metrics[s][f]['params'], metrics[s][f]['final_eval'])
                               for s in SIZES if metrics[s][f]['final_eval']])
               for f in FAMILIES}
    for f in FAMILIES:
        for size in SIZES:
            metrics[size][f]['scaling_gain'] = -scaling[f]['slope']
    scores = {s: composite_score(metrics[s], METRIC_SPEC) for s in SIZES}
    verdict = make_verdict(metrics, scores, scaling, PRIMARY, random_loss)
    print('  [phase core] done in %.0fs' % (time.perf_counter() - t_phase))
    return {'metrics': metrics, 'scaling': scaling, 'scores': scores,
            'verdict': verdict, 'primary': PRIMARY}


# ───────────────────────────── phase: context ─────────────────────────────
def _wiki_train_eval(seq_len, batch, seed):
    train_ds = WikiText2Dataset('train', seq_len)
    eval_ds = WikiText2Dataset('validation', seq_len)
    tr = DataLoader(train_ds, batch_size=batch, shuffle=True,
                    generator=torch.Generator().manual_seed(seed),
                    num_workers=0, pin_memory=HAS_CUDA)
    ev = DataLoader(eval_ds, batch_size=batch, shuffle=False,
                    num_workers=0, pin_memory=HAS_CUDA)
    return tr, ev


def phase_context(vocab):
    print('\n' + '=' * 72)
    print('PHASE context — length generalization: train @256, eval @256/512/1024')
    print('=' * 72)
    steps = plan_steps('context', 3, min(STEPS, 600))
    if steps <= 0:
        return {}
    rows = {}
    for fam in FAMILIES:
        spec = build_family_spec(fam, PRIMARY, vocab)
        torch.manual_seed(0)
        m = spec['builder'](vocab, spec['dim'], **spec['extra']).to(DEVICE)
        rescale_embed(m)
        tr, ev = _wiki_train_eval(SEQ_LEN, BATCH, SEEDS[-1])
        rec = train_one(m, vocab, steps, tr, ev, max(50, steps // 6),
                        ema_decay=EMA_DECAY, seed=SEEDS[-1])
        base = rec['eval_hist'][-1][1]
        lengths = {}
        for L in CONTEXT_LENS:
            ds = WikiText2Dataset('validation', L)
            lo = DataLoader(ds, batch_size=BATCH, shuffle=False,
                            num_workers=0, pin_memory=HAS_CUDA)
            lengths[L] = evaluate(m, lo, max_batches=8)
        rows[fam] = {'params': count_params(m), 'train_final': base,
                     'lengths': lengths,
                     'delta_2x': lengths[CONTEXT_LENS[1]] - base,
                     'delta_4x': lengths[CONTEXT_LENS[2]] - base}
        print('  [ctx] %-6s final %.3f | @256 %.3f  @512 %.3f  @1024 %.3f'
              % (fam, base, lengths[256], lengths[512], lengths[1024]))
        del m
        if HAS_CUDA:
            torch.cuda.empty_cache()
    return rows


# ───────────────────────────── phase: efficiency ─────────────────────────────
def _train_energy(model, vocab, steps, tr, ev, eval_gap, budget, seed,
                  ema_decay=0.0):
    model.train()
    opt, sched = make_optimizer(model, lr=3e-4,
                                warmup=min(50, max(steps // 10, 1)), total=steps)
    scaler = torch.amp.GradScaler(str(DEVICE)) if USE_AMP else None
    ema = EMA(model, ema_decay) if ema_decay > 0 else None
    it = seeded_train_loader(tr, seed)
    hist, eval_hist, nan = [], [], 0
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        try:
            x, y = next(it)
        except StopIteration:
            it = seeded_train_loader(tr, seed)
            x, y = next(it)
        x = x.to(DEVICE, non_blocking=HAS_CUDA)
        y = y.to(DEVICE, non_blocking=HAS_CUDA)
        opt.zero_grad()
        with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            logits = model(x, energy_budget=budget)['logits']
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, vocab), y.view(-1))
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            g = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if math.isfinite(g):
                scaler.step(opt)
            else:
                opt.zero_grad(set_to_none=True)
                nan += 1
            scaler.update()
        else:
            loss.backward()
            g = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if math.isfinite(g):
                opt.step()
            else:
                opt.zero_grad(set_to_none=True)
                nan += 1
        sched.step()
        if ema is not None:
            ema.update(model)
        hist.append(loss.detach())
        if step % eval_gap == 0 or step == steps:
            if ema is not None:
                ema.apply(model)
            eval_hist.append((step, evaluate(model, ev)))
            if ema is not None:
                ema.restore(model)
    if ema is not None:
        ema.apply(model)   # leave EMA weights in place for downstream evals
    if HAS_CUDA:
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    bs = getattr(tr, 'batch_size', 1)
    return {'loss_hist': [float(v) for v in hist], 'eval_hist': eval_hist,
            'tok_s': bs * SEQ_LEN * steps / max(wall, 1e-6),
            'ms_per_step': wall * 1000.0 / steps,
            'peak_mem': torch.cuda.max_memory_allocated() / 1e9 if HAS_CUDA else 0.0,
            'nan_steps': nan, 'wall_s': wall}


def phase_efficiency(vocab):
    print('\n' + '=' * 72)
    print('PHASE efficiency — NFRA energy-budget sweep on WikiText-2')
    print('=' * 72)
    steps = plan_steps('efficiency', len(ENERGY_BUDGETS), min(STEPS, 600))
    if steps <= 0:
        return {}
    spec = build_family_spec('nfra', PRIMARY, vocab)
    tr, ev = _wiki_train_eval(SEQ_LEN, BATCH, SEEDS[-1])
    rows = {}
    for b in ENERGY_BUDGETS:
        torch.manual_seed(0)
        m = spec['builder'](vocab, spec['dim'],
                            **spec['extra']).to(DEVICE)
        rescale_embed(m)
        rec = _train_energy(m, vocab, steps, tr, ev, max(50, steps // 6),
                            b, SEEDS[-1], ema_decay=EMA_DECAY)
        final = rec['eval_hist'][-1][1] if rec['eval_hist'] else None
        rows[str(b)] = {'budget': b, 'final_eval': final,
                        'tok_s': rec['tok_s'], 'ms_per_step': rec['ms_per_step'],
                        'peak_mem': rec['peak_mem']}
        print('  [energy] budget %.2f -> final %s  %.0f tok/s  %.2f GB'
              % (b, '%.3f' % final if final else 'NA', rec['tok_s'],
                 rec['peak_mem']))
    return rows


# ───────────────────────────── phase: ablate ─────────────────────────────
def phase_ablate(vocab):
    print('\n' + '=' * 72)
    print('PHASE ablate — NFRA "small but powerful" levers @ %dM' % PRIMARY)
    print('=' * 72)
    steps = plan_steps('ablate', len(ABLATE), min(STEPS, 600), min_steps=120)
    if steps <= 0:
        return {}
    train_loaders, eval_loader, _ext = make_loaders(SIZES.index(PRIMARY))
    results = {}
    for name, build_kw, train_kw, n_seeds in ABLATE:
        fam = build_kw.get('fam', 'nfra')
        if fam not in FAMILIES:
            print('  [skip] %-16s (family %s not in NFRA_OVN_FAMILIES)'
                  % (name, fam))
            continue
        seeds = SEEDS[:min(n_seeds, SEED_CNT)]
        recs, params = [], None
        build_kw = dict(build_kw)
        train_kw = dict(train_kw)
        for seed in seeds:
            t0 = time.perf_counter()
            fam = build_kw.pop('fam', 'nfra')
            m, params = _build_fam_ga(fam, PRIMARY, vocab, seed, **build_kw)
            m = m.to(DEVICE)
            rec = _run_ga(m, vocab, steps, train_loaders[seed], eval_loader,
                          max(50, steps // 6), seed=seed, **train_kw)
            recs.append(rec)
            print('  [train] %-16s seed %-4d final %s  %.0f tok/s  (%.0fs)'
                  % (name, seed,
                     '%.3f' % rec['eval_hist'][-1][1] if rec['eval_hist'] else 'NA',
                     rec['tok_s'], time.perf_counter() - t0))
        row = _agg_ga(recs)
        row['params'] = params
        row['seeds'] = len(recs)
        results[name] = row
    return results


# ───────────────────────────── phase: recall ─────────────────────────────
def phase_recall():
    print('\n' + '=' * 72)
    print('PHASE recall — memory-horizon diagnostic (associative recall)')
    print('=' * 72)
    from nfra.benchmark import recall_probe
    steps = plan_steps('recall', 2 * len(RECALL_KS), min(STEPS, 600),
                       min_steps=120)
    if steps <= 0:
        return {}
    dim = int(os.environ.get('NFRA_OVN_RECALL_DIM', '224'))
    rows = recall_probe._run_all(RECALL_KS, steps=steps, dim=dim, seq_len=256,
                                 batch=8, unique=4, depth=12, concurrent=True,
                                 families=tuple(FAMILIES))
    return {'rows': rows, 'vocab': recall_probe.V, 'dim': dim}


# ───────────────────────────── phase: deploy ─────────────────────────────
def phase_deploy(vocab):
    print('\n' + '=' * 72)
    print('PHASE deploy — INT8 quantization of primary-size models')
    print('=' * 72)
    steps = plan_steps('deploy', 3, min(STEPS, 300), min_steps=100)
    if steps <= 0:
        return {}
    ev_loader = _primary_val_loader()
    rows = {}
    for fam in FAMILIES:
        m = _cached_primary(fam, vocab, steps)
        fp_eval = evaluate(m, ev_loader, max_batches=6)
        mb_fp = mb(m)
        # Quantize a COPY so the fp32 cached model stays intact for perf/context.
        m8 = apply_int8_to_model(copy.deepcopy(m))
        mb_i8 = mb(m8)
        i8_eval = evaluate(m8, ev_loader, max_batches=6)
        cpus = cpu_prefill_tok_s(m8, vocab)
        rows[fam] = {'params': count_params(m), 'mb_fp32': mb_fp,
                     'mb_int8': mb_i8,
                     'size_saved_pct': (1 - mb_i8 / max(mb_fp, 1e-9)) * 100,
                     'eval_fp32': fp_eval, 'eval_int8': i8_eval,
                     'eval_delta': i8_eval - fp_eval,
                     'cpu_prefill_tok_s_int8': cpus}
        print('  [int8] %-6s %.1f -> %.1f MB (%.0f%% saved) | eval %.3f -> %.3f '
              '| CPU %.0f tok/s'
              % (fam, mb_fp, mb_i8, rows[fam]['size_saved_pct'], fp_eval,
                 i8_eval, cpus))
    return rows


# ───────────────────────────── phase: perf ─────────────────────────────
def phase_perf(vocab):
    print('\n' + '=' * 72)
    print('PHASE perf — inference battery + 2x extrapolation @ %dM' % PRIMARY)
    print('=' * 72)
    ext_loader = None
    if DATA == 'wikitext2':
        ext_ds = WikiText2Dataset('validation', SEQ_LEN * 2)
        ext_loader = DataLoader(ext_ds, batch_size=BATCH, shuffle=False,
                                num_workers=0, pin_memory=HAS_CUDA)
    rows = {}
    for fam in FAMILIES:
        m = _cached_primary(fam, vocab, min(STEPS, 300))
        pre = prefill_tok_s(m, max(arena.PRE_HEAD, 8), SEQ_LEN, vocab)
        gen = generate_metrics(m, vocab)
        ext = evaluate(m, ext_loader, max_batches=6) if ext_loader else None
        rows[fam] = {'prefill_tok_s': pre, 'gen_tok_s': gen['gen_tok_s'],
                     'ms_per_token': gen['ms_per_token'],
                     'infer_mem': gen['infer_mem'],
                     'extrap_loss_2x': ext}
        print('  [perf] %-6s prefill %5.0f | gen %5.1f tok/s | %5.2f ms/tok '
              '| %.2f GB | 2x ctx %.3f'
              % (fam, pre, gen['gen_tok_s'], gen['ms_per_token'],
                 gen['infer_mem'], ext if ext else float('nan')))
    return rows


# ───────────────────────────── phase: data2 (shakespeare) ─────────────────────
class CharTextDataset:
    """Generic char-level text dataset (real text, custom vocab)."""

    def __init__(self, text, vocab, seq_len):
        self.seq_len = seq_len
        ids = torch.tensor([vocab.get(c, 0) for c in text], dtype=torch.long)
        self.data = ids[: (len(ids) // seq_len) * seq_len + 1]
        self.num_seqs = len(self.data) // seq_len

    def __len__(self):
        return self.num_seqs

    def __getitem__(self, idx):
        s = idx * self.seq_len
        return self.data[s:s + self.seq_len], self.data[s + 1:s + self.seq_len + 1]


def _char_vocab(text):
    chars = sorted(set(text))
    return {c: i for i, c in enumerate(chars)}, len(chars)


def phase_data2(vocab=None):
    print('\n' + '=' * 72)
    print('PHASE data2 — cross-dataset robustness: TinyShakespeare (real text)')
    print('=' * 72)
    steps = plan_steps('data2', 3, min(STEPS, 500), min_steps=100)
    if steps <= 0:
        return {}
    path = ensure_shakespeare()
    with open(path, encoding='utf-8') as f:
        text = f.read()
    vocab2, V2 = _char_vocab(text)
    random_loss2 = math.log(V2)
    n_train = int(len(text) * 0.9)
    tr = CharTextDataset(text[:n_train], vocab2, SEQ_LEN)
    ev = CharTextDataset(text[n_train:], vocab2, SEQ_LEN)
    train_loader = DataLoader(tr, batch_size=BATCH, shuffle=True,
                              generator=torch.Generator().manual_seed(SEEDS[-1]),
                              num_workers=0, pin_memory=HAS_CUDA)
    eval_loader = DataLoader(ev, batch_size=BATCH, shuffle=False,
                             num_workers=0, pin_memory=HAS_CUDA)
    rows = {}
    for fam in FAMILIES:
        spec = build_family_spec(fam, PRIMARY, V2)
        torch.manual_seed(0)
        m = spec['builder'](V2, spec['dim'], **spec['extra']).to(DEVICE)
        rescale_embed(m)
        rec = train_one(m, V2, steps, train_loader, eval_loader,
                        max(50, steps // 6), ema_decay=EMA_DECAY, seed=SEEDS[-1])
        rows[fam] = {'params': count_params(m), 'vocab': V2,
                     'final_eval': rec['eval_hist'][-1][1] if rec['eval_hist'] else None,
                     'random_loss': random_loss2, 'tok_s': rec['tok_s']}
        print('  [data2] %-6s final %s (random %.2f)  %.0f tok/s'
              % (fam, '%.3f' % rows[fam]['final_eval']
                 if rows[fam]['final_eval'] else 'NA',
                 random_loss2, rec['tok_s']))
    return rows


# ───────────────────────────── CSV / report ─────────────────────────────
def write_csv(name, headers, rows):
    path = os.path.join(OUTDIR, name)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(','.join(map(str, headers)) + '\n')
        for r in rows:
            f.write(','.join('' if v is None else str(v) for v in r) + '\n')
    return path


def fmt(v, nd=3, suffix=''):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return '-'
    return f"{v:.{nd}f}{suffix}"


def md_table(headers, rows):
    out = ['| ' + ' | '.join(headers) + ' |',
           '|' + '|'.join('---' for _ in headers) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(str(c) for c in r) + ' |')
    return '\n'.join(out)


def build_report(data, env):
    L = []
    a = L.append
    a('# OVERNIGHT GRAND ARENA — final all-axes NFRA comparison\n')
    a('**Data:** WikiText-2 (char, real text)%s  |  **Vocab:** %d  |  '
      '**Sizes:** %sM  |  **Seeds:** %s  |  **Steps:** %d  |  **Mode:** %s\n'
      % (' + TinyShakespeare' if 'data2' in data else '', VOCAB, SIZES, SEEDS,
         STEPS, MODE))
    a('**Families:** %s (param-matched)  |  '
      '**Optimizer:** AdamW 3e-4 (warmup+cosine)\n' % ' vs '.join(FAMILIES))
    a('**Environment:** ' + ', '.join('%s=%s' % (k, v) for k, v in env.items()
                                      if v) + '\n')
    a('**Random-guess loss:** %.3f (ln vocab)\n\n' % RANDOM_LOSS)

    if 'core' in data:
        c = data['core']
        mets, scaling, scores = c['metrics'], c['scaling'], c['scores']
        a('\n## 1. Core — head-to-head + scaling\n')
        a('\n### 1a. Models (param-matched)\n')
        rows = []
        for size in sorted(SIZES):
            for fam in FAMILIES:
                s = mets[size][fam]['spec']
                rows.append([f'{size}M', fam, s['dim'], round(s['params'] / 1e6, 2),
                             mets[size][fam]['depth'],
                             s['extra'].get('unique_blocks', '-')])
        a(md_table(['size', 'family', 'dim', 'params (M)', 'depth',
                    'unique blocks (NFRA)'], rows) + '\n')
        a('\n### 1b. Scaling (bits of loss per doubling of params; neg = better)\n')
        rows = [[fam, fmt(scaling[fam]['slope'], 4), fmt(scaling[fam].get('r2'), 3),
                 fmt(scaling[fam].get('loss_100m'), 3), fmt(scaling[fam].get('loss_1b'), 3),
                 scaling[fam].get('n', 0)] for fam in FAMILIES]
        a(md_table(['family', 'slope', 'R²', 'extrap @100M', '@1B', 'points'],
                   rows) + '\n')
        for size in sorted(SIZES):
            m = mets[size]
            a(f'\n### 1c. Head-to-head @ {size}M\n')
            rows = [[fam, fmt(m[fam]['final_eval'], 3),
                     fmt(m[fam]['final_eval_sd'], 3), fmt(m[fam]['ppl'], 2),
                     fmt(m[fam]['sample_auc'], 3), f"{m[fam]['tok_s_train']:.0f}",
                     fmt(m[fam]['peak_mem'], 2),
                     fmt(m[fam].get('extrap_delta'), 3)] for fam in FAMILIES]
            a(md_table(['family', 'eval loss', 'mean±std', 'ppl', 'AUC',
                        'train tok/s', 'peak GB', '2x ctx Δ'], rows) + '\n')
            rows = []
            for s in METRIC_SPEC:
                w = winner_of(m, s['key'], s['dir'])
                if w is None:
                    rows.append([s['label'], '-', '-'])
                else:
                    v = m[w].get(s['key'])
                    rows.append([s['label'], w, fmt(v, 3) if isinstance(v, float) else v])
            a('**Who wins which aspect @ %dM:**\n' % size +
              md_table(['aspect', 'winner', 'value'], rows) + '\n')
        a(f'\n### 1d. Composite scores @ {PRIMARY}M\n')
        a(md_table(['family', 'score'],
                   [[fam, f"{scores[PRIMARY][fam]:.1f}"] for fam in FAMILIES]) + '\n')
        a('\n### 1e. Verdict\n')
        for cl in c['verdict']['claims']:
            ev = ' — ' + cl['evidence'] if cl.get('evidence') else ''
            a(f"- **{cl['claim']}:** {cl['family']} ({cl['status']}){ev}\n")
        r = c['verdict']['revo']
        a(f"- **Overall leader:** {r['overall_leader']} "
          f"(score {r['overall_score']:.1f} vs {r['worst_score']:.1f})\n")
        if 'nfra' in FAMILIES:
            a(f"- **NFRA Brain quality rank:** {r['nfra_quality_rank']} "
              f"(gap to best {r['nfra_quality_gap_to_best']:+.3f})\n")

    if 'context' in data:
        a('\n## 2. Context — length generalization (train @256)\n')
        rows = [[fam, fmt(v['train_final'], 3),
                 fmt(v['lengths'].get(256), 3), fmt(v['lengths'].get(512), 3),
                 fmt(v['lengths'].get(1024), 3),
                 fmt(v['delta_2x'], 3, ' Δ'), fmt(v['delta_4x'], 3, ' Δ')]
                for fam, v in data['context'].items()]
        a(md_table(['family', 'train', '@256', '@512', '@1024',
                    'Δ@2x', 'Δ@4x'], rows) + '\n')

    if 'efficiency' in data:
        a('\n## 3. Efficiency — NFRA energy-budget sweep @ %dM\n' % PRIMARY)
        rows = [[v['budget'], fmt(v['final_eval'], 3), f"{v['tok_s']:.0f}",
                 fmt(v['peak_mem'], 2)] for v in data['efficiency'].values()]
        a(md_table(['energy budget', 'eval loss', 'train tok/s', 'peak GB'],
                   rows) + '\n')

    if 'ablate' in data:
        a('\n## 4. Ablate — NFRA levers @ %dM\n' % PRIMARY)
        base = data['ablate'].get('nfra_baseline')
        rows = []
        for name, row in data['ablate'].items():
            if base and name != 'nfra_baseline':
                dE = (row['final_eval'] - base['final_eval']) if row['final_eval'] and base['final_eval'] else None
                dT = (row['tok_s'] / base['tok_s'] - 1.0) * 100 if base['tok_s'] else None
                rows.append([name, fmt(row['final_eval'], 3), fmt(dE, 3, ' Δ'),
                             f"{row['tok_s']:.0f}", fmt(dT, 1, '% Δ'),
                             row['seeds']])
            else:
                rows.append([name, fmt(row['final_eval'], 3), '-',
                             f"{row['tok_s']:.0f}", '-', row['seeds']])
        a(md_table(['variant', 'eval', 'Δ vs base', 'tok/s', 'Δ tok/s',
                    'seeds'], rows) + '\n')

    if 'recall' in data:
        r = data['recall']
        fams = [f for f in r['rows'] if f in FAMILIES]
        a('\n## 5. Recall — memory-horizon diagnostic (V=%d, dim=%d, floor %.3f)\n'
          % (r['vocab'], r['dim'], math.log(r['vocab'])))
        a('*Diagnostic on synthetic structured associative recall (learnable); '
          'NOT a language benchmark. A rising span-CE vs k = memory collapse.*\n')
        rows = []
        for k in RECALL_KS:
            row = [k]
            for f in fams:
                v = r['rows'][f].get(k, {})
                row += [v.get('span_ce'), v.get('span_acc')]
            rows.append(row)
        hdr = ['k'] + [x for f in fams for x in
                       (f + ' span CE', f + ' acc')]
        a(md_table(hdr, rows) + '\n')

    if 'deploy' in data:
        a('\n## 6. Deploy — INT8 quantization @ %dM\n' % PRIMARY)
        rows = [[fam, f"{v['params']/1e6:.1f}", f"{v['mb_fp32']:.1f}",
                 f"{v['mb_int8']:.1f}", f"{v['size_saved_pct']:.0f}%",
                 fmt(v['eval_fp32'], 3), fmt(v['eval_int8'], 3),
                 fmt(v['eval_delta'], 3, ' Δ'), f"{v['cpu_prefill_tok_s_int8']:.0f}"]
                for fam, v in data['deploy'].items()]
        a(md_table(['family', 'params M', 'fp32 MB', 'int8 MB', 'saved',
                    'eval fp32', 'eval int8', 'Δ', 'CPU tok/s (int8)'], rows) + '\n')

    if 'perf' in data:
        a('\n## 7. Perf — inference battery @ %dM\n' % PRIMARY)
        rows = [[fam, f"{v['prefill_tok_s']:.0f}", f"{v['gen_tok_s']:.1f}",
                 f"{v['ms_per_token']:.2f}", fmt(v['infer_mem'], 2),
                 fmt(v.get('extrap_loss_2x'), 3)]
                for fam, v in data['perf'].items()]
        a(md_table(['family', 'prefill tok/s', 'gen tok/s (b=1)', 'ms/token',
                    'peak infer GB', 'eval @2x ctx'], rows) + '\n')

    if 'data2' in data:
        a('\n## 8. Cross-data — TinyShakespeare (real text)\n')
        rows = [[fam, fmt(v['final_eval'], 3), fmt(v['random_loss'], 2),
                 f"{v['tok_s']:.0f}"] for fam, v in data['data2'].items()]
        a(md_table(['family', 'eval loss', 'random', 'train tok/s'], rows) + '\n')

    a('\n---\n*Methodology: param-matched families, identical data/optimizer/'
      'schedule/token budget, multiple seeds, multiple sizes, real text only; '
      'pure-PyTorch speed is a lower bound. Full per-run data in '
      'overnight_results.json.*\n')
    return '\n'.join(L)


# ───────────────────────────── orchestration ─────────────────────────────
STATE_FILE = os.path.join(OUTDIR, 'overnight_state.json')
RESULTS_FILE = os.path.join(OUTDIR, 'overnight_results.json')
REPORT_FILE = os.path.join(OUTDIR, 'overnight_report.md')


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, encoding='utf-8') as f:
                st = json.load(f)
            return set(st.get('completed', [])), st.get('data', {})
        except Exception:
            pass
    return set(), {}


def _json_default(o):
    if callable(o):
        return repr(o)
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().item()
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError('not JSON serializable: %r' % o)


def save_state(completed, data):
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump({'completed': sorted(completed)}, f, indent=2)
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump({'config': {'mode': MODE, 'steps': STEPS, 'sizes': SIZES,
                              'seeds': SEEDS, 'phases': PHASES,
                              'data': DATA, 'vocab': VOCAB},
                   'env': _env_snapshot(),
                   'data': data}, f, indent=2, default=_json_default)


def _env_snapshot():
    return {
        'torch': torch.__version__, 'numpy': np.__version__,
        'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0) if HAS_CUDA else None,
        'gpu_mem_gb': round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        if HAS_CUDA else None,
        'python': sys.version.split()[0],
    }


PHASE_FNS = {
    'core': lambda d: phase_core(VOCAB, RANDOM_LOSS),
    'context': lambda d: phase_context(VOCAB),
    'efficiency': lambda d: phase_efficiency(VOCAB),
    'ablate': lambda d: phase_ablate(VOCAB),
    'recall': lambda d: phase_recall(),
    'deploy': lambda d: phase_deploy(VOCAB),
    'perf': lambda d: phase_perf(VOCAB),
    'data2': lambda d: phase_data2(),
}


def main():
    if not HAS_CUDA:
        print('[WARN] no CUDA — an overnight run is only meaningful on a Kaggle '
              'GPU. Results here are a smoke test only.')
    completed, data = load_state()

    print('=' * 72)
    print('  OVERNIGHT GRAND ARENA — final all-axes NFRA comparison')
    print('  NFRA Brain vs %s  (param-matched, REAL text)'
          % ' vs '.join(FAMILIES))
    print('=' * 72)
    print('  mode     : %-8s steps: %d   sizes: %sM   seeds: %s'
          % (MODE, STEPS, SIZES, SEEDS))
    print('  data     : WikiText-2 (char)   vocab: %d   random loss: %.3f'
          % (VOCAB, RANDOM_LOSS))
    print('  phases   : %s' % ','.join(PHASES))
    print('  device   : %s'
          % (torch.cuda.get_device_name(0) if HAS_CUDA else 'CPU')
          + ('  (fp16 AMP)' if USE_AMP else ''))
    print('  budget   : %.0f min   out: %s' % (MAX_MIN, OUTDIR))
    print('=' * 72)
    if completed:
        print('  resuming: completed %s' % sorted(completed))

    t_all = time.perf_counter()
    for phase in PHASES:
        if phase in completed:
            print('\n[skip] %s already completed' % phase)
            continue
        if phase not in PHASE_FNS:
            print('\n[skip] unknown phase %r' % phase)
            continue
        if BUDGET.left() <= 60:
            print('\n[budget] %.0f min left — stopping before %s'
                  % (BUDGET.left() / 60, phase))
            break
        try:
            result = PHASE_FNS[phase](data)
            data[phase] = result
            completed.add(phase)
            save_state(completed, data)
            with open(REPORT_FILE, 'w', encoding='utf-8') as f:
                f.write(build_report(data, _env_snapshot()))
            print('[phase %s done] report + results saved' % phase)
        except Exception as e:
            print('[PHASE %s FAILED] %r (continuing)' % (phase, e))
            traceback.print_exc()

    # ── phase CSVs
    try:
        if 'core' in data:
            rows = []
            for size in sorted(SIZES):
                for fam in FAMILIES:
                    m = data['core']['metrics'][size][fam]
                    rows.append([size, fam, m['params'], m['final_eval'],
                                 m['final_eval_sd'], m['ppl'], m['sample_auc'],
                                 m['tok_s_train'], m['peak_mem'],
                                 data['core']['scaling'][fam]['slope']])
            write_csv('core.csv',
                      ['size_M', 'family', 'params', 'eval_loss', 'eval_sd',
                       'ppl', 'sample_auc', 'train_tok_s', 'peak_mem_gb',
                       'scaling_slope'], rows)
        if 'context' in data:
            rows = [[fam, v['train_final'], v['lengths'].get(256),
                     v['lengths'].get(512), v['lengths'].get(1024)]
                    for fam, v in data['context'].items()]
            write_csv('context.csv',
                      ['family', 'train_final', 'eval_256', 'eval_512', 'eval_1024'],
                      rows)
        if 'efficiency' in data:
            rows = [[v['budget'], v['final_eval'], v['tok_s'], v['peak_mem']]
                    for v in data['efficiency'].values()]
            write_csv('efficiency.csv',
                      ['energy_budget', 'eval_loss', 'train_tok_s', 'peak_mem_gb'],
                      rows)
        if 'ablate' in data:
            rows = [[name, v['final_eval'], v['tok_s'], v['seeds']]
                    for name, v in data['ablate'].items()]
            write_csv('ablate.csv', ['variant', 'eval_loss', 'train_tok_s', 'seeds'],
                      rows)
        if 'recall' in data:
            fams = [f for f in data['recall']['rows'] if f in FAMILIES]
            rows = []
            for k in RECALL_KS:
                row = [k]
                for f in fams:
                    row.append(data['recall']['rows'][f].get(k, {}).get('span_ce'))
                rows.append(row)
            write_csv('recall.csv', ['k'] + [f + '_span_ce' for f in fams], rows)
        if 'deploy' in data:
            rows = [[fam, v['mb_fp32'], v['mb_int8'], v['size_saved_pct'],
                     v['eval_fp32'], v['eval_int8'], v['cpu_prefill_tok_s_int8']]
                    for fam, v in data['deploy'].items()]
            write_csv('deploy.csv',
                      ['family', 'mb_fp32', 'mb_int8', 'size_saved_pct',
                       'eval_fp32', 'eval_int8', 'cpu_prefill_tok_s_int8'], rows)
        if 'perf' in data:
            rows = [[fam, v['prefill_tok_s'], v['gen_tok_s'], v['ms_per_token'],
                     v['infer_mem'], v.get('extrap_loss_2x')]
                    for fam, v in data['perf'].items()]
            write_csv('perf.csv',
                      ['family', 'prefill_tok_s', 'gen_tok_s', 'ms_per_token',
                       'infer_mem_gb', 'extrap_2x'], rows)
        if 'data2' in data:
            rows = [[fam, v['final_eval'], v['random_loss'], v['tok_s']]
                    for fam, v in data['data2'].items()]
            write_csv('data2.csv', ['family', 'eval_loss', 'random_loss', 'tok_s'],
                      rows)
        print('\n[out] CSVs written to %s' % OUTDIR)
    except Exception as e:
        print('[csv] error: %r' % e)

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(build_report(data, _env_snapshot()))

    print('\n' + '=' * 72)
    print('  OVERNIGHT GRAND ARENA DONE in %.0fs (%.0f min)'
          % (time.perf_counter() - t_all, BUDGET.used_min()))
    print('  -> %s' % REPORT_FILE)
    print('  -> %s' % RESULTS_FILE)
    print('  -> %s (resume state)' % STATE_FILE)
    print('=' * 72)


if __name__ == '__main__':
    main()
