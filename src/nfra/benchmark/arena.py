"""
╔══════════════════════════════════════════════════════════════════════════╗
║   NFRA ARENA — a global-standard, multi-dimension benchmark               ║
║                                                                           ║
║   Asks, and answers with evidence:                                        ║
║     • WHO is better ON WHICH aspect (quality, speed, memory, scaling,     ║
║       robustness, sample-efficiency, parameter-efficiency, latency)?      ║
║     • Is NFRA Brain really "revolutionary" — or a niche memory/speed      ║
║       specialist — vs Mamba-SSM and GPT-2 at matched params?              ║
║                                                                           ║
║   METHODOLOGY (credibility controls)                                      ║
║     • Param-matched models, identical data, identical optimizer+schedule. ║
║     • Multiple seeds → every quality metric reported as mean ± std.       ║
║     • Multiple model sizes → measured scaling slope (loss per doubling    ║
║       of params) + power-law extrapolation.                               ║
║     • Matched token budgets across families at every size.                ║
║     • Hardware metrics (tok/s, peak GB) measured on identical shapes.     ║
║     • Full config + raw per-seed data dumped to JSON.                     ║
║                                                                           ║
║   ENV                                                                     ║
║     NFRA_MODE       quick(150) | standard(600) | rigorous(1500) steps     ║
║     NFRA_SIZES      comma list of target sizes in M (default "5,20")      ║
║     NFRA_SEEDS      number of seeds (default 2, rigorous 3)               ║
║     NFRA_FAMILIES   comma list: nfra,mamba,gpt2 (default all)             ║
║     NFRA_DATA       synthetic | wikitext2                                 ║
║     NFRA_BATCH      override training batch size                          ║
║     NFRA_BANDS      NFRA Brain band count (H8 ablation: 2,4,8,16)         ║
║     NFRA_SCAN_KERNEL 0=torch, 1=auto Triton kernel, 2=force              ║
║                                                                           ║
║   Usage:  python -m nfra.benchmark.arena     (Kaggle T4 recommended)      ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import os, sys, time, math, json, warnings
import functools
print = functools.partial(print, flush=True)
warnings.filterwarnings('ignore')

import numpy as np
import torch
from torch.utils.data import DataLoader

from nfra import NFRAConfig, NFRAForCausalLM
from nfra.benchmark.compare import (
    MambaLM, GPT2ForCausalLM, HierarchicalDataset, WikiText2Dataset,
    count_params, rescale_embed, compute_loss, evaluate, make_optimizer,
    EMA, DEVICE, HAS_CUDA, USE_AMP, BATCH, D_STATE, SEQ_LEN, DATA_SOURCE,
    NFRA_DEPTH, WIKI_PATHS, CHAR_VOCAB,
)

# ─────────────────────────── config ───────────────────────────
MODE = os.environ.get('NFRA_MODE', 'standard')
STEP_CFG = {'quick': 150, 'standard': 600, 'rigorous': 1500}
STEPS = int(os.environ.get('NFRA_STEPS', STEP_CFG[MODE]))
if not HAS_CUDA:
    STEPS = min(STEPS, 80)

SIZES = [int(x) for x in os.environ.get('NFRA_SIZES', '5,20').split(',') if x.strip()]
if not SIZES:
    SIZES = [5, 20]
SEED_CNT = int(os.environ.get('NFRA_SEEDS', '3' if MODE == 'rigorous' else '2'))
SEED_LIST = [42, 7, 2026, 1337, 777][:SEED_CNT]
FAMILIES = [f.strip().lower() for f in
            os.environ.get('NFRA_FAMILIES', 'nfra,mamba,gpt2').split(',') if f.strip()]
# NFRA 3.2 feature toggles. EMA + surprise-weighted loss apply to ALL families
# (fair head-to-head); k-WTA is an NFRA architecture change only.
EMA_DECAY = float(os.environ.get('NFRA_EMA', '0'))          # 0 = off
SURPRISE = os.environ.get('NFRA_SURPRISE', '0') == '1'      # 1 = on
KWTA = float(os.environ.get('NFRA_KWTA', '0'))              # 0.0 = off
BANDS = int(os.environ.get('NFRA_BANDS', '16'))     # H8 band-count ablation knob
# Gradient checkpointing trades compute for memory; on a big GPU with a small
# model the recompute is pure overhead -> set 0 to raise tok/s.
CHECKPOINT = os.environ.get('NFRA_CHECKPOINT', '1') == '1'
EVAL_GAP = max(50, STEPS // 6)
EXT_FACTOR = 2                      # extrapolation test: eval at SEQ_LEN * EXT_FACTOR
GEN_LEN = 16
PROMPT_LEN = 64
PRE_HEAD = max(BATCH, 8)            # prefill throughput batch

DIM_GRID = {
    5:  [256, 224, 192, 160, 128],
    20: [512, 448, 384, 352, 320, 288, 256],
    50: [768, 704, 640, 576, 512],
}

# composite scoring: (metric key, direction, weight). direction +1 = higher better.
METRIC_SPEC = [
    dict(key='final_eval',   dir=-1, w=0.30, label='eval loss'),
    dict(key='sample_auc',   dir=-1, w=0.10, label='sample-efficiency AUC'),
    dict(key='param_eff',    dir=+1, w=0.10, label='loss-gain / M params'),
    dict(key='tok_s_train',  dir=+1, w=0.10, label='train tok/s'),
    dict(key='peak_mem',     dir=-1, w=0.10, label='peak train memory'),
    dict(key='gen_tok_s',    dir=+1, w=0.05, label='generation tok/s'),
    dict(key='infer_mem',    dir=-1, w=0.05, label='peak infer memory'),
    dict(key='extrap_delta', dir=-1, w=0.10, label='long-context extrapolation'),
    dict(key='scaling_gain', dir=+1, w=0.10, label='scaling gain / doubling'),
]


# ─────────────────────────── builders ───────────────────────────
def build_nfra(vocab, dim, unique_blocks, depth=NFRA_DEPTH, k_wta=None):
    if k_wta is None:
        k_wta = KWTA
    cfg = NFRAConfig(mode='brain', vocab_size=vocab, hidden_size=dim,
                     num_layers=depth, n_bands=BANDS, dropout=0.1,
                     depth_shared=True, unique_blocks=unique_blocks,
                     gradient_checkpointing=CHECKPOINT, k_wta_frac=k_wta)
    return NFRAForCausalLM(cfg)


def build_mamba(vocab, dim, n_layers, d_state=D_STATE):
    return MambaLM(vocab, dim, n_layers, d_state)


def build_gpt2(vocab, dim, n_layers, n_heads=8, pos_len=2048):
    return GPT2ForCausalLM(vocab, dim, n_layers, n_heads, pos_len=pos_len)


def tune_nfra_size(target, vocab, depth, dims):
    best = (float('inf'), None, None, None)
    for U in range(2, min(depth, 8) + 1):
        if depth % U:
            continue
        for d in dims:
            p = count_params(build_nfra(vocab, d, U, depth))
            err = abs(p - target)
            if err < best[0]:
                best = (err, U, d, p)
    if best[1] is None:
        return 1, dims[-1], count_params(build_nfra(vocab, dims[-1], 1, depth))
    return best[1], best[2], best[3]


def tune_layers_size(builder, target, vocab, dims, max_layers=40):
    best = (float('inf'), None, None, None)
    for d in dims:
        for L in range(1, max_layers):
            p = count_params(builder(vocab, d, L))
            err = abs(p - target)
            if err < best[0]:
                best = (err, d, L, p)
            if p > target * 1.2:
                break
    if best[1] is None:
        return dims[-1], 1, count_params(builder(vocab, dims[-1], 1))
    return best[1], best[2], best[3]


def build_family_spec(family, size, vocab):
    target = size * 1_000_000
    keys = sorted(DIM_GRID)
    dims = next((DIM_GRID[k] for k in keys if size <= k), DIM_GRID[keys[-1]])
    if family == 'nfra':
        U, dim, params = tune_nfra_size(target, vocab, NFRA_DEPTH, dims)
        spec = dict(builder=build_nfra, dim=dim, extra=dict(unique_blocks=U,
                                                            depth=NFRA_DEPTH),
                    params=params, depth=NFRA_DEPTH)
    elif family == 'mamba':
        dim, L, params = tune_layers_size(build_mamba, target, vocab, dims)
        spec = dict(builder=build_mamba, dim=dim,
                    extra=dict(n_layers=L, d_state=D_STATE),
                    params=params, depth=L)
    else:
        dim, L, params = tune_layers_size(build_gpt2, target, vocab, dims)
        spec = dict(builder=build_gpt2, dim=dim,
                    extra=dict(n_layers=L, n_heads=8, pos_len=2048),
                    params=params, depth=L)
    return spec


# ─────────────────────────── training ───────────────────────────
def train_one(model, vocab, steps, train_loader, eval_loader, eval_gap,
              ema_decay=0.0, surprise=False):
    model.train()
    opt, sched = make_optimizer(model, lr=3e-4,
                                warmup=min(50, max(steps // 10, 1)), total=steps)
    scaler = torch.amp.GradScaler(str(DEVICE)) if USE_AMP else None
    ema = EMA(model, ema_decay) if ema_decay > 0 else None
    it = iter(train_loader)
    if HAS_CUDA:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    loss_hist, eval_hist = [], []
    nan_steps = 0
    t_start = time.perf_counter()
    for step in range(1, steps + 1):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y = next(it)
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            loss = compute_loss(model, x, y, surprise=surprise)
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not math.isfinite(gnorm):
                opt.zero_grad(set_to_none=True)
                nan_steps += 1
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if math.isfinite(gnorm):
                opt.step()
            else:
                opt.zero_grad(set_to_none=True)
                nan_steps += 1
        sched.step()
        if ema is not None:
            ema.update(model)
        # Defer the loss materialization: reading a CUDA scalar syncs the
        # device, which would stall the pipeline every step. Collect detached
        # tensors now, flatten to floats once after the loop's single sync.
        loss_hist.append(loss.detach())
        if step % eval_gap == 0 or step == steps:
            if ema is not None:
                ema.apply(model)
            eval_hist.append((step, evaluate(model, eval_loader)))
            if ema is not None:
                ema.restore(model)
    if ema is not None:
        ema.apply(model)   # leave EMA weights in place for downstream evals
    if HAS_CUDA:
        torch.cuda.synchronize()   # single sync: drain queued work once
    wall = time.perf_counter() - t_start
    loss_hist = [float(v) for v in loss_hist]
    mem = torch.cuda.max_memory_allocated() / 1e9 if HAS_CUDA else 0.0
    bs = getattr(train_loader, 'batch_size', 1)
    seq = SEQ_LEN
    return {
        'loss_hist': loss_hist, 'eval_hist': eval_hist,
        'tok_s': bs * seq * steps / max(wall, 1e-6),
        'ms_per_step': wall * 1000.0 / steps,
        'peak_mem': mem, 'nan_steps': nan_steps, 'wall_s': wall,
    }


# ─────────────────────────── inference battery ───────────────────────────
@torch.no_grad()
def prefill_tok_s(model, batch, seq_len, vocab):
    model.eval()
    x = torch.randint(0, vocab, (batch, seq_len), device=DEVICE)
    for _ in range(2):
        model(x)
    if HAS_CUDA:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    model(x)
    if HAS_CUDA:
        torch.cuda.synchronize()
    return batch * seq_len / max(time.perf_counter() - t0, 1e-6)


@torch.no_grad()
def generate_metrics(model, vocab, prompt_len=PROMPT_LEN, gen_len=GEN_LEN):
    model.eval()
    ids = torch.randint(0, vocab, (1, prompt_len), device=DEVICE)
    model(ids)
    if HAS_CUDA:
        torch.cuda.synchronize()
    if HAS_CUDA:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(gen_len):
        logits = model(ids)['logits']
        nxt = logits[:, -1, :].argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
    if HAS_CUDA:
        torch.cuda.synchronize()
    dt = max(time.perf_counter() - t0, 1e-6)
    mem = torch.cuda.max_memory_allocated() / 1e9 if HAS_CUDA else 0.0
    return {'gen_tok_s': gen_len / dt, 'ms_per_token': dt / gen_len * 1000,
            'infer_mem': mem}


# ─────────────────────────── metrics / stats ───────────────────────────
def mean_std(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
    return m, sd


def sample_auc(eval_hist):
    if len(eval_hist) < 2:
        return None
    steps = [s for s, _ in eval_hist]
    losses = [l for _, l in eval_hist]
    span = max(steps[-1] - steps[0], 1)
    xs = [(s - steps[0]) / span for s in steps]
    return sum((a + b) / 2 * (xb - xa)
               for (xa, a), (xb, b) in zip(zip(xs, losses), zip(xs[1:], losses[1:])))


def fit_scaling(points):
    """points: [(params, eval_loss)]. Slope = bits of loss per doubling of
    params (negative = improving). OLS on log2(params) vs loss."""
    pts = sorted((float(p), float(l)) for p, l in points)
    if len(pts) < 2:
        return {'slope': 0.0, 'r2': None, 'loss_100m': None, 'loss_1b': None, 'n': len(pts)}
    xs = [math.log2(p) for p, _ in pts]
    ys = [l for _, l in pts]
    n = len(pts)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (my + slope * (x - mx))) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot else None
    pred = lambda p: my + slope * (math.log2(p) - mx)
    return {'slope': slope, 'r2': r2, 'loss_100m': pred(100e6),
            'loss_1b': pred(1e9), 'n': n}


def zscores(vals, direction):
    a = [v if (v is not None and math.isfinite(v)) else None for v in vals]
    good = [v for v in a if v is not None]
    if len(good) < 2:
        return [0.0] * len(a)
    mean = sum(good) / len(good)
    sd = (sum((v - mean) ** 2 for v in good) / max(len(good) - 1, 1)) ** 0.5
    if sd == 0:
        return [0.0] * len(a)
    return [((v - mean) / sd * direction) if v is not None else 0.0 for v in a]


def composite_score(metric_rows, spec):
    fams = list(metric_rows)
    scores = {f: 0.0 for f in fams}
    for s in spec:
        key, direction, w = s['key'], s['dir'], s['w']
        z = zscores([metric_rows[f].get(key) for f in fams], direction)
        for f, zi in zip(fams, z):
            scores[f] += w * zi
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return {f: 50.0 for f in fams}
    return {f: 100.0 * (v - lo) / (hi - lo) for f, v in scores.items()}


# ─────────────────────────── data ───────────────────────────
def make_loaders(size_idx):
    """Shared train loaders per (size, seed); eval + extrapolation per size.
    All families see byte-identical batches for a given (size, seed)."""
    use_wiki = DATA_SOURCE == 'wikitext2'
    wiki_ds = WikiText2Dataset('train', SEQ_LEN + 1) if use_wiki else None
    train_loaders = {}
    for si, seed in enumerate(SEED_LIST):
        gen = torch.Generator().manual_seed(1000 + size_idx * 100 + si * 10)
        if use_wiki:
            ds = wiki_ds
        else:
            ds = HierarchicalDataset(max(512, BATCH * 12), SEQ_LEN + 1,
                                     seed=100 + size_idx, seq_seed=200 + size_idx * 10 + si)
        train_loaders[seed] = DataLoader(ds, batch_size=BATCH, shuffle=True,
                                         generator=gen, num_workers=0)
    if use_wiki:
        eval_ds = WikiText2Dataset('validation', SEQ_LEN + 1)
        ext_ds = WikiText2Dataset('validation', SEQ_LEN * EXT_FACTOR + 1)
    else:
        eval_ds = HierarchicalDataset(max(256, BATCH * 6), SEQ_LEN + 1,
                                      seed=100 + size_idx, seq_seed=300 + size_idx)
        ext_ds = HierarchicalDataset(max(256, BATCH * 6), SEQ_LEN * EXT_FACTOR + 1,
                                     seed=100 + size_idx, seq_seed=400 + size_idx)
    eval_loader = DataLoader(eval_ds, batch_size=BATCH, shuffle=False, num_workers=0)
    ext_loader = DataLoader(ext_ds, batch_size=BATCH, shuffle=False, num_workers=0)
    return train_loaders, eval_loader, ext_loader


# ─────────────────────────── report ───────────────────────────
def fmt(v, nd=3, suffix=''):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return '-'
    return f"{v:.{nd}f}{suffix}"


def fmt_ms(v, sd, nd=3):
    if v is None:
        return '-'
    if sd and sd > 0:
        return f"{v:.{nd}f} ± {sd:.{nd}f}"
    return f"{v:.{nd}f}"


def md_table(headers, rows):
    out = ['| ' + ' | '.join(headers) + ' |',
           '|' + '|'.join('---' for _ in headers) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(str(c) for c in r) + ' |')
    return '\n'.join(out)


def winner_of(metric_rows, key, direction):
    best, bfam = None, None
    for fam, row in metric_rows.items():
        v = row.get(key)
        if v is None or not math.isfinite(v):
            continue
        if best is None or (v - best) * direction > 0:
            best, bfam = v, fam
    return bfam


# ─────────────────────────── main ───────────────────────────
def main():
    use_wiki = DATA_SOURCE == 'wikitext2'
    VOCAB = 96 if use_wiki else 4096
    RANDOM_LOSS = math.log(VOCAB)
    label = 'WikiText-2 (char)' if use_wiki else 'Synthetic hierarchical'

    t_all = time.perf_counter()
    print("=" * 72)
    print("  NFRA ARENA - global-standard multi-dimension comparison")
    print("  NFRA Brain  vs  Mamba-SSM  vs  GPT-2   (param-matched)")
    print("=" * 72)
    print(f"  mode     : {MODE} ({STEPS} steps)   data: {label}   vocab: {VOCAB}")
    print(f"  sizes    : {SIZES} M params    seeds: {SEED_LIST}")
    print(f"  families : {FAMILIES}    batch: {BATCH}    seq: {SEQ_LEN}")
    print(f"  device   : {'GPU ' + torch.cuda.get_device_name(0) if HAS_CUDA else 'CPU'}"
          + (f"   (fp16 AMP)" if USE_AMP else ""))
    feats = []
    if BANDS != 16:
        feats.append(f"bands={BANDS}")
    if EMA_DECAY > 0:
        feats.append(f"EMA={EMA_DECAY}")
    if SURPRISE:
        feats.append("surprise-weighted loss")
    if KWTA > 0:
        feats.append(f"k-WTA={KWTA}")
    if feats:
        print(f"  features : {', '.join(feats)}")
    print("=" * 72)

    env = {
        'torch': torch.__version__,
        'numpy': np.__version__,
        'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0) if HAS_CUDA else None,
        'gpu_mem_gb': round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        if HAS_CUDA else None,
        'python': sys.version.split()[0],
        'ema': EMA_DECAY,
        'surprise': int(SURPRISE),
        'kwta': KWTA,
    }

    # ── build specs per size
    specs = {}
    for size in SIZES:
        specs[size] = {f: build_family_spec(f, size, VOCAB) for f in FAMILIES}
        for f, s in specs[size].items():
            print(f"  [ok] {f:<6s} @ {size}M: dim {s['dim']:<4d} {s['params']/1e6:6.2f}M "
                  f"depth {s['depth']}  {s['extra']}")

    # ── run every (size × seed × family)
    print("\n[train] running %d training runs (%d sizes x %d seeds x %d families)..."
          % (len(SIZES) * SEED_CNT * len(FAMILIES), len(SIZES), SEED_CNT, len(FAMILIES)))
    runs = {}       # runs[size][seed][family] = train record
    battery = {}    # battery[size][family] = extrap + inference battery
    for size in SIZES:
        train_loaders, eval_loader, ext_loader = make_loaders(SIZES.index(size))
        runs[size] = {}
        for seed in SEED_LIST:
            runs[size][seed] = {}
            torch.manual_seed(seed)
            np.random.seed(seed)
            for family in FAMILIES:
                spec = specs[size][family]
                model = spec['builder'](VOCAB, spec['dim'], **spec['extra']).to(DEVICE)
                rescale_embed(model)
                rec = train_one(model, VOCAB, STEPS, train_loaders[seed],
                                eval_loader, EVAL_GAP,
                                ema_decay=EMA_DECAY, surprise=SURPRISE)
                runs[size][seed][family] = rec
                if seed == SEED_LIST[-1]:
                    ext = evaluate(model, ext_loader, max_batches=6)
                    b = {'extrap_loss': ext,
                         'extrap_delta': ext - rec['eval_hist'][-1][1]}
                    if size == SIZES[-1]:
                        pre = prefill_tok_s(model, PRE_HEAD, SEQ_LEN, VOCAB)
                        gen = generate_metrics(model, VOCAB)
                        b.update({'prefill_tok_s': pre, 'gen_tok_s': gen['gen_tok_s'],
                                  'ms_per_token': gen['ms_per_token'],
                                  'infer_mem': gen['infer_mem']})
                    battery.setdefault(size, {})[family] = b
                print(f"  [ok] {family:<6s} @ {size}M seed {seed:<4d} "
                      f"final {rec['eval_hist'][-1][1]:.3f}  "
                      f"{rec['tok_s']:.0f} tok/s  {rec['peak_mem']:.2f} GB")

    # ── aggregate metrics per (size, family)
    metrics = {}
    for size in SIZES:
        metrics[size] = {}
        for family in FAMILIES:
            recs = [runs[size][s][family] for s in SEED_LIST]
            finals = [r['eval_hist'][-1][1] for r in recs]
            m_final, sd_final = mean_std(finals)
            m_auc, _ = mean_std([sample_auc(r['eval_hist']) for r in recs])
            m_tok, _ = mean_std([r['tok_s'] for r in recs])
            m_ms, _ = mean_std([r['ms_per_step'] for r in recs])
            m_mem, _ = mean_std([r['peak_mem'] for r in recs])
            params = specs[size][family]['params']
            row = {
                'params': params,
                'depth': specs[size][family]['depth'],
                'spec': specs[size][family],
                'final_eval': m_final, 'final_eval_sd': sd_final,
                'final_eval_n': len(finals),
                'ppl': math.exp(min(m_final, 30)) if m_final else None,
                'sample_auc': m_auc,
                'tok_s_train': m_tok, 'ms_per_step': m_ms, 'peak_mem': m_mem,
                'nan_steps': sum(r['nan_steps'] for r in recs),
                'wall_s': sum(r['wall_s'] for r in recs),
                'param_eff': ((RANDOM_LOSS - m_final) / (params / 1e6)) if m_final else None,
                'est_flops_token': 6 * params,
            }
            if family in battery.get(size, {}):
                for k in ('extrap_loss', 'extrap_delta', 'prefill_tok_s',
                          'gen_tok_s', 'ms_per_token', 'infer_mem'):
                    if k in battery[size][family]:
                        row[k] = battery[size][family][k]
            metrics[size][family] = row

    # ── scaling fit per family (across sizes)
    scaling = {}
    for family in FAMILIES:
        pts = [(metrics[s][family]['params'], metrics[s][family]['final_eval'])
               for s in SIZES if metrics[s][family]['final_eval']]
        scaling[family] = fit_scaling(pts)
    for family in FAMILIES:
        gain = -scaling[family]['slope']
        for size in SIZES:
            metrics[size][family]['scaling_gain'] = gain

    # ── composite scores per size
    scores = {}
    for size in SIZES:
        scores[size] = composite_score(metrics[size], METRIC_SPEC)

    # ── verdict
    primary = max(SIZES)
    verdict = make_verdict(metrics, scores, scaling, primary, RANDOM_LOSS)

    # ── write outputs
    data = {
        'config': {'mode': MODE, 'steps': STEPS, 'sizes': SIZES, 'seeds': SEED_LIST,
                   'families': FAMILIES, 'data': DATA_SOURCE, 'vocab': VOCAB,
                   'batch': BATCH, 'seq_len': SEQ_LEN,
                   'optimizer': 'AdamW(lr=3e-4,beta=(0.9,0.95))',
                   'schedule': 'warmup+cosine', 'seed_list': SEED_LIST,
                   'ext_factor': EXT_FACTOR, 'metric_weights': [
                       {k: s[k] for k in ('key', 'dir', 'w')} for s in METRIC_SPEC]},
        'env': env,
        'specs': {str(s): {f: {k: v for k, v in specs[s][f].items()
                               if k != 'builder'} for f in FAMILIES} for s in SIZES},
        'metrics': {str(s): {f: {k: v for k, v in metrics[s][f].items()
                                 if k != 'spec' and not callable(v)}
                             for f in FAMILIES} for s in SIZES},
        'scaling': {f: scaling[f] for f in FAMILIES},
        'scores': {str(s): scores[s] for s in SIZES},
        'history': {str(s): {sd: {f: {'loss': runs[s][sd][f]['loss_hist'],
                                      'eval': runs[s][sd][f]['eval_hist']}
                                   for f in FAMILIES} for sd in SEED_LIST}
                    for s in SIZES},
        'verdict': verdict,
    }
    out_json = os.path.join(os.getcwd(), 'nfra_arena_results.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    report = build_report(metrics, scores, scaling, verdict, primary, RANDOM_LOSS,
                          label, env, data['config'])
    out_md = os.path.join(os.getcwd(), 'nfra_arena_report.md')
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n  done in {time.perf_counter() - t_all:.0f}s")
    print(f"  -> {out_json}")
    print(f"  -> {out_md}")


def make_verdict(metrics, scores, scaling, primary, random_loss):
    fams = list(metrics[primary])
    m = metrics[primary]
    by_loss = sorted(fams, key=lambda f: m[f]['final_eval'])
    best_q = by_loss[0]
    second_q = by_loss[1] if len(by_loss) > 1 else None
    q_delta = (m[second_q]['final_eval'] - m[best_q]['final_eval']) if second_q else 0.0
    best_mem = min(fams, key=lambda f: m[f]['peak_mem'])
    best_speed = max(fams, key=lambda f: m[f]['tok_s_train'])
    best_scale = max(fams, key=lambda f: -scaling[f]['slope'])
    best_score = max(fams, key=lambda f: scores[primary][f])
    worst_score = min(fams, key=lambda f: scores[primary][f])

    claims = [
        dict(claim='lowest eval loss (best quality)', family=best_q,
             status='confirmed' if q_delta >= 0.02 else 'marginal',
             evidence=f"Δ{abs(q_delta):.3f} vs runner-up at {primary}M"),
        dict(claim='most parameter-efficient (loss-gain per M param)',
             family=max(fams, key=lambda f: m[f]['param_eff']), status='confirmed',
             evidence=None),
        dict(claim='most sample-efficient (lowest learning-curve AUC)',
             family=min(fams, key=lambda f: m[f]['sample_auc']), status='confirmed',
             evidence=None),
        dict(claim='lowest peak training memory',
             family=best_mem, status='confirmed',
             evidence=f"{m[best_mem]['peak_mem']:.2f} GB vs "
                      f"{m[max(fams, key=lambda f: m[f]['peak_mem'])]['peak_mem']:.2f} GB"),
        dict(claim='fastest training throughput',
             family=best_speed, status='confirmed',
             evidence=f"{m[best_speed]['tok_s_train']:.0f} tok/s"),
        dict(claim='best scaling slope (loss per doubling of params)',
             family=best_scale, status='confirmed',
             evidence=f"{scaling[best_scale]['slope']:.3f} bits/doubling"),
        dict(claim='most robust to longer contexts (2× extrapolation)',
             family=min(fams, key=lambda f: m[f].get('extrap_delta', 0.0)),
             status='confirmed' if 'extrap_delta' in m[best_q] else 'unmeasured',
             evidence=None),
    ]
    for c in claims:
        if c['family'] not in fams:
            c['status'] = 'skipped'

    # revolution assessment
    nfra = 'nfra' if 'nfra' in fams else None
    revo = {
        'overall_leader': best_score,
        'overall_score': round(scores[primary][best_score], 1),
        'worst_score': round(scores[primary][worst_score], 1),
    }
    if nfra:
        revo['nfra_quality_rank'] = '1st' if best_q == nfra else \
            '2nd' if second_q == nfra else 'last'
        revo['nfra_quality_gap_to_best'] = round(
            m[nfra]['final_eval'] - m[best_q]['final_eval'], 3)
        revo['nfra_mem_gb'] = round(m[nfra]['peak_mem'], 2)
        revo['nfra_speed_tok_s'] = round(m[nfra]['tok_s_train'])
        revo['nfra_is_overall_winner'] = best_score == nfra
        revo['nfra_wins_quality'] = best_q == nfra
        revo['revolutionary'] = best_score == nfra
    return {'primary_size': primary, 'claims': claims, 'revo': revo,
            'winners': {'quality': best_q, 'memory': best_mem, 'speed': best_speed,
                        'scaling': best_scale, 'overall': best_score}}


def build_report(metrics, scores, scaling, verdict, primary, random_loss,
                 label, env, cfg):
    fams = list(metrics[primary])
    L = []
    a = L.append
    a(f"# NFRA ARENA — multi-dimension benchmark report\n")
    a(f"**Families:** {', '.join(fams)}  |  **Data:** {label}  |  "
      f"**Vocab:** {cfg['vocab']}  |  **Steps:** {cfg['steps']}  |  "
      f"**Batch:** {cfg['batch']}  |  **Seq:** {cfg['seq_len']}\n")
    a(f"**Sizes:** {cfg['sizes']}M params  |  **Seeds:** {cfg['seeds']}  |  "
      f"**Random-guess loss:** {random_loss:.3f} (ln vocab)\n")
    a(f"**Optimizer:** {cfg['optimizer']}  **Schedule:** {cfg['schedule']}\n")
    a("**Environment:** " + ", ".join(f"{k}={v}" for k, v in env.items() if v) + "\n")

    a("\n## 1. Models (param-matched builds)\n")
    rows = []
    for size in sorted(cfg['sizes']):
        for f in fams:
            s = metrics[size][f]['spec']
            rows.append([f"{size}M", f, s['dim'], s['params'] / 1e6,
                         metrics[size][f]['depth'],
                         s['extra'].get('unique_blocks', '-')])
    a(md_table(['size', 'family', 'dim', 'params (M)', 'depth',
                'unique blocks (NFRA)'], rows))
    a("")

    a("\n## 2. Scaling behaviour (quality vs params)\n")
    a("OLS fit of eval loss vs log2(params). **Slope** = bits of loss gained per "
      "doubling of params (more negative = better scaling).\n")
    rows = []
    for f in fams:
        sc = scaling[f]
        rows.append([f, fmt(sc['slope'], 4), fmt(sc.get('r2'), 3),
                     fmt(sc.get('loss_100m'), 3), fmt(sc.get('loss_1b'), 3),
                     sc.get('n', 0)])
    a(md_table(['family', 'slope (bits/doubling)', 'R²', 'extrap. loss @100M',
                '@1B', 'points'], rows))
    a("")

    sec = 3
    for size in sorted(cfg['sizes']):
        m = metrics[size]
        a(f"\n## {sec}. Head-to-head @ {size}M params\n")
        rows = []
        for f in fams:
            row = m[f]
            rows.append([f, fmt(row['final_eval'], 3),
                         fmt_ms(row['final_eval'], row['final_eval_sd'], 3),
                         fmt(row['ppl'], 2),
                         fmt(row['sample_auc'], 3),
                         f"{row['tok_s_train']:.0f}",
                         f"{row['ms_per_step']:.1f}",
                         fmt(row['peak_mem'], 2),
                         fmt(row['nan_steps'], 0)])
        a(md_table(['family', 'eval loss', 'mean ± std', 'ppl', 'AUC', 'train tok/s',
                    'ms/step', 'peak mem GB', 'NaN steps'], rows))
        a("")
        # inference battery only on the primary size
        if size == primary:
            rows = []
            for f in fams:
                row = m[f]
                if not all(k in row for k in ('prefill_tok_s', 'gen_tok_s',
                                              'ms_per_token', 'infer_mem',
                                              'extrap_loss', 'extrap_delta')):
                    continue
                rows.append([f, f"{row['prefill_tok_s']:.0f}",
                             f"{row['gen_tok_s']:.1f}",
                             f"{row['ms_per_token']:.2f}",
                             fmt(row['infer_mem'], 2),
                             fmt(row['extrap_loss'], 3),
                             fmt(row['extrap_delta'], 3)])
            a("**Inference battery + long-context extrapolation (primary size):**\n")
            a(md_table(['family', 'prefill tok/s', 'gen tok/s (b=1)', 'ms/token',
                        'peak infer GB', 'eval @2×ctx', 'Δ vs @1×ctx'], rows))
            a("")
        # per-aspect winners
        rows = []
        for s in METRIC_SPEC:
            w = winner_of(m, s['key'], s['dir'])
            if w is None:
                rows.append([s['label'], '-', '-'])
                continue
            val = m[w].get(s['key'])
            vstr = fmt(val, 3) if isinstance(val, float) else val
            rows.append([s['label'], w, vstr])
        a(f"**Who wins which aspect @ {size}M:**\n")
        a(md_table(['aspect', 'winner', 'value'], rows))
        a("")
        sec += 1

    a(f"\n## {sec}. Composite scores @ {primary}M (weighted z-scores, 0-100)\n")
    a("Weights: " + ", ".join(f"{s['label']}={s['w']:.2f}" for s in METRIC_SPEC) + "\n")
    rows = [[f, f"{scores[primary][f]:.1f}"] for f in fams]
    a(md_table(['family', 'score'], rows))
    a("")
    sec += 1

    a(f"\n## {sec}. Verdict\n")
    for c in verdict['claims']:
        ev = f" — {c['evidence']}" if c.get('evidence') else ""
        a(f"- **{c['claim']}:** {c['family']} ({c['status']}){ev}")
    a("")
    r = verdict['revo']
    a(f"- **Overall leader:** {r['overall_leader']} "
      f"(score {r['overall_score']:.1f} vs {r['worst_score']:.1f})\n")
    if 'nfra' in fams:
        a(f"- **NFRA Brain quality rank:** {r['nfra_quality_rank']} "
          f"(gap to best {r['nfra_quality_gap_to_best']:+.3f})")
        a(f"- **NFRA memory:** {r['nfra_mem_gb']} GB | **NFRA throughput:** "
          f"{r['nfra_speed_tok_s']} tok/s")
        a(f"- **'Revolutionary' verdict:** {'CONFIRMED' if r['revolutionary'] else 'NOT confirmed'}"
          f" — NFRA {'is' if r['revolutionary'] else 'is NOT'} the overall "
          f"winner at matched params, matched data, matched compute.\n")

    a("\n---\n")
    a("*Methodology: identical optimizer + schedule + token budget per family; "
      "multiple seeds (mean ± std); multiple sizes for measured scaling; "
      "pure-PyTorch implementations (no fused kernels — speed is a lower bound "
      "for Mamba/NFRA). Full per-seed data in nfra_arena_results.json.*")
    return '\n'.join(L)


if __name__ == '__main__':
    main()
