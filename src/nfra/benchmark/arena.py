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
║     NFRA_FAMILIES   comma list: nfra,rwkv,retnet,mamba,gpt2 (default all    ║
║     NFRA_DATA       synthetic | wikitext2                                 ║
║     NFRA_BATCH      override training batch size                          ║
║     NFRA_BANDS      NFRA Brain band count (H8 ablation: 2,4,8,16)         ║
║     NFRA_SCAN_KERNEL 0=torch, 1=auto Triton kernel, 2=force              ║
║                                                                           ║
║   Usage:  python -m nfra.benchmark.arena     (Kaggle T4 recommended)      ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import functools
import json
import math
import os
import sys
import time
import warnings

print = functools.partial(print, flush=True)
warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch.utils.data import DataLoader

from nfra import NFRAConfig, NFRAForCausalLM
from nfra.benchmark.compare import (
    BATCH,
    D_STATE,
    DATA_SOURCE,
    DEVICE,
    EMA,
    HAS_CUDA,
    NFRA_DEPTH,
    RWKVLM,
    SEQ_LEN,
    USE_AMP,
    GPT2ForCausalLM,
    HierarchicalDataset,
    MambaLM,
    RetNetLM,
    WikiText2Dataset,
    compute_loss,
    count_params,
    evaluate,
    make_optimizer,
    rescale_embed,
)

# ─────────────────────────── config ───────────────────────────
MODE = os.environ.get("NFRA_MODE", "standard")
STEP_CFG = {"quick": 150, "standard": 600, "rigorous": 1500}
STEPS = int(os.environ.get("NFRA_STEPS", STEP_CFG[MODE]))
if not HAS_CUDA:
    STEPS = min(STEPS, 80)

SIZES = [int(x) for x in os.environ.get("NFRA_SIZES", "5,20").split(",") if x.strip()]
if not SIZES:
    SIZES = [5, 20]
SEED_CNT = int(os.environ.get("NFRA_SEEDS", "3" if MODE == "rigorous" else "2"))
SEED_LIST = [42, 7, 2026, 1337, 777][:SEED_CNT]
FAMILIES = [
    f.strip().lower()
    for f in os.environ.get("NFRA_FAMILIES", "nfra,rwkv,retnet,gpt2").split(",")
    if f.strip()
]
# NFRA 3.2 feature toggles. EMA + surprise-weighted loss apply to ALL families
# (fair head-to-head); k-WTA is an NFRA architecture change only.
EMA_DECAY = float(os.environ.get("NFRA_EMA", "0"))  # 0 = off
SURPRISE = os.environ.get("NFRA_SURPRISE", "0") == "1"  # 1 = on
KWTA = float(os.environ.get("NFRA_KWTA", "0"))  # 0.0 = off
# "Small but powerful" brain levers (off by default; A/B-able overnight).
LOCAL_ROUTE = os.environ.get("NFRA_LOCALROUTE", "0") == "1"
DIV_NORM = os.environ.get("NFRA_DIVNORM", "0") == "1"
ASTRO = os.environ.get("NFRA_ASTRO", "0") == "1"
THETA = os.environ.get("NFRA_THETA", "0") == "1"
ACH_RETAIN = os.environ.get("NFRA_ACH_RETAIN", "0") == "1"
GAIN_NOV = os.environ.get("NFRA_GAIN_NOV", "0") == "1"
LORA_RANK = int(os.environ.get("NFRA_LORA_RANK", "0"))  # 0 = off (Space axis)
BANDS = int(os.environ.get("NFRA_BANDS", "16"))  # H8 band-count ablation knob
# NFRA 3.3 Cortex: 1 = build NFRA_Cortex_Block (3.3b retention-QK mixer + real
# exit gate) instead of the 3.2 Brain block. DEFAULT ON: the verified board
# (5.03M @ dim112/33, 20.00M @ dim224/33; losses 1.953/1.763 @ ~10.5k tok/s)
# and the isolation sweep were both measured on the Cortex block, and the
# NFRA_LEAN=1 / NFRA_ISO pruning flags only exist inside it. NFRA_CORTEX=0 is
# the legacy escape hatch that selects the old Brain block for A/B — it is NOT
# the default, otherwise a fresh clone silently ignores the lean pruning and
# regresses (5M: 2.140 @ 3.5k tok/s 3.77GB vs verified 1.953 @ 10.3k 1.40GB).
CORTEX = os.environ.get("NFRA_CORTEX", "1") == "1"
CORTEX_STATE = int(os.environ.get("NFRA_CORTEX_STATE", "8"))
EXIT_REG = float(os.environ.get("NFRA_EXIT_REG", "1e-3"))
# Isolation ablations (FUTURE_PLAN Part 11): NFRA_ISO=vgate,rgate,phase,gland,exit
# turns each 3.3b mechanism OFF (others stay on) to attribute the quality win.
# Empty -> the default architecture (see NFRA_LEAN below).
_ISO = frozenset(
    s.strip() for s in os.environ.get("NFRA_ISO", "").split(",") if s.strip()
)
# 3.3c LEAN (default, NFRA_LEAN=1): the post-prune architecture. The isolation
# sweep (b868477) proved ONLY the receptance gate carries the quality win
# (+0.038); the gland (+0.014), value gate (+0.006), phase (+0.005) and exit
# (+0.006) are within seed noise and all slow training 6-25%. Prune them, keep
# receptance. NFRA_LEAN=0 reproduces the full verified 3.3b block for A/B.
if _ISO:
    ISO_GLAND = "gland" in _ISO
    ISO_VGATE = "vgate" in _ISO
    ISO_RGATE = "rgate" in _ISO
    ISO_PHASE = "phase" in _ISO
    ISO_EXIT = "exit" in _ISO
elif os.environ.get("NFRA_LEAN", "1") == "1":
    ISO_GLAND = True  # neuromodulator pruned
    ISO_VGATE = True  # value gate pruned
    ISO_RGATE = False  # receptance gate KEPT (verified differentiator)
    ISO_PHASE = True  # phase modulation pruned
    ISO_EXIT = True  # adaptive-exit gate pruned
else:
    ISO_GLAND = ISO_VGATE = ISO_RGATE = ISO_PHASE = ISO_EXIT = False  # full 3.3b
# Gradient checkpointing trades compute for memory; on a big GPU with a small
# model the recompute is pure overhead -> set 0 to raise tok/s. Off by default
# for 3.3b: the lean retention block's activations are small (RetNet-shaped),
# so the recompute cost is pure speed loss.
CHECKPOINT = os.environ.get("NFRA_CHECKPOINT", "0") == "1"
# torch.compile(mode='reduce-overhead'): fuses the per-step kernel stream and
# captures it as CUDA graphs, killing Python/launch overhead (the main reason
# a small model idles the GPU). Auto-disables checkpointing (recompute conflicts
# with graph capture). Best combined with NFRA_SCAN_KERNEL=0 so the scan is
# plain torch and fuses into the same graph instead of forcing a graph break.
COMPILE = os.environ.get("NFRA_COMPILE", "0") == "1"
# Chunked retention: exact-equivalent reformulation of the decayed-QK^T mixer
# (within-chunk quadratic attention + cross-chunk linear state). Same math,
# ~C/S of the attention FLOPs and a fraction of the O(S^2) parallel form's
# memory — the Tier-1 lever to reach retnet-class tok/s AND retnet-class peak
# memory at 20M (verified loss 1.710 unchanged; the operator is identical).
# NFRA_CHUNK_SIZE=0 reproduces the verified parallel path bit-for-bit.
CHUNK_SIZE = int(os.environ.get("NFRA_CHUNK_SIZE", "64"))
# Recompute the two biggest GEMM activations (qkvr, MLP gate_up) in backward
# instead of storing them (~8 MB/layer saved, ~0.3 GB at depth 33). Trade
# compute for memory; forced OFF under torch.compile (checkpointed recompute
# breaks graph capture — pick one: speed via compile, or memory via ckpt).
CKPT_GEMM = os.environ.get("NFRA_CKPT_GEMM", "0") == "1" and not COMPILE
EVAL_GAP = max(50, STEPS // 6)
EXT_FACTOR = 2  # extrapolation test: eval at SEQ_LEN * EXT_FACTOR
GEN_LEN = 16
PROMPT_LEN = 64
PRE_HEAD = max(BATCH, 8)  # prefill throughput batch

DIM_GRID = {
    5: [256, 224, 192, 160, 128, 112, 96],
    20: [512, 448, 384, 352, 320, 288, 256, 224, 192, 160],
    50: [768, 704, 640, 576, 512],
}

# composite scoring: (metric key, direction, weight). direction +1 = higher better.
METRIC_SPEC = [
    {"key": "final_eval", "dir": -1, "w": 0.30, "label": "eval loss"},
    {"key": "sample_auc", "dir": -1, "w": 0.10, "label": "sample-efficiency AUC"},
    {"key": "param_eff", "dir": +1, "w": 0.10, "label": "loss-gain / M params"},
    {"key": "tok_s_train", "dir": +1, "w": 0.10, "label": "train tok/s"},
    {"key": "peak_mem", "dir": -1, "w": 0.10, "label": "peak train memory"},
    {"key": "gen_tok_s", "dir": +1, "w": 0.05, "label": "generation tok/s"},
    {"key": "infer_mem", "dir": -1, "w": 0.05, "label": "peak infer memory"},
    {
        "key": "extrap_delta",
        "dir": -1,
        "w": 0.10,
        "label": "long-context extrapolation",
    },
    {"key": "scaling_gain", "dir": +1, "w": 0.10, "label": "scaling gain / doubling"},
]


# ─────────────────────────── builders ───────────────────────────
def build_nfra(
    vocab,
    dim,
    unique_blocks,
    depth=NFRA_DEPTH,
    k_wta=None,
    local_route=None,
    div_norm=None,
    astro=None,
    theta=None,
    ach_retain=None,
    gain_nov=None,
    lora_rank=0,
    use_cortex=None,
    cortex_state=None,
    exit_reg=None,
    iso_gland=None,
    iso_vgate=None,
    iso_rgate=None,
    iso_phase=None,
    iso_exit=None,
    chunk_size=None,
    ckpt_gems=None,
):
    if k_wta is None:
        k_wta = KWTA
    if local_route is None:
        local_route = LOCAL_ROUTE
    if div_norm is None:
        div_norm = DIV_NORM
    if astro is None:
        astro = ASTRO
    if theta is None:
        theta = THETA
    if ach_retain is None:
        ach_retain = ACH_RETAIN
    if gain_nov is None:
        gain_nov = GAIN_NOV
    if not lora_rank:
        lora_rank = LORA_RANK
    if use_cortex is None:
        use_cortex = CORTEX
    if cortex_state is None:
        cortex_state = CORTEX_STATE
    if exit_reg is None:
        exit_reg = EXIT_REG
    if iso_gland is None:
        iso_gland = ISO_GLAND
    if iso_vgate is None:
        iso_vgate = ISO_VGATE
    if iso_rgate is None:
        iso_rgate = ISO_RGATE
    if iso_phase is None:
        iso_phase = ISO_PHASE
    if iso_exit is None:
        iso_exit = ISO_EXIT
    if chunk_size is None:
        chunk_size = CHUNK_SIZE
    if ckpt_gems is None:
        ckpt_gems = CKPT_GEMM
    cfg = NFRAConfig(
        mode="brain",
        vocab_size=vocab,
        hidden_size=dim,
        num_layers=depth,
        n_bands=BANDS,
        dropout=0.1,
        depth_shared=True,
        unique_blocks=unique_blocks,
        gradient_checkpointing=CHECKPOINT,
        k_wta_frac=k_wta,
        local_route=local_route,
        div_norm=div_norm,
        astro=astro,
        theta=theta,
        ach_retain=ach_retain,
        gain_nov=gain_nov,
        lora_rank=lora_rank,
        use_cortex=use_cortex,
        cortex_state=cortex_state,
        exit_reg=exit_reg,
        iso_gland=iso_gland,
        iso_vgate=iso_vgate,
        iso_rgate=iso_rgate,
        iso_phase=iso_phase,
        iso_exit=iso_exit,
        cortex_chunk_size=chunk_size,
        ckpt_gems=ckpt_gems,
    )
    return NFRAForCausalLM(cfg)


def build_mamba(vocab, dim, n_layers, d_state=D_STATE):
    return MambaLM(vocab, dim, n_layers, d_state)


def build_gpt2(vocab, dim, n_layers, n_heads=8, pos_len=2048):
    return GPT2ForCausalLM(vocab, dim, n_layers, n_heads, pos_len=pos_len)


def build_rwkv(vocab, dim, n_layers, dropout=0.1):
    return RWKVLM(vocab, dim, n_layers, dropout)


def build_retnet(vocab, dim, n_layers, n_heads=8, dropout=0.1):
    return RetNetLM(vocab, dim, n_layers, n_heads, dropout)


def tune_nfra_size(target, vocab, depth, dims, tol=0.20):
    """Pick (unique_blocks, dim) for NFRA. Prefer the MOST DISTINCT blocks
    (real layer diversity — deep-and-narrow stacks beat the 3.2 shallow
    recurrence on LM, see docs/OVERNIGHT_VERIFIED_RESULTS.md: retnet's 25
    distinct layers beat nfra's 6 re-used blocks at the same budget) among
    configs within `tol` of the param budget; tie-break by param proximity.
    `depth` is the number of effective layers; U = depth gives a plain
    distinct-layer stack (passes=1), U < depth keeps the depth-shared
    recurrence identity."""
    best = (None, None, None, None)  # (err, U, d, p)
    for U in range(depth, 1, -1):  # descending: max distinct blocks first
        if depth % U:
            continue
        for d in dims:
            p = count_params(build_nfra(vocab, d, U, depth))
            err = abs(p - target) / target
            if err <= tol:
                return U, d, p  # first (highest-U) config in budget
            if best[0] is None or err < best[0]:
                best = (err, U, d, p)
    if best[1] is None:
        return 1, dims[-1], count_params(build_nfra(vocab, dims[-1], 1, depth))
    return best[1], best[2], best[3]


def tune_layers_size(builder, target, vocab, dims, max_layers=40):
    best = (float("inf"), None, None, None)
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
    if family == "nfra":
        # Distinct-layer scaling: match the field's depth (retnet ~33, rwkv 24,
        # gpt2 24) instead of the 3.2 depth-shared shallow recurrence, so the
        # head-to-head measures real per-depth compute. Cortex depth 33 mirrors
        # retnet's winning 33-layer build at the same budget. tuner picks
        # U=depth (plain distinct stack) when params allow; depth-sharing stays
        # an option for memory-constrained configs via NFRA_DEPTH.
        depth = max(NFRA_DEPTH, 33)
        U, dim, params = tune_nfra_size(target, vocab, depth, dims)
        spec = {
            "builder": build_nfra,
            "dim": dim,
            "extra": {"unique_blocks": U, "depth": depth},
            "params": params,
            "depth": depth,
        }
        if CORTEX:
            spec["cortex"] = True
    elif family == "mamba":
        dim, L, params = tune_layers_size(build_mamba, target, vocab, dims)
        spec = {
            "builder": build_mamba,
            "dim": dim,
            "extra": {"n_layers": L, "d_state": D_STATE},
            "params": params,
            "depth": L,
        }
    elif family == "rwkv":
        dim, L, params = tune_layers_size(build_rwkv, target, vocab, dims)
        spec = {
            "builder": build_rwkv,
            "dim": dim,
            "extra": {"n_layers": L},
            "params": params,
            "depth": L,
        }
    elif family == "retnet":
        dim, L, params = tune_layers_size(build_retnet, target, vocab, dims)
        spec = {
            "builder": build_retnet,
            "dim": dim,
            "extra": {"n_layers": L, "n_heads": 8},
            "params": params,
            "depth": L,
        }
    else:
        dim, L, params = tune_layers_size(build_gpt2, target, vocab, dims)
        spec = {
            "builder": build_gpt2,
            "dim": dim,
            "extra": {"n_layers": L, "n_heads": 8, "pos_len": 2048},
            "params": params,
            "depth": L,
        }
    return spec


# ─────────────────────────── training ───────────────────────────
def train_one(
    model,
    vocab,
    steps,
    train_loader,
    eval_loader,
    eval_gap,
    ema_decay=0.0,
    surprise=False,
    seed=None,
):
    model.train()
    if COMPILE and HAS_CUDA:
        # Keep the caller's reference uncompiled (fine for later evals); train
        # the compiled copy here. Dropout is handled correctly by Dynamo (it
        # advances RNG between replays), so no loss of stochasticity.
        try:
            cfg = getattr(model, "config", None)
            ckpt_was = cfg.gradient_checkpointing if cfg is not None else None
            if cfg is not None:
                cfg.gradient_checkpointing = False
            model = torch.compile(model, mode="reduce-overhead", dynamic=False)
            print("  [compile] torch.compile(reduce-overhead) active")
        except Exception as e:  # pragma: no cover - depends on torch version
            print(f"  [warn] torch.compile failed ({e}) - eager fallback")
        finally:
            # Restore the caller's config: the compiled graph already captured
            # the flag at trace time (dynamic=False -> no re-trace), so this is
            # safe and keeps the caller's eager model/config unmutated.
            if cfg is not None:
                cfg.gradient_checkpointing = ckpt_was
    opt, sched = make_optimizer(
        model, lr=3e-4, warmup=min(50, max(steps // 10, 1)), total=steps
    )
    scaler = torch.amp.GradScaler(str(DEVICE)) if USE_AMP else None
    ema = EMA(model, ema_decay) if ema_decay > 0 else None
    if seed is not None:
        # Private identically-seeded generator: every caller training on the
        # same dataset with the same seed consumes byte-identical batches, so
        # families/models are compared on exactly the same stream. (Sharing one
        # DataLoader's iterator draws from the same RNG in an offset order.)
        loader = DataLoader(
            train_loader.dataset,
            batch_size=train_loader.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            num_workers=0,
            pin_memory=train_loader.pin_memory,
        )
        it = iter(loader)
    else:
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
        # Async H2D copies (non_blocking) overlap with the previous step's
        # kernels; they only actually run async if the loader uses pin_memory.
        x = x.to(DEVICE, non_blocking=HAS_CUDA)
        y = y.to(DEVICE, non_blocking=HAS_CUDA)
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
        ema.apply(model)  # leave EMA weights in place for downstream evals
    if HAS_CUDA:
        torch.cuda.synchronize()  # single sync: drain queued work once
    wall = time.perf_counter() - t_start
    loss_hist = [float(v) for v in loss_hist]
    mem = torch.cuda.max_memory_allocated() / 1e9 if HAS_CUDA else 0.0
    bs = getattr(train_loader, "batch_size", 1)
    seq = SEQ_LEN
    return {
        "loss_hist": loss_hist,
        "eval_hist": eval_hist,
        "tok_s": bs * seq * steps / max(wall, 1e-6),
        "ms_per_step": wall * 1000.0 / steps,
        "peak_mem": mem,
        "nan_steps": nan_steps,
        "wall_s": wall,
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
        logits = model(ids)["logits"]
        nxt = logits[:, -1, :].argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
    if HAS_CUDA:
        torch.cuda.synchronize()
    dt = max(time.perf_counter() - t0, 1e-6)
    mem = torch.cuda.max_memory_allocated() / 1e9 if HAS_CUDA else 0.0
    return {
        "gen_tok_s": gen_len / dt,
        "ms_per_token": dt / gen_len * 1000,
        "infer_mem": mem,
    }


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
    return sum(
        (a + b) / 2 * (xb - xa)
        for (xa, a), (xb, b) in zip(zip(xs, losses), zip(xs[1:], losses[1:]))
    )


def fit_scaling(points):
    """points: [(params, eval_loss)]. Slope = bits of loss per doubling of
    params (negative = improving). OLS on log2(params) vs loss."""
    pts = sorted((float(p), float(l)) for p, l in points)
    if len(pts) < 2:
        return {
            "slope": 0.0,
            "r2": None,
            "loss_100m": None,
            "loss_1b": None,
            "n": len(pts),
        }
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
    return {
        "slope": slope,
        "r2": r2,
        "loss_100m": pred(100e6),
        "loss_1b": pred(1e9),
        "n": n,
    }


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
        key, direction, w = s["key"], s["dir"], s["w"]
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
    train_one re-wraps each seed's train loader with a private, identically-
    seeded generator, so all families see byte-identical batches for a given
    (size, seed)."""
    use_wiki = DATA_SOURCE == "wikitext2"
    wiki_ds = WikiText2Dataset("train", SEQ_LEN) if use_wiki else None
    train_loaders = {}
    for si, seed in enumerate(SEED_LIST):
        gen = torch.Generator().manual_seed(1000 + size_idx * 100 + si * 10)
        if use_wiki:
            ds = wiki_ds
        else:
            ds = HierarchicalDataset(
                max(512, BATCH * 12),
                SEQ_LEN + 1,
                seed=100 + size_idx,
                seq_seed=200 + size_idx * 10 + si,
            )
        train_loaders[seed] = DataLoader(
            ds,
            batch_size=BATCH,
            shuffle=True,
            generator=gen,
            num_workers=0,
            pin_memory=HAS_CUDA,
        )
    if use_wiki:
        eval_ds = WikiText2Dataset("validation", SEQ_LEN)
        ext_ds = WikiText2Dataset("validation", SEQ_LEN * EXT_FACTOR)
    else:
        eval_ds = HierarchicalDataset(
            max(256, BATCH * 6),
            SEQ_LEN + 1,
            seed=100 + size_idx,
            seq_seed=300 + size_idx,
        )
        ext_ds = HierarchicalDataset(
            max(256, BATCH * 6),
            SEQ_LEN * EXT_FACTOR + 1,
            seed=100 + size_idx,
            seq_seed=400 + size_idx,
        )
    eval_loader = DataLoader(
        eval_ds, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=HAS_CUDA
    )
    ext_loader = DataLoader(
        ext_ds, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=HAS_CUDA
    )
    return train_loaders, eval_loader, ext_loader


# ─────────────────────────── report ───────────────────────────
def fmt(v, nd=3, suffix=""):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "-"
    return f"{v:.{nd}f}{suffix}"


def fmt_ms(v, sd, nd=3):
    if v is None:
        return "-"
    if sd and sd > 0:
        return f"{v:.{nd}f} ± {sd:.{nd}f}"
    return f"{v:.{nd}f}"


def md_table(headers, rows):
    out = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


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
    use_wiki = DATA_SOURCE == "wikitext2"
    VOCAB = 96 if use_wiki else 4096
    RANDOM_LOSS = math.log(VOCAB)
    label = "WikiText-2 (char)" if use_wiki else "Synthetic hierarchical"

    t_all = time.perf_counter()
    print("=" * 72)
    print("  NFRA ARENA - global-standard multi-dimension comparison")
    print("  NFRA Brain  vs  RWKV  vs  RetNet  vs  GPT-2  vs  Mamba (param-matched)")
    print("=" * 72)
    print(f"  mode     : {MODE} ({STEPS} steps)   data: {label}   vocab: {VOCAB}")
    print(f"  sizes    : {SIZES} M params    seeds: {SEED_LIST}")
    print(f"  families : {FAMILIES}    batch: {BATCH}    seq: {SEQ_LEN}")
    print(
        f"  device   : {'GPU ' + torch.cuda.get_device_name(0) if HAS_CUDA else 'CPU'}"
        + ("   (fp16 AMP)" if USE_AMP else "")
    )
    feats = []
    if BANDS != 16:
        feats.append(f"bands={BANDS}")
    if EMA_DECAY > 0:
        feats.append(f"EMA={EMA_DECAY}")
    if SURPRISE:
        feats.append("surprise-weighted loss")
    if KWTA > 0:
        feats.append(f"k-WTA={KWTA}")
    if LOCAL_ROUTE:
        feats.append("local cortical routing")
    if DIV_NORM:
        feats.append("divisive normalization")
    if ASTRO:
        feats.append("astrocytic homeostat")
    if feats:
        print(f"  features : {', '.join(feats)}")
    print("=" * 72)

    env = {
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if HAS_CUDA else None,
        "gpu_mem_gb": (
            round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            if HAS_CUDA
            else None
        ),
        "python": sys.version.split()[0],
        "ema": EMA_DECAY,
        "surprise": int(SURPRISE),
        "kwta": KWTA,
    }

    # ── build specs per size
    specs = {}
    for size in SIZES:
        specs[size] = {f: build_family_spec(f, size, VOCAB) for f in FAMILIES}
        for f, s in specs[size].items():
            print(
                f"  [ok] {f:<6s} @ {size}M: dim {s['dim']:<4d} {s['params']/1e6:6.2f}M "
                f"depth {s['depth']}  {s['extra']}"
            )

    # ── run every (size × seed × family)
    print(
        "\n[train] running %d training runs (%d sizes x %d seeds x %d families)..."
        % (len(SIZES) * SEED_CNT * len(FAMILIES), len(SIZES), SEED_CNT, len(FAMILIES))
    )
    runs = {}  # runs[size][seed][family] = train record
    battery = {}  # battery[size][family] = extrap + inference battery
    for size in SIZES:
        train_loaders, eval_loader, ext_loader = make_loaders(SIZES.index(size))
        runs[size] = {}
        for seed in SEED_LIST:
            runs[size][seed] = {}
            torch.manual_seed(seed)
            np.random.seed(seed)
            for family in FAMILIES:
                spec = specs[size][family]
                model = spec["builder"](VOCAB, spec["dim"], **spec["extra"]).to(DEVICE)
                rescale_embed(model)
                rec = train_one(
                    model,
                    VOCAB,
                    STEPS,
                    train_loaders[seed],
                    eval_loader,
                    EVAL_GAP,
                    ema_decay=EMA_DECAY,
                    surprise=SURPRISE,
                    seed=seed,
                )
                runs[size][seed][family] = rec
                if seed == SEED_LIST[-1]:
                    ext = evaluate(model, ext_loader, max_batches=6)
                    b = {
                        "extrap_loss": ext,
                        "extrap_delta": ext - rec["eval_hist"][-1][1],
                    }
                    if size == SIZES[-1]:
                        pre = prefill_tok_s(model, PRE_HEAD, SEQ_LEN, VOCAB)
                        gen = generate_metrics(model, VOCAB)
                        b.update(
                            {
                                "prefill_tok_s": pre,
                                "gen_tok_s": gen["gen_tok_s"],
                                "ms_per_token": gen["ms_per_token"],
                                "infer_mem": gen["infer_mem"],
                            }
                        )
                    battery.setdefault(size, {})[family] = b
                print(
                    f"  [ok] {family:<6s} @ {size}M seed {seed:<4d} "
                    f"final {rec['eval_hist'][-1][1]:.3f}  "
                    f"{rec['tok_s']:.0f} tok/s  {rec['peak_mem']:.2f} GB"
                )

    # ── aggregate metrics per (size, family)
    metrics = {}
    for size in SIZES:
        metrics[size] = {}
        for family in FAMILIES:
            recs = [runs[size][s][family] for s in SEED_LIST]
            finals = [r["eval_hist"][-1][1] for r in recs]
            m_final, sd_final = mean_std(finals)
            m_auc, _ = mean_std([sample_auc(r["eval_hist"]) for r in recs])
            m_tok, _ = mean_std([r["tok_s"] for r in recs])
            m_ms, _ = mean_std([r["ms_per_step"] for r in recs])
            m_mem, _ = mean_std([r["peak_mem"] for r in recs])
            params = specs[size][family]["params"]
            row = {
                "params": params,
                "depth": specs[size][family]["depth"],
                "spec": specs[size][family],
                "final_eval": m_final,
                "final_eval_sd": sd_final,
                "final_eval_n": len(finals),
                "ppl": math.exp(min(m_final, 30)) if m_final else None,
                "sample_auc": m_auc,
                "tok_s_train": m_tok,
                "ms_per_step": m_ms,
                "peak_mem": m_mem,
                "nan_steps": sum(r["nan_steps"] for r in recs),
                "wall_s": sum(r["wall_s"] for r in recs),
                "param_eff": (
                    ((RANDOM_LOSS - m_final) / (params / 1e6)) if m_final else None
                ),
                "est_flops_token": 6 * params,
            }
            if family in battery.get(size, {}):
                for k in (
                    "extrap_loss",
                    "extrap_delta",
                    "prefill_tok_s",
                    "gen_tok_s",
                    "ms_per_token",
                    "infer_mem",
                ):
                    if k in battery[size][family]:
                        row[k] = battery[size][family][k]
            metrics[size][family] = row

    # ── scaling fit per family (across sizes)
    scaling = {}
    for family in FAMILIES:
        pts = [
            (metrics[s][family]["params"], metrics[s][family]["final_eval"])
            for s in SIZES
            if metrics[s][family]["final_eval"]
        ]
        scaling[family] = fit_scaling(pts)
    for family in FAMILIES:
        gain = -scaling[family]["slope"]
        for size in SIZES:
            metrics[size][family]["scaling_gain"] = gain

    # ── composite scores per size
    scores = {}
    for size in SIZES:
        scores[size] = composite_score(metrics[size], METRIC_SPEC)

    # ── verdict
    primary = max(SIZES)
    verdict = make_verdict(metrics, scores, scaling, primary, RANDOM_LOSS)

    # ── write outputs
    data = {
        "config": {
            "mode": MODE,
            "steps": STEPS,
            "sizes": SIZES,
            "seeds": SEED_LIST,
            "families": FAMILIES,
            "data": DATA_SOURCE,
            "vocab": VOCAB,
            "batch": BATCH,
            "seq_len": SEQ_LEN,
            "optimizer": "AdamW(lr=3e-4,beta=(0.9,0.95))",
            "schedule": "warmup+cosine",
            "seed_list": SEED_LIST,
            "ext_factor": EXT_FACTOR,
            "metric_weights": [
                {k: s[k] for k in ("key", "dir", "w")} for s in METRIC_SPEC
            ],
        },
        "env": env,
        "specs": {
            str(s): {
                f: {k: v for k, v in specs[s][f].items() if k != "builder"}
                for f in FAMILIES
            }
            for s in SIZES
        },
        "metrics": {
            str(s): {
                f: {
                    k: v
                    for k, v in metrics[s][f].items()
                    if k != "spec" and not callable(v)
                }
                for f in FAMILIES
            }
            for s in SIZES
        },
        "scaling": {f: scaling[f] for f in FAMILIES},
        "scores": {str(s): scores[s] for s in SIZES},
        "history": {
            str(s): {
                sd: {
                    f: {
                        "loss": runs[s][sd][f]["loss_hist"],
                        "eval": runs[s][sd][f]["eval_hist"],
                    }
                    for f in FAMILIES
                }
                for sd in SEED_LIST
            }
            for s in SIZES
        },
        "verdict": verdict,
    }
    out_json = os.path.join(os.getcwd(), "nfra_arena_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    report = build_report(
        metrics,
        scores,
        scaling,
        verdict,
        primary,
        RANDOM_LOSS,
        label,
        env,
        data["config"],
    )
    out_md = os.path.join(os.getcwd(), "nfra_arena_report.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n  done in {time.perf_counter() - t_all:.0f}s")
    print(f"  -> {out_json}")
    print(f"  -> {out_md}")


def _arg_extreme(fams, key, direction):
    """argmin/argmax over fams by key(f), skipping None values; None if all
    None. Short/aborted runs produce None metrics (e.g. sample_auc needs >=2
    eval points, param_eff needs a final loss) -- a raw min()/max() on those
    crashed make_verdict and killed the whole phase."""
    cands = [(f, key(f)) for f in fams if key(f) is not None]
    if not cands:
        return None
    return (min if direction == "min" else max)(cands, key=lambda t: t[1])[0]


def make_verdict(metrics, scores, scaling, primary, random_loss):
    fams = list(metrics[primary])
    m = metrics[primary]
    finite = [f for f in fams if m[f]["final_eval"] is not None]
    by_loss = sorted(finite, key=lambda f: m[f]["final_eval"])
    best_q = by_loss[0] if by_loss else None
    second_q = by_loss[1] if len(by_loss) > 1 else None
    q_delta = (m[second_q]["final_eval"] - m[best_q]["final_eval"]) if second_q else 0.0
    best_mem = _arg_extreme(fams, lambda f: m[f]["peak_mem"], "min")
    worst_mem = _arg_extreme(fams, lambda f: m[f]["peak_mem"], "max")
    best_speed = _arg_extreme(fams, lambda f: m[f]["tok_s_train"], "max")
    best_scale = _arg_extreme(fams, lambda f: -scaling[f]["slope"], "max")
    best_score = _arg_extreme(fams, lambda f: scores[primary][f], "max")
    worst_score = _arg_extreme(fams, lambda f: scores[primary][f], "min")
    best_eff = _arg_extreme(fams, lambda f: m[f]["param_eff"], "max")
    best_sample_eff = _arg_extreme(fams, lambda f: m[f]["sample_auc"], "min")
    best_extrap = _arg_extreme(fams, lambda f: m[f].get("extrap_delta"), "min")

    def _ev(claim, fam, status, evidence):
        return {
            "claim": claim,
            "family": fam,
            "status": "skipped" if fam is None else status,
            "evidence": evidence,
        }

    claims = [
        _ev(
            "lowest eval loss (best quality)",
            best_q,
            "confirmed" if q_delta >= 0.02 else "marginal",
            f"Δ{abs(q_delta):.3f} vs runner-up at {primary}M",
        ),
        _ev(
            "most parameter-efficient (loss-gain per M param)",
            best_eff,
            "confirmed",
            None,
        ),
        _ev(
            "most sample-efficient (lowest learning-curve AUC)",
            best_sample_eff,
            "confirmed",
            None,
        ),
        _ev(
            "lowest peak training memory",
            best_mem,
            "confirmed",
            (
                f"{m[best_mem]['peak_mem']:.2f} GB vs "
                f"{m[worst_mem]['peak_mem']:.2f} GB"
                if best_mem and worst_mem
                else None
            ),
        ),
        _ev(
            "fastest training throughput",
            best_speed,
            "confirmed",
            f"{m[best_speed]['tok_s_train']:.0f} tok/s" if best_speed else None,
        ),
        _ev(
            "best scaling slope (loss per doubling of params)",
            best_scale,
            "confirmed",
            f"{scaling[best_scale]['slope']:.3f} bits/doubling" if best_scale else None,
        ),
        _ev(
            "most robust to longer contexts (2× extrapolation)",
            best_extrap,
            (
                "confirmed"
                if (best_extrap and "extrap_delta" in m[best_extrap])
                else "unmeasured"
            ),
            None,
        ),
    ]
    for c in claims:
        if c["family"] not in fams:
            c["status"] = "skipped"

    # revolution assessment
    nfra = "nfra" if "nfra" in fams else None
    revo = {
        "overall_leader": best_score,
        "overall_score": round(scores[primary][best_score], 1),
        "worst_score": round(scores[primary][worst_score], 1),
    }
    if nfra:
        revo["nfra_quality_rank"] = (
            "1st" if best_q == nfra else "2nd" if second_q == nfra else "last"
        )
        revo["nfra_quality_gap_to_best"] = (
            round(m[nfra]["final_eval"] - m[best_q]["final_eval"], 3)
            if best_q
            else None
        )
        revo["nfra_mem_gb"] = round(m[nfra]["peak_mem"], 2)
        revo["nfra_speed_tok_s"] = round(m[nfra]["tok_s_train"])
        revo["nfra_is_overall_winner"] = best_score == nfra
        revo["nfra_wins_quality"] = best_q == nfra
        revo["revolutionary"] = best_score == nfra
    return {
        "primary_size": primary,
        "claims": claims,
        "revo": revo,
        "winners": {
            "quality": best_q,
            "memory": best_mem,
            "speed": best_speed,
            "scaling": best_scale,
            "overall": best_score,
        },
    }


def build_report(
    metrics, scores, scaling, verdict, primary, random_loss, label, env, cfg
):
    fams = list(metrics[primary])
    L = []
    a = L.append
    a("# NFRA ARENA — multi-dimension benchmark report\n")
    a(
        f"**Families:** {', '.join(fams)}  |  **Data:** {label}  |  "
        f"**Vocab:** {cfg['vocab']}  |  **Steps:** {cfg['steps']}  |  "
        f"**Batch:** {cfg['batch']}  |  **Seq:** {cfg['seq_len']}\n"
    )
    a(
        f"**Sizes:** {cfg['sizes']}M params  |  **Seeds:** {cfg['seeds']}  |  "
        f"**Random-guess loss:** {random_loss:.3f} (ln vocab)\n"
    )
    a(f"**Optimizer:** {cfg['optimizer']}  **Schedule:** {cfg['schedule']}\n")
    a("**Environment:** " + ", ".join(f"{k}={v}" for k, v in env.items() if v) + "\n")

    a("\n## 1. Models (param-matched builds)\n")
    rows = []
    for size in sorted(cfg["sizes"]):
        for f in fams:
            s = metrics[size][f]["spec"]
            rows.append(
                [
                    f"{size}M",
                    f,
                    s["dim"],
                    s["params"] / 1e6,
                    metrics[size][f]["depth"],
                    s["extra"].get("unique_blocks", "-"),
                ]
            )
    a(
        md_table(
            ["size", "family", "dim", "params (M)", "depth", "unique blocks (NFRA)"],
            rows,
        )
    )
    a("")

    a("\n## 2. Scaling behaviour (quality vs params)\n")
    a(
        "OLS fit of eval loss vs log2(params). **Slope** = bits of loss gained per "
        "doubling of params (more negative = better scaling).\n"
    )
    rows = []
    for f in fams:
        sc = scaling[f]
        rows.append(
            [
                f,
                fmt(sc["slope"], 4),
                fmt(sc.get("r2"), 3),
                fmt(sc.get("loss_100m"), 3),
                fmt(sc.get("loss_1b"), 3),
                sc.get("n", 0),
            ]
        )
    a(
        md_table(
            [
                "family",
                "slope (bits/doubling)",
                "R²",
                "extrap. loss @100M",
                "@1B",
                "points",
            ],
            rows,
        )
    )
    a("")

    sec = 3
    for size in sorted(cfg["sizes"]):
        m = metrics[size]
        a(f"\n## {sec}. Head-to-head @ {size}M params\n")
        rows = []
        for f in fams:
            row = m[f]
            rows.append(
                [
                    f,
                    fmt(row["final_eval"], 3),
                    fmt_ms(row["final_eval"], row["final_eval_sd"], 3),
                    fmt(row["ppl"], 2),
                    fmt(row["sample_auc"], 3),
                    f"{row['tok_s_train']:.0f}",
                    f"{row['ms_per_step']:.1f}",
                    fmt(row["peak_mem"], 2),
                    fmt(row["nan_steps"], 0),
                ]
            )
        a(
            md_table(
                [
                    "family",
                    "eval loss",
                    "mean ± std",
                    "ppl",
                    "AUC",
                    "train tok/s",
                    "ms/step",
                    "peak mem GB",
                    "NaN steps",
                ],
                rows,
            )
        )
        a("")
        # inference battery only on the primary size
        if size == primary:
            rows = []
            for f in fams:
                row = m[f]
                if not all(
                    k in row
                    for k in (
                        "prefill_tok_s",
                        "gen_tok_s",
                        "ms_per_token",
                        "infer_mem",
                        "extrap_loss",
                        "extrap_delta",
                    )
                ):
                    continue
                rows.append(
                    [
                        f,
                        f"{row['prefill_tok_s']:.0f}",
                        f"{row['gen_tok_s']:.1f}",
                        f"{row['ms_per_token']:.2f}",
                        fmt(row["infer_mem"], 2),
                        fmt(row["extrap_loss"], 3),
                        fmt(row["extrap_delta"], 3),
                    ]
                )
            a("**Inference battery + long-context extrapolation (primary size):**\n")
            a(
                md_table(
                    [
                        "family",
                        "prefill tok/s",
                        "gen tok/s (b=1)",
                        "ms/token",
                        "peak infer GB",
                        "eval @2×ctx",
                        "Δ vs @1×ctx",
                    ],
                    rows,
                )
            )
            a("")
        # per-aspect winners
        rows = []
        for s in METRIC_SPEC:
            w = winner_of(m, s["key"], s["dir"])
            if w is None:
                rows.append([s["label"], "-", "-"])
                continue
            val = m[w].get(s["key"])
            vstr = fmt(val, 3) if isinstance(val, float) else val
            rows.append([s["label"], w, vstr])
        a(f"**Who wins which aspect @ {size}M:**\n")
        a(md_table(["aspect", "winner", "value"], rows))
        a("")
        sec += 1

    a(f"\n## {sec}. Composite scores @ {primary}M (weighted z-scores, 0-100)\n")
    a("Weights: " + ", ".join(f"{s['label']}={s['w']:.2f}" for s in METRIC_SPEC) + "\n")
    rows = [[f, f"{scores[primary][f]:.1f}"] for f in fams]
    a(md_table(["family", "score"], rows))
    a("")
    sec += 1

    a(f"\n## {sec}. Verdict\n")
    for c in verdict["claims"]:
        ev = f" — {c['evidence']}" if c.get("evidence") else ""
        a(f"- **{c['claim']}:** {c['family']} ({c['status']}){ev}")
    a("")
    r = verdict["revo"]
    a(
        f"- **Overall leader:** {r['overall_leader']} "
        f"(score {r['overall_score']:.1f} vs {r['worst_score']:.1f})\n"
    )
    if "nfra" in fams:
        a(
            f"- **NFRA Brain quality rank:** {r['nfra_quality_rank']} "
            f"(gap to best {r['nfra_quality_gap_to_best']:+.3f})"
        )
        a(
            f"- **NFRA memory:** {r['nfra_mem_gb']} GB | **NFRA throughput:** "
            f"{r['nfra_speed_tok_s']} tok/s"
        )
        a(
            f"- **'Revolutionary' verdict:** {'CONFIRMED' if r['revolutionary'] else 'NOT confirmed'}"
            f" — NFRA {'is' if r['revolutionary'] else 'is NOT'} the overall "
            f"winner at matched params, matched data, matched compute.\n"
        )

    a("\n---\n")
    a(
        "*Methodology: identical optimizer + schedule + token budget per family; "
        "multiple seeds (mean ± std); multiple sizes for measured scaling; "
        "pure-PyTorch implementations (no fused kernels — speed is a lower bound "
        "for Mamba/NFRA). Full per-seed data in nfra_arena_results.json.*"
    )
    return "\n".join(L)


if __name__ == "__main__":
    main()
