# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-CELL T4 FACTOR RUN — full multi-factor analysis in one go.
# Paste into a Kaggle T4 notebook cell (or `python factor_run.py`).
#
# Set the knobs in CONFIG below, then run. It re-evaluates the arena env knobs,
# trains every arm under the SAME seed/batches (byte-identical data), then emits:
#   1) a raw factor table (9 factors / arm),
#   2) per-factor deltas vs baseline with directional verdicts,
#   3) a composite score (METRIC_SPEC-weighted improvement vs baseline),
#   4) the stateful-gen headline: stateful O(1) tok/s vs slow re-eval tok/s.
#
# NOTE: export NFRA_PERTOKEN_GN=1 so the stateful dual is exact (gen_sf reports
# 'ok'); with it off, gen_sf shows '-' (the guard refuses wrong numbers).
# ─────────────────────────────────────────────────────────────────────────────
import os
import importlib

# ── CONFIG (edit here) ──────────────────────────────────────────────────────
CONFIG = {
    # flagship budget from the depth-8 sweep — trade a little loss for speed
    "NFRA_GATE_TARGET_M": "20",
    "NFRA_GATE_DEPTH": "8",
    "NFRA_BATCH": "24",
    "NFRA_CHECKPOINT": "1",
    "NFRA_SEQ": "256",
    "NFRA_PERTOKEN_GN": "1",   # enables the O(1) stateful decode (guarded)
}
# arms to run (each flips ONE factor off the baseline; keep the list small, T4
# trains each from scratch). Empty = all.
ARMS = ["baseline", "lsr", "int8_state", "depth_time", "triton_chunk", "rev"]

for _k, _v in CONFIG.items():
    os.environ[_k] = _v

import torch
import torch.nn.functional as F

has_cuda = torch.cuda.is_available()
print("CUDA:", torch.cuda.get_device_name(0) if has_cuda else "NONE — this needs a T4")

import nfra.benchmark.arena as A          # module-level knobs are read at import
importlib.reload(A)                        # re-read the env CONFIG above
from nfra.benchmark.arena import (
    build_nfra, make_loaders, evaluate, generate_metrics, sample_auc, train_one,
    tune_nfra_size, STEPS, EMA, EVAL_GAP, SEED_LIST, DIM_GRID, METRIC_SPEC,
    DEVICE, DATA_SOURCE,
)
if DATA_SOURCE == "wikitext2":
    VOCAB = 96
else:
    from nfra.benchmark.compare import HierarchicalDataset

    VOCAB = HierarchicalDataset.VOCAB_SIZE
from nfra.core.stateful import stateful_generate_metrics, supported

assert has_cuda, "factor run requires a CUDA GPU (T4)"

# arm recipe + which arena-level compile flag it needs (mirror experiments_gate)
ARM_RECIPE = {
    "baseline":     dict(chunk_size=0, triton=False, lsr=False, int8_state=False, depth_time=False),
    "triton_chunk": dict(chunk_size=64, triton=True),
    "lsr":          dict(lsr=True),
    "int8_state":   dict(chunk_size=64, int8_state=True),
    "depth_time":   dict(depth_time=True),
    "batch_pass":   dict(),
    "fuse_model":   dict(),
    "rev":          dict(lsr=True),
}
COMPILE_FLAG = {"batch_pass": "BATCH_PASSES", "fuse_model": "FUSE_MODEL", "rev": "BATCH_PASSES"}
arms = [n for n in ARMS if n in ARM_RECIPE] or list(ARM_RECIPE)
if not ARMS:
    arms = list(ARM_RECIPE)

M = int(os.environ.get("NFRA_GATE_TARGET_M", "20"))
DEPTH = int(os.environ.get("NFRA_GATE_DEPTH", "33"))
U, dim, params = tune_nfra_size(M * 1_000_000, VOCAB, DEPTH, DIM_GRID[M])
seed = SEED_LIST[0]
train_loaders, eval_loader, ext_loader = make_loaders(0)

print(f"battery: target={M}M depth={DEPTH} seq={A.SEQ_LEN} batch={A.BATCH} "
      f"ckpt={A.CHECKPOINT} per_token_gn={A.PER_TOKEN_GN} steps={STEPS} ema={EMA}")
print(f"model: unique={U} dim={dim} params={params/1e6:.2f}M\n")


def run_arm(name):
    kw = ARM_RECIPE[name]
    for _attr in ("FUSE_MODEL", "BATCH_PASSES"):
        setattr(A, _attr, _attr == COMPILE_FLAG.get(name, ""))
    build = lambda: build_nfra(VOCAB, dim, U, depth=DEPTH, use_cortex=True, **kw)
    m = build().to(DEVICE)
    # light sanity: forward+backward finite grads
    sm = build().to(DEVICE)
    _x = torch.randint(0, VOCAB, (A.BATCH, 32), device=DEVICE)
    _l = sm(_x, return_dict=True)["logits"][:, :-1].reshape(-1, VOCAB)
    _loss = F.cross_entropy(_l, _x[:, 1:].reshape(-1))
    _loss.backward()
    del sm
    torch.cuda.empty_cache()

    r = train_one(m, VOCAB, STEPS, train_loaders[seed], eval_loader,
                  eval_gap=EVAL_GAP, ema_decay=EMA, seed=seed)
    ev = evaluate(m, eval_loader)
    r["final_eval"] = float(ev)
    r["ext_eval"] = float(evaluate(m, ext_loader))
    r["sample_auc"] = sample_auc(r["eval_hist"])
    gm = generate_metrics(m, VOCAB)
    r["gen_tok_s"] = gm["gen_tok_s"]; r["infer_mem"] = gm["infer_mem"]
    try:
        sf = stateful_generate_metrics(m, VOCAB, device=DEVICE)
        r.update(sf)
    except Exception:
        r.update({"gen_sf": None, "sf_abs": None, "sf_rel": None, "sf_ok": None})
    r["param_M"] = params / 1e6
    del m
    torch.cuda.empty_cache()
    return r


results = {n: run_arm(n) for n in arms}

# ── build the factor table ───────────────────────────────────────────────────
FACTORS = ["final_eval", "ext_eval", "gen_tok_s", "gen_sf", "infer_mem",
           "peak_mem", "tok_s", "sample_auc", "ms_per_step"]
HDR = {  # (header, formatter)
    "final_eval": ("eval",      "{:.4f}"),
    "ext_eval":   ("ext@L×2",   "{:.4f}"),
    "gen_tok_s":  ("gen_slw/s", "{:.0f}"),
    "gen_sf":     ("gen_sf/s",  "{:.0f}"),
    "infer_mem":  ("infer GB",  "{:.3f}"),
    "peak_mem":   ("mem GB",    "{:.3f}"),
    "tok_s":      ("tok/s",     "{:.0f}"),
    "sample_auc": ("sampleAUC", "{:.3f}"),
    "ms_per_step":("ms/step",   "{:.1f}"),
}
print("\n── factor table ──────────────────────────────────────────────────────")
print(f"  {'arm':<11}" + "  ".join(f"{HDR[f][0]:>10}" for f in FACTORS) + "  param")
for name, r in results.items():
    cols = [HDR[f][1].format(r.get(f)) if r.get(f) is not None else f"{'-':>10}"
            for f in FACTORS]
    print(f"  {name:<11}  " + "  ".join(f"{c:>10}" for c in cols) +
          f"  {r['param_M']:.2f}M")

# ── per-factor deltas vs baseline (sign-corrected: + = good) ────────────────
DIRMAP = {"final_eval": -1, "ext_eval": -1, "gen_tok_s": +1, "gen_sf": +1,
          "infer_mem": -1, "peak_mem": -1, "tok_s": +1, "sample_auc": -1}
print("\n── per-factor deltas vs baseline (+ = better) ─────────────────────────")
print(f"  {'arm':<11}" + "  ".join(f"{HDR[f][0]:>10}" for f in FACTORS))
if "baseline" in results:
    base = results["baseline"]
    for name, r in results.items():
        if name == "baseline":
            continue
        cells = []
        for f in FACTORS:
            b, v = base.get(f), r.get(f)
            if b is None or v is None:
                cells.append(f"{'-':>10}"); continue
            d = (v - b) * DIRMAP.get(f, 1)
            cells.append(f"{d:+.3f}" if abs(d) < 100 else f"{d:+.0f}")
        print(f"  {name:<11}  " + "  ".join(f"{c:>10}" for c in cells))

# ── composite score (METRIC_SPEC-weighted improvement) ───────────────────────
def composite(r, base):
    tot, wsum = 0.0, 0.0
    for spec in METRIC_SPEC:
        k, dir_, w = spec["key"], spec["dir"], spec["w"]
        if k in ("extrap_delta", "scaling_gain", "param_eff"):
            continue  # composite uses the directly-measured factors only
        b, v = base.get(k), r.get(k)
        if b is None or v is None or abs(b) < 1e-9:
            continue
        rel = (v - b) / abs(b)          # fractional improvement
        tot += dir_ * rel * w
        wsum += w
    return tot / wsum if wsum else 0.0

print("\n── composite METRIC_SPEC score (0 = baseline, + = better) ─────────────")
if "baseline" in results:
    base = results["baseline"]
    for name, r in results.items():
        print(f"  {name:<11}  {composite(r, base):+.4f}")

# ── stateful-gen headline ────────────────────────────────────────────────────
print("\n── stateful O(1) decode headline ──────────────────────────────────────")
print(f"  {'arm':<11} {'slow/s':>7} {'stateful/s':>10} {'speedup':>8}  guard")
for name, r in results.items():
    slw = r.get("gen_tok_s"); sf_ = r.get("gen_sf"); ok = r.get("sf_ok")
    sp = (sf_ / slw) if (slw and sf_) else None
    guard = "ok" if ok else ("-" if ok is None else "X")
    print(f"  {name:<11} {slw if slw is not None else 0:>7.0f} "
          f"{sf_ if sf_ is not None else 0:>10.0f} "
          f"{sp:>7.1f}x" if sp else f"{'-':>8}" + f"    {guard}")