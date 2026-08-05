# ═══════════════════════════════════════════════════════════════════════════
# NFRA-2.0 · T4 MULTI-FACTOR RUN — one cell, paste into a Kaggle GPU notebook.
#  1) clones/pulls the repo + installs the package (idempotent)
#  2) trains every arm under the SAME seed/batches (byte-identical data)
#  3) prints: factor table → deltas vs baseline → composite → stateful O(1) decode
#
# Edit CONFIG / ARMS to taste. Needs a CUDA GPU (T4+). T4 time ≈ #arms × 1 train.
# ═══════════════════════════════════════════════════════════════════════════
import os, sys, subprocess, importlib

# ── 1) get the code from GitHub + install (safe to re-run) ──────────────────
REPO = "/kaggle/working/nfra-2.0"
if not os.path.isdir(REPO):
    subprocess.run(["git", "clone", "https://github.com/saurav3231/nfra-2.0.git", REPO], check=True)
else:
    subprocess.run(["git", "-C", REPO, "pull", "--ff-only"], check=True)
subprocess.run(["pip", "install", "-q", "-e", REPO], check=True)
if REPO + "/src" not in sys.path:
    sys.path.insert(0, REPO + "/src")

# ── 2) CONFIG ────────────────────────────────────────────────────────────────
CONFIG = {                     # flagship budget from the depth-8 sweep
    "NFRA_GATE_TARGET_M": "20", "NFRA_GATE_DEPTH": "8",
    "NFRA_BATCH": "24", "NFRA_CHECKPOINT": "1", "NFRA_SEQ": "256",
    "NFRA_PERTOKEN_GN": "1",    # O(1) stateful decode is exact (guard = 'ok')
    "NFRA_EMA": "0.99",         # EMA weight-averaging decay (0 = off)
}
ARMS = ["baseline", "lsr", "int8_state", "depth_time", "triton_chunk", "rev"]
for _k, _v in CONFIG.items():
    os.environ[_k] = _v

import torch
import torch.nn.functional as F
assert torch.cuda.is_available(), "factor run needs a CUDA GPU (T4)"
print("GPU:", torch.cuda.get_device_name(0))

import nfra.benchmark.arena as A
importlib.reload(A)                            # re-read CONFIG's env knobs
from nfra.benchmark.arena import (
    build_nfra, make_loaders, evaluate, generate_metrics, sample_auc, train_one,
    tune_nfra_size, STEPS, EMA_DECAY, EVAL_GAP, SEED_LIST, DIM_GRID, METRIC_SPEC,
    DEVICE, DATA_SOURCE,
)
if DATA_SOURCE == "wikitext2":
    VOCAB = 96
else:
    from nfra.benchmark.compare import HierarchicalDataset
    VOCAB = HierarchicalDataset.VOCAB_SIZE
from nfra.core.stateful import stateful_generate_metrics

# ── 3) arms: each flips ONE factor off the baseline ─────────────────────────
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

M = int(os.environ.get("NFRA_GATE_TARGET_M", "20"))
DEPTH = int(os.environ.get("NFRA_GATE_DEPTH", "33"))
U, dim, params = tune_nfra_size(M * 1_000_000, VOCAB, DEPTH, DIM_GRID[M])
seed = SEED_LIST[0]
train_loaders, eval_loader, ext_loader = make_loaders(0)
print(f"battery: target={M}M depth={DEPTH} seq={A.SEQ_LEN} batch={A.BATCH} "
      f"ckpt={A.CHECKPOINT} per_token_gn={A.PER_TOKEN_GN} steps={STEPS} ema={EMA_DECAY}")
print(f"model: unique={U} dim={dim} params={params/1e6:.2f}M\n")


def run_arm(name):
    kw = ARM_RECIPE[name]
    for _attr in ("FUSE_MODEL", "BATCH_PASSES"):
        setattr(A, _attr, _attr == COMPILE_FLAG.get(name, ""))
    build = lambda: build_nfra(VOCAB, dim, U, depth=DEPTH, use_cortex=True, **kw)
    m = build().to(DEVICE)
    sm = build().to(DEVICE)                      # sanity: finite-grad smoke
    _x = torch.randint(0, VOCAB, (A.BATCH, 32), device=DEVICE)
    _loss = F.cross_entropy(sm(_x, return_dict=True)["logits"][:, :-1].reshape(-1, VOCAB),
                            _x[:, 1:].reshape(-1))
    _loss.backward()
    del sm
    torch.cuda.empty_cache()

    r = train_one(m, VOCAB, STEPS, train_loaders[seed], eval_loader,
                  eval_gap=EVAL_GAP, ema_decay=EMA_DECAY, seed=seed)
    r["final_eval"] = float(evaluate(m, eval_loader))
    r["ext_eval"] = float(evaluate(m, ext_loader))
    r["sample_auc"] = sample_auc(r["eval_hist"])
    gm = generate_metrics(m, VOCAB)
    r["gen_tok_s"] = gm["gen_tok_s"]
    r["infer_mem"] = gm["infer_mem"]
    try:
        r.update(stateful_generate_metrics(m, VOCAB, device=DEVICE))
    except Exception:
        r.update({"gen_sf": None, "sf_abs": None, "sf_rel": None, "sf_ok": None})
    r["param_M"] = params / 1e6
    del m
    torch.cuda.empty_cache()
    return r


results = {n: run_arm(n) for n in arms}

# ── 4) factor table ──────────────────────────────────────────────────────────
FACTORS = ["final_eval", "ext_eval", "gen_tok_s", "gen_sf", "infer_mem",
           "peak_mem", "tok_s", "sample_auc", "ms_per_step"]
HDR = {"final_eval": ("eval", "{:.4f}"), "ext_eval": ("ext@Lx2", "{:.4f}"),
       "gen_tok_s": ("gen_slw/s", "{:.0f}"), "gen_sf": ("gen_sf/s", "{:.0f}"),
       "infer_mem": ("infer GB", "{:.3f}"), "peak_mem": ("mem GB", "{:.3f}"),
       "tok_s": ("tok/s", "{:.0f}"), "sample_auc": ("sampleAUC", "{:.3f}"),
       "ms_per_step": ("ms/step", "{:.1f}")}


def fmt(r, f):
    v = r.get(f)
    return HDR[f][1].format(v) if v is not None else "-"


print("\n── factor table ───────────────────────────────────────────────────────")
print("  " + f"{'arm':<11}" + "  ".join(f"{HDR[f][0]:>10}" for f in FACTORS) + f"  {'param':>6}")
for n, r in results.items():
    print("  " + f"{n:<11}" + "  ".join(f"{fmt(r, f):>10}" for f in FACTORS) + f"  {r['param_M']:.2f}M")

# ── 5) deltas vs baseline (+ = better) ───────────────────────────────────────
DIRMAP = {"final_eval": -1, "ext_eval": -1, "gen_tok_s": +1, "gen_sf": +1,
          "infer_mem": -1, "peak_mem": -1, "tok_s": +1, "sample_auc": -1}
print("\n── deltas vs baseline (+ = better) ─────────────────────────────────────")
print("  " + f"{'arm':<11}" + "  ".join(f"{HDR[f][0]:>10}" for f in FACTORS))
if "baseline" in results:
    base = results["baseline"]
    for n, r in results.items():
        if n == "baseline":
            continue
        cells = []
        for f in FACTORS:
            b, v = base.get(f), r.get(f)
            if b is None or v is None:
                cells.append(f"{'-':>10}")
            else:
                d = (v - b) * DIRMAP.get(f, 1)
                cells.append((f"{d:+.3f}" if abs(d) < 100 else f"{d:+.0f}").rjust(10))
        print("  " + f"{n:<11}" + "  ".join(cells))

# ── 6) composite METRIC_SPEC score ───────────────────────────────────────────
def composite(r, base):
    tot, ws = 0.0, 0.0
    for spec in METRIC_SPEC:
        k, d, w = spec["key"], spec["dir"], spec["w"]
        if k in ("extrap_delta", "scaling_gain", "param_eff"):
            continue
        b, v = base.get(k), r.get(k)
        if b is None or v is None or abs(b) < 1e-9:
            continue
        tot += d * (v - b) / abs(b) * w
        ws += w
    return tot / ws if ws else 0.0


print("\n── composite METRIC_SPEC score (0 = baseline, + = better) ──────────────")
if "baseline" in results:
    for n, r in results.items():
        print(f"  {n:<11}  {composite(r, results['baseline']):+.4f}")

# ── 7) stateful O(1) decode headline ─────────────────────────────────────────
print("\n── stateful O(1) decode headline ───────────────────────────────────────")
print(f"  {'arm':<11} {'slow/s':>7} {'stateful/s':>10} {'speedup':>8}  guard")
for n, r in results.items():
    slw, sf_, ok = r.get("gen_tok_s"), r.get("gen_sf"), r.get("sf_ok")
    sp = (sf_ / slw) if (slw and sf_) else None
    guard = "ok" if ok else ("-" if ok is None else "X")
    sp_str = f"{sp:>7.1f}x" if sp else f"{'-':>8}"
    print(f"  {n:<11} {slw or 0:>7.0f} {sf_ or 0:>10.0f} {sp_str}    {guard}")
