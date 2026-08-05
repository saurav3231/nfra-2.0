"""T4 A/B battery over the six Tier-1 experiment ideas (all opt-in, default off).

Runs 20M / NFRA_GATE_STEPS (default 300) with the SAME seed for every arm so all
arms consume byte-identical batches. Arm 0 is the verified baseline (every knob
off, exact 1.7 reproduction); each following arm flips exactly ONE idea:

  0 baseline     all flags off (verified parallel retention, eager)
  1 triton_chunk NFRA_CHUNK_SIZE=64 + NFRA_TRITON=1  (fused one-launch retention)
  2 lsr          NFRA_LSR=1        per-head learned long/short route (*loss)
  3 int8_state   NFRA_CHUNK_SIZE=64 + NFRA_INT8_STATE=1  int8 long-range state (*loss)
  4 depth_time   NFRA_DEPTH_TIME=1 continuous depth-pass FiLM (*loss)
  5 batch_pass   NFRA_BATCH_PASSES=1  fused shared-weight depth loop (compile)
  6 fuse_model   NFRA_FUSE_MODEL=1   whole-model single-graph fusion (compile)

Columns: train loss, final eval loss, tok/s, peak mem, ms/step. *loss arms are
allowed to move the loss — that is the point; the other arms must stay within
~0.02 of baseline or they are regressions.

Run on a Kaggle T4 (the one-cell bootstrap):
  !git clone https://github.com/saurav3231/nfra-2.0.git nfra && cd nfra
  !pip install -q -e . triton
  !python -u src/nfra/benchmark/experiments_gate.py
"""

import os
import sys

os.environ["NFRA_CHECKPOINT"] = os.environ.get("NFRA_GATE_CKPT", "0")
os.environ["NFRA_COMPILE"] = "0"
os.environ["NFRA_SCAN_KERNEL"] = "0"
os.environ["NFRA_CHUNK_SIZE"] = "0"
os.environ["NFRA_CKPT_GEMM"] = "0"
os.environ["NFRA_BATCH"] = os.environ.get("NFRA_GATE_BATCH") or os.environ.get("NFRA_BATCH") or ""
os.environ["NFRA_EMA"] = os.environ.get("NFRA_GATE_EMA", "0.99")
os.environ["NFRA_SEQ"] = os.environ.get("NFRA_GATE_SEQ") or os.environ.get("NFRA_SEQ") or "256"
os.environ["NFRA_DATA"] = os.environ.get("NFRA_GATE_DATA", "wikitext2").lower()
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Only propagate BATCH / SEQLEN overrides when a real (non-empty) value exists;
# leaving them unset lets compare.py / arena pick their own defaults.
for _k in ("NFRA_BATCH", "NFRA_SEQ"):
    if not os.environ.get(_k, "").strip():
        os.environ.pop(_k, None)

STEPS = int(os.environ.get("NFRA_GATE_STEPS", "300"))
EMA = float(os.environ["NFRA_EMA"])
# NFRA_GATE_TARGET_M: param budget to tune to (5/20/50). NFRA_GATE_ARMS: a
# comma-list of arm names to run (empty = all). Both let a single run target
# one goal — loss (50M, lsr, rigorous steps) vs memory (ckpt, small batch).
# NFRA_GATE_DEPTH: effective layer count (default 33 = the verified deep stack).
# The step is compute-bound across the sequential blocks, so fewer layers give
# an almost-linear tok/s win at a small loss cost (dim is re-tuned to keep the
# same 20M budget): depth 33 -> 16 roughly doubles tok/s toward ~30k.
TARGET_M = int(os.environ.get("NFRA_GATE_TARGET_M", "20"))
DEPTH = int(os.environ.get("NFRA_GATE_DEPTH", "33"))
ARMS = {
    "baseline":     dict(chunk_size=0, triton=False, lsr=False, int8_state=False, depth_time=False),
    "triton_chunk": dict(chunk_size=64, triton=True),
    "lsr":          dict(lsr=True),
    "int8_state":   dict(chunk_size=64, int8_state=True),
    "depth_time":   dict(depth_time=True),
    "batch_pass":   dict(),
    "fuse_model":   dict(),
    # "rev" = the optimum balanced recipe: lsr (proven -0.078 eval) + default
    # torch.compile (fuses the 33-block launch stream: the documented fix for
    # the 11.5k tok/s bottleneck, and it reuses buffers -> also cuts peak mem).
    # lsr and compile compose (compile wraps the whole model, lsr is interior).
    "rev":          dict(lsr=True),
}

import torch

from nfra.benchmark import arena
from nfra.benchmark.arena import (
    BATCH,
    CHECKPOINT,
    DIM_GRID,
    EVAL_GAP,
    SEQ_LEN,
    SEED_LIST,
    build_nfra,
    generate_metrics,
    make_loaders,
    sample_auc,
    train_one,
    tune_nfra_size,
)
from nfra.benchmark.compare import DEVICE, HAS_CUDA, evaluate

# Model vocab must match the ACTUAL data source. WikiText-2 char vocab is sized
# 96 (95 real entries + one unused row); the synthetic fallback
# (HierarchicalDataset) emits tokens up to VOCAB_SIZE-1, so building the model
# at 96 there crashes the embedding gather with "index out of bounds" on CUDA.
if arena.DATA_SOURCE == "wikitext2":
    VOCAB = 96
else:
    from nfra.benchmark.compare import HierarchicalDataset

    VOCAB = HierarchicalDataset.VOCAB_SIZE


def _sanity(build, label):
    """Forward + backward finite-grad smoke for an arm's model on CPU/GPU.

    Uses S=256 (> any chunk_size) so the chunked/Triton retention path is
    actually exercised — a kernel bug that only shows at S > chunk_size must
    abort here, never report a garbage loss. A loose CE bound on a random batch
    catches structurally-wrong kernels: the broken Triton path measured ~171,
    while a healthy arm stays well under 20. (Init logits are large on this
    architecture, so init CE can be ~0 via softmax saturation; only the upper
    bound is a real regression signal.)"""
    m = build().to(DEVICE)
    x = torch.randint(0, VOCAB, (2, 256), device=DEVICE)
    out = m(x)
    logits = out["logits"].float()
    ce = torch.nn.functional.cross_entropy(
        logits.view(-1, VOCAB), x.view(-1)
    ).item()
    assert ce == ce and ce < 20.0, f"{label} forward broken: CE {ce:.2f}"
    logits.mean().backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads), f"{label} grads"
    print(f"  sanity {label}: CE {ce:.2f} forward/backward OK")
    del m, grads, out, logits
    if HAS_CUDA:
        torch.cuda.empty_cache()


def main() -> int:
    if not HAS_CUDA:
        print("experiments_gate.py needs a CUDA GPU (T4). This box is CPU-only.")
        return 1
    print(
        f"T4 experiment battery  steps={STEPS} ema={EMA} data={arena.DATA_SOURCE} "
        f"target={TARGET_M}M depth={DEPTH} seq={SEQ_LEN} batch={BATCH} ckpt={arena.CHECKPOINT}"
    )

    U, dim, _params = tune_nfra_size(
        TARGET_M * 1_000_000, VOCAB, DEPTH, DIM_GRID[TARGET_M]
    )
    seed = SEED_LIST[0]
    train_loaders, eval_loader, ext_loader = make_loaders(0)

    select = [s.strip() for s in os.environ.get("NFRA_GATE_ARMS", "").split(",") if s.strip()]
    arms = [(n, k) for n, k in ARMS.items() if not select or n in select]
    compile_arm = {
        "batch_pass": "BATCH_PASSES",
        "fuse_model": "FUSE_MODEL",
        "rev": "BATCH_PASSES",
    }

    results = {}
    for name, kw in arms:
        if not torch.cuda.is_available():
            break
        torch.cuda.empty_cache()
        for attr in ("FUSE_MODEL", "BATCH_PASSES"):
            setattr(arena, attr, attr in (compile_arm.get(name, ""),))
        build = lambda n=name, k=dict(kw): build_nfra(
            VOCAB, dim, U, depth=DEPTH, use_cortex=True, **k,
        )
        _sanity(build, name)
        m = build().to(DEVICE)
        r = train_one(
            m, VOCAB, STEPS, train_loaders[seed], eval_loader,
            eval_gap=EVAL_GAP, ema_decay=EMA, seed=seed,
        )
        ev = evaluate(m, eval_loader)
        r["final_eval"] = float(ev)
        r["ext_eval"] = float(evaluate(m, ext_loader))  # long-context (seq*2)
        r["sample_auc"] = sample_auc(r["eval_hist"])
        gm = generate_metrics(m, VOCAB)
        r["gen_tok_s"] = gm["gen_tok_s"]
        r["infer_mem"] = gm["infer_mem"]
        results[name] = r
        ext_d = r["ext_eval"] - r["final_eval"]
        print(
            f"  {name:11s} train {r['loss_hist'][-1]:.4f}  eval {r['final_eval']:.4f}  "
            f"ext {r['ext_eval']:.4f}(d{ext_d:+.3f})  tok/s {r['tok_s']:.0f}  "
            f"mem {r['peak_mem']:.3f} GB  gen {r['gen_tok_s']:.0f}/s  "
            f"infer {r['infer_mem']:.3f} GB  auc {r['sample_auc']:.3f}  "
            f"ms/step {r['ms_per_step']:.1f}"
        )
        del m
        torch.cuda.empty_cache()

    if "baseline" in results:
        base = results["baseline"]
        print("\n── deltas vs baseline ──")
        for name, r in results.items():
            if name == "baseline":
                continue
            print(
                f"  {name:11s} eval {r['final_eval'] - base['final_eval']:+.4f}   "
                f"ext {r['ext_eval'] - base['ext_eval']:+.4f}   "
                f"tok/s {r['tok_s'] - base['tok_s']:+.0f}   "
                f"mem {r['peak_mem'] - base['peak_mem']:+.3f}   "
                f"gen_inf {r['gen_tok_s'] - base['gen_tok_s']:+.0f}   "
                f"infer_mem {r['infer_mem'] - base['infer_mem']:+.3f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
