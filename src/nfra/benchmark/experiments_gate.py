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

os.environ["NFRA_CHECKPOINT"] = "0"
os.environ["NFRA_COMPILE"] = "0"
os.environ["NFRA_SCAN_KERNEL"] = "0"
os.environ["NFRA_CHUNK_SIZE"] = "0"
os.environ["NFRA_CKPT_GEMM"] = "0"
os.environ["NFRA_EMA"] = os.environ.get("NFRA_GATE_EMA", "0.99")
os.environ["NFRA_DATA"] = os.environ.get("NFRA_GATE_DATA", "wikitext2").lower()
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

STEPS = int(os.environ.get("NFRA_GATE_STEPS", "300"))
EMA = float(os.environ["NFRA_EMA"])

import torch

from nfra.benchmark import arena
from nfra.benchmark.arena import (
    DIM_GRID,
    EVAL_GAP,
    SEED_LIST,
    build_nfra,
    make_loaders,
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
    """Forward + backward finite-grad smoke for an arm's model on CPU/GPU."""
    m = build().to(DEVICE)
    x = torch.randint(0, VOCAB, (2, 64), device=DEVICE)
    out = m(x)
    out["logits"].float().mean().backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads), f"{label} grads"
    print(f"  sanity {label}: forward/backward OK")
    del m, grads
    if HAS_CUDA:
        torch.cuda.empty_cache()


def main() -> int:
    if not HAS_CUDA:
        print("experiments_gate.py needs a CUDA GPU (T4). This box is CPU-only.")
        return 1
    print(f"T4 experiment battery  steps={STEPS} ema={EMA} data={arena.DATA_SOURCE}")

    U, dim, _params = tune_nfra_size(20_000_000, VOCAB, 33, DIM_GRID[20])
    seed = SEED_LIST[0]
    train_loaders, eval_loader, _ = make_loaders(0)

    arms = [
        ("baseline",     dict(chunk_size=0, triton=False, lsr=False, int8_state=False, depth_time=False)),
        ("triton_chunk", dict(chunk_size=64, triton=True)),
        ("lsr",          dict(lsr=True)),
        ("int8_state",   dict(chunk_size=64, int8_state=True)),
        ("depth_time",   dict(depth_time=True)),
        ("batch_pass",   dict()),
        ("fuse_model",   dict()),
    ]
    compile_arm = {"batch_pass": "BATCH_PASSES", "fuse_model": "FUSE_MODEL"}

    results = {}
    for name, kw in arms:
        if not torch.cuda.is_available():
            break
        torch.cuda.empty_cache()
        for attr in ("FUSE_MODEL", "BATCH_PASSES"):
            setattr(arena, attr, attr in (compile_arm.get(name, ""),))
        build = lambda n=name, k=dict(kw): build_nfra(
            VOCAB, dim, U, depth=33, use_cortex=True, **k,
        )
        _sanity(build, name)
        m = build().to(DEVICE)
        r = train_one(
            m, VOCAB, STEPS, train_loaders[seed], eval_loader,
            eval_gap=EVAL_GAP, ema_decay=EMA, seed=seed,
        )
        ev = evaluate(m, eval_loader)
        r["final_eval"] = float(ev)
        results[name] = r
        print(
            f"  {name:11s} train {r['loss_hist'][-1]:.4f}  eval {float(ev):.4f}  "
            f"tok/s {r['tok_s']:.0f}  mem {r['peak_mem']:.3f} GB  ms/step {r['ms_per_step']:.1f}"
        )
        del m
        torch.cuda.empty_cache()

    base = results["baseline"]
    print("\n── deltas vs baseline ──")
    for name, r in results.items():
        if name == "baseline":
            continue
        print(
            f"  {name:11s} eval {r['final_eval'] - base['final_eval']:+.4f}   "
            f"tok/s {r['tok_s'] - base['tok_s']:+.0f}   "
            f"mem {r['peak_mem'] - base['peak_mem']:+.3f} GB"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
