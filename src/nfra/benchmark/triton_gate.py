"""Kaggle T4 gate + A/B for the fused Triton chunked-retention kernel.

Phase 1 (GATE): run the fused kernel against the eager chunked reference on
CUDA across (fp32 / fp16-AMP) x (S in 64, 80, 256, 512) — forward max-diff
and gradient max-diff — plus a full-model logits max-diff vs the parallel
form. Any gate failure aborts before the benchmark so no loss-invalid number
is ever reported.

Phase 2 (A/B): 20M / 300 steps, eager-parallel (verified baseline) vs
triton-chunked, same seed -> byte-identical batch stream. Reports train loss,
eval loss, tok/s, peak mem, and the deltas. Loss drift > ~0.02 from the
parallel arm flags a regression.

Set env before the A/B:
  NFRA_GATE_STEPS  training steps per arm        (default 300)
  NFRA_GATE_EMA    EMA decay for train_one       (default 0.99, matches overnight)
  NFRA_GATE_DATA   wikitext2 | synthetic          (default wikitext2)

Run on a Kaggle T4 with:  python -u src/nfra/benchmark/triton_gate.py
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
from nfra.benchmark.compare import (
    DEVICE,
    HAS_CUDA,
    SEQ_LEN,
    USE_AMP,
    evaluate,
)
from nfra.core.retention import chunked_retention_eager
from nfra.core.retention_triton import _HAS_TRITON, chunked_retention

VOCAB = 96


def run_gate() -> None:
    """Raw-kernel forward + gradient gate on CUDA across dtype x seq combos."""
    if not (HAS_CUDA and _HAS_TRITON):
        raise RuntimeError(f"gate needs CUDA+Triton (cuda={HAS_CUDA} triton={_HAS_TRITON})")
    torch.manual_seed(0)
    H, Hd = 8, 32
    worst = {"fwd": 0.0, "grad": 0.0}
    for amp in (False, True):
        for S in (64, 80, 256, 512):
            q = torch.randn(4, H, S, Hd, device=DEVICE, requires_grad=True)
            k = torch.randn(4, H, S, Hd, device=DEVICE, requires_grad=True)
            v = torch.randn(4, H, S, Hd, device=DEVICE, requires_grad=True)
            l = torch.linspace(-5, 3, H, device=DEVICE).requires_grad_(True)
            with torch.amp.autocast(device_type=DEVICE.type, enabled=amp, dtype=torch.float16):
                yt = chunked_retention(q, k, v, l, 64, use_triton=True)
                ye = chunked_retention_eager(q, k, v, l, 64)
            if amp:
                yt, ye = yt.float(), ye.float()
            err_f = (yt - ye).abs().max().item()
            gt = torch.randn_like(ye)
            yt.backward(gt)
            gt_list = (q.grad, k.grad, v.grad, l.grad)
            for t in (q, k, v, l):
                t.grad = None
            ye.backward(gt)
            err_g = max(
                (gt_list[i] - t.grad).abs().max().item() for i, t in enumerate((q, k, v, l))
            )
            worst["fwd"] = max(worst["fwd"], err_f)
            worst["grad"] = max(worst["grad"], err_g)
            print(f"  gate amp={int(amp)} S={S}: fwd {err_f:.3e} grad {err_g:.3e}")
            assert err_f < 2e-2, f"forward mismatch amp={amp} S={S}: {err_f:.3e}"
            assert err_g < 2e-2, f"grad mismatch amp={amp} S={S}: {err_g:.3e}"
    print(f"GATE PASS (worst fwd {worst['fwd']:.3e}, grad {worst['grad']:.3e})")


def gate_model() -> None:
    """Full-model logits gate: triton-chunked vs parallel, identical weights."""
    torch.manual_seed(0)
    U, dim, params = tune_nfra_size(20_000_000, VOCAB, 33, DIM_GRID[20])
    a = build_nfra(VOCAB, dim, U, depth=33, use_cortex=True, chunk_size=0, triton=False).to(DEVICE)
    b = build_nfra(VOCAB, dim, U, depth=33, use_cortex=True, chunk_size=64, triton=True).to(DEVICE)
    with torch.no_grad():
        for pa, pb in zip(a.parameters(), b.parameters()):
            pb.copy_(pa)
    a.eval()
    b.eval()
    x = torch.randint(0, VOCAB, (4, SEQ_LEN), device=DEVICE)
    with torch.amp.autocast(device_type=DEVICE.type, dtype=torch.float16):
        la = a(x)["logits"].float()
        lb = b(x)["logits"].float()
    err = (la - lb).abs().max().item()
    print(f"  model logits max-diff {err:.3e} (params {params:.2f}M, dim {dim}, U {U})")
    assert err < 5e-2, f"model-level mismatch {err:.3e}"


def run_ab() -> None:
    """20M A/B: eager-parallel vs triton-chunked on identical batch streams."""
    U, dim, params = tune_nfra_size(20_000_000, VOCAB, 33, DIM_GRID[20])
    seed = SEED_LIST[0]
    # compare.py already falls back to synthetic when WikiText-2 is unavailable
    # (it flips DATA_SOURCE at import), so make_loaders can never raise here.
    train_loaders, eval_loader, _ = make_loaders(0)
    data = arena.DATA_SOURCE
    print(f"A/B data={data} seed={seed} params={params:.2f}M dim={dim} U={U} "
          f"steps={STEPS} ema={EMA} batch={train_loaders[seed].batch_size} seq={SEQ_LEN}")
    results = {}
    for name, chunk, triton in (
        ("parallel", 0, False),
        ("triton_chunked", 64, True),
    ):
        torch.cuda.empty_cache()
        m = build_nfra(
            VOCAB, dim, U, depth=33, use_cortex=True,
            chunk_size=chunk, triton=triton,
        ).to(DEVICE)
        r = train_one(
            m, VOCAB, STEPS, train_loaders[seed], eval_loader,
            eval_gap=EVAL_GAP, ema_decay=EMA, seed=seed,
        )
        ev = evaluate(m, eval_loader)
        r["final_eval"] = float(ev)
        results[name] = r
        print(
            f"  {name}: tok_s {r['tok_s']:.0f} mem {r['peak_mem']:.3f} GB "
            f"train {r['loss_hist'][-1]:.4f} eval {float(ev):.4f} "
            f"ms/step {r['ms_per_step']:.1f} nan {r['nan_steps']}"
        )
    print("── deltas (triton_chunked - parallel) ──")
    for key in ("tok_s", "peak_mem", "final_eval"):
        d = results["triton_chunked"][key] - results["parallel"][key]
        print(f"  {key}: {d:+.4f}")
    drift = results["triton_chunked"]["final_eval"] - results["parallel"]["final_eval"]
    if abs(drift) > 0.02:
        print(f"  WARN eval-loss drift {drift:+.4f} > 0.02 — check the gate")
    else:
        print("  eval-loss drift within tolerance")


def main() -> int:
    print(f"Triton fused chunked-retention gate+A/B  (triton={_HAS_TRITON}, cuda={HAS_CUDA}, "
          f"amp={USE_AMP}, steps={STEPS})")
    run_gate()
    gate_model()
    run_ab()
    return 0


if __name__ == "__main__":
    sys.exit(main())
