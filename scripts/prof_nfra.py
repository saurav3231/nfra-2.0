"""Profile nfra vs retnet forward+backward on the active device (Kaggle T4).

Mirrors arena.train_one exactly: dim 112 depth 33, batch 8, seq 256, fp16 AMP,
GradScaler, torch.compile(mode='reduce-overhead'). Usage:

    python scripts/prof_nfra.py          # nfra (default)
    PROF_ARCH=retnet python scripts/prof_nfra.py
    PROF_BOTH=1 python scripts/prof_nfra.py    # both, one after the other
"""

import math
import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NFRA_CORTEX", "1")
os.environ.setdefault("NFRA_SIZES", "5")
os.environ.setdefault("NFRA_SEEDS", "1")

from nfra.benchmark import arena
from nfra.benchmark.compare import compute_loss

torch.manual_seed(0)
torch.set_float32_matmul_precision("medium")

B, S, V = 8, 256, 96
STEPS = 10


def make(arch):
    spec = arena.build_family_spec(arch, 5, V)
    return spec["builder"](V, spec["dim"], **spec["extra"])


def bench(arch):
    print(f"\n=== building {arch} ... ===")
    model = make(arch).to(arena.DEVICE)
    model.train()
    if arena.COMPILE and arena.HAS_CUDA:
        try:
            cfg = getattr(model, "config", None)
            model = torch.compile(model, mode="reduce-overhead", dynamic=False)
            print("  [compile] reduce-overhead active")
        except Exception as e:
            print("  [warn] compile failed (%s)" % e)
    opt, sched = arena.make_optimizer(model, lr=3e-4, warmup=5, total=STEPS)
    scaler = torch.amp.GradScaler(str(arena.DEVICE)) if arena.USE_AMP else None

    x = torch.randint(0, V, (B, S), device=arena.DEVICE)
    y = torch.randint(0, V, (B, S), device=arena.DEVICE)

    def step():
        opt.zero_grad()
        with torch.amp.autocast(device_type=arena.DEVICE.type, enabled=arena.USE_AMP):
            loss = compute_loss(model, x, y)
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if math.isfinite(gnorm):
                opt.step()
        sched.step()

    for _ in range(4):
        step()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(STEPS):
        step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / STEPS * 1e3
    print(f"=== {arch}: {ms:.1f} ms/step  ({B*S*STEPS/ (time.perf_counter()-t0) * 1000:.0f} tok/s) ===")

    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            step()
        torch.cuda.synchronize()
    ev = prof.key_averages()
    top = sorted(ev, key=lambda e: e.self_device_time_total, reverse=True)[:16]
    for e in top:
        print(f"  {e.self_device_time_total/1e3:8.1f} ms  {e.key}")


if os.environ.get("PROF_BOTH", "0") == "1":
    bench("nfra")
    bench("retnet")
else:
    bench(os.environ.get("PROF_ARCH", "nfra"))
