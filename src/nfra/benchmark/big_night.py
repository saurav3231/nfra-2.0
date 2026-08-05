# ═══════════════════════════════════════════════════════════════════════════
# NFRA-2.0 · OVERNIGHT BIG-RUN — max T4, streaming output, credible cross-family
#   NFRA (recommended recipe)  vs  RetNet  vs  RWKV  vs  GPT-2   (NO Mamba)
#   • multi-size (5/20/50M) × multi-seed (mean±std) × matched params
#   • real WikiText-2 char data, AdamW + cosine, EMA, AMP, big steps
#   • wall-clock budget (NFRA_WALL min) so it always finishes before Kaggle's
#     session limit; prints frequent progress + ETA
#   • saves /kaggle/working/big_night_results.json
# Set once: CONFIG below. Paste the whole cell into a Kaggle GPU notebook.
# ═══════════════════════════════════════════════════════════════════════════
import os, sys, time, subprocess, importlib, json, math

REPO = "/kaggle/working/nfra-2.0"
if not os.path.isdir(REPO):
    subprocess.run(["git", "clone", "https://github.com/saurav3231/nfra-2.0.git", REPO], check=True)
else:
    subprocess.run(["git", "-C", REPO, "pull", "--ff-only"], check=True)
subprocess.run(["pip", "install", "-q", "-e", REPO], check=True)
if REPO + "/src" not in sys.path:
    sys.path.insert(0, REPO + "/src")

# ── CONFIG (edit here) ──────────────────────────────────────────────────────
CONFIG = {
    "NFRA_RECOMMENDED": "1",     # LSR + per-token GN (verified best recipe)
    "NFRA_DATA": "wikitext2",    # real char text (vocab 96) — credible
    "NFRA_SEQ": "256",
    "NFRA_BATCH": "16",
    "NFRA_EMA": "0.99",
    "NFRA_CHECKPOINT": "1",      # memory headroom for the 50M stack
    "NFRA_COMPILE": "0",         # identical eager treatment for ALL families (fair)
    "NFRA_AMP": "1",
    "NFRA_MODE": "rigorous",
}
for _k, _v in CONFIG.items():
    os.environ[_k] = _v
SIZES_M = [int(x) for x in os.environ.get("NFRA_BIG_SIZES", "20,50").split(",") if x.strip()]
SEED_CNT = int(os.environ.get("NFRA_BIG_SEEDS", "3"))
STEPS_BIG = int(os.environ.get("NFRA_BIG_STEPS", "3000"))
WALL_MIN = int(os.environ.get("NFRA_WALL", "480"))   # wall-clock budget (min)
FAMILIES = ["nfra", "retnet", "rwkv", "gpt2"]        # Mamba intentionally excluded
EVAL_GAP = max(50, STEPS_BIG // 8)

import torch
assert torch.cuda.is_available(), "needs a T4"
DEVICE = torch.device("cuda")
print("GPU:", torch.cuda.get_device_name(0), "| wall budget:", WALL_MIN, "min")

import nfra.benchmark.arena as A
import nfra.benchmark.compare as C
importlib.reload(C)
importlib.reload(A)
from nfra.benchmark.compare import (
    DataLoader, WikiText2Dataset, HierarchicalDataset, count_params, evaluate,
    make_optimizer, EMA, compute_loss, rescale_embed, measure_speed_memory,
    RetNetLM, RWKVLM, GPT2ForCausalLM, SEQ_LEN, BATCH, USE_AMP,
)
from nfra.benchmark.arena import build_nfra
from nfra.core.stateful import stateful_generate_metrics

VOCAB = 96
DIMS = [768, 704, 640, 576, 512, 448, 384, 352, 320, 288, 256, 224, 192, 160, 128]


def tune_nfra(depth, target):
    # unique == depth => depth_passes == 1, which the stateful O(1) dual requires
    # to be exact (see core/stateful.supported). Params are matched via dim.
    best = None
    for d in DIMS:
        p = count_params(build_nfra(VOCAB, d, depth, depth=depth))
        err = abs(p - target)
        if best is None or err < best[0]:
            best = (err, d, p)
    return best[1], best[2]


def tune_baseline(make_fn, dim, target):
    best = (1, float("inf"))
    for L in range(1, 64):
        p = count_params(make_fn(VOCAB, dim, L))
        if abs(p - target) < abs(best[1] - target):
            best = (L, p)
        if p >= target * 1.15:
            break
    return best[0]


def build_family(name, dim, target):
    if name == "nfra":
        depth = 8
        d, _ = tune_nfra(depth, target)
        m = build_nfra(VOCAB, d, depth, depth=depth)
        return m, f"8U×1p dim{d}"
    if name == "retnet":
        L = tune_baseline(RetNetLM, dim, target)
        return RetNetLM(VOCAB, dim, L), f"{L}L"
    if name == "rwkv":
        L = tune_baseline(RWKVLM, dim, target)
        return RWKVLM(VOCAB, dim, L, dropout=0.1), f"{L}L"
    if name == "gpt2":
        L = tune_baseline(GPT2ForCausalLM, dim, target)
        return GPT2ForCausalLM(VOCAB, dim, L), f"{L}L"
    raise ValueError(name)


def train(seed, steps, train_ds, eval_loader, model, vocab):
    model.train()
    opt, sched = make_optimizer(model, lr=3e-4, warmup=min(50, max(steps // 10, 1)), total=steps)
    scaler = torch.amp.GradScaler("cuda") if USE_AMP else None
    ema = EMA(model, float(C.EMA_DECAY)) if C.EMA_DECAY > 0 else None
    it = iter(DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                         generator=torch.Generator().manual_seed(seed),
                         num_workers=0, pin_memory=True))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    hist = {"loss": [], "eval": []}
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                                 generator=torch.Generator().manual_seed(seed),
                                 num_workers=0, pin_memory=True))
            x, y = next(it)
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        with torch.amp.autocast(device_type="cuda", enabled=USE_AMP):
            loss = compute_loss(model, x, y)
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if math.isfinite(gnorm):
                scaler.step(opt)
            else:
                opt.zero_grad(set_to_none=True)
            scaler.update()
        else:
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if math.isfinite(gnorm):
                opt.step()
            else:
                opt.zero_grad(set_to_none=True)
        sched.step()
        if ema is not None:
            ema.update(model)
        hist["loss"].append(loss.item())
        if step % EVAL_GAP == 0 or step == 1 or step == steps:
            if ema is not None:
                ema.apply(model)
            ev = evaluate(model, eval_loader, max_batches=30)
            if ema is not None:
                ema.restore(model)
            hist["eval"].append((step, ev))
            el = time.perf_counter() - t0
            sps = el / step
            eta = sps * (steps - step)
            print(
                f"    step {step:>5d}/{steps}  train {loss.item():.4f}  "
                f"eval {ev:.4f}  {sps*1000:6.1f} ms/step  "
                f"ETA {eta/60:5.1f}m  wall {el/60:5.1f}m", flush=True
            )
    return hist


wall0 = time.perf_counter()
if C.DATA_SOURCE == "wikitext2":  # real WikiText-2 char text (vocab 96)
    train_ds = WikiText2Dataset("train", SEQ_LEN)
    eval_ds = WikiText2Dataset("validation", SEQ_LEN)
    VOCAB = 96
else:  # synthetic hierarchical topics (vocab 4096) — mirror compare's fallback
    train_ds = HierarchicalDataset(max(4096, BATCH * 8), SEQ_LEN + 1, seed=42, seq_seed=42)
    eval_ds = HierarchicalDataset(512, SEQ_LEN + 1, seed=42, seq_seed=43)
    VOCAB = HierarchicalDataset.VOCAB_SIZE
RANDOM_LOSS = math.log(VOCAB)
eval_loader = DataLoader(eval_ds, batch_size=BATCH, shuffle=False, num_workers=0)

SEEDS = [42, 7, 2026, 1337, 777][:SEED_CNT]
all_results = {}
for size in SIZES_M:
    target = int(size * 1e6)
    dim = 448 if size <= 20 else 704
    print(f"\n{'='*72}\nSIZE {size}M  target={target/1e6:.0f}M  dim={dim}  "
          f"steps={STEPS_BIG}  seeds={SEEDS}  seq={SEQ_LEN}  batch={BATCH}\n{'='*72}")
    for fam in FAMILIES:
        for seed in SEEDS:
            if (time.perf_counter() - wall0) / 60 > WALL_MIN:
                print(f"[wall budget {WALL_MIN}m hit — stopping]", flush=True)
                break
            torch.cuda.empty_cache()
            m, desc = build_family(fam, dim, target)
            m = m.to(DEVICE)
            rescale_embed(m)
            n_p = count_params(m)
            print(f"\n── {fam:6s} seed {seed}  {n_p/1e6:6.2f}M ({desc})")
            tok_s, mem = measure_speed_memory(m, VOCAB)
            print(f"    throughput {int(tok_s):,} tok/s   peak {mem:.2f} GB")
            hist = train(seed, STEPS_BIG, train_ds, eval_loader, m, VOCAB)
            final = hist["eval"][-1][1]
            key = (fam, size, seed)
            r = {"final_eval": final, "ppl": math.exp(min(final, 30)),
                 "tok_s": int(tok_s), "mem": round(mem, 3), "history": hist}
            if fam == "nfra":
                sf = stateful_generate_metrics(m, VOCAB, device=DEVICE)
                r.update({"gen_sf": sf["gen_sf"], "sf_ok": sf["sf_ok"]})
            all_results[f"{fam}|{size}M|{seed}"] = r
            print(f"    DONE {fam} {size}M seed {seed}: final eval {final:.4f}"
                  + (f"  gen_sf {sf['gen_sf']:.0f}/s {sf['sf_ok']}" if fam == "nfra" else ""),
                  flush=True)
            del m
            torch.cuda.empty_cache()

print("\n" + "=" * 72)
print("  FINAL — mean ± std across seeds (eval loss, lower = better)")
print("=" * 72)
summary = {}
for size in SIZES_M:
    for fam in FAMILIES:
        vals = [all_results[f"{fam}|{size}M|{s}"]["final_eval"] for s in SEEDS
                if f"{fam}|{size}M|{s}" in all_results]
        if not vals:
            continue
        mean, std = sum(vals) / len(vals), (sum((v - sum(vals)/len(vals))**2 for v in vals)/len(vals))**0.5
        tok = all_results[f"{fam}|{size}M|{SEEDS[0]}"]["tok_s"]
        mem = all_results[f"{fam}|{size}M|{SEEDS[0]}"]["mem"]
        sf = all_results[f"{fam}|{size}M|{SEEDS[0]}"].get("gen_sf")
        line = f"  {fam:6s} {size:>3d}M  eval {mean:7.4f} ± {std:.4f}   ppl≈{math.exp(min(mean,30)):6.1f}  tok/s {tok:>7,}  mem {mem:5.2f}G"
        if sf:
            line += f"  gen_sf {sf:.0f}/s ok={all_results[f'{fam}|{size}M|{SEEDS[0]}']['sf_ok']}"
        print(line)
        summary[f"{fam}|{size}M"] = {"eval_mean": mean, "eval_std": std, "tok_s": tok, "mem": mem}

for size in SIZES_M:
    fams = [f for f in FAMILIES if f"{f}|{size}M" in summary]
    if fams:
        best = min(fams, key=lambda f: summary[f"{f}|{size}M"]["eval_mean"])
        print(f"  [{size}M] best quality: {best}  eval {summary[f'{best}|{size}M']['eval_mean']:.4f}")

out = os.path.join(os.getcwd(), "big_night_results.json")
with open(out, "w") as f:
    json.dump({"config": CONFIG, "sizes_M": SIZES_M, "seeds": SEEDS,
               "steps": STEPS_BIG, "summary": summary, "all": all_results}, f, indent=2)
print(f"\n  results saved → {out}")
