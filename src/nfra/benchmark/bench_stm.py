# ═══════════════════════════════════════════════════════════════════════════
# bench_stm — measure the RSM short-term working-tag ring vs the baseline.
#
# The ring-`on` model is a deepcopy of the `off` baseline with a windowed ring
# attached (output zero-init), so both START from identical weights and the on
# model is bit-identical at init. Both train same steps/data/optimizer, then:
#   * init parity   (max|off - on| at init ~= 0)
#   * eval delta    (end eval off vs on -> must not regress)
#   * speed         (ms/step off vs on)
#   * O(1) stateful (sf_ok / max_rel for the ring model)
# Env: NFRA_STM_STEPS, NFRA_STM_WINDOW, NFRA_STM_DIM, NFRA_STM_SIZE_M
# Paste into a Kaggle GPU notebook; or run on CPU with tiny NFRA_STM_STEPS.
# ═══════════════════════════════════════════════════════════════════════════
import os, sys, math, copy, time
sys.path.insert(0, r"A:\Project\NFRA-2.0\src")
os.environ.update({
    "NFRA_DATA": "synthetic", "NFRA_SEQ": "256", "NFRA_MODE": "standard",
    "NFRA_EMA": "0.99", "NFRA_PERTOKEN_GN": "1", "NFRA_RECOMMENDED": "1",
})
import importlib
import torch
import nfra.benchmark.arena as A
import nfra.benchmark.compare as C
importlib.reload(C)
importlib.reload(A)
from nfra.benchmark.compare import (
    HierarchicalDataset, DataLoader, count_params, SEQ_LEN, BATCH,
)
from nfra.core.cortex import CortexWorkingMemory
from nfra.core.stateful import stateful_generate_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
steps = int(os.environ.get("NFRA_STM_STEPS", "200"))
WINDOW = int(os.environ.get("NFRA_STM_WINDOW", "16"))
STM_DIM = int(os.environ.get("NFRA_STM_DIM", "32"))
SIZE_M = int(os.environ.get("NFRA_STM_SIZE_M", "20"))
VOCAB = HierarchicalDataset.VOCAB_SIZE if C.DATA_SOURCE == "synthetic" else 96
SEED = int(os.environ.get("NFRA_STM_SEED", "42"))


def build():
    from nfra import NFRAConfig, NFRAForCausalLM
    dim = 448 if SIZE_M <= 20 else 704
    depth = 8
    cfg = NFRAConfig(
        mode="brain", vocab_size=VOCAB, hidden_size=dim, num_layers=depth,
        depth_shared=False, unique_blocks=depth, use_cortex=True, n_bands=16,
        gradient_checkpointing=True, k_wta_frac=0.0, cortex_lsr=False,
        iso_gland=False, iso_vgate=True, iso_phase=True, iso_exit=False,
        cortex_per_token_gn=True, cortex_stm_ring=0, cortex_stm_dim=STM_DIM,
        dropout=0.1,
    )
    m = NFRAForCausalLM(cfg).to(DEVICE).train()
    return m


def make_loaders():
    ds_tr = HierarchicalDataset(max(4096, BATCH * 8), SEQ_LEN + 1, seed=SEED, seq_seed=SEED)
    ds_ev = HierarchicalDataset(512, SEQ_LEN + 1, seed=SEED, seq_seed=SEED + 1)
    tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True,
                    generator=torch.Generator().manual_seed(SEED), num_workers=0)
    ev = DataLoader(ds_ev, batch_size=BATCH, shuffle=False, num_workers=0)
    return tr, ev


# ── baseline + ring copy (identical init; ring read adds 0)
base = build()
on = copy.deepcopy(base)
for layer in on.layers:
    wm = CortexWorkingMemory(base.config.hidden_size, WINDOW, STM_DIM).to(DEVICE).train()
    layer.mixer.stm = wm  # forward() + stateful dual read it via mixer.stm

x = torch.randint(0, VOCAB, (2, 16))
with torch.no_grad():
    base.eval(); on.eval()
    p0 = float((base(x)["logits"] - on(x)["logits"]).abs().max())
    base.train(); on.train()
print(f"[init parity] max|off - on| = {p0:.3e}   (must be ~0: zero-init read)")

tr, ev = make_loaders()


def run(model):
    from nfra.benchmark.arena import train_one
    t0 = time.perf_counter()
    h = train_one(model, VOCAB, steps, tr, ev, eval_gap=max(20, steps // 4),
                  ema_decay=float(C.EMA_DECAY), surprise=False, seed=SEED)
    dt = time.perf_counter() - t0
    final = h["eval_hist"][-1][1] if h["eval_hist"] else float("nan")
    return h, final, dt * 1000 / steps  # ms/step


h_off, f_off, ms_off = run(base)
h_on, f_on, ms_on = run(on)

n_ring = sum(p.numel() for layer in on.layers
             for p in layer.mixer.stm.parameters()) if getattr(on.layers[0].mixer, "stm", None) else 0
print("\n────────────── RSM STM result ──────────────")
print(f"  baseline(off) eval {f_off:.3f}   {ms_off:.1f} ms/step")
print(f"  ring(open)    eval {f_on:.3f}   {ms_on:.1f} ms/step")
print(f"  Δ eval (on - off) = {f_on - f_off:+.3f}   (<=0 = no regression)")
print(f"  speed Δ          = {ms_on / ms_off * 100 - 100:+.1f}%")
print(f"  ring params      = {n_ring:,}")
sf = stateful_generate_metrics(on, VOCAB, prompt_len=64, gen_len=16, device=DEVICE)
print(f"  O(1) stateful    = sf_ok {sf['sf_ok']}  max_rel {sf['sf_rel']:.2e}  gen_sf {sf['gen_sf']:.1f}/s")
print("BENCH_DONE")