# ═══════════════════════════════════════════════════════════════════════════
# recall_probe — RSM STM recall/generalization A/B (stable hierarchical task)
#
# The pure copy-recall task (density 1.0, single lag) destabilized training
# (logits blew up -> eval ~138) and is NOT a reliable probe. This version uses
# the same HierarchicalDataset pipeline that big_night trains stably to ~8.3:
#   train = seq_seed SEED   |   eval = seq_seed SEED+1  (UNSEEN content, same
# grammar) -> real generalization/recall, not memorization.
# Ring-off vs ring-on, identical init, identical batches, longer training.
# Verdict: HELPS (Δ<=-0.004) / PASS (-0.004<Δ<=+0.004) / REGRESSED (Δ>+0.004).
# Env: NFRA_RECALL_STEPS, NFRA_STM_WINDOW, NFRA_STM_DIM, NFRA_STM_SIZE_M
# ═══════════════════════════════════════════════════════════════════════════
import os, sys, copy, time
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
from nfra.benchmark.compare import HierarchicalDataset, DataLoader, SEQ_LEN, BATCH
from nfra.core.cortex import CortexWorkingMemory
from nfra.core.stateful import stateful_generate_metrics

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STEPS   = int(os.environ.get("NFRA_RECALL_STEPS", "2000"))
WINDOW  = int(os.environ.get("NFRA_STM_WINDOW", "16"))
STM_DIM = int(os.environ.get("NFRA_STM_DIM", "32"))
SIZE_M  = int(os.environ.get("NFRA_STM_SIZE_M", "20"))
SEED    = int(os.environ.get("NFRA_STM_SEED", "42"))
VOCAB   = HierarchicalDataset.VOCAB_SIZE  # 4096


def make_loaders():
    ds_tr = HierarchicalDataset(max(4096, BATCH * 8), SEQ_LEN + 1, seed=SEED, seq_seed=SEED)
    ds_ev = HierarchicalDataset(512, SEQ_LEN + 1, seed=SEED, seq_seed=SEED + 1)  # unseen content
    tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True,
                    generator=torch.Generator().manual_seed(SEED), num_workers=0)
    ev = DataLoader(ds_ev, batch_size=BATCH, shuffle=False, num_workers=0)
    return tr, ev


def build():
    from nfra import NFRAConfig, NFRAForCausalLM
    dim = 448 if SIZE_M <= 20 else 704
    cfg = NFRAConfig(mode="brain", vocab_size=VOCAB, hidden_size=dim, num_layers=8,
        depth_shared=False, unique_blocks=8, use_cortex=True, n_bands=16,
        gradient_checkpointing=True, k_wta_frac=0.0, cortex_lsr=False,
        iso_gland=True, iso_vgate=True, iso_rgate=False, iso_phase=True, iso_exit=True,
        cortex_per_token_gn=True, cortex_stm_ring=0, cortex_stm_dim=STM_DIM,
        dropout=0.1)
    return NFRAForCausalLM(cfg).to(DEVICE).train()


if __name__ == "__main__":
    base = build()
    on = copy.deepcopy(base)
    for layer in on.layers:
        layer.mixer.stm = CortexWorkingMemory(base.config.hidden_size, WINDOW, STM_DIM).to(DEVICE).train()

    def run(model):
        from nfra.benchmark.arena import train_one
        _tr, _ev = make_loaders()  # fresh, same seed => identical batches both arms
        t0 = time.perf_counter()
        h = train_one(model, VOCAB, STEPS, _tr, _ev, eval_gap=max(20, STEPS // 4),
                      ema_decay=float(C.EMA_DECAY), surprise=False, seed=SEED)
        dt = time.perf_counter() - t0
        fin = h["eval_hist"][-1][1] if h["eval_hist"] else float("nan")
        tr_last = h["loss_hist"][-1] if h["loss_hist"] else float("nan")
        return fin, tr_last, dt * 1000 / STEPS

    lm_off, tr_off, ms_off = run(base)
    lm_on, tr_on, ms_on = run(on)
    delta = lm_on - lm_off

    print("\n──────────── RSM recall/generalization A/B (unseen content) ────────────")
    print(f"  ring window = {WINDOW}    steps = {STEPS}")
    print(f"  train-loss final  off {tr_off:.3f}  on {tr_on:.3f}")
    print(f"  eval (unseen)     off {lm_off:.3f}  on {lm_on:.3f}   Δ {delta:+.3f}")
    print(f"  speed             off {ms_off:.1f}  on {ms_on:.1f} ms/step   (Δ {ms_on/ms_off*100-100:+.1f}%)")
    sf = stateful_generate_metrics(on, VOCAB, prompt_len=64, gen_len=16, device=DEVICE)
    if sf["sf_ok"] is None:
        print("  O(1) stateful = UNSUPPORTED")
    else:
        print(f"  O(1) stateful = sf_ok {sf['sf_ok']}  max_rel {sf['sf_rel']:.2e}  gen_sf {sf['gen_sf']:.1f}/s")
    verdict = "HELPS" if delta <= -0.004 else ("REGRESSED" if delta > 0.004 else "PASS")
    print("PROBE_DONE " + verdict)