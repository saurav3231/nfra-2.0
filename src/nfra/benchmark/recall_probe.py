# ═══════════════════════════════════════════════════════════════════════════
# recall_probe — RSM STM single-lag recall test (auxiliary memory objective)
#
# Copy-recall with ONE lag, copy-density = 1.0, so the task is learnable
# (every position is a copy of the token `lag` back). Ring-off vs ring-on, same
# init, same batches. Accuracy measured on UNSEEN content at the same lag.
#   * run with NFRA_RECALL_LAG <= NFRA_STM_WINDOW  -> ring should dominate
#   * run with NFRA_RECALL_LAG >>  NFRA_STM_WINDOW -> both rely on matrix state
# Env: NFRA_RECALL_STEPS, NFRA_RECALL_LAG, NFRA_STM_WINDOW, NFRA_STM_DIM,
#      NFRA_STM_SIZE_M
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
from torch.utils.data import Dataset, DataLoader
from nfra.benchmark.compare import HierarchicalDataset
from nfra.core.cortex import CortexWorkingMemory

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STEPS   = int(os.environ.get("NFRA_RECALL_STEPS", "1500"))
LAG     = int(os.environ.get("NFRA_RECALL_LAG", "4"))
WINDOW  = int(os.environ.get("NFRA_STM_WINDOW", "16"))
STM_DIM = int(os.environ.get("NFRA_STM_DIM", "32"))
SIZE_M  = int(os.environ.get("NFRA_STM_SIZE_M", "20"))
SEED    = int(os.environ.get("NFRA_STM_SEED", "42"))
VOCAB   = HierarchicalDataset.VOCAB_SIZE  # 4096
SEQ     = 257


class CopyDataset(Dataset):
    """Next-token copy-recall with one lag, density `frac`.
    row[i] = row[i-lag] with prob `frac` (i>=lag). Returns (x, y) where
    y[j] == x[j+1-lag] at copied positions."""

    def __init__(self, num, seq_len, vocab, lag, frac, seed):
        g = torch.Generator().manual_seed(seed)
        self.data = torch.randint(0, vocab, (num, seq_len), generator=g)
        for i in range(lag, seq_len):
            if torch.rand(1, generator=g).item() < frac:
                self.data[:, i] = self.data[:, i - lag]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data[idx]
        return row[:-1], row[1:]


def make_loader(num, lag, frac, seed, shuffle=True, batch=16):
    ds = CopyDataset(num, SEQ, VOCAB, lag, frac, seed)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed) if shuffle else None)


def recall_accuracy(model, loader, lag, max_batches=25):
    """Fraction of copied positions (target = token `lag` back) predicted exactly."""
    model.eval()
    corr = tot = 0
    with torch.no_grad():
        for bi, (x, y) in enumerate(loader):
            if bi >= max_batches:
                break
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)["logits"].argmax(-1)          # (B, S)
            S = x.shape[1]
            j = torch.arange(S, device=DEVICE)
            mask = j >= lag - 1
            idx = mask.nonzero().squeeze(-1)
            predsel = pred[:, mask]
            targsel = x[:, idx + (1 - lag)]
            corr += (predsel == targsel).sum().item()
            tot += predsel.numel()
    model.train()
    return corr / max(tot, 1)


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
        tr = make_loader(2048, LAG, 1.0, SEED, shuffle=True)         # learnable copy task
        ev = make_loader(512, LAG, 0.0, SEED + 7, shuffle=False)   # LM-free (no copies)
        t0 = time.perf_counter()
        h = train_one(model, VOCAB, STEPS, tr, ev, eval_gap=max(20, STEPS // 4),
                      ema_decay=float(C.EMA_DECAY), surprise=False, seed=SEED)
        dt = time.perf_counter() - t0
        fin = h["eval_hist"][-1][1] if h["eval_hist"] else float("nan")
        return fin, dt * 1000 / STEPS

    lm_off, ms_off = run(base)
    lm_on, ms_on = run(on)

    ev_off = make_loader(512, LAG, 1.0, SEED + 99, shuffle=False)  # unseen content, pure copy
    ev_on = make_loader(512, LAG, 1.0, SEED + 99, shuffle=False)
    a_off = recall_accuracy(base, ev_off, LAG)
    a_on = recall_accuracy(on, ev_on, LAG)

    print("\n──────────── RSM single-lag recall ────────────")
    print(f"  task: copy token {LAG} back   |   ring window = {WINDOW}")
    print(f"  LM eval   off {lm_off:.3f}  on {lm_on:.3f}    {ms_off:.1f}/{ms_on:.1f} ms/step")
    print(f"  recall @lag={LAG}   off {a_off*100:6.2f}%   on {a_on*100:6.2f}%   Δ {a_on-a_off:+.2%}")
    verdict = ("RING-HYPO-CONFIRMED" if LAG <= WINDOW and a_on > a_off
               else "RING-HYP-REVERSED" if a_on < a_off else "RING-TIE")
    print("PROBE_DONE " + verdict)