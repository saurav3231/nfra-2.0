# ═══════════════════════════════════════════════════════════════════════════
# recall_probe — RSM STM recall test (auxiliary memory objective, off-LM-loss)
#
# Task: copy-recall. Sequences are injected so that at many positions the next
# token EQUALS the token `lag` positions earlier. The model must retrieve it.
#   * ring-off vs ring-on start identical (zero-init ring), train same batches.
#   * eval: per-lag copy accuracy on UNSEEN content (fresh seed, frac=1.0).
#   * hypothesis: windowed STM ring makes lags <= window near-trivial to copy;
#     both arms rely on the matrix state for lags > window.
# Env: NFRA_RECALL_STEPS, NFRA_STM_WINDOW, NFRA_STM_DIM, NFRA_STM_SIZE_M
# ═══════════════════════════════════════════════════════════════════════════
import os, sys, copy, math, time
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STEPS  = int(os.environ.get("NFRA_RECALL_STEPS", "300"))
WINDOW = int(os.environ.get("NFRA_STM_WINDOW", "16"))
STM_DIM = int(os.environ.get("NFRA_STM_DIM", "32"))
SIZE_M  = int(os.environ.get("NFRA_STM_SIZE_M", "20"))
SEED    = int(os.environ.get("NFRA_STM_SEED", "42"))
LAGS    = [int(x) for x in os.environ.get("NFRA_RECALL_LAGS", "2,8,16,32,64,128").split(",")]
FRAC    = float(os.environ.get("NFRA_RECALL_FRAC", "0.5"))
VOCAB   = HierarchicalDataset.VOCAB_SIZE  # synthetic 4096
SEQ     = 257


class RecallDataset(Dataset):
    """Next-token copy-recall: row[i] = row[i-lag] with prob `frac` (i>=lag).
    __getitem__ returns (x, y) like the other datasets: y[j] copies x[j+1-lag]
    at injected positions."""

    def __init__(self, num, seq_len, vocab, lags, frac, seed):
        g = torch.Generator().manual_seed(seed)
        self.data = torch.randint(0, vocab, (num, seq_len), generator=g)
        for i in range(1, seq_len):
            if torch.rand(1, generator=g).item() < frac:
                lag = int(lags[torch.randint(0, len(lags), (), generator=g).item()])
                if i >= lag:
                    self.data[:, i] = self.data[:, i - lag]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data[idx]
        return row[:-1], row[1:]


def make_loader(num, lags, frac, seed, shuffle=True, batch=16):
    ds = RecallDataset(num, SEQ, VOCAB, lags, frac, seed)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed) if shuffle else None)


@torch.no_grad()
def recall_accuracy(model, loader, lag, max_batches=25):
    """Fraction of copy positions (target = token `lag` back) predicted exactly."""
    model.eval()
    corr = tot = 0
    for bi, (x, y) in enumerate(loader):
        if bi >= max_batches:
            break
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x)["logits"].argmax(-1)          # (B, S)
        S = x.shape[1]
        j = torch.arange(S, device=DEVICE)
        mask = j >= lag - 1                            # positions with a valid copy target
        idx = mask.nonzero().squeeze(-1)               # j values
        predsel = pred[:, mask]                        # (B, M)
        targsel = x[:, idx + (1 - lag)]                # (B, M) the token `lag` back
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
        tr = make_loader(2048, LAGS, FRAC, SEED, shuffle=True)      # mixed-lag train
        ev = make_loader(512, LAGS, 0.0, SEED + 7, shuffle=False)   # LM-free (no copies)
        t0 = time.perf_counter()
        h = train_one(model, VOCAB, STEPS, tr, ev, eval_gap=max(20, STEPS // 4),
                      ema_decay=float(C.EMA_DECAY), surprise=False, seed=SEED)
        dt = time.perf_counter() - t0
        fin = h["eval_hist"][-1][1] if h["eval_hist"] else float("nan")
        return fin, dt * 1000 / STEPS

    lm_off, ms_off = run(base)
    lm_on, ms_on = run(on)

    print("\n──────────── RSM recall probe ────────────")
    print(f"  LM eval   off {lm_off:.3f}  on {lm_on:.3f}   {ms_off:.1f}/{ms_on:.1f} ms/step")
    print(f"  ring window = {WINDOW}   (lags <= window should favor ring-on)")
    print(f"  lag    off_acc    on_acc    Δ(on-off)")
    for lag in LAGS:
        ev = make_loader(512, [lag], 1.0, SEED + 99 + lag, shuffle=False)  # unseen, pure copy
        a_off = recall_accuracy(base, ev, lag)
        a_on = recall_accuracy(on, ev, lag)
        print(f"  {lag:>4}   {a_off*100:6.2f}%   {a_on*100:6.2f}%   {a_on-a_off:+.2%}")
    print("PROBE_DONE")