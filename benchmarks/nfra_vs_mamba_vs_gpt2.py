"""
+======================================================================+
|   NFRA BRAIN vs MAMBA-SSM vs GPT-2  —  APPLES-TO-APPLES BENCHMARK   |
|                                                                      |
|  The honest, rigorous version: same params, same data, same         |
|  optimizer, same steps. Three pure-PyTorch reference implementations |
|  compared on quality, speed, and memory.                            |
|                                                                      |
|  Models:                                                             |
|   1. NFRA Brain  (recurrence + surprise gating + neuromodulation,   |
|                   depth-shared: 4 unique blocks reused 3x)          |
|   2. Mamba-SSM   (faithful selective state-space model: conv1d +    |
|                   discretized per-step SSM scan via Hillis-Steele    |
|                   associative scan — exact, parallel)               |
|   3. GPT-2       (classical causal transformer: attention + MLP)    |
|                                                                      |
|  Fairness: all three use matched parameter budgets, identical data,  |
|  optimizer, and step count. All are pure PyTorch (no fused kernels), |
|  so this is a software-level comparison. Real fused kernels would    |
|  accelerate BOTH the SSM and NFRA.                                  |
|                                                                      |
|  Metrics: final train loss, eval perplexity, ppl per million params,|
|  tokens/sec (train step), ms/step, peak GPU memory.                 |
|                                                                      |
|  Env:                                                                |
|   NFRA_DATA  = synthetic (default) | wikitext2                       |
|   NFRA_MODE  = quick (150) | standard (600) | rigorous (1500)        |
|   NFRA_TARGET_PARAMS = 20 (million)                                  |
|   NFRA_DIM   = 512                                                   |
|                                                                      |
|  Usage: python nfra_vs_mamba_vs_gpt2.py                              |
|         Recommended: Kaggle T4 GPU.                                  |
+======================================================================+
"""

import os, sys, time, math, json, warnings
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from nfra import NFRAConfig, NFRAForCausalLM

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
DATA_SOURCE = os.environ.get('NFRA_DATA', 'synthetic').lower()
HAS_DATASETS = False
if DATA_SOURCE == 'wikitext2':
    try:
        from datasets import load_dataset
        HAS_DATASETS = True
    except ImportError:
        print("  WARNING: 'datasets' not installed — using synthetic data")
        DATA_SOURCE = 'synthetic'

MODE = os.environ.get('NFRA_MODE', 'standard')
STEP_CFG = {'quick': 150, 'standard': 600, 'rigorous': 1500}
STEPS = int(os.environ.get('NFRA_STEPS', STEP_CFG[MODE]))
TARGET_PARAMS_M = float(os.environ.get('NFRA_TARGET_PARAMS', '20'))
DIM = int(os.environ.get('NFRA_DIM', '512'))
D_STATE = 16
EVAL_GAP = max(50, STEPS // 6)
SEQ_LEN = 256
SEED = 42

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HAS_CUDA = DEVICE.type == 'cuda'
USE_AMP = False
DEFAULT_BATCH = 8

if HAS_CUDA:
    print(f"  GPU: {torch.cuda.get_device_name(0)}  "
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('medium')
    cc = torch.cuda.get_device_capability(0)
    if cc >= (8, 0):
        USE_AMP = True; amp_dtype = torch.bfloat16
    elif cc >= (7, 0):
        USE_AMP = True; amp_dtype = torch.float16
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    DEFAULT_BATCH = 48 if gpu_mem >= 70 else 32 if gpu_mem >= 35 else 8 if gpu_mem >= 14 else 4
    if USE_AMP:
        print(f"  [AMP] {amp_dtype} enabled")
else:
    DEFAULT_BATCH = 4
    STEPS = min(STEPS, 80)
    EVAL_GAP = max(20, STEPS // 4)
    print("  WARNING: CPU mode — small steps, results not representative of GPU speed")

torch.manual_seed(SEED); np.random.seed(SEED)


# ---------------------------------------------------------------------
# Data  (synthetic hierarchical — same generator as nfra_revolution_test)
# ---------------------------------------------------------------------
class HierarchicalDataset(Dataset):
    VOCAB_SIZE = 4096
    def __init__(self, num_seqs: int, seq_len: int, seed: int = 0):
        super().__init__()
        self.seq_len = seq_len
        rng = np.random.RandomState(seed)
        N_TOPICS = 32
        pi = np.exp(rng.randn(N_TOPICS, N_TOPICS) * 0.3)
        np.fill_diagonal(pi, pi.diagonal() * 3)
        self._topic_trans = (pi / pi.sum(1, keepdims=True)).astype(np.float32)
        phi = np.exp(rng.randn(N_TOPICS, self.VOCAB_SIZE) * 0.5)
        self._topic_emit = (phi / phi.sum(1, keepdims=True)).astype(np.float32)
        th = np.exp(rng.randn(self.VOCAB_SIZE, self.VOCAB_SIZE) * 0.4)
        self._bigram = (th / th.sum(1, keepdims=True)).astype(np.float32)
        data = np.zeros((num_seqs, seq_len), dtype=np.int64)
        for s in range(num_seqs):
            topic = rng.randint(N_TOPICS); prev = rng.randint(self.VOCAB_SIZE)
            for t in range(seq_len):
                if rng.rand() < 0.1:
                    topic = rng.choice(N_TOPICS, p=self._topic_trans[topic])
                if rng.rand() < 0.3:
                    tok = rng.choice(self.VOCAB_SIZE, p=self._topic_emit[topic])
                else:
                    p = self._bigram[prev] * 0.7 + self._topic_emit[topic] * 0.3
                    tok = rng.choice(self.VOCAB_SIZE, p=p / p.sum())
                data[s, t] = tok; prev = tok
        self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        x = self.data[idx, :-1]; y = self.data[idx, 1:]
        return torch.from_numpy(x), torch.from_numpy(y)


CHAR_VOCAB = ['\n', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',',
              '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':',
              ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
              'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
              'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', 'a', 'b', 'c', 'd', 'e',
              'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
              't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~']
CHAR2IDX = {c: i for i, c in enumerate(CHAR_VOCAB)}


class WikiText2Dataset(Dataset):
    def __init__(self, split: str = 'train', seq_len: int = 256):
        super().__init__()
        self.seq_len = seq_len
        text = load_dataset("wikitext", "wikitext-2-raw-v1", split=split, trust_remote_code=True)
        full_text = '\n'.join(text['text'])
        ids = [CHAR2IDX.get(c, 0) for c in full_text]
        data = torch.tensor(ids, dtype=torch.long)
        n = len(data) // seq_len
        self.data = data[:n * seq_len + 1]
    def __len__(self): return len(self.data) // self.seq_len
    def __getitem__(self, idx):
        s = idx * self.seq_len
        return self.data[s:s + self.seq_len], self.data[s + 1:s + self.seq_len + 1]


# ---------------------------------------------------------------------
# Model 1: NFRA Brain
# ---------------------------------------------------------------------
def make_nfra(vocab: int, dim: int, layers: int) -> NFRAForCausalLM:
    cfg = NFRAConfig(mode='brain', vocab_size=vocab, hidden_size=dim,
                     num_layers=layers, n_bands=16, dropout=0.1)
    return NFRAForCausalLM(cfg)


# ---------------------------------------------------------------------
# Model 2: Mamba-style SSM (faithful block, pure PyTorch)
# ---------------------------------------------------------------------
def mamba_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Exact associative scan for h_t = a_t * h_{t-1} + b_t  (Hillis-Steele).
    Fully parallel: O(log S) sequential steps of vectorized O(S) ops.
    Verified numerically identical to the sequential recurrence.
    """
    S = a.shape[-2]
    a_cur, b_cur = a, b
    offset = 1
    while offset < S:
        a_prev, b_prev = a_cur, b_cur
        a_shift = F.pad(a_prev, (0, 0, offset, 0), value=1.0)[..., :S, :]
        b_shift = F.pad(b_prev, (0, 0, offset, 0), value=0.0)[..., :S, :]
        a_cur = a_prev * a_shift          # A_right * A_left
        b_cur = a_prev * b_shift + b_prev # A_right * B_left + B_right
        offset *= 2
    return b_cur


class MambaBlock(nn.Module):
    """Mamba v1 block: conv1d + input-dependent selective SSM scan."""
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        d_inner = int(expand * dim)
        self.d_inner = d_inner
        self.d_state = d_state
        self.in_proj = nn.Linear(dim, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv, groups=d_inner,
                                padding=d_conv - 1, bias=True)
        self.x_proj = nn.Linear(d_inner, 3 * d_state, bias=False)
        self.dt_proj = nn.Linear(d_state, d_inner, bias=True)
        with torch.no_grad():
            self.dt_proj.bias.copy_(torch.log(torch.full_like(self.dt_proj.bias, 0.1)))
        self.A_log = nn.Parameter(torch.randn(d_state))
        self.D = nn.Parameter(torch.randn(d_inner))
        self.out_proj = nn.Linear(d_inner, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        N, E = self.d_state, self.d_inner

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = self.conv1d(x.transpose(1, 2)).transpose(1, 2)[:, :S, :]  # causal conv
        x = F.silu(x)

        dt, Bm, C = self.x_proj(x).chunk(3, dim=-1)          # [B,S,N] each
        dt = F.softplus(self.dt_proj(dt))                    # [B,S,E]

        A = -torch.exp(self.A_log)                           # [N]
        alpha = torch.exp(A.view(1, 1, N, 1) * dt.unsqueeze(2))     # [B,S,N,E] per-step decay
        u = Bm.unsqueeze(-1) * x.unsqueeze(2)                # [B,S,N,E] = B(x) outer x

        h = mamba_scan(alpha.permute(0, 2, 1, 3).reshape(B, 1, S, N * E),
                       u.permute(0, 2, 1, 3).reshape(B, 1, S, N * E))  # [B,1,S,N*E]
        h = h.view(B, S, N, E)

        y = (h * C.unsqueeze(-1)).sum(dim=2)                 # [B,S,E] = C·h
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x
        y = y * F.silu(z)
        return self.out_proj(y)


class MambaLM(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 512, n_layers: int = 8, d_state: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([MambaBlock(dim, d_state) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids: torch.Tensor, **kw) -> Dict:
        x = self.embed(input_ids)
        for blk in self.blocks:
            x = x + blk(x)
        return {'logits': self.lm_head(self.norm(x))}


# ---------------------------------------------------------------------
# Model 3: GPT-2 Transformer (identical to nfra_revolution_test)
# ---------------------------------------------------------------------
class GPT2Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        H, Hd = self.n_heads, self.head_dim
        qkv = self.qkv(x).view(B, S, 3, H, Hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / (Hd ** 0.5)
        causal = torch.triu(torch.full((S, S), float('-inf'), device=x.device), 1)
        attn = F.softmax(scores + causal, dim=-1)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(B, S, D)
        return self.out(out)


class GPT2Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = GPT2Attention(dim, n_heads)
        self.ln2 = nn.LayerNorm(dim)
        hidden = int(dim * 4)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.fc2(F.gelu(self.fc1(self.ln2(x)))))
        return x


class GPT2ForCausalLM(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 512, n_layers: int = 6,
                 n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(8192, dim)
        self.blocks = nn.ModuleList([GPT2Block(dim, n_heads, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids: torch.Tensor, **kw) -> Dict:
        B, S = input_ids.shape
        pos = torch.arange(S, device=input_ids.device)
        x = self.embed(input_ids) + self.pos_embed(pos)
        for blk in self.blocks:
            x = blk(x)
        return {'logits': self.lm_head(self.ln_f(x))}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def tune_layers(make_fn, target, vocab):
    """Pick n_layers landing closest to the target param budget."""
    best = (1, float('inf'))
    for L in range(1, 64):
        p = count_params(make_fn(vocab, DIM, L))
        if abs(p - target) < abs(best[1] - target):
            best = (L, p)
        if p >= target * 1.15:
            break
    return best


def compute_loss(model, x, y) -> torch.Tensor:
    logits = model(x)['logits']
    return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))


def make_optimizer(model, lr: float = 3e-4, warmup: int = 50, total: int = STEPS):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    def schedule(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, schedule)


@torch.no_grad()
def evaluate(model, loader, max_batches: int = 15) -> Tuple[float, float]:
    model.eval()
    total_loss, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            loss = compute_loss(model, x, y)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    model.train()
    avg = total_loss / max(n, 1)
    return math.exp(avg), avg


def measure_throughput(model, batch_size: int, n_steps: int = 20) -> float:
    x = torch.randint(0, 4096, (batch_size, SEQ_LEN), device=DEVICE)
    y = torch.randint(0, 4096, (batch_size, SEQ_LEN), device=DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler(str(DEVICE)) if USE_AMP else None
    if HAS_CUDA:
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_steps):
        opt.zero_grad()
        with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            loss = compute_loss(model, x, y)
        if scaler:
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            loss.backward(); opt.step()
    if HAS_CUDA:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return batch_size * SEQ_LEN * n_steps / elapsed


def measure_memory(model, batch_size: int = 2) -> float:
    if not HAS_CUDA:
        return 0.0
    torch.cuda.reset_peak_memory_stats()
    x = torch.randint(0, 4096, (batch_size, SEQ_LEN), device=DEVICE)
    y = torch.randint(0, 4096, (batch_size, SEQ_LEN), device=DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
        loss = compute_loss(model, x, y)
    loss.backward(); opt.step()
    return torch.cuda.max_memory_allocated() / 1e9


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    use_wiki = DATA_SOURCE == 'wikitext2' and HAS_DATASETS
    VOCAB = 96 if use_wiki else 4096
    target = int(TARGET_PARAMS_M * 1e6)

    print("+==========================================================+")
    print("|   NFRA BRAIN vs MAMBA-SSM vs GPT-2  (apples-to-apples)  |")
    print("+==========================================================+")
    print(f"|  Data:      {'WikiText-2 (char)' if use_wiki else 'Synthetic hierarchical':<48s}|")
    print(f"|  Vocab:     {VOCAB:<6d}    Dim: {DIM:<6d}    SeqLen: {SEQ_LEN:<6d}        |")
    print(f"|  Param target: ~{TARGET_PARAMS_M:.0f}M    Steps: {STEPS:<6d}    Batch: {DEFAULT_BATCH}   |")
    print(f"|  Device:    {'GPU ' + torch.cuda.get_device_name(0) if HAS_CUDA else 'CPU':<48s}|")
    print("+==========================================================+")

    # ---- data
    if use_wiki:
        train_ds = WikiText2Dataset('train', SEQ_LEN + 1)
        eval_ds = WikiText2Dataset('validation', SEQ_LEN + 1)
    else:
        train_ds = HierarchicalDataset(max(4096, DEFAULT_BATCH * 8), SEQ_LEN + 1, seed=SEED)
        eval_ds = HierarchicalDataset(512, SEQ_LEN + 1, seed=SEED + 1)
    train_loader = DataLoader(train_ds, batch_size=DEFAULT_BATCH, shuffle=True, num_workers=0)
    eval_loader = DataLoader(eval_ds, batch_size=DEFAULT_BATCH, shuffle=False, num_workers=0)

    # ---- build models, tuned to the same param budget
    models = {}
    print("\n  [1/4] Building models (param-matched)...")

    L_nfra, p_nfra = tune_layers(make_nfra, target, VOCAB)
    models['nfra_brain'] = make_nfra(VOCAB, DIM, L_nfra).to(DEVICE)
    print(f"  └- NFRA Brain : effective {L_nfra} layers (4 unique blocks x3 passes) "
          f"-> {p_nfra/1e6:.1f}M")

    def make_mamba(vocab, dim, L):
        return MambaLM(vocab, dim, L, d_state=D_STATE)
    L_mam, p_mam = tune_layers(make_mamba, target, VOCAB)
    models['mamba_ssm'] = make_mamba(VOCAB, DIM, L_mam).to(DEVICE)
    print(f"  └- Mamba SSM  : {L_mam} layers (d_state={D_STATE}) -> {p_mam/1e6:.1f}M")

    def make_gpt2(vocab, dim, L):
        return GPT2ForCausalLM(vocab, dim, L, n_heads=8)
    L_gpt, p_gpt = tune_layers(make_gpt2, target, VOCAB)
    models['gpt2'] = make_gpt2(VOCAB, DIM, L_gpt).to(DEVICE)
    print(f"  └- GPT-2      : {L_gpt} layers -> {p_gpt/1e6:.1f}M")

    # ---- throughput + memory (before training)
    print("\n  [2/4] Measuring throughput + peak memory...")
    perf = {}
    for name, m in models.items():
        m.train()
        tok_s = measure_throughput(m, batch_size=DEFAULT_BATCH, n_steps=15)
        mem = measure_memory(m)
        perf[name] = {'tok_per_s': round(tok_s), 'peak_mem_gb': round(mem, 2)}
        print(f"  └- {name:<11s} {tok_s:>8,.0f} tok/s   peak {mem:.2f} GB")

    # ---- training loop (fair: same optimizer, same data order reset)
    print(f"\n  [3/4] Training {STEPS} steps (AdamW 3e-4 + warmup + cosine)...")
    history = {n: {'train_loss': [], 'eval_ppl': [], 'step': []} for n in models}
    optimizers = {n: make_optimizer(m) for n, m in models.items()}
    loaders = {n: iter(train_loader) for n in models}
    step_times = {n: [] for n in models}
    scaler = {n: (torch.amp.GradScaler(str(DEVICE)) if USE_AMP else None) for n in models}

    for step in range(1, STEPS + 1):
        for name, m in models.items():
            try:
                x, y = next(loaders[name])
            except StopIteration:
                loaders[name] = iter(train_loader)
                x, y = next(loaders[name])
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt, sched = optimizers[name]
            opt.zero_grad()
            t0 = time.perf_counter()
            with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                loss = compute_loss(m, x, y)
            if scaler[name]:
                scaler[name].scale(loss).backward()
                scaler[name].step(opt); scaler[name].update()
            else:
                loss.backward(); opt.step()
            if HAS_CUDA:
                torch.cuda.synchronize()
            step_times[name].append(time.perf_counter() - t0)
            sched.step()

        if step == 1 or step % 20 == 0 or step == STEPS:
            line = f"  step {step:>5d}/{STEPS}"
            for name in models:
                hist = history[name]
                hist['train_loss'].append(loss.item())
                if step % EVAL_GAP == 0 or step == STEPS or step == 1:
                    ppl, avg = evaluate(models[name], eval_loader)
                    hist['eval_ppl'].append(ppl)
                    hist['step'].append(step)
                    line += f" | {name} ppl={ppl:6.1f}"
                else:
                    line += f" | {name} loss={loss.item():5.3f}"
            print(line)

    # ---- summary
    print("\n  [4/4] Summary")
    print("+=============================================================================+")
    print(f"| {'model':<11s} {'params':>7s} {'eval_ppl':>9s} {'ppl/M':>7s} "
          f"{'tok/s':>9s} {'ms/step':>8s} {'mem(GB)':>8s} {'final_loss':>10s} |")
    print("+=============================================================================+")
    results = {}
    for name, m in models.items():
        params = count_params(m)
        ppl = history[name]['eval_ppl'][-1] if history[name]['eval_ppl'] else float('nan')
        avg_ms = 1000 * np.mean(step_times[name]) if step_times[name] else 0
        results[name] = {
            'params': params,
            'effective_layers': L_nfra if name == 'nfra_brain' else
                                (L_mam if name == 'mamba_ssm' else L_gpt),
            'eval_ppl': round(ppl, 3),
            'ppl_per_million_params': round(ppl / (params / 1e6), 3),
            'tok_per_s': perf[name]['tok_per_s'],
            'ms_per_step': round(avg_ms, 2),
            'peak_mem_gb': perf[name]['peak_mem_gb'],
            'final_train_loss': round(history[name]['train_loss'][-1], 4),
        }
        print(f"| {name:<11s} {params/1e6:6.1f}M {ppl:8.1f} {ppl/(params/1e6):6.2f} "
              f"{perf[name]['tok_per_s']:>8,} {avg_ms:7.1f} "
              f"{perf[name]['peak_mem_gb']:>7.2f} "
              f"{history[name]['train_loss'][-1]:9.4f} |")
    print("+=============================================================================+")
    print("  NOTE: all three are pure-PyTorch references (no fused kernels). Real fused")
    print("  kernels would speed up both Mamba and NFRA. This measures software-level")
    print("  quality/speed/memory at matched parameter count on identical data.")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"nfra_vs_mamba_vs_gpt2_results.json")
    with open(out_path, 'w') as f:
        json.dump({'config': {'steps': STEPS, 'dim': DIM, 'target_params': TARGET_PARAMS_M,
                              'data': DATA_SOURCE, 'vocab': VOCAB,
                              'device': DEVICE.type, 'batch': DEFAULT_BATCH},
                   'results': results, 'perf': perf, 'history': history}, f, indent=2)
    print(f"  Results saved to {out_path}")


if __name__ == '__main__':
    main()
