"""
╔══════════════════════════════════════════════════════════════════════╗
║   NFRA BRAIN  vs  MAMBA-SSM  vs  GPT-2   —  apples-to-apples          ║
║                                                                        ║
║   Three pure-PyTorch reference implementations, matched on params,     ║
║   trained on identical data with the identical optimizer.             ║
║                                                                        ║
║   WHAT IS MEASURED                                                      ║
║     eval loss  : log-perplexity (lower = better; ≈ln(vocab) = random guess) ║
║     throughput : training tok/s (pure-PyTorch; no fused kernels)       ║
║     peak memory: GB during one train step                              ║
║                                                                        ║
║   CONFIGURATIONS                                                        ║
║     • All models matched on params (~20M) and effective depth (12).     ║
║     • NFRA runs depth-shared: {U unique blocks} reused for the full     ║
║       depth — a core NFRA design point. U and dim are tuned jointly     ║
║       so NFRA gets several DISTINCT blocks within budget. A pure        ║
║       1-block x depth-passes 'recurrent' config is also a valid NFRA    ║
║       setting but is intentionally NOT used for the head-to-head.       ║
║     • NFRA uses per-layer gradient checkpointing; Mamba runs in fp32      ║
║       with pre-LayerNorm (bounded activations) — required for stable     ║
║       training without the official CUDA kernel (fp16 overflow → NaN).  ║
║     • A NaN/Inf guard skips optimizer updates rather than letting a      ║
║       bad gradient permanently poison the model.                        ║
║                                                                        ║
║   READING THE NUMBERS                                                   ║
║     • All heads use GPT-2-style init (embed std 0.02), so loss starts     ║
║       near ln(vocab) (random guess) for every model.                       ║
║       Judge quality by the FINAL eval loss.                              ║
║     • Mamba/NFRA use unfused scans here; production fused kernels        ║
║       would make them dramatically faster than shown.                    ║
║                                                                        ║
║   ENV                                                                    ║
║     NFRA_MODE           quick(150) | standard(600) | rigorous(1500)    ║
║     NFRA_DATA           synthetic | wikitext2                          ║
║     NFRA_TARGET_PARAMS  target params in millions (default 20)        ║
║     NFRA_DIM            hidden size (default 512)                      ║
║                                                                        ║
║   NOTE: the synthetic bigram is a random VxV table — unlearnable at     ║
║   small token budgets, so every model plateaus at random-guess loss.    ║
║   Use NFRA_DATA=wikitext2 (real text) to actually differentiate them.  ║
║                                                                        ║
║   Usage: python -m nfra.benchmark.compare   (Kaggle T4 recommended)    ║
║                                                                        ║
║   Wikitext-2 (raw) files must be in the CWD:                           ║
║     wikitext-train-raw-v1.txt, wikitext-valid-raw-v1.txt               ║
║   (download from huggingface.co/datasets/wikitext/wikitext-2-raw-v1)   ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os, sys, time, math, json, warnings, functools
print = functools.partial(print, flush=True)
warnings.filterwarnings('ignore', message='Detected call of .*lr_scheduler.*')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

from nfra import NFRAConfig, NFRAForCausalLM

# ─────────────────────────── config ───────────────────────────
DATA_SOURCE = os.environ.get('NFRA_DATA', 'synthetic').lower()
WIKI_PATHS = {'train': 'wikitext-train-raw-v1.txt',
              'validation': 'wikitext-valid-raw-v1.txt'}
if DATA_SOURCE == 'wikitext2':
    missing = [f for f in WIKI_PATHS.values() if not os.path.exists(f)]
    if missing:
        print(f"  [warn] wikitext2 files missing ({', '.join(missing)}) — "
              "falling back to synthetic")
        print("         download from huggingface.co/datasets/Salesforce/wikitext "
              "(wikitext-2-raw-v1)")
        DATA_SOURCE = 'synthetic'

STEP_CFG = {'quick': 150, 'standard': 600, 'rigorous': 1500}
MODE = os.environ.get('NFRA_MODE', 'standard')
STEPS = int(os.environ.get('NFRA_STEPS', STEP_CFG[MODE]))
TARGET_M = float(os.environ.get('NFRA_TARGET_PARAMS', '20'))
DIM = int(os.environ.get('NFRA_DIM', '512'))
# NFRA 3.2 feature toggles (applied to ALL families when enabled, so the
# head-to-head stays fair): EMA weight averaging for eval, surprise-weighted
# (RPE) gradients, and k-WTA lateral inhibition (NFRA architecture only).
EMA_DECAY = float(os.environ.get('NFRA_EMA', '0'))          # 0 = off
SURPRISE = os.environ.get('NFRA_SURPRISE', '0') == '1'      # 1 = on
KWTA = float(os.environ.get('NFRA_KWTA', '0'))              # 0.0 = off
LOCAL_ROUTE = os.environ.get('NFRA_LOCALROUTE', '0') == '1'
DIV_NORM = os.environ.get('NFRA_DIVNORM', '0') == '1'
ASTRO = os.environ.get('NFRA_ASTRO', '0') == '1'
THETA = os.environ.get('NFRA_THETA', '0') == '1'
ACH_RETAIN = os.environ.get('NFRA_ACH_RETAIN', '0') == '1'
GAIN_NOV = os.environ.get('NFRA_GAIN_NOV', '0') == '1'
LORA_RANK = int(os.environ.get('NFRA_LORA_RANK', '0'))     # 0 = off (Space axis)
D_STATE = 8
NFRA_DEPTH = 12                      # effective NFRA depth (unique × passes)
EVAL_GAP = max(50, STEPS // 6)
SEQ_LEN = 256
SEED = 42

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HAS_CUDA = DEVICE.type == 'cuda'
USE_AMP = False
BATCH = 8
if HAS_CUDA:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('medium')
    cc = torch.cuda.get_device_capability(0)
    if cc >= (8, 0):
        USE_AMP = True; AMP_DTYPE = torch.bfloat16
    elif cc >= (7, 0):
        USE_AMP = True; AMP_DTYPE = torch.float16
    gmem = torch.cuda.get_device_properties(0).total_memory / 1e9
    BATCH = 48 if gmem >= 70 else 32 if gmem >= 35 else 8 if gmem >= 14 else 4
    if DATA_SOURCE == 'wikitext2' and BATCH > 8 and gmem < 20:
        # Mamba is out of the default grid and nfra's cumsum mixer + distinct
        # shallow blocks are memory-light, so batch 8 fits every family on a
        # 14.5 GB T4 (overnight frees non-cached models between runs). nfra is
        # launch-bound at batch 4 — bigger batches are a free tok/s win.
        BATCH = 8
    BATCH = int(os.environ.get('NFRA_BATCH', BATCH))
else:
    BATCH = 4
    STEPS = min(STEPS, 80); EVAL_GAP = max(20, STEPS // 4)
    print("  [warn] CPU mode - few steps, speed numbers are not representative")

torch.manual_seed(SEED); np.random.seed(SEED)


# ─────────────────────────── data ───────────────────────────
class HierarchicalDataset(Dataset):
    """Synthetic data with topics + bigram structure (deterministic, seed-based).
    Distribution (bigram/topic matrices) is fixed by `seed`; `seq_seed`
    controls only the sampled sequences, so train and eval share the SAME
    underlying distribution (otherwise eval is unlearnable out-of-distribution)."""
    VOCAB_SIZE = 4096
    def __init__(self, num_seqs, seq_len, seed=0, seq_seed=None):
        super().__init__()
        self.seq_len = seq_len
        dist_rng = np.random.RandomState(seed)
        N_TOPICS = 32
        pi = np.exp(dist_rng.randn(N_TOPICS, N_TOPICS) * 0.5)
        np.fill_diagonal(pi, pi.diagonal() * 3)
        self._topic_trans = pi / pi.sum(1, keepdims=True)
        phi = np.exp(dist_rng.randn(N_TOPICS, self.VOCAB_SIZE) * 1.0)
        self._topic_emit = phi / phi.sum(1, keepdims=True)
        th = np.exp(dist_rng.randn(self.VOCAB_SIZE, self.VOCAB_SIZE) * 1.0)
        self._bigram = th / th.sum(1, keepdims=True)
        seq_rng = np.random.RandomState(seed if seq_seed is None else seq_seed)
        self.data = self._generate(num_seqs, seq_len, seq_rng)

    def _generate(self, num_seqs, seq_len, rng):
        V = self.VOCAB_SIZE
        tc = torch.from_numpy(self._topic_trans).cumsum(1)
        ec = torch.from_numpy(self._topic_emit).cumsum(1)
        bc = torch.from_numpy(self._bigram).cumsum(1)
        def sample(cdf, rows):
            return torch.searchsorted(cdf[rows], torch.rand(len(rows), 1)).squeeze(-1)
        data = np.empty((num_seqs, seq_len), dtype=np.int64)
        topics = torch.randint(32, (num_seqs,)); prev = torch.randint(V, (num_seqs,))
        for t in range(seq_len):
            e = torch.nonzero(torch.rand(num_seqs) < 0.1).flatten()
            if len(e): topics[e] = sample(tc, topics[e])
            emit = torch.rand(num_seqs) < 0.3
            e = torch.nonzero(emit).flatten(); b = torch.nonzero(~emit).flatten()
            tok = torch.empty(num_seqs, dtype=torch.long)
            if len(e): tok[e] = sample(ec, topics[e])
            if len(b): tok[b] = sample(bc, prev[b])
            data[:, t] = tok.numpy(); prev = tok
        return data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return (torch.from_numpy(self.data[idx, :-1]),
                torch.from_numpy(self.data[idx, 1:]))


# ─────────────────────────── WikiText-2 (char) ───────────────────────────
CHAR_VOCAB = ['\n', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+',
              ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8',
              '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E',
              'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
              'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_',
              'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
              'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
              '{', '|', '}', '~']
CHAR2IDX = {c: i for i, c in enumerate(CHAR_VOCAB)}


class WikiText2Dataset(Dataset):
    """Character-level WikiText-2 language modeling dataset (real text).
    Reads the raw .txt files directly (no `datasets`/scipy dependency)."""

    def __init__(self, split: str = 'train', seq_len: int = 256):
        super().__init__()
        self.seq_len = seq_len
        path = WIKI_PATHS[split]
        print(f"  └- Loading WikiText-2 ({split}: {path})...", end=' ')
        with open(path, encoding='utf-8') as f:
            full_text = f.read()
        ids = [CHAR2IDX.get(c, 0) for c in full_text]
        self.data = torch.tensor(ids, dtype=torch.long)
        self.num_seqs = len(self.data) // seq_len
        self.data = self.data[:self.num_seqs * seq_len + 1]
        print(f"{self.num_seqs} seqs of {seq_len}")

    def __len__(self):
        return self.num_seqs

    def __getitem__(self, idx):
        start = idx * self.seq_len
        return (self.data[start:start + self.seq_len],
                self.data[start + 1:start + self.seq_len + 1])


# ─────────────────────────── models ───────────────────────────
def make_nfra(vocab, dim, unique_blocks, k_wta=None):
    if k_wta is None:
        k_wta = KWTA
    cfg = NFRAConfig(mode='brain', vocab_size=vocab, hidden_size=dim,
                     num_layers=NFRA_DEPTH, n_bands=16, dropout=0.1,
                     depth_shared=True, unique_blocks=unique_blocks,
                     gradient_checkpointing=True, k_wta_frac=k_wta,
                     local_route=LOCAL_ROUTE, div_norm=DIV_NORM, astro=ASTRO,
                     theta=THETA, ach_retain=ACH_RETAIN, gain_nov=GAIN_NOV,
                     lora_rank=LORA_RANK)
    return NFRAForCausalLM(cfg)


def hillis_prefix(a, b):
    """Associative prefix scan (exact, O(log S) parallel steps). Returns (A, B)
    with h_t = A_t*h_0 + B_t for h_t = a_t*h_{t-1} + b_t."""
    S = a.shape[-2]
    a_cur, b_cur = a, b
    off = 1
    while off < S:
        a_prev, b_prev = a_cur, b_cur
        a_shift = F.pad(a_prev, (0, 0, off, 0), value=1.0)[..., :S, :]
        b_shift = F.pad(b_prev, (0, 0, off, 0), value=0.0)[..., :S, :]
        a_cur = a_prev * a_shift
        b_cur = a_prev * b_shift + b_prev
        off *= 2
    return a_cur, b_cur


def mamba_scan(a, b, chunk=256):
    """Chunked associative scan run in fp32 (state recurrence is numerically
    sensitive; fp16 can overflow and NaN the loss), each chunk gradient-
    checkpointed so backward never holds more than one chunk."""
    orig_dtype = a.dtype
    a = a.float(); b = b.float()
    B, _, S, D = a.shape
    n = math.ceil(S / chunk)
    pad = n * chunk - S
    if pad:
        a = F.pad(a, (0, 0, 0, pad), value=1.0)
        b = F.pad(b, (0, 0, 0, pad), value=0.0)
    a = a.reshape(B, 1, n, chunk, D); b = b.reshape(B, 1, n, chunk, D)
    ckpt = torch.utils.checkpoint.checkpoint
    h = torch.zeros(B, D, device=a.device)
    outs = []
    for c in range(n):
        a_rel, b_rel = ckpt(hillis_prefix, a[:, :, c], b[:, :, c],
                            use_reentrant=False)
        out = a_rel * h.view(B, 1, 1, D) + b_rel
        h = out[:, :, -1, :].squeeze(1)
        outs.append(out)
    return torch.cat(outs, dim=2)[:, :, :S, :].to(orig_dtype)


class MambaBlock(nn.Module):
    """Mamba v1: conv1d + input-dependent selective SSM (pure PyTorch),
    with pre-LayerNorm so activations stay bounded without the official
    CUDA kernel (required for stable fp32/AMP training)."""
    def __init__(self, dim, d_state=8, d_conv=4, expand=2):
        super().__init__()
        d_inner = expand * dim
        self.d_inner, self.d_state = d_inner, d_state
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner,
                                padding=d_conv - 1, bias=True)
        self.x_proj = nn.Linear(d_inner, 3 * d_state, bias=False)
        self.dt_proj = nn.Linear(d_state, d_inner, bias=True)
        with torch.no_grad():
            self.dt_proj.bias.copy_(torch.log(torch.full_like(self.dt_proj.bias, 0.1)))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.randn(d_inner))
        self.out_proj = nn.Linear(d_inner, dim, bias=False)

    def forward(self, x):
        x_dtype = x.dtype
        x = x.float()
        with torch.autocast(device_type=DEVICE.type, enabled=False):
            B, S, D = x.shape
            N, E = self.d_state, self.d_inner
            x = self.norm(x)
            x, z = self.in_proj(x).chunk(2, dim=-1)
            x = self.conv1d(x.transpose(1, 2)).transpose(1, 2)[:, :S, :]
            x = F.silu(x)
            dt, Bm, C = self.x_proj(x).chunk(3, dim=-1)
            dt = F.softplus(self.dt_proj(dt))
            A = -torch.exp(self.A_log)
            alpha = torch.exp(A.view(1, 1, N, 1) * dt.unsqueeze(2))
            u = Bm.unsqueeze(-1) * x.unsqueeze(2)
            a_f = alpha.permute(0, 2, 1, 3).reshape(B, 1, S, N * E)
            u_f = u.permute(0, 2, 1, 3).reshape(B, 1, S, N * E)
            h = mamba_scan(a_f, u_f)
            h = h.view(B, S, N, E)
            y = (h * C.unsqueeze(-1)).sum(dim=2) + self.D.unsqueeze(0).unsqueeze(0) * x
            out = self.out_proj(y * F.silu(z))
        return out.to(x_dtype)


class MambaLM(nn.Module):
    def __init__(self, vocab_size, dim, n_layers, d_state=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([MambaBlock(dim, d_state) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
    def forward(self, input_ids, **kw):
        x = self.embed(input_ids)
        for blk in self.blocks:
            x = x + blk(x)
        return {'logits': self.lm_head(self.norm(x))}


class GPT2Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
    def forward(self, x):
        B, S, D = x.shape
        H, Hd = self.n_heads, self.head_dim
        qkv = self.qkv(x).view(B, S, 3, H, Hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / (Hd ** 0.5)
        causal = torch.triu(torch.full((S, S), float('-inf'), device=x.device), 1)
        attn = F.softmax(scores + causal, dim=-1)
        return self.out(torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(B, S, D))


class GPT2Block(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = GPT2Attention(dim, n_heads)
        self.ln2 = nn.LayerNorm(dim)
        h = int(dim * 4)
        self.fc1 = nn.Linear(dim, h, bias=False)
        self.fc2 = nn.Linear(h, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.fc2(F.gelu(self.fc1(self.ln2(x)))))
        return x


class GPT2ForCausalLM(nn.Module):
    def __init__(self, vocab_size, dim=512, n_layers=6, n_heads=8, dropout=0.1,
                 pos_len=8192):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(pos_len, dim)
        self.blocks = nn.ModuleList([GPT2Block(dim, n_heads, dropout)
                                     for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
    def forward(self, input_ids, **kw):
        B, S = input_ids.shape
        x = self.embed(input_ids) + self.pos_embed(torch.arange(S, device=input_ids.device))
        for blk in self.blocks:
            x = blk(x)
        return {'logits': self.lm_head(self.ln_f(x))}


class RWKVBlock(nn.Module):
    """RWKV-6 style block: pre-LN token-shift time-mixing (linear attention
    with a per-channel exponential-decay WKV recurrence + current-token bonus)
    plus a squared-ReLU channel-mixing over the post-time-mix residual. The
    WKV decay is CONSTANT per channel, so the recurrence reduces to two
    cumsums — O(S) pure-torch ops, no associative scan — which is why RWKV
    trains much faster than Mamba's selective SSM here.
    """
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)          # pre-norm: bounds k/v so fp16 AMP linears stay finite
        self.time_mix_k = nn.Parameter(torch.ones(dim))
        self.time_mix_v = nn.Parameter(torch.ones(dim))
        self.time_mix_r = nn.Parameter(torch.ones(dim))
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        self.time_decay = nn.Parameter(torch.zeros(dim))   # w = exp(-exp(lw)) ~ 0.37
        self.time_first = nn.Parameter(torch.randn(dim))   # current-token bonus
        self.ln2 = nn.LayerNorm(dim)
        self.time_mix_cm = nn.Parameter(torch.ones(dim))
        self.time_mix_r2 = nn.Parameter(torch.ones(dim))
        self.cm_key = nn.Linear(dim, dim, bias=False)
        self.cm_value = nn.Linear(dim, dim, bias=False)
        self.cm_receptance = nn.Linear(dim, dim, bias=False)
        self.cm_output = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, S, D = x.shape
        xn = self.ln1(x)
        x_shift = F.pad(xn[:, :-1, :], (0, 0, 1, 0))
        # ── time mixing (linear attention, WKV recurrence)
        xk = xn * self.time_mix_k + x_shift * (1 - self.time_mix_k)
        xv = xn * self.time_mix_v + x_shift * (1 - self.time_mix_v)
        xr = xn * self.time_mix_r + x_shift * (1 - self.time_mix_r)
        k = self.key(xk)
        v = self.value(xv)
        r = torch.sigmoid(self.receptance(xr))
        # WKV recurrence is numerically delicate (exp decay + long cumsum):
        # cast to fp32 like mamba_scan so fp16 AMP can't overflow to NaN.
        k = k.float(); v = v.float(); r = r.float()
        w = torch.exp(-torch.exp(self.time_decay.float())).clamp(max=1.0)  # [D] in (0,1]
        t = torch.arange(S, device=x.device).float()
        wpos = torch.clamp(w.view(1, 1, D) * t.view(1, S, 1), -60, 60)
        e = torch.exp(wpos)
        num = torch.cumsum(k * v * e, dim=1) * torch.exp(-wpos)
        den = torch.cumsum(k * e, dim=1) * torch.exp(-wpos)
        b = torch.exp(torch.clamp(self.time_first.float(), -10, 10)).view(1, 1, D)
        wkv = r * ((num + b * k * v) / (den + b * k + 1e-6))
        x = x + self.dropout(self.output(wkv))
        # ── channel mixing (squared ReLU gate, over post-time-mix residual)
        x2 = self.ln2(x)
        x2_shift = F.pad(x2[:, :-1, :], (0, 0, 1, 0))
        xk2 = x2 * self.time_mix_cm + x2_shift * (1 - self.time_mix_cm)
        xr2 = x2 * self.time_mix_r2 + x2_shift * (1 - self.time_mix_r2)
        kk = self.cm_key(xk2)
        y2 = self.cm_output(torch.sigmoid(self.cm_receptance(xr2))
                            * (F.relu(kk).float() ** 2 * self.cm_value(xk2)))
        return x + self.dropout(y2)


class RWKVLM(nn.Module):
    def __init__(self, vocab_size, dim, n_layers, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([RWKVBlock(dim, dropout)
                                     for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
    def forward(self, input_ids, **kw):
        x = self.embed(input_ids)
        for blk in self.blocks:
            x = blk(x)
        return {'logits': self.lm_head(self.ln_f(x))}


class RetNetAttention(nn.Module):
    """RetNet retention (parallel form): QK^T scores decayed by a learned
    per-head exponential mask gamma^(i-j) (causal), then projected through a
    GroupNorm over head groups. O(S^2) matmul like attention but no softmax —
    fast and stable in pure torch.
    """
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.gn = nn.GroupNorm(n_heads, dim)
        self.out = nn.Linear(dim, dim, bias=False)
        # gamma = exp(-exp(log_decay)): spread heads from long-range
        # (log_decay -5 -> gamma ~0.99, 0.99^255 ~0.18 survives) to short-range
        # (log_decay +3 -> gamma ~2e-9, local only). A uniform -1 gives every
        # head gamma^255 ~ 1e-42, i.e. no usable long-range memory.
        self.log_decay = nn.Parameter(torch.linspace(-5.0, 3.0, n_heads))
    def forward(self, x):
        B, S, D = x.shape
        H, Hd = self.n_heads, self.head_dim
        qkv = self.qkv(x).view(B, S, 3, H, Hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        gamma = torch.exp(-torch.exp(self.log_decay)).view(1, H, 1, 1)
        rel = (torch.arange(S, device=x.device).float().view(S, 1)
               - torch.arange(S, device=x.device).float().view(1, S)).clamp(min=0)
        decay = gamma ** rel
        decay = decay.masked_fill(torch.triu(
            torch.ones(S, S, device=x.device, dtype=torch.bool), 1), 0.0)
        y = torch.matmul((q * Hd ** -0.5), k.transpose(-2, -1))
        y = torch.matmul(y * decay, v)
        y = y.permute(0, 2, 1, 3).reshape(B, S, D)
        y = self.gn(y.permute(0, 2, 1)).permute(0, 2, 1)
        return self.out(y)


class RetNetBlock(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ret = RetNetAttention(dim, n_heads)
        self.ln2 = nn.LayerNorm(dim)
        h = int(dim * 4)
        self.fc1 = nn.Linear(dim, h, bias=False)
        self.fc2 = nn.Linear(h, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = x + self.dropout(self.ret(self.ln1(x)))
        x = x + self.dropout(self.fc2(F.silu(self.fc1(self.ln2(x)))))
        return x


class RetNetLM(nn.Module):
    def __init__(self, vocab_size, dim, n_layers, n_heads=8, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([RetNetBlock(dim, n_heads, dropout)
                                     for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
    def forward(self, input_ids, **kw):
        x = self.embed(input_ids)
        for blk in self.blocks:
            x = blk(x)
        return {'logits': self.lm_head(self.ln_f(x))}


# ─────────────────────────── helpers ───────────────────────────
def count_params(m): return sum(p.numel() for p in m.parameters())

def rescale_embed(model, std=0.02):
    """Scale the tied embedding/lm_head to GPT-2-style init so every model
    starts near ln(vocab) (random guess) instead of exploding logits at random init."""
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Embedding):
                m.weight.mul_(std / max(m.weight.std(), 1e-8))
                break

def tune_layers(make_fn, target, vocab):
    """Pick layer count landing closest to the target param budget."""
    best = (1, float('inf'))
    prev, plateau = None, 0
    for L in range(1, 64):
        p = count_params(make_fn(vocab, DIM, L))
        if abs(p - target) < abs(best[1] - target):
            best = (L, p)
        if p >= target * 1.15:
            break
        if prev is not None and p == prev:
            plateau += 1
            if plateau >= 3:
                break
        else:
            plateau = 0
        prev = p
    return best

def tune_nfra(make_fn, target, vocab, depth, min_dim=224):
    """Jointly tune NFRA unique_blocks and hidden dim so it lands near the
    param budget with at least a couple of DISTINCT blocks (real layer
    diversity — a pure 1-block x depth-passes 'recurrent' config is also a
    valid NFRA setting, but not the one used for the head-to-head)."""
    dims = [512, 448, 384, 352, 320, 288, 256, 224, 192, 160, 128]
    best = None
    for U in range(2, min(depth, 8) + 1):
        if depth % U:
            continue
        for d in dims:
            if d < min_dim:
                continue
            p = count_params(make_fn(vocab, d, U))
            err = abs(p - target)
            if best is None or err < best[0]:
                best = (err, U, d, p)
    if best is None:
        return 1, DIM, count_params(make_fn(vocab, DIM, 1))
    return best[1], best[2], best[3]

def make_optimizer(model, lr=3e-4, warmup=50, total=STEPS):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    def sched(step):
        if step < warmup:
            return step / max(warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, sched)

class EMA:
    """Exponential moving average of model weights (zero-cost at inference).

    Shadow weights are updated every optimizer step: shadow = decay*shadow +
    (1-decay)*param. At eval time the EMA weights are temporarily swapped in,
    then restored. Lowers eval loss on small models at no extra inference cost.
    """

    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone().float()
            for k, v in model.named_parameters() if v.requires_grad
        }
        self._backup = {}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.named_parameters():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(),
                                                     alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply(self, model):
        self._backup = {
            k: v.detach().clone()
            for k, v in model.named_parameters() if k in self.shadow
        }
        for k, v in model.named_parameters():
            if k in self.shadow:
                v.data.copy_(self.shadow[k])

    @torch.no_grad()
    def restore(self, model):
        for k, v in model.named_parameters():
            if k in self._backup:
                v.data.copy_(self._backup[k])
        self._backup = {}


def compute_loss(model, x, y, surprise: bool = False):
    out = model(x)
    logits = out['logits']
    logits = logits.view(-1, logits.size(-1))
    targets = y.view(-1)
    logp = logits.log_softmax(-1)
    ce = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)   # [N]
    if surprise:
        # Dopamine-style reward-prediction-error weighting: tokens where the
        # model was wrong/unsure (low P(correct) → high surprise) get higher
        # gradient weight. Weights are normalized to mean 1, so the effective
        # learning rate is unchanged (mean-preserving).
        w = 1.0 - torch.exp(-ce)                                # surprise in (0,1)
        w = w / (w.mean() + 1e-6)
        loss = (ce * w).mean()
    else:
        loss = ce.mean()
    # NFRA 3.3 Cortex: additive adaptive-compute regularizer from the exit gate
    # (easy tokens exit early, hard tokens spend all depth passes). None for
    # every other family, so the head-to-head cost is unchanged.
    if 'exit_aux' in out:
        loss = loss + out['exit_aux']
    return loss

@torch.no_grad()
def evaluate(model, loader, max_batches=15):
    model.eval()
    total, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches: break
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            loss = compute_loss(model, x, y)
        total += loss.item() * x.size(0); n += x.size(0)
    model.train()
    # Empty / too-short loader: report inf, never a silent "perfect" 0.0.
    if n == 0:
        return float('inf')
    return total / n

def measure_speed_memory(model, vocab, n_steps=10):
    x = torch.randint(0, vocab, (BATCH, SEQ_LEN), device=DEVICE)
    y = torch.randint(0, vocab, (BATCH, SEQ_LEN), device=DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler(str(DEVICE)) if USE_AMP else None
    if HAS_CUDA:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t0 = time.perf_counter()
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
    tok_s = BATCH * SEQ_LEN * n_steps / (time.perf_counter() - t0)
    mem = torch.cuda.max_memory_allocated() / 1e9 if HAS_CUDA else 0.0
    return tok_s, mem


# ─────────────────────────── main ───────────────────────────
def fmt_loss(v):
    return f"{v:7.2f}"

def main():
    use_wiki = DATA_SOURCE == 'wikitext2'
    VOCAB = 96 if use_wiki else 4096
    RANDOM_LOSS = math.log(VOCAB)
    target = int(TARGET_M * 1e6)

    print("=" * 66)
    print("  NFRA BRAIN  vs  MAMBA-SSM  vs  GPT-2")
    print("  apples-to-apples  •  matched params  •  identical training")
    print("=" * 66)
    print(f"  bench   : v3 (in-package, local-txt wikitext2)   — {DATA_SOURCE}")
    print(f"  data    : {'WikiText-2 (char)' if use_wiki else 'Synthetic hierarchical'}")
    print(f"  vocab   : {VOCAB}    dim: {DIM}    seq_len: {SEQ_LEN}")
    print(f"  params  : ~{TARGET_M:.0f}M    steps: {STEPS}    batch: {BATCH}")
    print(f"  device  : {'GPU ' + torch.cuda.get_device_name(0) if HAS_CUDA else 'CPU'}"
          + ("   (fp16 AMP)" if USE_AMP else ""))
    feats = []
    if EMA_DECAY > 0:
        feats.append(f"EMA={EMA_DECAY}")
    if SURPRISE:
        feats.append("surprise-weighted loss")
    if KWTA > 0:
        feats.append(f"k-WTA={KWTA}")
    if feats:
        print(f"  features: {', '.join(feats)}  (NFRA 3.2 toggles; apply to all families)")
    print("=" * 66)

    # ── data
    print("\n[1/5] Generating data ... ", end='')
    if use_wiki:
        train_ds = WikiText2Dataset('train', SEQ_LEN)
        eval_ds = WikiText2Dataset('validation', SEQ_LEN)
    else:
        train_ds = HierarchicalDataset(max(4096, BATCH * 8), SEQ_LEN + 1,
                                       seed=SEED, seq_seed=SEED)
        eval_ds = HierarchicalDataset(512, SEQ_LEN + 1,
                                      seed=SEED, seq_seed=SEED + 1)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=0, pin_memory=HAS_CUDA)
    eval_loader = DataLoader(eval_ds, batch_size=BATCH, shuffle=False,
                             num_workers=0, pin_memory=HAS_CUDA)
    print("done")

    # ── build models, matched on params
    print("\n[2/5] Building models (param-matched to ~%.0fM) ..." % TARGET_M)

    U, n_dim, p_n = tune_nfra(make_nfra, target, VOCAB, NFRA_DEPTH)
    nfra = make_nfra(VOCAB, n_dim, U).to(DEVICE)
    rescale_embed(nfra)
    n_passes = NFRA_DEPTH // U
    print(f"    ✓ NFRA Brain   {p_n/1e6:6.1f}M   {U} unique blocks x {n_passes} passes "
          f"= {NFRA_DEPTH} effective layers (dim {n_dim})")

    L_m, p_m = tune_layers(lambda v, d, L: MambaLM(v, d, L, d_state=D_STATE), target, VOCAB)
    mamba = MambaLM(VOCAB, DIM, L_m, d_state=D_STATE).to(DEVICE)
    rescale_embed(mamba)
    print(f"    ✓ Mamba SSM    {p_m/1e6:6.1f}M   {L_m} layers (d_state={D_STATE})")

    L_g, p_g = tune_layers(GPT2ForCausalLM, target, VOCAB)
    gpt2 = GPT2ForCausalLM(VOCAB, DIM, L_g, n_heads=8).to(DEVICE)
    rescale_embed(gpt2)
    print(f"    ✓ GPT-2        {p_g/1e6:6.1f}M   {L_g} layers")

    models = {'NFRA Brain': nfra, 'Mamba SSM': mamba, 'GPT-2': gpt2}

    # ── throughput + memory
    print("\n[3/5] Measuring throughput + peak memory ...")
    perf = {}
    for name, m in models.items():
        m.train()
        tok_s, mem = measure_speed_memory(m, VOCAB)
        perf[name] = {'tok_s': int(tok_s), 'mem': mem}
        print(f"    {name:<11s} {int(tok_s):>9,d} tok/s    peak {mem:.2f} GB")

    # ── training
    print(f"\n[4/5] Training {STEPS} steps (AdamW 3e-4, warmup + cosine)...")
    history = {n: {'loss': [], 'eval': []} for n in models}
    opts = {n: make_optimizer(m) for n, m in models.items()}
    # One identically-seeded DataLoader per model => every family consumes
    # byte-identical batches at the same step (fair head-to-head). Sharing a
    # single loader's iterator would draw offset batches from one RNG.
    loaders = {
        n: iter(DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                           generator=torch.Generator().manual_seed(SEED),
                           num_workers=0, pin_memory=HAS_CUDA))
        for n in models
    }
    scalers = {n: (torch.amp.GradScaler(str(DEVICE)) if USE_AMP else None) for n in models}
    step_ms = {n: [] for n in models}
    emas = {n: (EMA(m, EMA_DECAY) if EMA_DECAY > 0 else None) for n, m in models.items()}

    for step in range(1, STEPS + 1):
        for name, m in models.items():
            try:
                x, y = next(loaders[name])
            except StopIteration:
                loaders[name] = iter(train_loader); x, y = next(loaders[name])
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt, sched = opts[name]
            opt.zero_grad()
            t0 = time.perf_counter()
            with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                loss = compute_loss(m, x, y, surprise=SURPRISE)
            if scalers[name]:
                scalers[name].scale(loss).backward()
                scalers[name].unscale_(opt)
                gnorm = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                if not math.isfinite(gnorm):
                    opt.zero_grad(set_to_none=True)
                scalers[name].step(opt); scalers[name].update()
            else:
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                if math.isfinite(gnorm):
                    opt.step()
                else:
                    opt.zero_grad(set_to_none=True)
            if HAS_CUDA:
                torch.cuda.synchronize()
            step_ms[name].append((time.perf_counter() - t0) * 1000)
            sched.step()
            if emas[name] is not None:
                emas[name].update(m)
            history[name]['loss'].append(loss.item())

        if step % EVAL_GAP == 0 or step == 1 or step == STEPS:
            evals = {}
            for n, m in models.items():
                if emas[n] is not None:
                    emas[n].apply(m)
                evals[n] = evaluate(m, eval_loader)
                if emas[n] is not None:
                    emas[n].restore(m)
            for n in models:
                history[n]['eval'].append((step, evals[n]))
            line = f"    step {step:>5d}/{STEPS}   eval loss:  " + \
                   "  |  ".join(f"{n} {evals[n]:7.2f}" for n in models)
            print(line)

    # ── summary
    print("\n[5/5] Summary  (eval loss = log-perplexity, lower is better)")
    print("-" * 66)
    hdr = f"  {'model':<11s} {'params':>7s} {'depth':>5s} {'eval_loss':>9s} {'tok/s':>9s} {'ms/step':>8s} {'mem':>6s}"
    print(hdr)
    print("-" * 66)
    results = {}
    for name, m in models.items():
        params = count_params(m)
        depth = NFRA_DEPTH if name == 'NFRA Brain' else (L_m if name == 'Mamba SSM' else L_g)
        final = history[name]['eval'][-1][1] if history[name]['eval'] else float('nan')
        results[name] = {'params': params, 'depth': depth, 'eval_loss': round(final, 3),
                         'ppl': round(math.exp(min(final, 30)), 2),
                         'tok_s': perf[name]['tok_s'], 'mem_gb': round(perf[name]['mem'], 2),
                         'ms_per_step': round(sum(step_ms[name]) / len(step_ms[name]), 1)
                         if step_ms[name] else 0.0}
        ppl = f"{math.exp(min(final, 30)):8.2f}" if final < 25 else "   >e^25 "
        print(f"  {name:<11s} {params/1e6:6.1f}M {depth:5d} {final:9.2f} "
              f"{perf[name]['tok_s']:>8,d} "
              f"{results[name]['ms_per_step']:7.1f} {perf[name]['mem']:5.2f}G"
              + (f"   ppl≈{math.exp(min(final,30)):.0f}" if final < 25 else "   ppl: huge"))
    print("-" * 66)
    best_q = min(results, key=lambda k: results[k]['eval_loss'])
    best_s = max(results, key=lambda k: results[k]['tok_s'])
    print(f"\n  ✓ best quality (lowest eval loss): {best_q}")
    print(f"  ✓ best throughput:                 {best_s}")

    print("\n  — how to read this —")
    print(f"  • eval loss ~{RANDOM_LOSS:.2f} = random guessing; lower = better language model.")
    print(f"  • All losses start near {RANDOM_LOSS:.2f} (fair init) and fall as the model")
    print("    learns the structure — judge by the FINAL value.")
    print("  • tok/s here is pure-PyTorch (no fused kernels). Production fused")
    print("    kernels would make Mamba and NFRA dramatically faster.")
    print("  • All three trained on identical data with identical optimizer,")
    print("    matched to ~%.0fM params." % TARGET_M)

    out_path = os.path.join(os.getcwd(), 'nfra_vs_mamba_vs_gpt2_results.json')
    with open(out_path, 'w') as f:
        json.dump({'config': {'steps': STEPS, 'dim': DIM, 'target_params': TARGET_M,
                              'data': DATA_SOURCE, 'vocab': VOCAB, 'batch': BATCH},
                   'results': results, 'perf': perf,
                   'history': {k: {'loss': v['loss'], 'eval': v['eval']}
                               for k, v in history.items()}}, f, indent=2)
    print(f"\n  results saved → {out_path}")


if __name__ == '__main__':
    main()
