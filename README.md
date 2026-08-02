# NFRA — Nonlinear Factorized Recurrent Attention

**An efficient recurrent language-model block that beats RetNet and RWKV on
quality at matched parameters — verified head-to-head on a single modest GPU.**

NFRA is a depth-shared recurrent block built from two well-cited atoms — a
decayed query-key **retention** mixer ([RetNet, Sun et al., 2023](https://arxiv.org/abs/2307.08621))
and a token-wise **receptance gate** ([RWKV, Peng et al., 2023](https://arxiv.org/abs/2305.13048))
— plus a SwiGLU feed-forward. A per-gate isolation sweep showed the receptance
gate is the mechanism that actually carries the quality win, so it ships on and
the rest of the "brain" machinery ships off by default.

**Built by SAURAV BHANDARI** — conceived, designed, and developed with AI assistance.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Ruff](https://img.shields.io/badge/linter-ruff-D7FF64.svg)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/tests-49%20passing-brightgreen.svg)]()

---

## Table of Contents

- [Why NFRA?](#why-nfra)
- [Verified Results](#verified-results)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Benchmarks](#benchmarks)
- [Documentation](#documentation)
- [Development](#development)
- [Author & License](#author--license)

---

## Why NFRA?

Recurrent models promise linear-complexity sequence modeling, but the field's
two strongest recurrent families have distinct weaknesses that show up clearly
at small scale:

- **RetNet** is fast and stable, but its decayed-retention mixer has no
  input-dependent selectivity.
- **RWKV** has strong selectivity (its receptance gate), but its
  per-channel fixed decay limits effective context, and its training is
  numerically delicate.

NFRA combines the two: RetNet-style retention (parallelizable, stable) with an
RWKV-style receptance read-gate (input-dependent selection). The result, at
matched parameters on identical data, beats both on next-token loss while
keeping memory far below a full Transformer.

NFRA is designed for **LLM research and deployment on modest hardware** —
next-token prediction, sequence modeling, and autoregressive generation on a
single small GPU — and ships as a drop-in comparison subject against
Transformer and SSM baselines.

---

## Verified Results

All numbers below are **real measured runs**, produced by this repository's own
benchmark on a **Kaggle T4 GPU**, on **WikiText-2 (character-level, vocab 96)**,
with every family **param-matched** and trained on **identical data, optimizer,
EMA, and schedule** (600 steps, seeds 42/7, fp16 AMP). Full log:
[`docs/OVERNIGHT_VERIFIED_RESULTS.md`](docs/OVERNIGHT_VERIFIED_RESULTS.md).

### Core head-to-head (verified, commit `defe8b2`)

| size | family | eval loss (seed 42 / 7) | mean | train tok/s |
|------|--------|------------------------:|-----:|------------:|
| 5M | **nfra** | 1.961 / 1.945 | **1.953** | 10,320 / 10,495 |
| 5M | retnet | 2.127 / 2.143 | 2.135 | 17,681 / 17,764 |
| 5M | gpt2 | 3.212 / 3.204 | 3.208 | 33,157 / 33,192 |
| 5M | rwkv | 4.275 / 4.267 | 4.271 | 13,364 / 13,447 |
| 20M | **nfra** | 1.763 / 1.763 | **1.763** | 10,484 / 10,500 |
| 20M | retnet | 1.811 / 1.810 | 1.811 | 24,880 / 24,794 |
| 20M | gpt2 | 2.962 / 2.935 | 2.949 | 51,554 / 51,552 |
| 20M | rwkv | 3.931 / 4.090 | 4.011 | 10,755 / 10,708 |

### Fresh-clone re-verification of the lean default (commit `38a3f20`)

The lean 3.3c default (Cortex block, `NFRA_CORTEX=1` by default) was re-run
from a **fresh `git clone`** on a Kaggle T4 — the exact path a new user hits.
It **reproduces and slightly beats the board**, confirming the `38a3f20` fix
(arena previously defaulted to the legacy Brain block, silently ignoring the
lean pruning; see `git log 38a3f20`):

| size | seed | nfra | retnet | board nfra ref |
|------|------|-----:|-------:|---------------:|
| 5M | 42 | **1.923** | 2.134 | 1.953 |
| 5M | 7 | **1.936** | 2.120 | 1.945 |
| 20M | 42 | **1.715** | 1.802 | 1.763 |
| 20M | 7 | **1.710** | 1.816 | 1.763 |

nfra beats retnet at both sizes, both seeds, no overlap; tok/s restored to
17.1–17.5k @5M / 15.2k @20M (vs the 3.5k regression) and memory to 1.26 /
1.92 GB. The lean build lands slightly larger (5.99M / 23.88M params) because
the tuner prefers max distinct blocks (U=33) over exact param match.

**Verified facts:**

- **nfra beats retnet on loss at both sizes, both seeds, no overlap** — by
  −0.18 nats @5M (1.953 vs 2.135) and −0.048 nats @20M (1.763 vs 1.811), at
  **bit-identical geometry** (dim 112/depth 33 @5M, dim 224/depth 33 @20M).
- **nfra is the only family that improves at 4× context length** (1.759→1.719
  when eval @1024 vs train @256); gpt2 collapses (+0.44) and retnet barely moves.
- **Memory** stays small: 1.40 GB @5M / 2.16 GB @20M peak, far below the 16 GB T4.

### Isolation sweep — what actually carries the win (commit `b868477`)

One mechanism turned off at a time, everything else at the exact verified build
(5M / 600 steps / seed 42, eager). Seed noise is ~±0.01:

| config | loss | Δ loss | tok/s | verdict |
|--------|-----:|-------:|------:|---------|
| baseline | 1.966 | +0.000 | 8,059 | — |
| neuromodulator gland OFF | 1.980 | +0.014 | 10,041 | wash (~seed noise) |
| value gate OFF | 1.971 | +0.006 | 9,469 | wash |
| **receptance gate OFF** | **2.004** | **+0.038** | 8,341 | **KEEP — carries the win** |
| phase modulation OFF | 1.971 | +0.005 | 8,759 | wash |
| exit gate OFF | 1.972 | +0.006 | 8,539 | wash |

**Conclusion:** the **receptance gate** (RWKV-style read gate) is the only
mechanism that clears the +0.02 bar — the real differentiator, and cheap. Every
"brain" gate individually is within seed noise while costing 6–25% speed, so
the shipped default (**`NFRA_LEAN=1`**) turns them all off: the block is
**retention + receptance gate + SwiGLU**.

---

## Architecture

The lean block (`NFRA_Cortex_Block`):

```
x → LN → Retention-QK mixer ─→ σ(r) receptance gate → proj_out → + x
                        ↓
x → LN → SwiGLU → + x
```

- **Retention mixer** — decayed `Q·Kᵀ` with per-head exponential decay masks,
  GroupNorm per head, no softmax ([RetNet, 2023](https://arxiv.org/abs/2307.08621)).
- **Receptance gate** — input-dependent read gate `y = proj_out(retention · σ(r))`
  ([RWKV, 2023](https://arxiv.org/abs/2305.13048)).
- **Multi-scale decay heads** — `log_decay −5…+3`, the source of the verified
  4×-length improvement.
- **Depth sharing** — `unique_blocks` unique blocks reused `depth_passes` times
  with per-pass FiLM adapters: params scale with `unique_blocks`, effective depth
  with `num_layers`.

The full 3.3b block (neuromodulator, value gate, phase modulation, adaptive
exit) is retained behind flags for reproducibility — set `NFRA_LEAN=0` — but is
**not** the recommended architecture: the isolation sweep showed it adds cost,
not quality, at this budget.

---

## Installation

Requires **Python 3.9+** and **PyTorch 2.0+**.

```bash
# From GitHub (does NOT reinstall/override your existing torch build)
pip install --no-deps git+https://github.com/saurav3231/nfra-2.0.git

# Or clone and install for development
git clone https://github.com/saurav3231/nfra-2.0.git
cd nfra-2.0
pip install -e .
```

Verify:

```python
import nfra
print(nfra.__version__, nfra.__author__)   # 3.3.0 SAURAV BHANDARI
```

---

## Quick Start

```python
import torch
from nfra import NFRAConfig, NFRAForCausalLM

config = NFRAConfig(
    vocab_size=32000,
    hidden_size=512,
    num_layers=12,
    unique_blocks=4,        # 4 distinct blocks reused depth-shared
    depth_shared=True,
    use_cortex=True,        # NFRA_Cortex_Block (the verified architecture)
)
model = NFRAForCausalLM(config)
print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

x = torch.randint(0, 32000, (2, 64))
logits = model(x)["logits"]
print(logits.shape)         # [2, 64, 32000]
```

A dependency-light **NFRA Lite** variant (`NFRALiteForCausalLM`) is also
shipped for very old/low-power CPUs:

```python
from nfra.models import create_nfra_lite
model = create_nfra_lite()
```

---

## Benchmarks

Two credible, reproducible benchmarks ship **inside the package** (no external
scripts).

### 1. `nfra.benchmark.compare` — quick apples-to-apples head-to-head

```bash
python -m nfra.benchmark.compare
```

NFRA vs RetNet vs RWKV vs GPT-2, param-matched, identical training, real
WikiText-2 char data. Outputs final eval loss, perplexity, throughput, and peak
memory.

### 2. `nfra.benchmark.overnight` — the verified multi-phase benchmark

```bash
python -m nfra.benchmark.overnight
```

The phased, resumable run that produced the verified results above
(`core, context, efficiency, ablate, recall, deploy, perf` phases). Answers
"who wins on which aspect" with mean ± std over seeds, measured scaling slopes,
inference latency, memory, and long-context extrapolation.

#### Environment reference

| Env var | Default | Meaning |
|---------|---------|---------|
| `NFRA_OVN_MODE` | `standard` | `quick` (300) / `standard` (600) / `big` (1500 steps) |
| `NFRA_OVN_SIZES` | `5,20` | Target model sizes in millions of params |
| `NFRA_OVN_SEEDS` | `2` | Independent seeds for mean ± std |
| `NFRA_OVN_FAMILIES` | `nfra,rwkv,retnet,gpt2` | Architectures to include |
| `NFRA_OVN_PHASES` | all | Comma list of phases: `core,context,efficiency,ablate,recall,deploy,perf,data2` |
| `NFRA_OVN_DATA` | `wikitext2` | Data source (only `wikitext2` is allowed for the headline run) |
| `NFRA_LEAN` | `1` | `0` = full 3.3b block (all gates), `1` = lean (receptance gate only) |
| `NFRA_CORTEX` | `1` | `1` = Cortex block (verified architecture, default), `0` = legacy Brain block A/B |
| `NFRA_COMPILE` | `1` | `1` = `torch.compile(mode='reduce-overhead')` (auto-disables when unstable) |

#### Outputs

- `overnight_results.json` — full per-seed data, config fingerprint, machine-readable verdict
- `overnight_report.md` — a publishable Markdown report with tables, scaling fits, and per-aspect winners

> Full Kaggle step-by-step guide: **[docs/BENCHMARK.md](docs/BENCHMARK.md)**

---

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/OVERNIGHT_VERIFIED_RESULTS.md](docs/OVERNIGHT_VERIFIED_RESULTS.md) | All verified benchmark results + identity audit + isolation sweep |
| [docs/BENCHMARK.md](docs/BENCHMARK.md) | Step-by-step benchmark guide (Kaggle T4) + methodology |
| [docs/RESEARCH_PAPER.md](docs/RESEARCH_PAPER.md) | Research paper draft |
| [docs/FUTURE_PLAN.md](docs/FUTURE_PLAN.md) | Verified roadmap and open levers |
| [FAQ.md](docs/FAQ.md) | Frequently asked questions (usage, results, troubleshooting) |

---

## Project Structure

```
nfra-2.0/
├── src/nfra/
│   ├── __init__.py            # public API + metadata
│   ├── benchmark/
│   │   ├── compare.py         # quick head-to-head
│   │   ├── arena.py           # shared harness (loaders, trainers, families)
│   │   └── overnight.py       # phased, resumable multi-phase benchmark
│   ├── core/                  # blocks: cortex.py (verified), legacy mixers
│   ├── models/                # NFRAForCausalLM, NFRAConfig, NFRA Lite
│   ├── training/              # trainer, losses
│   ├── evaluation/            # metrics
│   ├── kernels/               # selective-scan backend
│   └── utils/                 # hardware info, quantization
├── scripts/                   # research helpers (iso_sweep.py, prof_nfra.py, ...)
├── docs/                      # documentation
├── examples/                  # usage examples
├── tests/                     # unit tests
├── pyproject.toml             # packaging + tooling (Black, Ruff)
├── CONTRIBUTING.md
└── LICENSE
```

---

## Development

```bash
pip install -e ".[dev]"

# Lint + format
ruff check src
black --check src

# Tests (CPU smoke: model construction, forward/backward, save/load,
# param-scaling with unique_blocks, and benchmark scoring math)
pytest
```

- Code style: **Black** (line-length 88), linted with **Ruff**.
- Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).
- Reporting issues or asking questions? Open a GitHub issue.

---

## Author & License

**Author:** SAURAV BHANDARI

**License:** MIT — see [LICENSE](LICENSE).
