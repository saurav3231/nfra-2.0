# 🧠 NFRA 2.0 — NeuroFractal Resonance Architecture

**A brain-inspired neural network for quality AI on modest hardware — benchmarked apples-to-apples against Mamba-SSM and GPT-2.**

> Built on **PyTorch** for **large language model (LLM)** research and deployment: next-token prediction, sequence modeling, and autoregressive text generation on a single modest GPU. A drop-in comparison subject against **Transformer** and **State Space Model (SSM)** baselines such as Mamba, with low-memory training and inference.

> Built by **SAURAV BHANDARI** — conceived, designed, and developed with AI assistance.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/badge/linter-ruff-D7FF64.svg)](https://github.com/astral-sh/ruff)
[![Status](https://img.shields.io/badge/status-research--alpha-orange.svg)]()

---

## Table of Contents

- [What is NFRA?](#-what-is-nfra)
- [Measured Results (Real Benchmarks)](#-measured-results)
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Benchmarks](#-benchmarks)
- [Project Structure](#-project-structure)
- [Documentation](#-documentation)
- [Development](#-development)
- [Author & License](#-author--license)

---

## 🌍 What is NFRA?

NFRA (NeuroFractal Resonance Architecture) is a family of neural-network building blocks inspired by how biological brains use **fractal self-similarity**, **resonance-based sparse activation**, and **predictive coding** to process sequences efficiently.

It ships in two flavours:

| Flavour | What it is | Best for |
|---------|-----------|----------|
| **NFRA Brain** (`mode="brain"`) | Multi-band sequence mixer + fractal gated MLPs, local attention, selective decay, per-pass modulation | Quality-vs-resource benchmarks (`nfra.benchmark`) |
| **NFRA Lite** (`NFRALiteForCausalLM`) | Single-file, dependency-light model | 2012–2018 CPUs, Raspberry Pi, microcontrollers |

**The goal**: make capable AI accessible to the billions of devices that cannot afford high-end GPUs — and to prove *with evidence* how it compares to mainstream architectures.

---

## 📊 Measured Results

These are **real measured numbers**, not projections — produced by this repository's own benchmark on a Kaggle T4 GPU, on real text (WikiText-2, character-level), with all models **param-matched (~20M)**, trained on **identical data, optimizer, and schedule** (600 steps).

| Model | Eval loss (↓) | Perplexity (↓) | Train tok/s (↑) | Peak memory (↓) |
|-------|--------------:|---------------:|----------------:|----------------:|
| **NFRA Brain** | **2.13** | ≈ 8 | 2,042 | **0.62 GB** |
| **Mamba SSM** (ref) | **1.59** | ≈ 5 | 845 | 5.09 GB |
| **GPT-2** (ref) | 3.19 | ≈ 24 | 37,570 | 0.95 GB |

**What this shows:**
- **Quality:** Mamba wins (1.59), NFRA beats GPT-2 by a large margin (2.13 vs 3.19).
- **Memory:** NFRA uses **8.2× less peak memory** than Mamba and **1.5× less** than GPT-2.
- **Speed:** NFRA trains **2.4× faster** than Mamba (pure-PyTorch implementations; fused kernels would raise both).

> The **NFRA Arena** benchmark extends this into a global-standard, multi-dimension comparison — multiple sizes, multiple seeds (mean ± std), scaling slopes, inference latency, and an evidence-based verdict. See [Benchmarks](#-benchmarks).

---

## ✨ Key Features

- **Multi-band sequence mixing** — recurrence at several temporal resolutions in a single block (α ∈ [0.90, 0.995])
- **Fractal gated MLPs** — hierarchical, structurally sparse feed-forward networks
- **Selective (input-dependent) decay** — the model decides how much to remember per token (SSM-style, but built on parallel scans)
- **Resonance-guided local attention** — cheap, windowed attention gated by neuromodulation
- **Per-pass FiLM adapters + global brain state** — depth-shared blocks are modulated per pass like cortical layers
- **Gradient checkpointing & fp32-safe scans** — stable training on modest GPUs (fp16 scan overflow → NaN is guarded)
- **Legacy-hardware-friendly Lite variant** — single-file, no heavy dependencies

---

## 💾 Installation

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
print(nfra.__version__, nfra.__author__)   # 3.1.0 SAURAV BHANDARI
```

---

## 🚀 Quick Start

### NFRA Brain (full model)

```python
import torch
from nfra import NFRAConfig, NFRAForCausalLM

config = NFRAConfig(
    mode="brain",
    vocab_size=32000,
    hidden_size=512,
    num_layers=12,
    unique_blocks=4,        # 4 distinct blocks reused depth-shared
    depth_shared=True,
)
model = NFRAForCausalLM(config)
print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

x = torch.randint(0, 32000, (2, 64))
logits = model(x)["logits"]
print(logits.shape)         # [2, 64, 32000]
```

### NFRA Lite (legacy hardware)

```python
from nfra import NFRAConfig
from nfra.models.nfra_lite import NFRALiteForCausalLM
model = NFRALiteForCausalLM(NFRAConfig(mode="lite", vocab_size=50257))
```

---

## 🏆 Benchmarks

Two credible, reproducible benchmarks ship **inside the package** (no external scripts).

### 1. `nfra.benchmark.compare` — quick apples-to-apples head-to-head

```bash
python -m nfra.benchmark.compare
```

NFRA Brain vs Mamba-SSM vs GPT-2, param-matched (~20M), identical training, real Wikitext-2 char data. Outputs final eval loss, perplexity, throughput, and peak memory.

### 2. `nfra.benchmark.arena` — global-standard multi-dimension comparison

```bash
python -m nfra.benchmark.arena
```

The credible one. Answers **"who wins on which aspect"** and **"is NFRA really revolutionary?"** across:

- **Quality** — eval loss & perplexity, multiple seeds (reported as **mean ± std**)
- **Scaling** — multiple model sizes → measured **power-law slope** (bits of loss per doubling of params) + extrapolation to 100M
- **Efficiency** — sample-efficiency AUC, parameter efficiency, est. FLOPs/token
- **Speed** — train tok/s, ms/step, prefill & autoregressive generation tok/s, ms/token
- **Memory** — peak training & inference GB
- **Robustness** — eval at 2× context length, NaN/stability events
- **Composite score** — weighted z-scores across all dimensions, plus a structured evidence-based **verdict**

#### Environment reference

| Env var | Default | Meaning |
|---------|---------|---------|
| `NFRA_MODE` | `standard` | `quick` (150) / `standard` (600) / `rigorous` (1500 steps) |
| `NFRA_DATA` | `synthetic` | `synthetic` or `wikitext2` (real text; requires the two local `.txt` files) |
| `NFRA_SIZES` | `5,20` | Target model sizes in millions of params |
| `NFRA_SEEDS` | `2` | Number of independent seeds for mean ± std |
| `NFRA_FAMILIES` | `nfra,mamba,gpt2` | Which architectures to include |
| `NFRA_BATCH` | auto | Override training batch size |
| `NFRA_TARGET_PARAMS` | `20` | Target params (M) for `compare` |
| `NFRA_DIM` | `512` | Hidden size for `compare` |
| `NFRA_EMA` | `0` | EMA weight-averaging decay (e.g. `0.999`); eval uses averaged weights. Applied to all families for a fair head-to-head |
| `NFRA_SURPRISE` | `0` | `1` = surprise-weighted (dopamine-RPE) gradients; mean-preserving weights. Applied to all families |
| `NFRA_KWTA` | `0` | k-WTA lateral inhibition fraction for NFRA (e.g. `0.5`); `0` = off |
| `NFRA_SCAN_KERNEL` | `1` | Selective-scan backend: `0` = torch closed-form, `1` = auto (Triton CUDA kernel when available), `2` = force Triton |
| `NFRA_BANDS` | `16` | Recurrence band/head count for NFRA Brain (H8 ablation: `2,4,8,16`; `16` = hierarchical `[8,4,2,1]+router`) |
| `NFRA_RECALL_KS` | `4,16,64,128` | Spans for the H3 memory-horizon probe (`python -m nfra.benchmark.recall_probe`) |
| `NFRA_RECALL_STEPS` | `400` | Train steps per (span, model) in the recall probe |

#### Outputs

- `nfra_arena_results.json` — full per-seed data, config fingerprint, machine-readable verdict
- `nfra_arena_report.md` — a publishable Markdown report with tables, scaling fits, winners-per-aspect, and the verdict

> Full Kaggle step-by-step guide: **[docs/BENCHMARK.md](docs/BENCHMARK.md)**

---

## 📁 Project Structure

```
nfra-2.0/
├── src/nfra/
│   ├── __init__.py            # public API + metadata
│   ├── benchmark/
│   │   ├── compare.py         # quick head-to-head (v3)
│   │   └── arena.py           # global-standard multi-dimension benchmark
│   ├── core/                  # building blocks: mixers, scans, energy, neuro
│   ├── models/                # NFRAForCausalLM, NFRAConfig, NFRA Lite
│   ├── training/              # trainer, losses
│   ├── evaluation/            # metrics
│   └── utils/                 # config, IO, dataset helpers
├── benchmarks/                # standalone legacy benchmark scripts
├── notebooks/                 # Kaggle / Colab notebooks
├── docs/                      # documentation (research draft, benchmark guide)
├── examples/                  # usage examples
├── configs/                   # YAML configs
├── scripts/                   # helper scripts
├── tests/                     # unit tests
├── pyproject.toml             # packaging + tooling (Black, Ruff)
├── CONTRIBUTING.md
└── LICENSE
```

---

## 📚 Documentation

| Doc | Contents |
|-----|----------|
| [docs/BENCHMARK.md](docs/BENCHMARK.md) | Full step-by-step benchmark guide (Kaggle T4) + methodology + how to read results |
| [FAQ.md](FAQ.md) | Frequently asked questions (usage, results, comparison, troubleshooting) |
| [docs/NFRA_2.0_Research_Paper_Draft.md](docs/NFRA_2.0_Research_Paper_Draft.md) | Research paper draft |
| [docs/PAPER_OUTLINE.md](docs/PAPER_OUTLINE.md) | Paper outline |
| [docs/DATASETS.md](docs/DATASETS.md) | Dataset notes |
| [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) | Detailed install guide |

---

## 🛠️ Development

```bash
pip install -e ".[dev]"

# Lint + format
ruff check src
black --check src

# Tests
# CPU smoke tests: model construction, forward/backward, save/load IO,
# param-scaling with unique_blocks, and the benchmark scoring math.
pytest
```

- Code style: **Black** (line-length 88), linted with **Ruff**.
- Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).
- Reporting issues or asking questions? Open a GitHub issue.

---

## 👤 Author & License

**Author:** SAURAV BHANDARI

**License:** MIT — see [LICENSE](LICENSE).

---

*"The future of AI should be measured not only by capability, but by accessibility and sustainability."*

— **SAURAV BHANDARI**, July 2026
