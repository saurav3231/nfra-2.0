# NFRA — Frequently Asked Questions (FAQ)

## 1. Basics

**Q1.1 — What is NFRA?**

NFRA (**Nonlinear Factorized Recurrent Attention**) is a recurrent
language-model block: a decayed query-key **retention** mixer
([RetNet](https://arxiv.org/abs/2307.08621)) gated by a token-wise
**receptance** gate ([RWKV](https://arxiv.org/abs/2305.13048)), plus a SwiGLU
feed-forward. It is designed to be verified honestly on modest hardware.

**Q1.2 — Is NFRA a Transformer?**

No. It does not use quadratic self-attention. It is a linear-complexity
recurrent model in the family of state-space models (RetNet, Mamba), with
input-dependent selectivity from the receptance gate.

**Q1.3 — Why the name?**

The acronym NFRA stands for **Nonlinear Factorized Recurrent Attention**,
which describes the actual mechanism. (Earlier project iterations used a
different expansion; the architecture was redesigned based on verified
measurements, and the name was reframed to match what is real.)

## 2. Architecture

**Q2.1 — What is the verified architecture?**

The lean block: `x → LN → Retention-QK mixer → σ(r) receptance gate →
proj_out → + x` followed by `LN → SwiGLU → + x`. Blocks are depth-shared
(`unique_blocks` unique blocks reused `depth_passes` times) with per-pass FiLM
adapters. Multi-scale decay heads (`log_decay −5…+3`) give long-range memory.

**Q2.2 — What happened to the "brain" mechanisms?**

A per-gate isolation sweep (verified, commit `b868477`) measured each mechanism
one at a time. The neuromodulator gland, value gate, phase modulation, and
adaptive-exit gate were all within seed noise (~±0.01 loss) while costing
6–25% training speed. Only the **receptance gate** cleared the +0.02 bar.
So the shipped default (`NFRA_LEAN=1`) is the lean block; the full block is
available behind `NFRA_LEAN=0` for reproducibility but is not recommended.

**Q2.3 — How does the model scale?**

Params scale with `unique_blocks`; effective depth scales with `num_layers`.
This gives near-Transformer effective depth at a fraction of the parameters.

## 3. Benchmarks & results

**Q3.1 — What are the verified results?**

The headline numbers (Kaggle T4, WikiText-2 char, param-matched, identical
data/optimizer/EMA/schedule, seeds 42/7, 600 steps):

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

Full log: [docs/OVERNIGHT_VERIFIED_RESULTS.md](OVERNIGHT_VERIFIED_RESULTS.md).

**Q3.2 — Does NFRA beat its baselines?**

On **quality at matched parameters**: yes — it beats RetNet on loss at both
sizes, both seeds, no overlap (−0.18 nats at 5M, −0.048 nats at 20M), and GPT-2
by a wide margin. On **training speed**: no — RetNet and GPT-2 are faster
(10.5k vs 17.7–24.9k / 33–51k tok/s). NFRA's claim is quality at matched
parameters and long-context generalization, not speed.

**Q3.3 — How are these numbers produced?**

By the repository's own benchmark (`nfra.benchmark.overnight`), with strict
controls: every family is param-matched at bit-identical geometry to RetNet,
consumes identical characters (so token budgets are equal), and runs the same
optimizer, EMA, and step count. This is documented in
[docs/BENCHMARK.md](BENCHMARK.md).

**Q3.4 — Is NFRA revolutionary?**

No single component is new — that is true of essentially every modern
architecture. The defensible claims are the measured synthesis (it beats its
parents at matched params) and the honest methodology. See the identity audit
in [docs/OVERNIGHT_VERIFIED_RESULTS.md](OVERNIGHT_VERIFIED_RESULTS.md) §13.

## 4. Comparison with other models

**Q4.1 — How is NFRA different from RetNet?**

NFRA adds an RWKV-style receptance gate on top of RetNet retention. The
isolation sweep showed this gate is what carries the quality edge (+0.038 when
removed).

**Q4.2 — How is NFRA different from RWKV?**

NFRA uses RetNet's decayed-retention operator (stable, parallelizable,
GroupNorm heads) instead of RWKV's WKV recurrence, which limits effective
context and is numerically delicate.

**Q4.3 — How is NFRA different from Mamba?**

NFRA does not use a matrix state (SSD-style B/C write-read); it uses the
cheaper retention operator. Mamba was used as a comparison family in earlier
benchmarks; it is not in the default grid (its pure-PyTorch implementation is
~700 tok/s and burns runtime).

## 5. Installation & usage

**Q5.1 — How do I install?**

```bash
pip install --no-deps git+https://github.com/saurav3231/nfra-2.0.git
```

Requires Python 3.9+ and PyTorch 2.0+. See the README for details.

**Q5.2 — How do I run a quick check?**

```python
import torch
from nfra import NFRAConfig, NFRAForCausalLM

config = NFRAConfig(vocab_size=32000, hidden_size=512, num_layers=12,
                    unique_blocks=4, depth_shared=True, use_cortex=True)
model = NFRAForCausalLM(config)
logits = model(torch.randint(0, 32000, (2, 64)))["logits"]
print(logits.shape)  # [2, 64, 32000]
```

## 6. Hardware, memory & performance

**Q6.1 — What GPU does this need?**

The verified runs use a single Kaggle T4 (16 GB). Peak memory is small: 1.40 GB
at 5M, 2.16 GB at 20M.

**Q6.2 — Is there a CPU variant?**

Yes — a dependency-light **NFRA Lite** (`NFRALiteForCausalLM`) ships for very
old/low-power CPUs, though it is not the primary verified architecture.

## 7. Troubleshooting

**Q7.1 — `NFRA_COMPILE=1` crashes on the neuromodulator cumsum codegen.**

Known issue with `torch.compile` on some torch builds (the fresh Kaggle torch
Inductor crashes on the cumsum path). Run eager with `NFRA_COMPILE=0` (and, if
using the full block, `NFRA_SCAN_KERNEL=0`). The lean default (`NFRA_LEAN=1`)
removes the neuromodulator entirely and compiles cleanly.

**Q7.2 — RWKV shows NaN during training.**

Fixed by a ratio clamp and EMA NaN guard; RWKV's weaker quality is a
small-model context limitation, not a stability bug.

## 8. Project & community

**Q8.1 — Where is the verified roadmap?**

[docs/FUTURE_PLAN.md](FUTURE_PLAN.md) — including the resolved Part 11 prune
plan and the current lean-block verification run.

**Q8.2 — What are the open problems?**

- Training throughput is ~1.7× lower than RetNet (the quality win is real;
  the next lever is speed).
- Results are verified at 5M/20M over 600 steps; larger-scale behavior is not
  yet measured.
- No sub-word tokenizer or downstream-task evaluation yet.

**Q8.3 — How do I contribute?**

See [CONTRIBUTING.md](../CONTRIBUTING.md).
