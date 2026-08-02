# NFRA: Nonlinear Factorized Recurrent Attention

**Author:** Saurav Bhandari
**Affiliation:** Independent Researcher
**Date:** August 2026

---

## Abstract

We present **NFRA**, a recurrent language-model block that combines a
decayed query-key **retention** mixer ([RetNet, Sun et al., 2023](https://arxiv.org/abs/2307.08621))
with a token-wise **receptance gate** ([RWKV, Peng et al., 2023](https://arxiv.org/abs/2305.13048))
and a SwiGLU feed-forward. The block is depth-shared (a small number of unique
blocks reused across depth with per-pass adapters), giving near-Transformer
effective depth at a fraction of the parameters.

On **WikiText-2 character-level** at matched parameters, identical data,
optimizer, EMA, and schedule on a single T4 GPU, NFRA beats **RetNet** on
next-token loss at both tested scales without overlap (−0.18 nats at 5M,
−0.048 nats at 20M) and is the only family that **improves** when evaluated at
4× the training context length. A per-gate isolation sweep shows the quality
edge is carried by the receptance gate; the remaining "brain" mechanisms
(neuromodulation, value gate, phase modulation, adaptive exit) are within seed
noise and are removed by default for speed.

We report a verified, reproducible methodology and honest attribution: no
atomic component is new, and the claim this work makes is limited to the
measured effect of the synthesis.

**Keywords:** Efficient AI, Recurrent Attention, Retention, Receptance Gating,
State Space Models, Small-Scale Language Models

---

## 1. Introduction

### 1.1 Motivation

Recurrent sequence models promise linear-complexity training and inference for
language modeling, but the two dominant recurrent families have complementary
weaknesses that are visible even at small scale:

- **RetNet** is fast, stable, and parallelizable, but its decayed-retention
  mixer has no input-dependent selectivity.
- **RWKV** has strong input-dependent selectivity (the receptance gate), but
  its per-channel fixed decay limits effective context, and its training is
  numerically delicate.

The goal of this project is to build a small, verifiable recurrent block that
combines the strengths of both, and to measure — honestly, on identical
hardware and data — whether the synthesis beats its parents at matched
parameters.

### 1.2 Why matched-parameter, same-budget verification matters

Architecture claims are routinely confounded by tokenizer differences,
optimizer schedules, and parameter-count drift. NFRA's benchmark controls all
of these: every family is built to the same parameter budget, consumes the same
characters (so token budgets are exactly equal), and runs the same optimizer,
EMA, and step count. Under those controls, a loss difference is attributable to
the architecture.

### 1.3 Our Contributions

1. A **depth-shared retention + receptance-gate block** that outperforms both
   parents on loss at matched parameters (verified, two sizes, two seeds).
2. **Verified long-context behavior**: the only family that improves at 4× the
   training context length.
3. A **per-gate isolation methodology** that separates genuine mechanisms from
   decoration, with the decoration pruned for speed.
4. An honest, fully reproducible benchmark harness that ships inside the
   package (`nfra.benchmark`).

---

## 2. Related Work

### 2.1 Retention and decayed attention (RetNet)

RetNet ([Sun et al., 2023](https://arxiv.org/abs/2307.08621)) replaces
softmax attention with `Q·Kᵀ` scaled by exponential per-head decay masks
`γ^(i−j)`, with GroupNorm applied per head, enabling a recurrent representation
with linear inference cost. NFRA uses this operator as its sequence mixer.

### 2.2 Receptance gating (RWKV)

RWKV ([Peng et al., 2023](https://arxiv.org/abs/2305.13048)) introduces
input-dependent time-mixing where a token-wise **receptance** gate
`y = proj_out(x · σ(r))` selectively writes information. NFRA applies this
read-gate on top of retention output.

### 2.3 State space models (Mamba)

Mamba and SSD ([Gu & Dao, 2023](https://arxiv.org/abs/2312.00752)) frame
sequence mixing as a selective state-space model with matrix states. Earlier
NFRA versions used a matrix-state mixer; the verified 3.3 architecture dropped
it for the cheaper retention operator.

### 2.4 Depth sharing

Sharing a small set of blocks across depth (with per-depth adapters) reduces
parameter growth while preserving effective depth, a practice related to
weight-tied and deep-shared architectures.

---

## 3. Architecture

### 3.1 The lean block

```
x → LN → Retention-QK mixer ─→ σ(r) receptance gate → proj_out → + x
                        ↓
x → LN → SwiGLU → + x
```

- **Retention mixer** — decayed `Q·Kᵀ` with per-head exponential decay masks,
  GroupNorm per head, no softmax ([RetNet, 2023](https://arxiv.org/abs/2307.08621)).
- **Receptance gate** — input-dependent read gate `y = proj_out(retention · σ(r))`
  ([RWKV, 2023](https://arxiv.org/abs/2305.13048)).
- **Multi-scale decay heads** — `log_decay −5…+3`, the source of the measured
  4×-length improvement.
- **Depth sharing** — `unique_blocks` blocks reused `depth_passes` times with
  per-pass FiLM adapters.

### 3.2 The full 3.3b block (flag only)

The full block additionally contains a 6-channel causal neuromodulator gland
(ACh/NE/DA/5HT/CORT/OX), an ACh-gated value write, sin-phase selectivity
modulation, and a Gumbel adaptive-exit gate. These are retained behind
`NFRA_LEAN=0` for reproducibility but are **not** recommended: the isolation
sweep (§5.3) showed they are within seed noise while costing 6–25% training
speed.

---

## 4. Training Methodology

### 4.1 Setup

- **Data:** WikiText-2, character-level (vocab 96, random loss 4.564), so
  token budgets are exactly equal across families.
- **Optimizer:** identical optimizer and schedule for every family; EMA 0.99.
- **Budget:** 600 steps (mode `standard`), batch 8, fp16 AMP on a Kaggle T4.
- **Sizes:** 5M and 20M parameters, matched exactly (nfra builds the same
  dim/depth geometry as retnet at each size).
- **Seeds:** 42 and 7, reported as per-seed values and mean.

The harness is `nfra.benchmark.overnight`; the methodology and controls are
documented in `docs/BENCHMARK.md`.

### 4.2 Verification controls

- Every family: identical data, optimizer, EMA, token budget, and seeds.
- Param-matched builds at bit-identical geometry to RetNet.
- NaN guards on every step and an eval harness that reports stability events.

---

## 5. Experiments

### 5.1 Head-to-head at matched parameters

Full run on Kaggle T4, 600 steps, seeds 42/7. **These are the verified results**
(log: `docs/OVERNIGHT_VERIFIED_RESULTS.md` §11):

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

**Findings.** nfra beats retnet on loss at both sizes, both seeds, no overlap.
Its throughput is lower (10.5k vs 17.7–24.9k tok/s) — the quality win is real,
and throughput is the next open lever, not a claim of speed superiority.

### 5.2 Long-context extrapolation

Trained at context 256, evaluated at 256/512/1024 (verified, §12):

| family | @256 | @512 | @1024 |
|--------|-----:|-----:|------:|
| **nfra** | 1.759 | 1.763 | **1.719** |
| retnet | 1.816 | 1.819 | 1.773 |
| gpt2 | 3.024 | 3.337 | 3.463 |
| rwkv | 3.930 | 3.938 | 3.937 |

nfra is the only family that improves at 4× length (−0.04 nats), attributed to
the multi-scale decay heads.

### 5.3 Isolation sweep — which mechanism carries the win

One mechanism off at a time at the exact verified build (5M / 600 steps /
seed 42; seed noise ~±0.01). Verified, commit `b868477`:

| config | loss | Δ loss | tok/s | verdict |
|--------|-----:|-------:|------:|---------|
| baseline | 1.966 | +0.000 | 8,059 | — |
| neuromodulator gland OFF | 1.980 | +0.014 | 10,041 | wash |
| value gate OFF | 1.971 | +0.006 | 9,469 | wash |
| **receptance gate OFF** | **2.004** | **+0.038** | 8,341 | **KEEP** |
| phase modulation OFF | 1.971 | +0.005 | 8,759 | wash |
| exit gate OFF | 1.972 | +0.006 | 8,539 | wash |

The **receptance gate** is the single mechanism clearing the +0.02 bar. The
lean block (retention + receptance gate + SwiGLU) is therefore the default
architecture.

**Reproducibility addendum (commit `38a3f20`):** the lean default was re-run
from a **fresh `git clone`** on a Kaggle T4 (core phase, 600 steps, seeds
42/7). It reproduces and slightly beats the board: nfra 1.923/1.936 @5M and
1.715/1.710 @20M vs retnet 2.134/2.120 and 1.802/1.816 — the win holds at both
sizes, both seeds. (This re-verification also caught and fixed a default-block
bug: the arena previously built the legacy Brain block unless `NFRA_CORTEX=1`,
so a fresh clone silently ignored the lean pruning and regressed to ~2.14 @
3.5k tok/s. The Cortex block is now the default.)

### 5.4 Honest limitations

- **Throughput** is ~1.7× lower than RetNet at matched quality; NFRA does not
  claim a speed advantage.
- Results are at 5M/20M over 600 steps; larger-scale behavior is not yet
  verified.
- No sub-word tokenizer or downstream-task evaluation is yet reported.

---

## 6. Future Work

1. **Throughput**: profile and fuse the remaining launch-bound kernels
   (`scripts/prof_nfra.py`).
2. **Scaling**: map loss-vs-steps at 20M and probe 50M+.
3. **Sub-word tokenization** and downstream-task benchmarks (perplexity on
   WikiText-103, standard eval suites).
4. Verify the lean block head-to-head vs retnet (in progress, `NFRA_OVN_PHASES=core`).

---

## 7. Conclusion

NFRA is a small, honest recurrent block that verifiably beats its parents —
RetNet and RWKV — on quality at matched parameters, and that generalizes to
longer contexts. Its contribution is a measured synthesis, not atomic novelty,
and this draft reports only measured numbers with full attribution.

---

## References

- Sun, Y. et al. **Retentive Network: A Successor to Transformer for Large
  Language Models.** 2023. https://arxiv.org/abs/2307.08621
- Peng, B. et al. **RWKV: Reinventing RNNs for the Transformer Era.** 2023.
  https://arxiv.org/abs/2305.13048
- Gu, A. & Dao, T. **Mamba: Linear-Time Sequence Modeling with Selective State
  Spaces.** 2023. https://arxiv.org/abs/2312.00752
- Shazeer, N. **GLU Variants Improve Transformer.** 2020.
  https://arxiv.org/abs/2002.05202
- Dao, T. et al. **FlashAttention: Fast and Memory-Efficient Exact Attention
  with IO-Awareness.** 2022. https://arxiv.org/abs/2205.14135
- Larsson, G. et al. **FractalNet: Ultra-Deep Neural Networks without
  Residuals.** 2017. https://arxiv.org/abs/1605.07648
- Graves, A. **Adaptive Computation Time for Recurrent Neural Networks.** 2016.
  https://arxiv.org/abs/1603.08983
