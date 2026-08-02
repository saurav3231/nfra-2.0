# NFRA Research Paper Outline

**Title:**
**NFRA: Nonlinear Factorized Recurrent Attention — a Verified, Small-Scale
Recurrent Language-Model Block**

**Author:** Saurav Bhandari
**Affiliation:** Independent Researcher
**Date:** August 2026

---

## Abstract

We present NFRA, a recurrent language-model block that combines a decayed
query-key retention mixer (RetNet) with a token-wise receptance gate (RWKV)
and a SwiGLU feed-forward. At matched parameters on identical data it beats
both parents on next-token loss and is the only family that improves at 4× the
training context length. A per-gate isolation sweep identifies the receptance
gate as the mechanism that carries the win.

---

## 1. Introduction

- Motivation: recurrent models promise linear complexity; RetNet lacks
  selectivity, RWKV lacks stable context
- Goal: a small, verifiable block combining both strengths
- Why matched-parameter, same-budget verification matters
- Our contributions (synthesis + verified measurements + isolation methodology)

## 2. Related Work

- Retention and decayed attention (RetNet)
- Receptance gating (RWKV)
- State space models (Mamba/SSD)
- Depth sharing / weight tying

## 3. Architecture

### 3.1 The lean block (retention + receptance gate + SwiGLU)
### 3.2 Multi-scale decay heads
### 3.3 Depth sharing with per-pass adapters
### 3.4 The full 3.3b block (flag-only, pruned)

## 4. Training Methodology

- Data: WikiText-2 character-level (vocab 96) — equal token budgets
- Identical optimizer / EMA / schedule for every family
- Param-matched builds at bit-identical geometry to RetNet
- Sizes 5M/20M, seeds 42/7, 600 steps, fp16 AMP, T4

## 5. Experiments

### 5.1 Head-to-head at matched parameters (verified)
### 5.2 Long-context extrapolation at 4× length (verified)
### 5.3 Isolation sweep — which mechanism carries the win (verified)

## 6. Ablation Studies

- Per-gate isolation (gland / value gate / receptance gate / phase / exit)
- Stacked-levers baseline comparison

## 7. Discussion & Limitations

- Throughput gap vs RetNet (quality win is the claim, not speed)
- Small-scale verification only; scaling not yet measured
- No sub-word tokenizer / downstream-task evaluation yet

## 8. Conclusion & Future Work

- Throughput profiling and fusion
- Loss-vs-steps scaling probe
- Sub-word tokenization and standard eval suites
- Lean-block head-to-head verification (in progress)

## References

- Sun et al. Retentive Network (2023)
- Peng et al. RWKV (2023)
- Gu & Dao. Mamba (2023)
- Shazeer. GLU Variants (2020)
- Graves. Adaptive Computation Time (2016)

---

**Note:** All experimental results in this outline are real, measured numbers
produced by the repository's own benchmark on a Kaggle T4 GPU. See
`docs/OVERNIGHT_VERIFIED_RESULTS.md` for the full log.
