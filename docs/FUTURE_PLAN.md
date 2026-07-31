# NFRA 2.0 — Future Plan (Theory, Critique, Roadmap)

**Status:** working plan — updated through 5 criticism rounds
**Motive anchor:** *quality AI on modest hardware* — every idea must keep **latency, peak memory, and training speed low**, and keep the benchmark **credible**.
**Baseline facts (measured, Kaggle T4, WikiText-2 char, ~20M params, 600 steps):**

| Model | Eval loss (↓) | Peak mem (↓) | Train tok/s (↑) |
|-------|--------------:|-------------:|----------------:|
| NFRA Brain | 2.13 | **0.62 GB** | 2,042 |
| Mamba SSM | **1.59** | 5.09 GB | 845 |
| GPT-2 | 3.19 | 0.95 GB | 37,570 |

At 5M: NFRA 2.51 / 0.14 GB / 3,178 tok/s vs Mamba 1.81 / 3.66 GB / 669 tok/s.

---

## Part 0 — The core theory (v0)

> **NFRA's loss gap is a capacity-placement problem, not a mixing problem.**
> NFRA wins where it *doesn't* spend (memory ×26, speed ×4.7). It loses on loss because capacity is mis-placed: depth is diluted (`unique_blocks` 6 vs Mamba's 30 real layers), dynamics are frozen (fixed α grid), and sparsity is uniform (easy tokens cost the same as hard ones).

v0 levers:
- **A. AFC-α** — input-adaptive band decay (multi-scale selectivity)
- **B. AFC-LoRA** — per-pass low-rank adapters on depth-shared blocks
- **C. AFC-MoE** — token-routed experts (code already exists, not wired in)
- **D. Training** — EMA + multi-epoch + iso-wall-clock reporting
- **E. Memory-matched scaling** — NFRA's real metric is loss-per-megabyte
- **F. Difficulty-conditional compute** — gate bands/experts by predicted entropy

---

## Part 1 — Criticism round 1: measurement & benchmark validity

**Attacked:** We are optimizing against one metric (WikiText-2 char, 600 steps, 2 seeds). The gap may be partly a *measurement artifact*, not an architecture fact.

- Fixed 600 steps favors fast-converging models; NFRA might be slower-converging, not worse-capacity.
- 2 seeds ⇒ high variance; a "gap" of 0.3 could be noise.
- Char-level vocab 96 under-represents real token-level difficulty.

**Survives:** capacity-placement as the leading hypothesis (memory/speed edge is structural, real).

**Changed / advanced:**
- Every lever must be validated on **≥3 seeds, mean ± std**, and on **2+ datasets** (add a token-level corpus later).
- Report **both** iso-step and iso-wall-clock (NFRA's 4.7× speed is a real advantage, not a dodge).

**New idea — H1. Hyperparameter fairness protocol.** Each family gets a *small* HP search (LR × schedule × warmup, same token budget). Otherwise "NFRA improved" is confounded by LR luck. This is the difference between a credible result and a blog post.

---

## Part 2 — Criticism round 2: training-dynamics critique

**Attacked:** 600 steps on char-WikiText is *deeply under-trained* (5M loss 2.5 is far from converged). We may be treating a **convergence problem as a capacity problem**. Also, EMA/multi-epoch (v0 D) are tricks Mamba could equally use — not NFRA-specific principles.

**Survives:** A/B/C (capacity levers) remain valid, but *only after* convergence parity.

**Changed / advanced:**
- Re-sequence the plan: **Phase 0 = convergence parity first.** Measure loss-vs-steps curves and sample-efficiency AUC (arena already computes AUC). Only when NFRA and Mamba are at comparable optimization maturity do capacity levers matter.
- Frame D honestly: it is *leveraging NFRA's speed* (a structural property), not a trick that Mamba denies.

**New idea — H2. Convergence-rate gradient shaping.** The band recurrence has a spectral structure; long-α bands have ill-conditioned gradients. Add per-band **gradient scaling / norm shaping** (normalize band-mixer grads by α scale) so memory-horizon information flows earlier. Expected: faster loss drop at fixed steps + better wall-clock efficiency. Cheap, no inference cost.

**New idea — H3. Empirical memory-horizon probe.** Before building adaptive α, *measure* NFRA's effective recall with synthetic probes (associative recall, copy, retrieval at distance k). This decides the diagnosis: if recall collapses at distance d, the gap is **memory** (→ AFC-α); if recall is fine but loss lags, the gap is **capacity** (→ AFC-LoRA/MoE). A 30-line script that tells us which lever to bet on.

---

## Part 3 — Criticism round 3: latency & cost realism

**Attacked:** "Neutral latency" claims are too loose. MoE routing and LoRA adapters add *real kernel overhead*; adaptive α adds a scan dependency that may hurt fp16 (the exact thing the NaN guard protects). On T4 this is minor; on Lite's CPU target it is fatal. The motive is modest hardware — overhead is a first-class concern.

**Survives:** A (adaptive α) and B (LoRA) — but with explicit budgets.

**Changed / advanced:**
- Hard acceptance budget: each lever must keep **tok/s within 10%** of baseline and **peak mem within +15%**; else rejected regardless of loss gain.
- **Fusion requirements:** all band α predictions = one batched GEMM; LoRA = single fused `A·B·x` matmul; experts routed at **chunk level** (16–32 tokens) for GEMM-friendly shapes, not per-token scalar routing.

**New idea — H4. Band-drop routing (compute on demand, without MoE overhead).** Instead of experts, a binary per-token "skip this band" gate (LayerSkip-style). Skipping = zero FLOPs, zero extra kernels, single boolean. This realizes difficulty-conditional compute with *less* overhead than MoE. More latency-honest for the Lite/CV audience.

**New idea — H5. Two-track realization.** Same principle ("spend compute where error is") implemented twice:
- **Brain track** (GPU/T4): *dynamic* — adaptive α, LoRA, band-drop, MoE.
- **Lite track** (CPU/i5-337U): *static* — fused ops, fixed α, INT8, no routing.
This prevents the Brain complexity from cannibalizing the Lite story (the motive's literal target).

---

## Part 4 — Criticism round 4: motive & narrative alignment

**Attacked:** The theory risks drifting from the motive. "Memory-matched scaling" (v0 E) can read as moving goalposts; an everything-and-the-kitchen-sink roadmap weakens the "one clean idea" narrative; the project sells *elegance* (fractal resonance), and bolting on LoRA/MoE could make it look like imitation.

**Survives:** E is legitimate — but must be framed as a **complementary axis**, never a dodge; F is the heart of the story (attention as resource allocation) and must stay central.

**Changed / advanced:**
- Reframe all levers as one coherent mechanism: **"the model allocates computation by its own prediction error."** Adaptive α allocates *memory*, band-drop allocates *compute*, MoE allocates *capacity*, neuromodulator allocates *context*. One sentence, five instantiations — this preserves the brain metaphor and the thesis.
- Every Brain-track idea must have a documented "what does Lite get?" answer, or it is not project-aligned.

**New idea — H6. Energy-budget transfer (unification).** The per-token difficulty signal (entropy from Part 0 F) *is* the input the existing `DynamicEnergyBudgetAllocator` already takes. One mechanism, two outputs: gates band-drop/experts (compute) and modulates FiLM/resonance (context). This merges v0 A + C + F into a single module instead of three scattered features.

**New idea — H7. Static sparsity portfolio for Lite.** Give Lite a *frozen* difficulty map learned once on GPU ("train the router on T4, deploy the router as a table") — dynamic-style efficiency on CPU with zero runtime routing. Keeps the "one mechanism" story true across both tracks.

---

## Part 5 — Criticism round 5: evidence, risk & falsifiability

**Attacked:** Expected gains are guesses. Risks are unranked: (1) adaptive α destabilizes fp16 scans → NaN; (2) LoRA/MoE balloon code complexity; (3) the fairness attack — EMA/multi-epoch change benchmark semantics; (4) **the existential risk: the gap may be fundamental** (Mamba's fully-learned continuous dynamics genuinely out-express band recurrence), in which case no placement fix closes it.

**Survives:** The plan — but now as a *falsifiable, phase-gated* program with pre-registered outcomes.

**Changed / advanced:**
- Every lever becomes a **go/no-go experiment** with a threshold (table below).
- Each gate checks loss gain **and** tok/s, mem, and NaN count — no single-metric victories.
- **Pre-committed negative result:** if after Phase 1–3 the gap remains > 0.3, we *stop chasing* and publish the honest verdict: NFRA owns the memory/latency Pareto point; the comparison becomes memory-matched, not param-matched. A negative result here is still a publishable, credible result — it is the "apples-to-apples honesty" the project was built on.

**New ideas:**

**H8. Band-count ablation (the cheap truth-probe).** Before building adaptive α, run n_bands ∈ {1, 2, 4, 8, 16} at fixed params. If 1 band ≈ 16 bands, multi-band recurrence itself is the wrong hypothesis and we pivot to a *single fully-learned selective recurrence* (closer to Mamba). This is the cheapest way to discover which architectural hypothesis is true.

**H9. Zero-cost structural wins first.** Audit current design for free wins before any new feature: tie input embedding + LM head at char level (reallocates params to blocks), remove dead compute (scanner/mixture modules unused by Brain), verify dropout placement doesn't waste the shared-block capacity. These are the true "loss ↓ at zero latency cost" items — they belong at the very top of the roadmap.

**H10. Stability hardening for adaptive dynamics.** Clamp predicted α to [0.90, 0.999], apply fp32 accumulation on the scan path, and run a stress suite (fp16, long seq) in every gate. Protects the existing "no NaN" guarantee that the memory advantage depends on.

---

## Part 6 — Final integrated roadmap (phased, gated)

| Phase | Contents | Go/no-go gate |
|-------|----------|---------------|
| **0. Free wins** | H9 (tie head, remove dead compute), H3 recall probe, H8 band ablation, H1 fairness protocol | Probe results decide Phase 2 direction |
| **1. Convergence parity** | D (EMA, multi-epoch, schedule), H2 gradient shaping | loss gain ≥ 0.05 at ≤5% latency, ≥3 seeds |
| **2. Adaptive memory** | A (AFC-α), H10 stability | gain ≥ 0.05 at ≤5% latency, 0 NaN |
| **3. Adaptive capacity** | B (AFC-LoRA), then C (MoE, wire existing code) or H4 (band-drop) | gain ≥ 0.10 at ≤10% latency, mem ≤ +15% |
| **4. Unified energy** | H6 energy-budget transfer, F difficulty-conditional compute | gain ≥ 0.05, or speed gain ≥ 10% |
| **5. Memory-matched story** | E + H5 two-track (Brain dynamic / Lite static H7) | gap ≤ 0.3 after Ph 1–4, or honest pivot |

Each phase = A/B in `compare.py`/`arena.py`; keep only Pareto winners.

---

## Part 7 — Anti-goals (what we will NOT do — motive protection)

1. **No quadratic attention / dense global attention.** Breaks O(L) latency and the memory story.
2. **No per-token scalar routing on CPU.** Kills Lite; chunk-level routing only (H4/H5).
3. **No silent benchmark redefinition.** Every reporting change (iso-wall-clock, memory-matched) is announced as a separate axis.
4. **No NaN trade.** Any lever that weakens the fp16 scan safety net without H10 hardening is rejected.
5. **No feature accretion without ablation.** Nothing enters Brain unless it wins its gate; nothing enters Lite unless it has a static/deployable form (H5/H7).
6. **No chasing a gap we cannot close.** The plan pre-commits to owning the memory/latency axis if Phase 1–4 fail to close within 0.3.

---

## Part 8 — Metric dashboard (every experiment reports these)

- Eval loss & ppl (≥3 seeds, mean ± std), tok/s, ms/token (prefill + AR gen), peak train & infer memory GB, NaN/stability events, sample-efficiency AUC, scaling slope, extrap_delta.
- Plus the fairness record: HP budget per family, token budget, seeds, wall-clock — so every number is reproducible and no family is handicapped.

---

## Part 9 — Brain-inspired ideas (neuroscience → NFRA)

Every idea below maps a *specific* brain mechanism to a concrete change. All are checked against the motive (low latency/memory, modest hardware) and the anti-goals in Part 7.

### Tier 1 — cheap, high-impact, fully aligned

1. **Dopamine reward-prediction-error → surprise-weighted gradients.**
   Brain: learning driven by the gap between predicted and actual outcome (RPE).
   NFRA: weight each token's gradient by |confidence − outcome| ("RPE").
   Impact: capacity spent on hard/ambiguous tokens; loss ↓ with zero added params.
   Practical: one scalar/token, training-only, no inference cost. **High.**

2. **Cortical lateral inhibition → k-WTA sparsity in fractal MLP.**
   Brain: neighboring neurons suppress each other; sparsity is input-dependent, not fixed.
   NFRA: replace uniform fractal masks with data-driven top-k activation per feature group.
   Impact: capacity lands where input needs it — most of MoE's benefit, none of its routing overhead.
   Practical: one k-select op, O(L), hardware-friendly. **High.**

3. **Acetylcholine novelty signal → band-drop gating.**
   Brain: ACh marks novelty/uncertainty and steers top-down vs bottom-up attention.
   NFRA: predicted novelty/entropy gates whole bands off for easy tokens (biologically grounded H4).
   Impact: compute-on-demand — same speed curve, better loss on hard tokens.
   Practical: one novelty head + boolean gate. **High.**

4. **Norepinephrine arousal → global gain modulation.**
   Brain: NE sets global responsiveness — a single gain over the whole system.
   NFRA: predicted arousal scalar (from recent error rate) scales activations via the existing global-brain-state modulation.
   Impact: robustness to distribution shift and bursty text; stabilized dynamics.
   Practical: one scalar/segment, reuses existing modulation layers. **Very high.**

5. **NREM sharp-wave replay → offline replay of hard sequences.**
   Brain: during sleep the hippocampus replays salient sequences to the cortex for consolidation.
   NFRA: between epochs, re-train on the highest-loss sequences (prioritized replay).
   Impact: sample efficiency on exactly the data that hurts the loss.
   Practical: training-only, reuses forward pass. **Very high.**

### Tier 2 — medium effort, strong alignment

6. **Hippocampal episodic memory → bounded associative context cache.**
   Brain: hippocampus stores episodes separately from slow cortical weights (Complementary Learning Systems).
   NFRA: small, fast-learning key-value memory of recent context the model attends to (compressed episodic buffer) layered over slow shared weights.
   Impact: long-range recall — the exact axis where Mamba beats NFRA.
   Practical: bounded KV buffer + one attention op; needs a memory-budget guard. **Medium.**

7. **Cerebellar forward model → fast low-rank corrector.**
   Brain: cerebellum learns to predict and cancel systematic errors quickly.
   NFRA: low-rank module updated rapidly, correcting the block's systematic output error (learned residual prediction).
   Impact: faster convergence (synergy with H2); fewer steps to same loss.
   Practical: one LoRA-like module + fast update rule. **Medium.**

8. **Mirror-simulation / world model → latent future prediction loss.**
   Brain: predicts its own future internal states, not only external outputs.
   NFRA: auxiliary loss predicting the *next hidden state* in addition to next token.
   Impact: forces a better internal model → better token loss at zero inference cost.
   Practical: extra head + MSE on internals, training-only. **Medium-high.**

9. **Grid cells → multi-scale positional encoding.**
   Brain: hippocampal grid cells encode space at nested scales (hexagonal hierarchies).
   NFRA: give each band a matching-resolution positional embedding — slow bands see coarse positions, fast bands fine ones.
   Impact: sharper position info per temporal scale → better band mixing.
   Practical: additive embeddings, trivial to add. **High.**

10. **DMN task/rest alternation → interleaved consolidation phases.**
    Brain: alternates task mode with internally-focused rest.
    NFRA: schedule training as "predict next token" phases alternating with "predict your own hidden representations" phases.
    Impact: better representations, less overfit to surface statistics.
    Practical: a training-schedule switch. **Medium.**

### Tier 3 — bold, experimental

11. **Burst coding → skip + burst compute.**
    Brain: neurons stay silent, then burst when input is surprising (event-driven computation).
    NFRA: confidence-gated early-exit; on high surprise allow an extra refinement pass (burst) over the block.
    Impact: both speed up (skips) *and* quality up (bursts) — the rare dual win.
    Practical: **Low-medium** — exit/burst policies are hard to stabilize.

12. **Neural synchrony → per-band oscillatory gating.**
    Brain: gamma (fast) and theta (slow) oscillations bind information within their timescale.
    NFRA: multiply each band's gate by a fixed-frequency sinusoidal phase matching band scale.
    Impact: possible better temporal binding/mixing; speculative.
    Practical: **Low** — cheap to add but may not help; needs an H8-style ablation first.

13. **Belief-state inference → latent generative state.**
    Brain: perception is inference of hidden causes, not just prediction.
    NFRA: make the recurrence's hidden state an explicit latent belief (normalized/regularized, optionally variational).
    Impact: better sample efficiency, cleaner long-range state.
    Practical: **Low-medium** — adds training complexity; keep deterministic first.

### Cross-cutting — efficiency (the brain's real lesson)

14. **Cortex runs at ~4.7 bits/weight → QAT + 4-bit NFRA.**
    NFRA: quantization-aware training to 4-bit weights → halve memory per param → ~2× bigger `hidden_size` in the same envelope (realizes E / memory-matched scaling).
    Impact: the most motive-aligned idea of all — the brain's efficiency *is* its memory economy.
    Practical: PyTorch QAT infra exists; benchmark-gated. **High.**

**Suggested order:** Tier 1 (1→5) first — all cheap, training/inference-safe, directly attack the loss gap or speed curve → Tier 2 (9, 8, 6, 7) — attack the Mamba long-range/capacity gap → Tier 3 (11, 12, 13) only if the H3/H8 probes justify.
