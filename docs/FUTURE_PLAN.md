# NFRA 2.0 — Future Plan (Theory, Critique, Roadmap)

**Status:** working plan — updated through 5 criticism rounds
**Motive anchor:** *quality AI on modest hardware* — every idea must keep **latency, peak memory, and training speed low**, and keep the benchmark **credible**.

**Implementation log (dates newest-first):**
- **2026-07-31 — kernels + Phase 0/H8/H3 tooling.** Custom fused selective-scan Triton kernel `nfra.kernels.scan` (one-pass forward, exact closed-form backward, torch fallback; `NFRA_SCAN_KERNEL` 0/1/2). `BrainMixer` now honors an explicit band count (`NFRA_BANDS`, H8 ablation: 2/4/8/16; 16 keeps the legacy `[8,4,2,1]+router` hierarchy) — `n_bands` was previously dead in Brain mode. H3 memory-horizon probe (`nfra.benchmark.recall_probe`) ships to decide memory-vs-capacity.
- **2026-07 — NFRA 3.2 feature toggles.** EMA, surprise-weighted (dopamine-RPE) loss, k-WTA lateral inhibition (`NFRA_EMA`/`NFRA_SURPRISE`/`NFRA_KWTA`), each A/B-able and applied to all families except k-WTA (NFRA-only). Script `compare_versions` isolates 3.1-parity vs 3.2-full on identical seed/init/data.

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

**New idea — H3. Empirical memory-horizon probe.** Before building adaptive α, *measure* NFRA's effective recall with synthetic probes (associative recall, copy, retrieval at distance k). This decides the diagnosis: if recall collapses at distance d, the gap is **memory** (→ AFC-α); if recall is fine but loss lags, the gap is **capacity** (→ AFC-LoRA/MoE). A 30-line script that tells us which lever to bet on. **RESULT: see Part 10 — the original probe was unlearnable by ANY causal model (keys hidden) and NFRA leaked future context; both fixed, probe redesigned with observable keys, and re-run is pending on Kaggle.**

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

---

## Part 10 — Experiment log (recorded results)

### 2026-07-31 — H3 recall probe, concurrent, dim 224 / unique 4 / 600 steps / V=16 / floor ln(16)=2.773

Config: `NFRA_RECALL_KS=4,16,64,128 NFRA_RECALL_DIM=224 NFRA_RECALL_CONCURRENT=1`; 8 trainings (3 NFRA + 3 Mamba, ks 4/16/64/128), 2228.4s, 4411 tok/s aggregate.

| model | k | train first→last | span_ce | span_acc | pad_ce |
|-------|---|------------------|---------|----------|--------|
| nfra  | 4   | 3.2200 → 2.7706 | 2.7728 | 0.0644 | 2.7713 |
| nfra  | 16  | 3.1852 → 2.7673 | 2.7729 | 0.0660 | 2.7714 |
| nfra  | 64  | 3.1741 → 2.7669 | 2.7726 | 0.0643 | 2.7697 |
| nfra  | 128 | 3.1743 → 2.7661 | 2.7721 | 0.0675 | 2.7687 |
| mamba | 4   | 2.8367 → 2.0812 | 3.0258 | 0.0637 | 3.0240 |
| mamba | 16  | 2.8391 → 2.0737 | 3.0967 | 0.0627 | 3.0966 |
| mamba | 64  | 2.8396 → 2.0789 | 3.1041 | 0.0619 | 3.1029 |
| mamba | 128 | 2.8411 → 2.2537 | 3.0396 | 0.0645 | 3.0375 |

**Reading:**
- **NFRA is flat at the floor for every k** (span_ce ≈ ln16, acc ≈ 1/16 chance). k=4 `3.2200 → 2.7706` exactly reproduces the earlier sequential dim-224/600 run → also validates the concurrent harness (build/init parity + results).
- **Mamba trains far below the floor (2.08–2.25) but evals ABOVE it (3.03–3.10, acc chance).** Eval CE > ln(16) = confidently-wrong → the signature of train-set memorization, not rule-learning. So even the model with working memory machinery does not *generalize* the `value=key+1` rule in 600 steps.
- **Decision logic refined (NOT yet resolved):** the flat-high reading is ambiguous because NFRA fails at the base case (k=4) and k=1 is untested. Two confounds: (a) NFRA at ~5M is in a known no-learning regime (dim-192/600 sits at random loss on wikitext too), vs (b) NFRA's recurrence specifically cannot form delay lines. Mamba's overfit-no-generalize also proves this probe is a harsh *generalization* test, so "span at floor" ≠ "no capacity for the rule."
- **Next gate (env-only, no code change):** `NFRA_RECALL_KS=1,2,4 NFRA_RECALL_DIM=512 NFRA_RECALL_CONCURRENT=1` — run at the ~20M size where NFRA demonstrably learns (dim-512 reached eval 2.13 on wikitext). k=1 is a per-token `+1` map (zero memory needed): if NFRA-512 nails k=1 but flattens at k=2/4 → **memory-formability** (→ AFC-α / memory levers); if it fails even k=1 → **scale / learning-dynamics** (→ Phase 1 convergence-parity levers, not capacity).

### 2026-07-31 (night) — static audit: found a real defect, root cause NOT yet confirmed

- **Found + fixed a real defect** (`eda3919+`): `NFRA_Brain_Block` used `residual = prediction` with `n = ln1(x - predictor(x))` — a SELF-prediction (predict x from x). Identity is a degenerate attractor: if `predictor -> I`, then `error -> 0`, `ln1(0) = 0` with ZERO gradient, so the recurrence/attention inputs die and the block reduces to a per-token pass-through with no memory — exactly the H3 symptom. Fix: `residual = x`, `n = ln1(x)`; the prediction error now only gates gist-vs-deep (free-energy intent preserved).
- **Honest caveat:** this is NOT confirmed as the H3 root cause. A tiny CPU test (`test_brain_learns_short_recall`, k=2/dim 64/40 steps) passes under BOTH old and new code — the H3 failure is regime-specific (dim 224 / 600 steps / k≥4) and only reproduces on GPU. The fix is kept because it is strictly more expressive (removes the attractor) and cannot hurt capacity.
- **Decisive next run (one Kaggle script, `python -m nfra.benchmark.recall_diag`, ~3 variants concurrent in the exact failing regime):** `fix k=4` vs `k1` vs `noshare` (12 distinct blocks). If `fix k=4` learns → fix confirmed. If `fix` floors but `noshare` learns → depth weight-sharing is the culprit (→ different fix). If `k1` also floors → capacity/optimization at dim 224 (→ dim-512 probe as planned). Tracks `pred|W-I|` every 50 steps to watch the collapse develop.

### 2026-07-31 (late night) — three brain-inspired mechanisms + overnight global arena (`17b0e50`)

- **Novel mechanisms (off by default, near-zero cost, A/B-able via env `NFRA_LOCALROUTE` / `NFRA_DIVNORM` / `NFRA_ASTRO`):**
  - `local_route` — cortical microcircuit routing: BrainMLP router reads a causal sliding-window pool (`local_win=64`) of the token's LOCAL context, blended with the sequence-global pool.
  - `div_norm` — divisive (contrast) normalization of MLP hidden units by pooled intensity (`hidden / (1 + pooled_act)`).
  - `astro` — astrocytic timescale homeostat: a slow per-sequence signal (`astro_proj Linear(dim,1)`) scales ALL band decays (`alpha *= 1 + 0.2·tanh(astro_proj(pool))`), shifting the global memory horizon.
- **Overnight script:** `python -m nfra.benchmark.global_arena` — 4 resumable phases (core head-to-head+scaling / 10-variant ablate / recall diag / perf battery+extrapolation), env-driven, JSON + Markdown report with verdict logic. This replaces the manual Kaggle cell sequence.
- **Verification:** 3 new CPU-safe tests (local-pool vs manual sliding window; feature toggles fwd/bwd; toggles off by default); `tests/test_model.py` 11/11. Static import/signature audit of global_arena against arena/compare; config+forward smoke at dim 224 passed locally. NO local training runs.
- **Pending:** run `recall_diag` (quick, decisive for the H3 root-cause question) and/or `global_arena` on Kaggle T4 overnight; the ablate phase then answers which levers actually pay.

### 2026-07-31 (post-dim512-run) — TWO real defects found: probe unlearnable by design + future leak (both fixed)

**1. The recall probe could NOT be learned by ANY causal model (design bug).** The stream contained only values `(keys[t-k]+1)%V`; the `keys[t-k]` needed to predict `y[t]` never appear in the observed prefix, so H(y|prefix)=ln(16) at every position. No causal model can drop below the floor — Mamba flooring proves it (a strong SSM, so not capacity). Every past "flat at floor" reading is void, including the dim-224 note that Mamba "trains 2.08–2.25" (that was memorization of the *padding* rule, not memory).

**2. NFRA Brain leaked FUTURE context into every position** (whole-sequence global pools), tainting every autoregressive number:
- `NeuroModulator`: `x.mean(dim=1)` hormones broadcast to all tokens.
- `GlobalBrainState`: whole-sequence pool → GRU; cross-pass carry used `state[:, -1]`, whose last-position state had seen the full sequence, then broadcast to every position.
- `BrainMixer.astro`: `x.mean(dim=1)` scaled all band decays.
- `BrainMLP` router pool: `x.mean(dim=1)`.
- Isolated empirically (dim 224 / 12-depth): changing token@200 moved logits at positions 1–180 by 0.0465 (causal ⇒ exactly 0.0); changing a token moves its own-position logits 44.1. Explains the old anomaly NFRA k=2/dim-64 train 1.99 < floor 2.77 (leak-based overfit) yet eval 3.24 > floor (future-peek never generalizes).

**Fixes (verified, 13/13 tests pass):**
- All global pools → causal per-token prefix means (`cumsum/cnt`): NeuroModulator (prefix mean + prefix-variance novelty), GlobalBrainState (vectorized causal prefix state, per-position cross-pass prior — GRU removed), astro (prefix mean [B,S,1] applied as [B,1,S,1]), BrainMLP router pool (prefix).
- Probe redesigned: the stream IS the keys; `y[t] = (key[t-k]+1)%V` for t≥k, random padding otherwise — a genuinely learnable memory test (k=1 is the memory-free baseline that MUST learn).
- New regression tests: `test_brain_no_future_leak` (perturb a future token ⇒ earlier logits unchanged <1e-5), `test_recall_dataset_keys_observable` (target = value of the key k back; key visible in the prefix), `test_brain_learns_short_recall` now trains on the corrected dataset.
- Decision (user): fix both now, then overnight `global_arena` run on Kaggle T4.

**Next:** commit+push → Kaggle: `pip install` → restart kernel → `python -m nfra.benchmark.recall_diag` (quick decisive) → `python -m nfra.benchmark.global_arena`; report back `global_arena_report.md` + JSON.

### 2026-07-31 (post-fix) — structural changes + four new levers (all off-by-default)

**Structural (shipped, 15/15 tests pass):**
- **`prefix_pool` / `prefix_var` helpers** (`neuro.py`): one causal `cumsum` reduction shared by NeuroModulator / GlobalBrainState / astro / BrainMLP — removes 4 duplicated reductions, keeps the causality rule in one place (the leak fix's contract).
- **AFC-LoRA per-pass (`lora_rank`, `NFRA_LORA_RANK`)** — the v0 plan's Space-axis lever, now real: one low-rank adapter per depth pass on the shared block (`y = x + (x@A)@B`, `B=0` ⇒ exact identity at init). Direct structural fix for the depth-dilution critique (6 unique blocks vs Mamba's 30 real layers) at ~0.9% params (r=8, 5M).

**New creative levers (one axis each, identity-init, near-zero cost, in `global_arena` ablate):**
- **`theta` (`NFRA_THETA`)** — Time axis. Per-band learnable theta rhythm modulates the decay *pre-scan* (`alpha *= 1 + amp·sin(2πft/S + φ)`): memory windows rhythmically open/close (hippocampal theta-gamma coupling). Causal (t-only); `amp=0` at init.
- **`ach_retain` (`NFRA_ACH_RETAIN`)** — Time axis. One-line polarity toggle: high ACh → *hold* memory (`dt / (1+0.5·ACh)`) vs legacy high ACh → *forget* (`dt · (1+0.5·ACh)`). Tests the encoding hypothesis; zero params.
- **`gain_nov` (`NFRA_GAIN_NOV`)** — Gain axis. Causal prefix-variance scales the recurrence *write* (`value *= 1 + w·Var(x[0..t])`, `w` learnable, init 0): high-contrast tokens are written in harder (surprise-modulated plasticity).

**Verified:** 15/15 tests (new: levers-identity-init — enabling theta/gain_nov/lora is bit-identical to baseline at init; no-future-leak-with-all-levers — still exactly 0.0; levers forward/backward). Levers confirmed *live* (perturbing identity params moves logits). All-levers model still learns the corrected recall task (CE 1.79 < floor 2.77). Docs: `BRAIN_LEVERS.md` now documents all seven levers; ablate table has 14 variants (added `nfra_theta`, `nfra_achretain`, `nfra_gainnov`, `nfra_lora8`). No local training runs (CPU-only box).

**Next:** overnight `global_arena` on Kaggle decides which levers pay; the corrected `recall_diag` decides the H3 root-cause question.
