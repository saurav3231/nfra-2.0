# NFRA Brain Levers — Adaptive Computation on the Space / Gain / Time Axis

**Design, examples, use cases, advantages, and what success would mean — for the three near-zero-cost brain-inspired mechanisms added to NFRA Brain:**

| Lever | Env flag | `NFRAConfig` field | Brain component | Axis it adapts |
|-------|----------|--------------------|-----------------|----------------|
| **Local cortical routing** | `NFRA_LOCALROUTE` | `local_route=True` | `BrainMLP` | *Space* — where capacity is placed per token |
| **Divisive (contrast) normalization** | `NFRA_DIVNORM` | `div_norm=True` | `BrainMLP` | *Gain* — how strongly each unit responds |
| **Astrocytic timescale homeostat** | `NFRA_ASTRO` | `astro=True` | `BrainMixer` | *Time* — how long memory persists |

All three are **off by default**, individually toggleable, GPU-parallel friendly, and cost effectively **zero extra parameters** (one has 0, one has `dim+1` per block, one has 0). They are the *first* of a "small but powerful" program: can a network that **adapts its own computation** — the way a cortex does — close the quality gap to much larger models, at a fraction of the memory?

---

## 1. Why three new levers?

NFRA Brain already encodes six neuroscience principles (neuromodulation, cortical columns, thalamic gating, oscillatory phase, homeostatic energy, prediction error). Those are mostly *structural* — the shape of the network is fixed and the "adaptation" is baked into parameters.

These three levers add something qualitatively different: **the network changes its own computation on the fly, per token and per sequence, without being told to.** Fixed-shape networks waste capacity: a static MLP routes the same fraction of units into the same groups no matter what the input is; a static recurrence forgets at the same rate no matter what the sequence needs.

Biology does not work that way:

- A cortical column routes signals through *different* microcircuits depending on the **local** input neighborhood — not a single global summary.
- Cortical neurons do not fire at a fixed gain; their response is **normalized by the contrast of the surrounding activity** (divisive normalization is arguably *the* canonical cortical computation, Heeger 1992).
- Neurons do not hold memory with a fixed half-life; **glial cells** — the "support cells" biology long ignored — regulate synaptic strength on slow timescales, shifting the whole network's memory horizon to match the situation.

These levers are a minimal, honest translation of all three into GPU code, at near-zero cost, so they can be **A/B-tested on identical hardware and data** — no hand-waving.

---

## 2. The unifying idea: adaptive computation on three axes

Think of a language model as a computation that must place capacity (where), spend energy (how much), and retain information (how long). Every fixed architecture hard-codes these. These levers make each axis input-dependent:

```
      WHERE?              HOW MUCH?                 HOW LONG?
   local_route          div_norm                   astro
   ─────────────────    ─────────────────          ─────────────────
   routing context =    hidden /= (1 + <h²>)       alpha *= (1 + 0.2·tanh(pool))
   global pool + local  → each unit is rescaled    → one slow per-sequence
   sliding-window pool  → by its own local         → signal raises/lowers
   (window 64)          → contrast                 → every band's decay
   ─────────────────    ─────────────────          ─────────────────
   SPACE AXIS           GAIN AXIS                  TIME AXIS
```

Three properties they share (the design rules for every future lever):

1. **Context-dependent** — the behavior differs per token / per sequence, driven by the input itself, not by learned static masks.
2. **Near-zero cost** — no new big tensors, no new heavy ops (one cumsum, one mean-square, one `dim→1` linear).
3. **Composable & falsifiable** — each can be turned on alone, with others, or off; the `global_arena` ablate phase measures each in isolation and in combination.

---

## 3. Lever 1 — `local_route`: cortical microcircuit routing (the Space axis)

### 3.1 Neuroscience basis

The neocortex is not a bag of independent columns. A cortical column's input is shaped by *lateral connections* — nearby neurons telling each other what is happening **right here, right now** — long before the whole brain's summary comes back down. Local circuitry provides fast, input-specific context; the global state provides a stable, coarse prior. Routing decisions (which microcircuit handles this signal) are made by fusing both.

Most neural networks ignore this: a router (in MoE, or here in `BrainMLP`) reads a **sequence-global** pool — one number per dimension averaged over the entire context. That collapses "this token is at position 500 in a 5000-token document" and "this token is at position 10" into the same signal.

### 3.2 Design

`BrainMLP` already has a tiny router that reads a pooled summary of `x` and produces a soft group-weighting over the fractal groups. `local_route` changes **what the router reads**:

```
pooled = x.mean(dim=1)                    # sequence-global pool  (stable prior)
pooled = pooled + _local_pool(x)          # + causal sliding-window mean (local context)
routing = router(pooled)                  # per-token routing decisions
```

`_local_pool` is a causal mean over the last `local_win = 64` tokens:

```
window_sum[t] = sum(x[t-63 .. t])         # one cumsum pair
window_mean[t] = window_sum[t] / min(t+1, 64)   # corrected at sequence start
```

Why causal? The network predicts the next token from the past — the local context that can *inform* routing is the recent past only.

### 3.3 Implementation

```python
def _local_pool(self, x):
    B, S, D = x.shape
    w = min(self.local_win, S)
    cum = x.cumsum(1)                                       # [B, S, D]
    left = torch.cat([torch.zeros(B, w, D, device=x.device, dtype=x.dtype),
                      cum[:, :-w]], dim=1)                  # [B, S, D]
    wsum = cum - left                                       # sum over [t-w+1 .. t]
    cnt = torch.arange(1, S + 1, device=x.device).clamp(max=w).float()
    return wsum / cnt.view(1, S, 1)
```

And in `forward`:

```python
pooled = x.mean(dim=1, keepdim=True)
if self.local_route:
    pooled = pooled + self._local_pool(x)      # local + global blend
routing = self.router(pooled)
```

**Parameters added: 0.** **FLOPs added:** one cumsum pair ≈ `S·D` adds — trivial against the `S·D²` of the MLP itself.

### 3.4 Worked example

Take `dim = 224`, sequence length `S = 256`, hidden `D = 224`.

- **Global pool:** one vector `[224]` = mean over all 256 tokens → the same for every token. Token 7 and token 250 see the *identical* routing context.
- **Local pool (on):** token `t` now sees a `[224]` vector that is the mean of `x[t-63..t]` (and only what has been seen, at the start). Token 7 sees ~8 tokens; token 250 sees the last 64. Token 7 is routing on "the very recent local neighborhood", token 250 on "a settled local context" — plus both still carry the global prior.

Concrete consequence: inside a code block or a list of numbers, recent local structure is highly predictive (indent level, current list element). The router can now distinguish "I am deep inside local structure" from "I am reading a globally repetitive document" — and allocate fractal-group capacity accordingly, per token.

### 3.5 Advantages

- **Input-dependent capacity placement** without MoE's per-expert parameter count or dispatch cost — the router stays tiny (one 64-wide hidden layer).
- **Cheaper than adding context**: a bigger attention window is `O(S²)`; a longer local window here is `O(S·D)`.
- **Stable coarse prior preserved** — the global mean is still there, so the local signal *refines* rather than replaces; this avoids the well-known instability of pure-local routing in MoE literature.
- **Zero params, zero new states** — no KV cache growth, no per-expert bookkeeping.

### 3.6 Use cases

- **Long-document modeling** (books, transcripts, logs): global summary is constant for thousands of tokens; local routing keeps per-token sensitivity without quadratic cost.
- **Code & structured text** (JSON, YAML, indentation-sensitive formats): local structure is the dominant signal.
- **Streaming / online inference**: only the last `local_win` tokens matter for the routing update — friendly to incremental decoding.

### 3.7 If it succeeds

If `nfra_local` beats `nfra_baseline` (and especially if it also improves extrapolation to 2× context), it is evidence that **spatial/lateral adaptation is worth more than raw context length** at the same cost. That generalizes far beyond NFRA: any transformer/SSM with a router (MoE, mixture-of-attention) could adopt local+global routing. It is the first concrete step toward "conditional computing for free".

---

## 4. Lever 2 — `div_norm`: divisive (contrast) normalization (the Gain axis)

### 4.1 Neuroscience basis

Divisive normalization (Heeger 1992, Carandini & Heeger 2012) is the best-supported canonical computation in cortex: a neuron's response is **divided by a measure of the pooled activity of its neighbors**. It explains contrast adaptation, surround suppression, and why neural responses are remarkably stable across input intensities. Every neuron's gain is set by the *context* it is embedded in — bright surroundings turn down a neuron's response, not just the neuron itself.

LLM MLPs have no such gain control. A hidden unit's magnitude is whatever the projection produced, and nothing rescales the layer to the current input's "intensity".

### 4.2 Design

After the gated hidden activation is computed (per token), divide the whole hidden vector by `1 +` its own pooled intensity:

```
hidden = silu(gate_proj(x)) * up_proj(x)
pooled_act = mean over dim of hidden²          # per-token scalar "contrast"
hidden = hidden / (1 + pooled_act)
```

The `+1` keeps the operation smooth and prevents division blow-up when `pooled_act → 0`. The result: tokens whose MLP activation is globally large (high-contrast, "loud") are **uniformly gain-reduced**; tokens whose activation is small (quiet, low-contrast) keep near-full gain. Relative differences between units are preserved — it is a gain normalization, not a sparsification.

### 4.3 Implementation

```python
if self.div_norm:
    pooled_act = hidden.square().mean(dim=-1, keepdim=True)
    hidden = hidden / (1.0 + pooled_act)
```

**Parameters added: 0.** **FLOPs added:** one square + one mean per token ≈ `S·D`.

### 4.4 Worked example

`hidden_dim = 896` (dim 224 × 4). Suppose at token A the 896 activations average squared value 4.0 (a loud token), at token B it is 0.5 (a quiet token).

- **Without `div_norm`:** token A's units feed downstream at 4× the energy of B — even if B's *pattern* of relative activations is the informative one. The network must learn to tolerate wild gain variance across tokens.
- **With `div_norm`:** token A is divided by 5.0 → units scaled ×0.2; token B is divided by 1.5 → ×0.67. Both are brought into a comparable dynamic range **while keeping their internal contrast** (a 2× unit difference in A stays 2×).

This is precisely the "stable responses across input intensity" property of cortex — and for a model it means the downstream `down_proj` sees well-conditioned inputs regardless of how loud each token's MLP became. Conditioning, not sparsity, is the point.

### 4.5 Advantages

- **Better-conditioned hidden activations** → typically faster, more stable convergence (smaller effective Hessian variance across tokens).
- **Theoretically principled** — it is not a hack; it is the same normalization that makes cortical circuits robust, and it is a form of adaptive gain control.
- **No parameters, no new hyperparameters** (the `+1` and the full-vector pooling are fixed and simple).
- **Complements, not competes with, LayerNorm**: LayerNorm normalizes the *input to* a layer; `div_norm` normalizes the *activation of* the layer's internal computation.

### 4.6 Use cases

- **Any high-dynamic-range input** — text with very different per-token "loudness" (mixed languages, code comments vs code, interleaved tables).
- **Long training runs** where stable per-token gradients matter more than raw speed.
- **A cheap "free" regularizer** in the same spirit as RMSNorm but on the internal gain rather than the residual stream.

### 4.7 If it succeeds

If `nfra_divnorm` improves eval loss and/or sample-efficiency (AUC) with zero params, it is a clean, generalizable result: **a canonical cortical computation transfers to language-model MLPs.** Because it has no parameters and no moving parts, it is trivially portable to GPT-style, Mamba-style, and MoE models — a small idea with broad reach. It also pairs naturally with future *contrastive* (two-population excitation/inhibition) mechanisms.

---

## 5. Lever 3 — `astro`: astrocytic timescale homeostat (the Time axis)

### 5.1 Neuroscience basis

For decades, memory was assumed to live only in neurons. **Astrocytes** — star-shaped glial cells that outnumber neurons — wrap synapses and regulate their strength on *seconds* timescales via calcium waves and released gliotransmitters. They do not fire action potentials; they are the *slow modulator* that decides how sticky synapses are right now. The brain's memory horizon is not fixed — it is continuously set by glial state.

`BrainMixer` has fast, per-token, input-dependent decay (`dt_proj`, Mamba-style selectivity). What it lacked is the *slow, global* knob — the glia. `astro` adds exactly that: one tiny linear that reads the whole sequence and shifts **every band's decay** together.

### 5.2 Design

```
astro = tanh(astro_proj(mean over tokens of x))     # one slow scalar per sequence
alpha = alpha * (1 + 0.2 * astro)                   # rescale ALL band decays
```

where `alpha` is the per-head, per-dim decay used by the recurrence scan `h_t = alpha_t · h_{t-1} + gate_t · value_t`, and the scan then clamps `alpha` to a numerically-safe `[0.75, 0.9995]`.

The `tanh` bounds the shift to ±20% of the decay, so the homeostat can push the network toward **remembering more** (astro → +1, decays rise toward the 0.9995 clamp, horizons lengthen) or **forgetting faster** (astro → −1, decays drop toward 0.75). It is orthogonal to per-token `dt` selectivity: `dt` decides *per token whether to keep this exact token's state*; `astro` decides *per sequence what the whole network's default horizon is*.

### 5.3 Implementation

```python
self.astro_proj = nn.Linear(dim, 1, bias=False) if astro else None
...
if self.astro_proj is not None:
    astro = torch.tanh(self.astro_proj(x.mean(dim=1, keepdim=True)))
    alpha = alpha * (1.0 + 0.2 * astro.view(B, 1, 1, 1))
```

**Parameters added: `dim + 1` per block** (here 225) — 0.004% of a 5M model. **FLOPs added:** one `dim→1` dot per sequence.

### 5.4 Worked example

Effective memory horizon ≈ `1 / (1 − alpha)` steps. Content heads start around `alpha ≈ 0.90–0.995` → horizons ≈ 10–200 tokens.

- **astro → +1 (0.2 boost):** a head at 0.90 → 1.08 → clamped; a head at 0.98 → 1.176 → clamped to 0.9995 → horizon ~2000. The whole network is told "this sequence needs long memory."
- **astro → −1 (0.2 cut):** a head at 0.98 → 0.784 → horizon ~5. The network is told "this sequence is high-turnover, keep little."

Because the same signal touches every band, one sequence can demand "long-horizon mode" (e.g., a legal contract where a clause matters 500 tokens later) and the next can demand "short-horizon mode" (e.g., a chat whose relevant state is the last 20 tokens) — without the per-token mechanism having to learn it.

### 5.5 Advantages

- **Global, cheap memory-horizon adaptation** — the first mechanism in NFRA that modulates *the whole recurrence's* timescale, not a single head or token.
- **Truly novel lineage** — glial/astrocytic inspiration is essentially absent from the LLM literature; success here is a genuinely new mechanism class (slow non-neuronal modulation), not a re-derivation of an existing trick.
- **Orthogonal to all existing knobs** — composes with ACh decay, `dt` selectivity, k-WTA, etc., because it touches a different part of the computation.
- **Near-free** — one 225-param linear per block.

### 5.6 Use cases

- **Heterogeneous corpora** — mixed short-form (chat, tweets) and long-form (documents, code files): the homeostat can switch horizon per document.
- **Long-context generalization** — if the horizon adapts to sequence length, extrapolation to longer sequences should degrade more gracefully.
- **Memory-pressure control** — on memory-constrained devices, `astro → −1` is the network *choosing* to use less state when the task does not need it.

### 5.7 If it succeeds

If `nfra_astro` improves long-context extrapolation or sample efficiency, it establishes **slow glial-scale modulation as a load-bearing component** of sequence models. That is a publishable, defensible novelty: "an LLM with a homeostat." It also opens a design space — astro could later gate attention windows, KV retention, or optimizer state, not just recurrence decay.

---

## 6. How the levers compose

The three levers touch three different parts of `NFRA_Brain_Block`:

```
NFRA_Brain_Block
├── LN → ThalamicGate + BrainMixer  ── astro lives here (decay/horizon)
└── LN → BrainMLP  ── local_route (routing context) + div_norm (gain) live here
```

Because they act on disjoint axes, they can be combined without interacting adversarially — and the `nfra_all` variant in the ablate suite tests exactly that. Expected synergies:

- **local_route + div_norm:** routing decides *which groups get capacity*; normalization decides *how loud they are*. Together: "put capacity in the right place, at the right gain."
- **astro + local_route:** the network simultaneously chooses *how long to remember* (astro) and *where to focus locally* (routing) — long-memory mode for a contract, then local attention for its numbered clauses.
- **all three + k-WTA / ema / surprise:** the ablate suite measures whether the brain-inspired stack beats each lever alone, and whether it beats plain NFRA.

---

## 7. Cost accounting (the "small" promise)

At a 5M model (dim 224, 12 depth passes), measured bookkeeping:

| Lever | Extra params | Extra FLOPs/token | Extra memory |
|-------|-------------|--------------------|--------------|
| `local_route` | 0 | ~`S·D` (one cumsum) | one `[B,S,D]` temp |
| `div_norm` | 0 | ~`S·D` (square+mean) | one scalar |
| `astro` | 225/block | ~`D` (one dot) | one scalar |
| **Total** | **~2,700** (0.05%) | **~3× `S·D`** (≪ the `S·D²` of any MLP layer) | negligible |

The design rule that keeps this true: **no new sequence states, no new attention, no new per-head tensors.** Everything is a reduction over existing tensors or a `dim→1` projection. This is what makes them honest "free lever" candidates rather than "new capacity that happens to help."

---

## 8. How to enable and test

### In code

```python
from nfra import NFRAConfig, NFRAForCausalLM
cfg = NFRAConfig(
    mode="brain",
    local_route=True,   # Space  axis
    div_norm=True,      # Gain   axis
    astro=True,         # Time   axis
)
model = NFRAForCausalLM(cfg)
```

### Via environment (benchmarking)

```bash
NFRA_LOCALROUTE=1 NFRA_DIVNORM=1 NFRA_ASTRO=1 python -m nfra.benchmark.compare
```

Each flag is independent; leave any unset to keep that lever off.

### Correctness guards (already in the test suite)

- `test_local_pool_matches_sliding_window` — `_local_pool` equals a naive manual window, including sequence-start (fewer than `local_win` tokens seen) and window edges.
- `test_brain_feature_toggles_forward_backward` — each lever changes the forward output and still trains (backward runs, gradients finite).
- `test_brain_feature_toggles_off_by_default` — default `NFRAConfig` keeps all three off, so existing benchmarks/results are bit-identical.

---

## 9. The experiments that will decide

The **`ablate` phase of `global_arena`** trains 10 variants on identical data/optimizer/schedule (param-matched at the primary size):

| Variant | Levers | Asks |
|---------|--------|------|
| `nfra_baseline` | none | reference |
| `nfra_local` | `local_route` | does spatial adaptation help alone? |
| `nfra_divnorm` | `div_norm` | does gain normalization help alone? |
| `nfra_astro` | `astro` | does the homeostat help alone? |
| `nfra_kwta` | `k_wta` | existing sparsity lever, for context |
| `nfra_ema` / `nfra_surprise` | EMA / surprise loss | existing training levers, for context |
| `nfra_all` | all levers + ema + surprise | does the full brain stack help most? |
| `mamba_ema` / `mamba_surprise` | — | fairness: the same training tricks on the baseline |

Each is scored on eval loss, sample-efficiency AUC, param-efficiency, throughput, memory, and extrapolation. The report auto-prints the **best variant vs baseline** and whether the gain beats Mamba+EMA.

The **`recall` phase** (recall_diag) separately pins the H3 root-cause question (self-prediction fix vs depth weight-sharing vs capacity) on the strict held-out recall probe.

---

## 10. Success criteria

A lever "wins" if, at the primary size, it improves on `nfra_baseline` on the composite score — most importantly **eval loss and sample-efficiency** — with the cost accounting of §7 intact (no meaningful speed or memory regression). Priority ordering of evidence:

1. **Quality:** eval loss drops (and stays dropped across seeds).
2. **Sample-efficiency:** learning-curve AUC improves (learns more from fewer tokens).
3. **Extrapolation:** 2×-context delta improves (adaptation generalizes, not just fits).
4. **Composite:** the net score beats baseline and, ideally, closes part of the NFRA→Mamba quality gap.

`nfra_all` winning *more* than each lever alone is the strong signal that the three axes are complementary, not redundant.

---

## 11. Failure modes — and what each one teaches us

| Observation | Interpretation | Action |
|-------------|----------------|--------|
| A lever helps at 20M but not 5M | adaptation needs capacity to exploit | test at larger size / longer horizon |
| A lever helps in-distribution but hurts extrapolation | over-adaptation to training context | weaken the axis (e.g., smaller local window, smaller astro gain) |
| `nfra_all` ≈ best single lever | levers overlap; keep the simplest | drop the redundant ones (design principle: smallest winning set) |
| A lever helps quality but costs speed | cost accounting breaks | revisit: usually means a reduction was not as cheap as claimed |
| Nothing helps vs baseline | adaptive computation is not the missing ingredient at this scale | pivot to the capacity/optimization axes (dim-512 probe, memory levers) |

Negative results here are still valuable: NFRA is committed to *evidence*, and a documented "this axis does not pay at 5M" is a finding, not a failure.

---

## 12. If all three succeed — the "small but powerful" outcome

The project's thesis is that **capable AI can be made dramatically more efficient**, and that the brain's principles — not raw scale — are the path. If the three levers stack (best case: `nfra_all` gains on quality *and* sample-efficiency *and* extrapolation at 0.05% extra parameters), the claim becomes concrete and portable:

- NFRA Brain stays at ~0.6 GB peak memory while reducing the quality gap to Mamba — **brain-inspired adaptive computation as a substitute for parameter count.**
- Every lever is a standalone, trivially portable patch (zero or 225 params each) — a gift to the wider ML community: *"cortical routing, divisive normalization, and a glial homeostat, each worth studying in your own architecture."*
- The mechanisms open the next design questions the plan already flags: AFC-α memory, contrastive two-population gating, local-expert routing — each to be added under the same near-zero-cost discipline.

And it directly serves the stated mission: **quality AI on modest hardware, for everyone** — because a 0.05%-parameter addition that improves learning is exactly the kind of win that helps a phone, a laptop, or a Raspberry Pi far more than 100 extra million parameters does.

---

## 13. Design rules for future levers (this file is the contract)

Any future mechanism admitted into `NFRA_Brain_Block` under the "lever" umbrella must satisfy:

1. **Off by default** — never change shipped behavior unless toggled.
2. **One axis** — adapt WHERE / HOW MUCH / HOW LONG, cleanly separable from existing knobs.
3. **Near-zero cost** — reductions and `dim→1` projections only; no new sequence states, no new attention, no per-head tensors.
4. **Individually A/B-able** — must have a standalone env flag and a standalone ablate variant.
5. **Neuroscience-traceable** — the docstring and this file must state the biological analogue, because the analogue is the point.
6. **Falsifiable** — a stated success criterion and a stated failure interpretation.

---

## 14. FAQ

**Q: Why not just make the model bigger?**
A: The whole thesis is *evidence* that adaptation beats raw scale at the same cost. Each lever's cost is 0–225 params; a comparable quality gain from scale would cost millions. That is the bet this file documents.

**Q: Are these always on in released configs?**
A: No — all three default to `False`. The baseline NFRA results in the README are unchanged. The ablate experiment decides what becomes default.

**Q: `div_norm` and LayerNorm both normalize — redundant?**
A: No. LayerNorm normalizes the *input* of a layer across the dim axis; `div_norm` normalizes the *activation gain* of the hidden units by their own pooled intensity per token. One fixes input scale, the other fixes output gain. They are on different tensors and are trivially composable.

**Q: Does `astro` break the parallel scan?**
A: No. It multiplies `alpha` *before* the closed-form scan, and the scan's `alpha_min`/`alpha_max` clamp bounds the result, so the time-varying scan stays numerically safe (same guarantee the H3/audit work relies on).

**Q: Why a 64-token window for `local_route`?**
A: It is the sweet spot for the regimes tested (sequences 256–1024): short enough to be genuinely "local" per token, long enough to capture meaningful recent structure. It is a single constant and a candidate for a future ablation.

**Q: Can I run these on CPU?**
A: Yes — all three are pure PyTorch reductions, no CUDA-specific code. (Real training/benchmarks should still run on the T4 as documented.)

**Q: Where do the results get reported?**
A: `python -m nfra.benchmark.global_arena` produces `global_arena_report.md` (human-readable) and `global_arena_results.json` (machine-readable); the ablate section lists every lever's delta vs baseline.
