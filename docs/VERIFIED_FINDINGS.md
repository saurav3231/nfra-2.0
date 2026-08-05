# NFRA-2.0 — Verified Findings (measured on T4, this project)

This page records what has been **measured and verified on hardware** in this
project. Everything here is a run result, not a claim — numbers are from the
T4 battery (`nfra.benchmark.experiments_gate` / `factor_run`) with deterministic
seeds, so runs are reproducible.

> The repo **defaults remain byte-identical to the verified baseline** (all
> architecture knobs off) so a fresh clone reproduces the board exactly. The
> recommended production recipe is opt-in via `NFRA_RECOMMENDED=1` (see below).

---

## 1. The flagship operating point (best measured on WikiText-2, vocab 96)

| config | eval | train tok/s | peak train mem |
|---|---|---|---|
| depth 33 (deep stack) | ~1.66 – 1.70 | ~24k | ~2.9 GB |
| **depth 8 · batch 24 · ckpt on · seq 256 · EMA 0.99** | **1.6455** | **28.9k** | **0.78 GB** |
| long-context @512 | ext Δ +0.0149 | — | infer 0.216 GB |

Takeaways
- **Depth is the speed/memory lever**: fewer sequential blocks → near-linear
  tok/s for roughly constant quality. The depth-8 sweep beat the deep stack on
  both speed and memory at ~equal loss.
- **Gradient checkpointing** buys −62% memory for +~23% step time (recompute).
  The flagship pairs it with depth-8 + batch 24 to get 0.78 GB.

## 2. The stateful O(1) generator (this project's core deliverable)

**Mechanism.** Retention (decayed-causal QK^T, no softmax) has an exact
recurrent dual `y_h = (q/√head_dim)·R_h`, `R ← γ·R + kᵀv`, `γ = exp(-exp(log_decay))`,
so after a one-time prefill the model decodes each new token in O(1) — it never
re-reads the context. The barrier was the mixer **GroupNorm**, which normalizes
over the whole sequence (a future-leak). **Per-token GroupNorm**
(`cortex_per_token_gn` / `NFRA_PERTOKEN_GN=1`) removes that coupling, leaving
retention `R` as the only cross-token state.

**Guard.** `nn.select` → `nfra.core.stateful.supported()` + the equivalence
check. The fast path is only trusted (battery prints `gen_sf … ok`) when it
reproduces the full model to ~1e-5:
- CPU selftest: max rel err **1.1e-5** (ok), single-step match **6e-6**.

**Measured on T4 (depth-8, batch-1 decode):**

| arm | slow re-eval tok/s | stateful tok/s | speedup | guard |
|---|---|---|---|---|
| trained | 97 | 167 | 1.7× | ok |
| lsr | 111 | 156 | 1.4× | ok |
| int8_state | 55 | 165 | 3.0× | **X** (refused) |
| depth_time | 111 | 149 | 1.3× | ok |
| triton_chunk | 71 | 161 | 2.3× | ok |
| rev | 105 | 157 | 1.5× | ok |

- `int8_state` → `X`: its quantized long-range state breaks exactness, so the
  guard **refuses** to report it — the fast path never silently lies.
- Gen speedup grows with depth (the slow path slows ~5× at depth-33 while the
  stateful cost stays flat). Absolute ~150 tok/s is bounded by batch-1
  small-GEMM Python decoding on the T4, not by the O(1) theory.

## 3. Per-factor sweep (20M, depth-8, batch-24, ckpt, seq-256, synthetic vocab-4096)

`trained` = plain baseline. Deltas **+ = better** (so a positive `eval` delta is
a *lower loss*).

| arm | eval | eval Δ | composite | sampleAUC | train tok/s | ms/step | guard |
|---|---|---|---|---|---|---|---|
| **trained** | 9.9839 | — | ±0.0000 | 75.80 | 30932 | 198.6 | ok |
| **lsr** | 9.9652 | **+0.019** | **+0.0138** | best | 29193 | 210.5 | ok |
| int8_state | 9.7480 | **+0.236** | −0.0250 | worse | 27610 | 222.5 | X |
| depth_time | 9.9837 | ~0 | +0.0107 | better | 29370 | 209.2 | ok |
| triton_chunk | 9.9506 | +0.033 | −0.0225 | worse | 18937 | 324.4 | ok |
| rev | 9.9676 | +0.016 | +0.0078 | better | 25573 | 240.3 | ok |

Findings:
- **`lsr` is the balanced, safe winner** — best composite, best sample-AUC,
  ~neutral-to-better eval, small speed cost, and it keeps the exact stateful
  guard (`ok`). Recommended default of production.
- **`int8_state` wins big on eval** (+0.24) but loses the composite (worse
  sample-AUC, slower train) and *breaks exact stateful gen* (`X`) — a pure
  memory/precision trade, not a free direction.
- **`triton_chunk` wins eval but collapses training throughput** (−12k tok/s,
  +126 ms/step at S=256) — the serial chunk loop is launch-bound in eager mode.
  It's only worth it at long context where the quadratic-memory win dominates.
- **Extrapolation holds**: every arm's `ext@L×2` is better (lower loss) than its
  `eval` — the linear core generalizes over longer context rather than degrading.

## 4. Non-wins we measured and kept as-is (honest list)

- **LSR short-run artifact**: on short budgets LSR can look neutral/negative
  (+0.006 at 1200 steps) yet wins at longer horizons (−0.078 wikitext). Timescale
  dynamics; judge it over the full budget.
- **Fused Triton retention backward** (dq/dk/dv): correct (self-test max rel
  6.7e-4) but **not faster** at head-dim 32 (414 vs 364 ms/step) — kept as a
  memory/activation save, not a speed win.
- **torch.compile `max_autotune_gemm`**: **backfires on T4** (not enough SMs)
  → 10.2k tok/s / 2.96 GB. Only `mode='default'` compile is beneficial.

## 5. Recommended production recipe (opt-in, `NFRA_RECOMMENDED=1`)

The default stays byte-identical to the verified baseline. To run the verified
**recommended** config, set `NFRA_RECOMMENDED=1` (turns on LSR + per-token GN;
an explicit `NFRA_LSR=0` / `NFRA_PERTOKEN_GN=0` still wins over the preset):

```
NFRA_RECOMMENDED=1  NFRA_GATE_TARGET_M=20  NFRA_GATE_DEPTH=8  NFRA_BATCH=24
NFRA_CHECKPOINT=1  NFRA_SEQ=256  NFRA_EMA=0.99  NFRA_GATE_ARMS=trained,lsr,rev
python src/nfra/benchmark/experiments_gate.py
```
(or `python src/nfra/benchmark/factor_run.py` with `ARMS=["trained","lsr","rev"]` for
the compact factor table + deltas table + `lsr` composite + stateful headline.)

### Reproducibility note
- Seeds + batches are byte-identical across arms within a run, so deltas
  (`lsr`/`rev` vs `trained`) are internally consistent and deterministic.
- The factor-table `eval` (~9.98) is on the **synthetic** corpus (vocab 4096);
  do not compare it to the WikiText-2/vocab-96 flagship (1.6455). Those are
  different data regimes — the deltas vs `trained` are the apples-to-apples measure.
- The repo **default** is deliberately the byte-identical verified baseline
  (knobs off). `NFRA_RECOMMENDED=1` applies the recommended recipe without
  ever overriding an explicit knob.