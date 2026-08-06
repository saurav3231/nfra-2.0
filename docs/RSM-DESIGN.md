# RSM — remembering state machine (memory retrofitted onto NFRA)

Status legend:

- **[built]** = code present + structurally verified on CPU.
- **[verified]** = numbers confirmed on the T4 board.
- **[design]** = not yet implemented (RFC only).

## 1. What RSM adds to NFRA

NFRA's `CortexMixer` (matrix-state mixer + resonance) is already O(1) stateful and writes
every token into a fixed per-layer state. That state has **no decayed relevance** — every
past token competes equally on read — and **no write control** — nothing decides which
inputs are worth remembering vs. forgetting. RSM layers two memory tiers on top as *gates
into the existing state*, not a replacement.

| tier | mechanism | capacity | write | read | status |
|------|-----------|----------|-------|------|--------|
| L0 core | existing `CortexMixer` matrix-state (per-layer `R`, running `μ`) | unbounded, distributed | always | exact O(1) | **[built]** |
| L1 STM | `CortexWorkingMemory` windowed causal ring | `window` tags/layer | learned | softmax recency | **[prototype]** |
| L2 LTM | sleep-scheduled engram consolidation | grows with sleep | scheduled | associative | **[design]** |

The core RSM idea: **decay is a bias, not a loss** — old but relevant state must stay
recoverable. STM is a *read-side* recall assist (peek at recent tags); LTM is a
*write-side* consolidation that promotes important past engrams back to the current read.

## 2. STM working-tag ring (`CortexWorkingMemory`)

A small windowed attention over the last `window` mixer inputs, per layer:

```python
class CortexWorkingMemory(nn.Module):
    def __init__(self, dim, window, stm_dim=32):
        self.window, self.stm_dim = window, stm_dim
        self.wq    = nn.Linear(dim, stm_dim, bias=False)  # live
        self.wk    = nn.Linear(dim, stm_dim, bias=False)  # live
        self.wv    = nn.Linear(dim, dim,    bias=False)   # live
        self.w_out = nn.Linear(dim, dim,   bias=False)    # ZERO-INIT read gate
```

`forward(x)` at position `S` behaves as windowed attention over the last `window` cached
keys aligned to `S` (rel = `s - S`, mask `0 <= rel <= window`), softmax weights
`softmax((x wq)(cache wk)ᵀ / sqrt(stm_dim))`, then value read `wv(cache)ᵀ w` → `w_out(...)`
added to the mixer output after the receptance gate.

`read_step(x1, ctx)` runs the same windowed softmax over the cached ring and produces the
identical `y_out` **without rebuilding the full sequence** — O(1) exact decode.

### Why zero-init `w_out` (the non-obvious part)
First draft zero-ed both `wv` and `w_out` → dead gate: with 0 input the *gate input is a
constant*, so no gradient reaches `wq/wk` and the ring never learns. Fix: zero only
`w_out`. At init the read is 0 → the model is **bit-identical to baseline** (parity 0.0).
Once training starts, `wq/wk/wv` receive real gradients inside a fitted read signal and the
ring learns a function while starting as the pure bottom baseline → **zero-regression**.

### Verified (CPU structural, this repo)
```
[1] init zero-regression  : enabled == disabled, max_abs = 0.0        → bit-identical
[2] stateful O(1) dual    : sf_ok=True, max_rel=4.39e-07, active ring → exact
[3] ring learns (3 steps) : wv.grad ~1.8, w_out.grad ~11.9, output moved
```
The eval-Δ / speed / regression numbers require the T4; see the `bench_stm` cell.

### Wiring
- `NFRAConfig.cortex_stm_ring: int = 0` (0=off), `cortex_stm_dim: int = 32`.
- `block_kwargs["stm_ring"/"stm_dim"]` → `CortexMixer(stm_ring=…)`.
- Env: `NFRA_STM_RING`, `NFRA_STM_DIM` in `arena.build_nfra`.
- O(1) stateful dual (`stateful.py`): per-layer `stm_ctx` cache; `_mixer_step` runs the
  ring read and trims → exact decode.

## 3. L2 LTM engram consolidation — **[design]**

Not implemented. RFC direction:

- **trigger**: idle/sleep window; a per-slot `read-count` tracks how often a write slot's
  tag is read. High-read slots are promoted, low ones decayed.
- **action**: copy a strong past engram `(tag,state)` back into the current state via a
  low-rank put-only adapter, scheduled at sleep.
- **recall probe**: an auxiliary memory objective, separate from LM loss, so memory
  capacity is measured independently of perplexity.

## 4. Evaluation contract (all on T4)

1. **Zero-regression gate** (`bench_stm`): off vs ring-on identical init → `Δ eval <= 0.004`
   nats and step-time `% ` <= `+5%`. If regresses, drop `w_out` lr or reduce window.
2. **Throughput**: keep the ~29k tok/s·0.8 GB ·20M baseline.
3. **Recall probe**: windowed copy-recall, per family (`nfra` vs `retnet` vs `gpt2`).
4. **Final**: `big_night` multi-seed run.

## 5. Open questions
- One ring per layer (current) vs a single shared top-level ring (half the memory, loses
  per-depth relevance)?
- Always-cache vs a gradual read-workload-selected tag decay to free ring space?

## Milestones
1. [x] prototype + O(1) dual + zero-init verified on CPU (`test_stm.py`, `bench_stm.py`)
2. [ ] zero-regression A/B + speed on T4 (`bench_stm` → `BENCH_DONE`)
3. [ ] recall probe (auxiliary memory objective), per family
4. [ ] LTM sleep-scheduled consolidation + slow-copy ring
5. [ ] `big_night` multi-seed headline run