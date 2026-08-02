# NFRA Benchmark Guide

How to run credible, reproducible, apples-to-apples comparisons of **NFRA vs
RetNet vs RWKV vs GPT-2** — on a Kaggle T4 GPU.

Three benchmark tools ship inside the package:

| Tool | Command | Purpose |
|------|---------|---------|
| **compare** | `python -m nfra.benchmark.compare` | Quick head-to-head at ~20M params (eval loss, ppl, tok/s, memory) |
| **arena** | `python -m nfra.benchmark.arena` | Multi-dimension comparison (scaling, seeds, latency, verdict) |
| **overnight** | `python -m nfra.benchmark.overnight` | **The verified phased run** (`core,context,efficiency,ablate,recall,deploy,perf`) |

---

## 1. Method (credibility controls)

All benchmarks enforce the same fairness rules:

- **Param-matched models** — every architecture is tuned (layers / unique-blocks
  / dim) to land on the same parameter budget; nfra builds bit-identical
  geometry to RetNet at each size.
- **Identical data** — all models train on the *same* character streams
  (WikiText-2, vocab 96), so token budgets are exactly equal.
- **Identical optimizer & schedule** — AdamW, warmup + cosine.
- **Matched token budgets** — same steps × batch × sequence length per family.
- **GPT-2-style init** — every model starts near `ln(vocab)` (random-guess
  loss), so final loss is comparable.
- **Multiple seeds** — quality reported as per-seed values and mean, never a
  single lucky run.
- **Multiple sizes** — a measured scaling slope, not a guess.

---

## 2. Setup on Kaggle

Create a **New Notebook** → Accelerator: **GPU T4** → Internet: **On**.

**Cell 1 — install from GitHub** (use `--no-deps` so the preinstalled CUDA
torch is not replaced):

```python
!pip install -q --no-deps git+https://github.com/saurav3231/nfra-2.0.git
!python -c "import nfra; print(nfra.__version__, nfra.__author__)"
```

**Cell 2 — download WikiText-2 data**

The official `wikitext/wikitext-2-raw-v1` dataset is now private (HTTP 401).
Use the `Salesforce/wikitext` mirror and convert the Parquet files to plain
text in the working directory:

```python
import os
!pip install -q pandas pyarrow
import pandas as pd

for name in ['train', 'validation']:
    url = f'https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/{name}-00000-of-00001.parquet?download=true'
    df = pd.read_parquet(url)
    fname = 'wikitext-train-raw-v1.txt' if name == 'train' else 'wikitext-valid-raw-v1.txt'
    open(fname, 'w', encoding='utf-8').write('\n'.join(df['text']))
    print(fname, round(os.path.getsize(fname) / 1e6, 2), 'MB')
```

Both files must sit in the notebook's current working directory.

---

## 3. Run the benchmarks

### 3a. Quick head-to-head (compare)

```bash
!NFRA_DATA=wikitext2 python -m nfra.benchmark.compare
```

Output: console summary + `nfra_vs_mamba_vs_gpt2_results.json`.

### 3b. Multi-dimension (arena)

Sanity check first (fast, verifies the pipeline):

```bash
!NFRA_MODE=quick NFRA_DATA=wikitext2 python -m nfra.benchmark.arena
```

### 3c. The verified phased run (overnight) — RECOMMENDED

This is the run that produced the numbers in the README and
`docs/OVERNIGHT_VERIFIED_RESULTS.md`:

```bash
!NFRA_OVN_MODE=standard NFRA_OVN_PHASES=core NFRA_OVN_FAMILIES=nfra,retnet python -m nfra.benchmark.overnight
```

To run the full verified 8-phase suite:

```bash
!NFRA_OVN_MODE=standard python -m nfra.benchmark.overnight
```

Phases are resumable via `overnight_state.json`. Outputs:
`overnight_results.json` and `overnight_report.md`.

---

## 4. Env reference (overnight)

| Env var | Default | Meaning |
|---------|---------|---------|
| `NFRA_OVN_MODE` | `standard` | `quick`=300, `standard`=600, `big`=1500 steps |
| `NFRA_OVN_SIZES` | `5,20` | target model sizes in millions of params |
| `NFRA_OVN_SEEDS` | `2` | independent seeds → mean ± std |
| `NFRA_OVN_FAMILIES` | `nfra,rwkv,retnet,gpt2` | architectures to compare |
| `NFRA_OVN_PHASES` | all | `core,context,efficiency,ablate,recall,deploy,perf,data2` |
| `NFRA_OVN_DATA` | `wikitext2` | data source (only `wikitext2` allowed) |
| `NFRA_LEAN` | `1` | `1` = lean block (verified default); `0` = full 3.3b |
| `NFRA_COMPILE` | `1` | `1` = `torch.compile` (auto-disables when unstable) |

### Env reference (arena)

| Env var | Default | Meaning |
|---------|---------|---------|
| `NFRA_MODE` | `standard` | `quick`=150, `standard`=600, `rigorous`=1500 steps |
| `NFRA_SIZES` | `5,20` | target model sizes in millions of params |
| `NFRA_SEEDS` | `2` | independent seeds |
| `NFRA_FAMILIES` | all | `nfra,rwkv,retnet,mamba,gpt2` |
| `NFRA_DATA` | `synthetic` | `synthetic` or `wikitext2` |
| `NFRA_BATCH` | auto | override batch size |

---

## 5. Outputs & interpretation

`overnight` writes two files into the working directory:

- **`overnight_results.json`** — full per-seed history, config/env fingerprint,
  per-size metrics, scaling fit, composite scores, structured verdict.
- **`overnight_report.md`** — a publishable report with:

  1. **Model builds** — params, dim, depth per family (verifies param-matching).
  2. **Head-to-head per size** — eval loss (per-seed + mean), ppl, train tok/s,
     ms/step, peak memory, NaN steps.
  3. **Context extrapolation** — eval at 256/512/1024 (train @256).
  4. **Recall / deploy (INT8) / perf** phases — memory-horizon, quantization,
     inference battery.
  5. **Winners-per-aspect** — which architecture is best on *which* dimension.

### Reading the verdict

Loss numbers are in **nats** (natural log of perplexity), which is the
quantity that matters for a matched-parameter comparison. A gap of −0.18 nats
(5M) or −0.048 nats (20M) with no overlap across seeds is a real, measurable
architecture difference — see the verified results in
[`docs/OVERNIGHT_VERIFIED_RESULTS.md`](OVERNIGHT_VERIFIED_RESULTS.md).
