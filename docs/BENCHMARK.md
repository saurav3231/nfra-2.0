# NFRA Benchmark Guide

How to run credible, reproducible, apples-to-apples comparisons of **NFRA Brain vs Mamba-SSM vs GPT-2** — on a Kaggle T4 GPU.

Two benchmarks ship inside the package:

| Tool | Command | Purpose |
|------|---------|---------|
| **compare** | `python -m nfra.benchmark.compare` | Quick head-to-head at ~20M params (eval loss, ppl, tok/s, memory) |
| **arena** | `python -m nfra.benchmark.arena` | Global-standard multi-dimension comparison (scaling, seeds, latency, verdict) |

---

## 1. Method (credibility controls)

Both benchmarks enforce the same fairness rules:

- **Param-matched models** — every architecture is tuned (layers / unique-blocks / dim) to land on the same parameter budget.
- **Identical data** — all models train on the *same* token streams.
- **Identical optimizer & schedule** — AdamW (lr 3e-4, β=(0.9, 0.95)), warmup + cosine.
- **Matched token budgets** — same steps × batch × sequence length per family.
- **GPT-2-style init** — every model starts near `ln(vocab)` (random-guess loss), so final loss is comparable.
- **Multiple seeds** (arena) — quality reported as **mean ± std**, never a single lucky run.
- **Multiple sizes** (arena) — a measured scaling slope, not a guess.

---

## 2. Setup on Kaggle

Create a **New Notebook** → Accelerator: **GPU T4** → Internet: **On**.

**Cell 1 — install from GitHub** (use `--no-deps` so the preinstalled CUDA torch is not replaced):

```python
!pip install -q --no-deps git+https://github.com/saurav3231/nfra-2.0.git
!python -c "import nfra; print(nfra.__version__, nfra.__author__)"
```

**Cell 2 — download WikiText-2 data**

The official `wikitext/wikitext-2-raw-v1` dataset is now private (HTTP 401). Use the `Salesforce/wikitext` mirror and convert the Parquet files to plain text in the working directory:

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
!NFRA_MODE=standard NFRA_DATA=wikitext2 python -m nfra.benchmark.compare
```

Output: console summary + `nfra_vs_mamba_vs_gpt2_results.json`.

### 3b. Global-standard multi-dimension (arena)

**Sanity check first** (fast, verifies the pipeline):

```bash
!NFRA_MODE=quick NFRA_DATA=wikitext2 python -m nfra.benchmark.arena
```

**Credible run** (~3–3.5 h on T4; Mamba's pure-PyTorch fp32 scan is the bottleneck):

```bash
!NFRA_MODE=standard NFRA_DATA=wikitext2 NFRA_SIZES=5,20 NFRA_SEEDS=2 python -m nfra.benchmark.arena
```

**Gold-standard headline run** (~8 h; 3 sizes for a real scaling fit, 3 seeds for tight stats):

```bash
!NFRA_MODE=rigorous NFRA_DATA=wikitext2 NFRA_SIZES=5,20,50 NFRA_SEEDS=3 python -m nfra.benchmark.arena
```

---

## 4. Env reference (arena)

| Env var | Default | Meaning |
|---------|---------|---------|
| `NFRA_MODE` | `standard` | `quick`=150, `standard`=600, `rigorous`=1500 steps |
| `NFRA_SIZES` | `5,20` | target model sizes in millions of params |
| `NFRA_SEEDS` | `2` (3 for rigorous) | independent seeds → mean ± std |
| `NFRA_FAMILIES` | `nfra,mamba,gpt2` | architectures to compare |
| `NFRA_DATA` | `synthetic` | `synthetic` or `wikitext2` |
| `NFRA_BATCH` | auto (4 on T4 wikitext2) | override batch size |

---

## 5. Outputs & interpretation

`arena` writes two files into the working directory:

- **`nfra_arena_results.json`** — full per-seed history, config/env fingerprint, per-size metrics, scaling fit, composite scores, structured verdict.
- **`nfra_arena_report.md`** — a publishable report with:

  1. **Model builds** — params, dim, depth per family (verifies param-matching).
  2. **Scaling behaviour** — OLS fit of eval loss vs `log2(params)`; slope = bits of loss gained per doubling (more negative = better scaling) + extrapolated loss @100M.
  3. **Head-to-head per size** — eval loss (mean ± std), ppl, sample-efficiency AUC, train tok/s, ms/step, peak memory, NaN steps.
  4. **Inference battery** — prefill tok/s, autoregressive gen tok/s, ms/token, peak inference memory, eval at 2× context (extrapolation).
  5. **Winners-per-aspect** — which architecture is best on *which* dimension.
  6. **Composite scores** — weighted z-scores (0–100) across all dimensions.
  7. **Verdict** — evidence-based claims + the "revolutionary?" assessment.

### Reading the verdict

- `eval loss ≈ ln(vocab)` = random guessing (4.56 for the 96-char wikitext vocab; 8.32 for synthetic 4096).
- Each claim lists its winner **and** the measured delta, so the verdict is auditable.
- A "revolutionary" verdict of *not confirmed* is still a valid result — the report shows precisely where the architecture stands.

---

## 6. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `wikitext2 files missing` | re-run the data cell from the same directory as the benchmark |
| Dataset HTTP 401 | use the `Salesforce/wikitext` mirror (see Cell 2) |
| `ModuleNotFoundError: nfra` | install via pip (Cell 1); do **not** rely on `sys.path` from a notebook cell — it does not reach `!python` subprocesses |
| Low GPU util (~40%) | expected — batch 4 + per-step sync + fp32 scans |
| OOM on larger sizes | keep wikitext2 batch auto-shrink, or drop the 50M size |
| Changed env vars not applying | each `!python` is a fresh process, so re-running applies them; if importing in-kernel, restart the kernel |

---

*Benchmark code lives in `src/nfra/benchmark/`. Author: SAURAV BHANDARI.*
