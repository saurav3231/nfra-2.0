# Running the Overnight NFRA Benchmark on Kaggle (GPU)

This guide runs `nfra.benchmark.overnight` — the phased, all-axes comparison of
**NFRA vs RetNet vs RWKV vs GPT-2** on **real WikiText-2** character text — on a
Kaggle GPU session. It is designed to run unattended overnight: phases are resumable,
each phase is wrapped in try/except, and an adaptive time budget prunes work instead
of crashing.

> Only run this on a GPU. On CPU the 5M-param models train so slowly the run looks
> "stuck" — that is normal, not a hang. The script prints a warning when CUDA is absent.

## What the script does

| Phase | Axis measured |
|---|---|
| `core` | Head-to-head eval loss / PPL / sample-AUC + scaling curve across sizes |
| `context` | Performance vs context length (256/512/1024) |
| `efficiency` | Eval loss under energy budget (0.25–1.0) |
| `ablate` | NFRA feature toggles (EMA, surprise, k-WTA, routing, …) |
| `recall` | Span-recall probes (k = 4/16/64/128) |
| `deploy` | FP32 vs INT8 size + CPU prefill speed |
| `perf` | Prefill / generation tok/s + inference memory |
| `data2` | Out-of-domain generalization (TinyShakespeare) |

## Step 1 — Create the Kaggle notebook

1. Go to **kaggle.com → Create → New Notebook**.
2. On the right sidebar pick **GPU T4 x2** (or better) under Accelerator.
3. Set **Language: Python**.

## Step 2 — Pull the repo

In a notebook cell:

```bash
!git clone https://github.com/saurav3231/nfra-2.0.git /kaggle/working/nfra-2.0
```

## Step 3 — Install dependencies

```bash
!pip install -q torch torchvision datasets transformers tqdm numpy
```

`torch` is usually already present on Kaggle; the line is a safety net.

## Step 3 — Install dependencies

```bash
!pip install -q torch torchvision datasets transformers tqdm numpy
```

`torch` is usually already present on Kaggle; the line is a safety net.

## Step 4 — Install the package

The repo uses a `src/` layout, so `nfra` is not importable until installed. In a cell:

```bash
!cd /kaggle/working/nfra-2.0 && pip install -e . --quiet
```

(If you skip this, `python -m nfra.benchmark.overnight` fails with
`ModuleNotFoundError: No module named 'nfra'`.)

## Step 5 — Run the benchmark

```bash
!cd /kaggle/working/nfra-2.0 && \
  python -m nfra.benchmark.overnight
```

The script will automatically **download WikiText-2-raw** (two mirrored URLs, falls
back if the canonical S3 link 301s) and **TinyShakespeare**, then run every phase.

### Tuning with env vars

```bash
!cd /kaggle/working/nfra-2.0 && \
  NFRA_OVN_MODE=big \
  NFRA_OVN_STEPS=600 \
  NFRA_OVN_SIZES=5,20,50 \
  NFRA_OVN_SEEDS=3 \
  NFRA_OVN_MAX_MIN=500 \
  NFRA_OVN_OUTDIR=/kaggle/working/out \
  python -m nfra.benchmark.overnight
```

| Var | Default | Meaning |
|---|---|---|
| `NFRA_OVN_MODE` | `standard` | `quick` / `standard` / `big` presets |
| `NFRA_OVN_STEPS` | from mode | training steps per (size, seed, family) |
| `NFRA_OVN_SIZES` | `5,20,50` | target param sizes in M |
| `NFRA_OVN_SEEDS` | 3 | number of seeds |
| `NFRA_OVN_MAX_MIN` | 400 | total wall-clock budget in minutes |
| `NFRA_OVN_OUTDIR` | CWD | output directory |
| `NFRA_OVN_DATA` | `wikitext2` | must stay `wikitext2` (script exits otherwise) |
| `NFRA_OVN_PHASES` | all 8 | comma list to restrict phases |
| `NFRA_COMPILE` | `1` | `torch.compile` fusion (1.5–3× fewer launches; auto-fallback) |
| `NFRA_CHECKPOINT` | `0` | gradient checkpointing; `0` = faster, no memory need at these sizes |
| `NFRA_SCAN_KERNEL` | `0` | `0` = pure-torch scan so the compiled graph stays fused |
| `NFRA_EMA` | `0.99` | EMA weight-averaging decay applied to all families (~0.1–0.3 nats loss gain); `0` = off |

Data guard: if `NFRA_OVN_DATA` is anything but `wikitext2`, the script refuses to run —
real text only, never the synthetic "unlearnable" set.

## Step 6 — Resume / avoid the 9-hour Kaggle limit

Phases checkpoint after each completion into `overnight_state.json`. If the session
expires, start a new notebook, re-clone the repo, and re-run the same command — it
will skip completed phases and continue. Keep output in `/kaggle/working/out` and
re-upload that folder between sessions to preserve resume state.

## Step 7 — Fetch results

Add a final cell to package everything as a downloadable archive:

```bash
!cd /kaggle/working && tar czf overnight_out.tar.gz out && echo DONE
```

Then use the notebook's **Output → Commit → Save** or download `overnight_out.tar.gz`
from the file panel.

## Outputs

- `overnight_results.json` — full structured results
- `overnight_report.md` — human-readable report (tables + verdicts)
- `overnight_state.json` — resume state
- `core.csv`, `context.csv`, `efficiency.csv`, `ablate.csv`, `recall.csv`,
  `deploy.csv`, `perf.csv`, `data2.csv` — per-phase tables

## Troubleshooting

- **"IndexError: index out of range in self"** — impossible now: the script forces
  `DATA_SOURCE=wikitext2` on the shared modules (the real-data fix) and pre-checks
  every dataset's max token against the model vocab (`assert_tokens_in_range`).
- **WikiText-2 download fails** — the script tries the canonical S3 URL first, then a
  verified HF LFS mirror. If both fail, manually place
  `wikitext-train-raw-v1.txt` + `wikitext-valid-raw-v1.txt` in the run directory.
- **Looks stuck / no progress** — you are on CPU. Use a GPU accelerator.
- **Non-ASCII output garbled** — the script reconfigures stdout to UTF-8; if your
  terminal still mangles it, set `PYTHONIOENCODING=utf-8` in the notebook.
