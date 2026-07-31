# NFRA 2.0 — Frequently Asked Questions (FAQ)

A plain-language companion to the [README](README.md). Questions are grouped by topic; if yours is missing, open a GitHub issue.

---

## 1. Basics

**Q1.1 — What is NFRA?**
NFRA (**NeuroFractal Resonance Architecture**) is a family of neural-network building blocks for language modeling, inspired by how biological brains use fractal self-similarity, resonance-based sparse activation, and predictive coding to process sequences efficiently.

**Q1.2 — What does "2.0" mean?**
It's the current generation of the architecture. 2.0 introduces the **NFRA Brain** mode (the benchmarked, GPU-focused design) alongside the long-standing **NFRA Lite** variant for legacy hardware.

**Q1.3 — Is NFRA a transformer?**
No. It does not rely on quadratic self-attention. It is closer in spirit to **state-space models (SSMs)** like Mamba — it uses recurrent-style mixing built on parallel scans — but with its own distinctive ingredients (multi-band decay, fractal gated MLPs, resonance-gated local attention, per-pass modulation).

**Q1.4 — Is NFRA a production-ready library?**
No. It is a **research project** (status: research-alpha). It exists to test a thesis: *capable AI on modest hardware*. Use it for experiments, benchmarking, and study — not for serving production traffic.

**Q1.5 — Who built it?**
**SAURAV BHANDARI**, conceived, designed, and developed with AI assistance.

---

## 2. Architecture

**Q2.1 — What is "multi-band sequence mixing"?**
The block maintains several recurrences at different temporal resolutions (bands), each with a decay rate α in [0.90, 0.995]. Slow bands remember far, fast bands react quickly — like multiple memory horizons running at once.

**Q2.2 — What is "selective decay"?**
The model decides, per token, how much of the past to remember — input-dependent forgetting. This is the SSM-style trick that gives Mamba its long-range edge; NFRA applies it across its bands.

**Q2.3 — What are "fractal gated MLPs"?**
Hierarchical, structurally sparse feed-forward networks. Instead of one dense MLP, the MLP is organized in a fractal/recursive pattern, which reduces parameters and FLOPs while preserving representational capacity.

**Q2.4 — What is "resonance-guided local attention"?**
A cheap, windowed attention mechanism whose strength is gated by a neuromodulation signal. It gives the model some token-to-token interaction without quadratic cost.

**Q2.5 — What does "depth-shared blocks" / `unique_blocks` mean?**
Instead of 12 independent layers, the model defines a small set of distinct blocks (e.g., `unique_blocks=6`) and reuses them across depth, giving each pass a different per-pass FiLM modulation plus a shared "global brain state." This is a big source of NFRA's memory savings.

**Q2.6 — What is NFRA Brain vs NFRA Lite?**
- **NFRA Brain** (`mode="brain"`): full multi-band design, the one used in the benchmarks.
- **NFRA Lite** (`NFRALiteForCausalLM`): single-file, dependency-light, INT8-friendly; targets 2012–2018 CPUs, Raspberry Pi, and microcontrollers.

**Q2.7 — Does NFRA support fp16?**
Yes. Training uses fp16 AMP. The scan path is fp32-safe — there is a guard against fp16 overflow (NaN) on the cumulative scans.

**Q2.8 — Does it support long contexts?**
The recurrent/scan-based parts are O(L) per token, so memory cost is linear in sequence length (no quadratic attention blow-up). The arena also tests robustness at 2× context length.

---

## 3. Benchmarks & results

**Q3.1 — What do the headline numbers mean?**
On WikiText-2 (character-level), ~20M params, 600 steps, param-matched models, identical data/optimizer/schedule, on a Kaggle T4:

| Model | Eval loss (↓) | Perplexity (↓) | Train tok/s (↑) | Peak memory (↓) |
|-------|--------------:|---------------:|----------------:|----------------:|
| **NFRA Brain** | **2.13** | ≈ 8 | 2,042 | **0.62 GB** |
| **Mamba SSM** | **1.59** | ≈ 5 | 845 | 5.09 GB |
| **GPT-2** | 3.19 | ≈ 24 | 37,570 | 0.95 GB |

**Q3.2 — Why does Mamba win on quality?**
Mamba has ~30 independent layers with fully-learned continuous dynamics at the same param budget, while NFRA trades capacity for memory (depth-shared blocks, fixed band grid). At equal params, that capacity difference shows up as loss. See the [FUTURE_PLAN](docs/FUTURE_PLAN.md) for the theory and roadmap to close it.

**Q3.3 — Is NFRA "better" than Mamba?**
On **memory and training speed**: yes, dramatically (26× less peak memory at 5M, ~4.7× faster pure-PyTorch training). On **loss at equal params**: no, Mamba is better. NFRA's honest thesis is that it wins on the **loss-per-megabyte** axis — see the memory-matched story in the future plan.

**Q3.4 — What is the NFRA Arena?**
`nfra.benchmark.arena` — a global-standard, multi-dimension benchmark: multiple sizes, multiple seeds (mean ± std), scaling slopes, inference latency, memory, extrapolation to 100M, robustness, and an evidence-based **verdict** on whether NFRA is "revolutionary." It writes `nfra_arena_results.json` + `nfra_arena_report.md`.

**Q3.5 — What is the verdict?**
The arena's verdict is data-driven, not asserted. It is printed in `nfra_arena_report.md` after a full run. At publication time the honest interim verdict is: *NFRA is not yet a quality leader at matched params, but it is a memory/efficiency leader — the strongest evidence so far is its memory and speed advantage, and the open question is whether scaling closes the quality gap.*

**Q3.6 — Why character-level WikiText-2?**
It is a small, reproducible corpus (no dataset-download flakiness) with a tiny vocab (96 chars), making quality differences visible quickly on modest GPUs. It is a *scientific* benchmark, not a claim about real-world performance.

**Q3.7 — Why only 2 seeds in the standard run?**
`NFRA_SEEDS` defaults to 2 (3 in `rigorous` mode). More seeds = tighter confidence intervals but linearly more GPU time. The report shows mean ± std so you can judge the spread.

**Q3.8 — How is "param-matched" ensured?**
The benchmark sweeps hidden size (and depth/`unique_blocks` for NFRA) until each model lands within a small tolerance of the target (e.g., ~20M). Mamba's `n_layers`/`d_state`, GPT-2's layers/heads, and NFRA's `dim`/`unique_blocks` are all tuned to hit the same budget.

**Q3.9 — What is the scaling slope / extrapolation?**
The arena fits a power law (loss vs params) across the tested sizes and extrapolates to 100M params. A steeper slope means a model improves faster per doubling of params — a key quality-per-param signal.

**Q3.10 — What is sample-efficiency AUC?**
The area under the loss-vs-tokens curve. It rewards models that reach low loss with fewer training tokens — a fairer "how fast does it learn" metric than the final loss alone.

**Q3.11 — Can I reproduce the numbers?**
Yes — that's the point. Follow [docs/BENCHMARK.md](docs/BENCHMARK.md). Every run records the config fingerprint, env vars, seeds, and wall-clock, so nothing is hidden.

---

## 4. Comparison with other models

**Q4.1 — How is NFRA different from Mamba?**
| | NFRA Brain | Mamba |
|---|---|---|
| Mixing | Multi-band decay + resonance attention | Single selective SSM |
| MLP | Fractal gated (structurally sparse) | Dense gated |
| Layers | Depth-shared blocks + per-pass modulation | Independent layers |
| Memory | ~0.14–0.62 GB at 5–20M | ~3.7–5.1 GB at 5–20M |
| Train speed | ~3,200 / ~2,040 tok/s (5M/20M) | ~670 / ~845 tok/s |

**Q4.2 — How is NFRA different from GPT-2?**
GPT-2 uses full quadratic self-attention + dense MLPs. It is extremely fast on GPU (37k tok/s) because of highly-optimized attention kernels, but its memory grows quadratically with context and it lacks a recurrence, so long-range recall requires the attention window. NFRA's recurrence gives linear memory and a different efficiency profile.

**Q4.3 — Is NFRA related to BrainMixer?**
NFRA 2.0's core blocks were inspired by the BrainMixer family (per previous development notes), then extended with resonance gating, per-pass FiLM adapters, a global brain state, and the multi-band scan design. See the [research paper draft](docs/NFRA_2.0_Research_Paper_Draft.md).

**Q4.4 — Does NFRA beat GPT-2?**
On WikiText-2 char at equal params: yes, clearly (2.13 vs 3.19). On wall-clock training throughput, GPT-2 wins (it has decades of kernel optimization behind it). On memory, NFRA wins (0.62 vs 0.95 GB).

---

## 5. Installation & usage

**Q5.1 — What do I need?**
Python 3.9+ and PyTorch 2.0+.

**Q5.2 — How do I install from GitHub?**
```bash
pip install --no-deps git+https://github.com/saurav3231/nfra-2.0.git
```
The `--no-deps` flag is important on Kaggle/Colab: it installs NFRA **without replacing** your prebuilt CUDA torch.

**Q5.3 — How do I install for development?**
```bash
git clone https://github.com/saurav3231/nfra-2.0.git
cd nfra-2.0
pip install -e ".[dev]"
```

**Q5.4 — How do I verify the install?**
```python
import nfra
print(nfra.__version__, nfra.__author__)   # 3.1.0 SAURAV BHANDARI
```

**Q5.5 — How do I build a model?**
```python
from nfra import NFRAConfig, NFRAForCausalLM
config = NFRAConfig(mode="brain", vocab_size=32000, hidden_size=512,
                    num_layers=12, unique_blocks=4, depth_shared=True)
model = NFRAForCausalLM(config)
```

**Q5.6 — How do I run the quick comparison?**
```bash
python -m nfra.benchmark.compare
```

**Q5.7 — How do I run the full arena?**
```bash
python -m nfra.benchmark.arena
```
Environment variables control it — see the README env-reference table (`NFRA_MODE`, `NFRA_DATA`, `NFRA_SIZES`, `NFRA_SEEDS`, `NFRA_FAMILIES`, `NFRA_BATCH`, `NFRA_STEPS`, `NFRA_TARGET_PARAMS`, `NFRA_DIM`).

**Q5.8 — Where can I find real WikiText-2 data?**
The official HF `wikitext/wikitext-2-raw-v1` dataset is currently private (returns 401). Use the working mirror documented in [docs/BENCHMARK.md](docs/BENCHMARK.md): download `Salesforce/wikitext` Parquet, join the `text` column with `'\n'`, and save `wikitext-train-raw-v1.txt` / `wikitext-valid-raw-v1.txt` in the working directory.

**Q5.9 — Can I use NFRA on CPU?**
Yes — the whole codebase runs on CPU (tests do). Lite is explicitly built for weak CPUs. The arena numbers above are from a T4 GPU; CPU numbers will differ.

**Q5.10 — Are there pretrained weights?**
No downloadable pretrained checkpoints yet. The project's focus is architecture + benchmarking, and training is research-scale.

---

## 6. Hardware, memory & performance

**Q6.1 — Why is NFRA's memory so low?**
Depth-shared blocks (few unique weights), structurally sparse MLPs, local (windowed) attention, and no quadratic attention cache all shrink the footprint. At 5M it peaks at ~0.14 GB vs Mamba's ~3.66 GB.

**Q6.2 — Why is NFRA faster than Mamba in this benchmark?**
Mamba's scan is implemented in pure Python/PyTorch here (no fused kernels), while NFRA's parallel scans are structured to run efficiently. Both would speed up with fused kernels; the comparison is apples-to-apples because both are pure PyTorch.

**Q6.3 — What GPU do I need for the benchmarks?**
A Kaggle T4 (16 GB) runs the full standard arena comfortably. `quick` mode (150 steps) is for smoke-testing.

**Q6.4 — Can NFRA fit on very small GPUs / edge devices?**
That's the Lite target: single-file, INT8-friendly, works on 2012–2018 CPUs, Raspberry Pi, and microcontrollers.

**Q6.5 — What does "batch auto-shrinks to 4 on T4" mean?**
When `NFRA_DATA=wikitext2`, the arena lowers the training batch to 4 automatically to fit the longer char-level sequences comfortably on 16 GB.

---

## 7. Troubleshooting

**Q7.1 — `ModuleNotFoundError: No module named 'nfra'`**
The package isn't on `sys.path`. Run `pip install -e .` (dev) or `pip install --no-deps git+https://github.com/saurav3231/nfra-2.0.git`. Note: on Kaggle, `!python` subprocesses do **not** see notebook `sys.path` — install the package rather than relying on a clone+`sys.path` hack.

**Q7.2 — `UnicodeEncodeError` in console output**
The console locale (e.g., cp1252 on Windows/Kaggle) can't print non-ASCII glyphs. The benchmark intentionally prints ASCII-only; write files with `encoding='utf-8'`. Use a UTF-8 terminal if you must show fancy glyphs.

**Q7.3 — `NameError: name 'os' is not defined` in the data-download snippet**
The mirror snippet needs `import os` at the top of the cell. The current [docs/BENCHMARK.md](docs/BENCHMARK.md) version already includes it.

**Q7.4 — The benchmark crashes with OOM on my GPU**
Lower the batch: `NFRA_BATCH=2 python -m nfra.benchmark.arena`, or use a smaller size list (`NFRA_SIZES=5`) and `quick` mode.

**Q7.5 — I see NaN during training**
NFRA guards against fp16 scan overflow, but if you see NaN: run in fp32, lower the learning rate, and confirm your torch build has working fp16 AMP. Open an issue with the config fingerprint if it persists.

**Q7.6 — `401` downloading WikiText-2**
Use the mirror in [docs/BENCHMARK.md](docs/BENCHMARK.md) (Question 5.8).

**Q7.7 — How do I confirm an interrupted arena run saved results?**
It doesn't — by design, results are written only on completion (`nfra_arena_results.json` + `nfra_arena_report.md`). Re-run to completion.

**Q7.8 — Where do results files go?**
Current working directory: `nfra_arena_results.json`, `nfra_arena_report.md`, and (for `compare`) the result JSON. Both are git-ignored so runs don't pollute the repo.

---

## 8. Project & community

**Q8.1 — What is the license?**
MIT — see [LICENSE](LICENSE).

**Q8.2 — How can I contribute?**
See [CONTRIBUTING.md](CONTRIBUTING.md). Code style: Black (line-length 88) + Ruff. Run the CPU tests (`pytest`) before opening a PR.

**Q8.3 — Is there a research paper?**
A draft is in [docs/NFRA_2.0_Research_Paper_Draft.md](docs/NFRA_2.0_Research_Paper_Draft.md), with an outline in [docs/PAPER_OUTLINE.md](docs/PAPER_OUTLINE.md).

**Q8.4 — What is the roadmap?**
See [docs/FUTURE_PLAN.md](docs/FUTURE_PLAN.md): a phased, gated plan to close the quality gap (adaptive bands, per-pass low-rank adapters, routed sparsity) plus brain-inspired ideas (surprise-weighted gradients, k-WTA sparsity, episodic replay, arousal gating), all while protecting the memory/latency advantage.

**Q8.5 — What are NFRA's current limitations?**
- Trails Mamba on loss at equal params (the open problem).
- Research-grade code; not fully battle-tested for production.
- Benchmarks are character-level / small-scale; real-world scaling is unproven.
- Speed comparison is against pure-PyTorch baselines, not fused-kernel Mamba.
- Only small checkpoints; no large pretrained models yet.

**Q8.6 — What is the project's core belief?**
*"The future of AI should be measured not only by capability, but by accessibility and sustainability."* The goal is quality AI on hardware ordinary people own — and credible, evidence-based proof of how NFRA compares.

**Q8.7 — Why should I care about NFRA if Mamba has better loss?**
Because "better loss at 20M params" is only one point on a multi-dimensional Pareto frontier. NFRA sits on the **memory-efficient, hardware-friendly** end of that frontier — the region that matters for edge devices, low-power inference, and researchers who care about loss-per-megabyte. If you have a T4, a Pi, or a 2015 laptop and want a *serious* language model, that's NFRA's space.
