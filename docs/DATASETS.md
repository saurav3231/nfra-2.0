# Dataset Guide

The benchmark harness (`nfra.benchmark`) runs on **real text**: WikiText-2 at
character level. This page describes the data pipeline and how to reproduce it.

## Benchmark data (WikiText-2, character-level)

The headline benchmark uses the two local plain-text files:

- `wikitext-train-raw-v1.txt`
- `wikitext-valid-raw-v1.txt`

Download them from the Hugging Face dataset `Salesforce/wikitext` (
`wikitext-2-raw-v1`), place them in the working directory, and run with:

```bash
export NFRA_DATA=wikitext2
python -m nfra.benchmark.overnight
```

The harness splits text into characters (`vocab 96`, random loss 4.564), so
results are directly comparable across model families at identical token
budgets.

### Character vs sub-word tokens

Character-level tokenization is used deliberately:

- **Apples-to-apples**: every family consumes the same characters, so
  per-token comparisons are exact and no tokenizer variant skews the head-to-head.
- **Small budget**: the whole corpus fits in a few MB, enabling rapid,
  reproducible 5M/20M param runs on a free T4.

The trade-off is that character-level perplexity is higher and less intuitive
than word-level; the README reports raw eval loss (nats) which is the quantity
that matters for the matched-parameter comparison.

## Synthetic fallback

For quick smoke tests without network access, `NFRA_DATA=synthetic` (the
default in `nfra.benchmark.compare`) trains on random synthetic data. It is
useful for debugging and timing only — it is intentionally unlearnable and is
**not** used for the headline verified results.

## Notes on larger corpora

The verified results are produced at 5M/20M params on WikiText-2. For
scale-up experiments, larger corpora (WikiText-103, OpenWebText) require a
sub-word tokenizer and a longer schedule; this is an open research direction
tracked in [docs/FUTURE_PLAN.md](FUTURE_PLAN.md).
