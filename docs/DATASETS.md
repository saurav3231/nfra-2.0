# Supported Public Datasets for NFRA 2.0

**Maintained by Saurav Bhandari**

## Recommended Datasets

### 1. WikiText-2 / WikiText-103
- **Source**: `wikitext` on Hugging Face
- **Best for**: Initial training and benchmarking
- **Size**: Small to medium

### 2. C4 (Colossal Clean Crawled Corpus)
- **Source**: `c4` on Hugging Face
- **Best for**: Large-scale pretraining
- **Note**: Use subset for free Kaggle/Colab

### 3. OpenWebText
- **Source**: `openwebtext` 
- **Best for**: High-quality language modeling

### 4. The Pile (subset)
- Good for research

## How to Use

```python
from datasets import load_dataset

# WikiText-2
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# C4 (small subset)
dataset = load_dataset("c4", split="train[:10000]")
```

All configs in `configs/` are compatible with these datasets.