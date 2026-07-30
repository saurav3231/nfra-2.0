# Additional Public Datasets for NFRA 2.0

## High-Quality Datasets

### 1. **C4 (Colossal Clean Crawled Corpus)**
```python
from datasets import load_dataset
dataset = load_dataset("c4", split="train[:50000]")
```

### 2. **OpenWebText**
```python
dataset = load_dataset("openwebtext", split="train[:30000]")
```

### 3. **The Pile (subset)**
```python
dataset = load_dataset("EleutherAI/pile", split="train[:20000]")
```

### 4. **RedPajama**
```python
dataset = load_dataset("togethercomputer/RedPajama-Data-1T", split="train[:10000]")
```

### 5. **FineWeb**
```python
dataset = load_dataset("HuggingFaceFW/fineweb", split="train[:15000]")
```

## Multilingual Datasets

- **OSCAR** (multilingual)
- **mC4**

## Domain-Specific

- **PubMed** (biomedical)
- **ArXiv** (scientific papers)
- **CodeParrot** (code)

All datasets above are publicly available on Hugging Face and compatible with NFRA 2.0 training scripts.