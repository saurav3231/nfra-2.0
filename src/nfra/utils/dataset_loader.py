"""
Dataset loading utilities for NFRA 2.0

Supports multiple public datasets.
Created by Saurav Bhandari
"""

import torch


def load_and_tokenize(
    dataset_name: str = "wikitext",
    config: str = "wikitext-2-raw-v1",
    split: str = "train",
    tokenizer_name: str = "gpt2",
    max_length: int = 512,
    max_samples: int = None
):
    """Load and tokenize a public dataset."""
    from datasets import load_dataset
    from transformers import AutoTokenizer
    
    print(f"Loading dataset: {dataset_name} ({config})")
    raw_dataset = load_dataset(dataset_name, config, split=split)
    
    if max_samples:
        raw_dataset = raw_dataset.select(range(min(max_samples, len(raw_dataset))))
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize(examples):
        return tokenizer(
            examples["text"] if "text" in examples else examples,
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
    
    tokenized = raw_dataset.map(tokenize, batched=True, remove_columns=raw_dataset.column_names)
    
    input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long)
    # Autoregressive targets: predict the NEXT token. labels[t] = input_ids[t+1];
    # the final column has no successor (-100 = ignored). PAD positions (from
    # padding="max_length") are also ignored so the model isn't trained to emit
    # or evaluated on trivial pad-to-pad predictions.
    labels = torch.full_like(input_ids, -100)
    labels[:, :-1] = input_ids[:, 1:]
    labels = labels.masked_fill(labels == tokenizer.pad_token_id, -100)
    
    return torch.utils.data.TensorDataset(input_ids, labels)