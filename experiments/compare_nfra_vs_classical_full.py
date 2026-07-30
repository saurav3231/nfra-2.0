#!/usr/bin/env python3
"""
NFRA Lite vs Classical Transformer - Full Comparison

This single script:
1. Trains NFRA Lite
2. Trains a Classical Transformer
3. Compares both models on your hardware

Created by Saurav Bhandari
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from transformers import AutoTokenizer
import time
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from nfra.models import create_nfra_lite


def get_dataset(max_samples=3000, max_length=128):
    print("Loading dataset...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Use tiny_shakespeare instead of wikitext (more reliable)
    try:
        dataset = load_dataset("tiny_shakespeare", split="train")
    except:
        # Fallback to a simple text dataset
        print("Using simple text dataset...")
        texts = ["hello world " * 50] * max_samples
        from datasets import Dataset
        dataset = Dataset.from_dict({"text": texts})
    
    # Take only needed samples
    if len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))
    
    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length, padding="max_length")
    
    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    input_ids = torch.tensor(tokenized["input_ids"])
    labels = input_ids.clone()
    
    return TensorDataset(input_ids, labels), tokenizer


def train_nfra_lite(train_loader, epochs=3):
    print("\n" + "="*50)
    print("Training NFRA Lite")
    print("="*50)
    
    model = create_nfra_lite()
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            input_ids, labels = batch
            optimizer.zero_grad()
            
            outputs = model(input_ids, energy_budget=0.6)
            loss = criterion(outputs["logits"].view(-1, outputs["logits"].size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
    
    training_time = time.time() - start_time
    return model, training_time, avg_loss


def train_classical(train_loader, epochs=3):
    print("\n" + "="*50)
    print("Training Classical Transformer")
    print("="*50)
    
    vocab_size = 50257
    hidden_size = 384
    num_layers = 8
    
    embedding = nn.Embedding(vocab_size, hidden_size)
    transformer = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model=hidden_size, nhead=6, batch_first=True),
        num_layers=num_layers
    )
    lm_head = nn.Linear(hidden_size, vocab_size)
    lm_head.weight = embedding.weight
    
    params = list(embedding.parameters()) + list(transformer.parameters()) + list(lm_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            input_ids, labels = batch
            optimizer.zero_grad()
            
            emb = embedding(input_ids)
            out = transformer(emb)
            logits = lm_head(out)
            
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
    
    training_time = time.time() - start_time
    return embedding, transformer, lm_head, training_time, avg_loss, params


def compare_inference(nfra_model, embedding, transformer, lm_head):
    print("\n" + "="*50)
    print("INFERENCE COMPARISON")
    print("="*50)
    
    input_ids = torch.randint(0, 50257, (1, 128))
    
    # NFRA Lite
    nfra_model.eval()
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            out = nfra_model(input_ids, energy_budget=0.5)
    nfra_time = (time.time() - start) / 10 * 1000
    
    # Classical
    embedding.eval()
    transformer.eval()
    lm_head.eval()
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            emb = embedding(input_ids)
            out = transformer(emb)
            logits = lm_head(out)
    classical_time = (time.time() - start) / 10 * 1000
    
    print(f"\nNFRA Lite:")
    print(f"  Parameters: {sum(p.numel() for p in nfra_model.parameters()):,}")
    print(f"  Avg Inference Time: {nfra_time:.2f} ms")
    
    total_classical = sum(p.numel() for p in embedding.parameters()) + \
                      sum(p.numel() for p in transformer.parameters()) + \
                      sum(p.numel() for p in lm_head.parameters())
    
    print(f"\nClassical Transformer:")
    print(f"  Parameters: {total_classical:,}")
    print(f"  Avg Inference Time: {classical_time:.2f} ms")
    
    print(f"\nSpeed Difference: {classical_time/nfra_time:.2f}x")
    
    print("\n" + "="*50)
    print("KEY ADVANTAGES OF NFRA LITE")
    print("="*50)
    print("✓ Built-in energy awareness (energy_budget parameter)")
    print("✓ High sparsity during inference")
    print("✓ Designed for old hardware (i5-337U)")
    print("✓ Predictive coding reduces computation")
    print("✓ Fractal structure for efficient processing")


def main():
    print("=" * 60)
    print("NFRA Lite vs Classical Transformer - Full Comparison")
    print("=" * 60)
    
    # Load dataset
    train_dataset, tokenizer = get_dataset(max_samples=3000)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    print(f"\nDataset: {len(train_dataset)} samples")
    
    # Train both models
    nfra_model, nfra_time, nfra_loss = train_nfra_lite(train_loader, epochs=3)
    embedding, transformer, lm_head, classical_time, classical_loss, classical_params = train_classical(train_loader, epochs=3)
    
    # Final comparison
    print("\n" + "="*60)
    print("TRAINING RESULTS")
    print("="*60)
    print(f"\nNFRA Lite:")
    print(f"  Training Time: {nfra_time:.1f} seconds")
    print(f"  Final Loss:    {nfra_loss:.4f}")
    
    print(f"\nClassical Transformer:")
    print(f"  Training Time: {classical_time:.1f} seconds")
    print(f"  Final Loss:    {classical_loss:.4f}")
    
    # Inference comparison
    compare_inference(nfra_model, embedding, transformer, lm_head)
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()