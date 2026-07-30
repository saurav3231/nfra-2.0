#!/usr/bin/env python3
"""
NFRA Lite vs Classical Neural Network Comparison

This script trains a very small NFRA Lite model and a classical Transformer
of similar size, then compares their performance.

Created by Saurav Bhandari
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time
from datasets import load_dataset
from transformers import AutoTokenizer

from nfra.models import create_nfra_lite, NFRAConfig, NFRAForCausalLM


def get_tiny_dataset(tokenizer, max_samples=2000, max_length=128):
    """Load a very small subset of WikiText-2 for quick testing."""
    print("Loading tiny WikiText-2 subset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:2000]")
    
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
    
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    input_ids = torch.tensor(tokenized["input_ids"])
    labels = input_ids.clone()
    
    return TensorDataset(input_ids, labels)


def create_classical_transformer(vocab_size=50257, hidden_size=256, num_layers=4):
    """Create a small classical Transformer for comparison."""
    config = type('Config', (), {
        'vocab_size': vocab_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'max_position_embeddings': 128,
        'dropout': 0.1
    })()
    
    # Simple Transformer decoder
    model = nn.TransformerDecoder(
        nn.TransformerDecoderLayer(d_model=hidden_size, nhead=4, batch_first=True),
        num_layers=num_layers
    )
    
    # Add embedding and output layer
    embed = nn.Embedding(vocab_size, hidden_size)
    lm_head = nn.Linear(hidden_size, vocab_size)
    lm_head.weight = embed.weight
    
    return embed, model, lm_head


def train_nfra_lite(train_loader, epochs=2):
    print("\n=== Training NFRA Lite ===")
    model = create_nfra_lite()
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    start = time.time()
    
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
    
    elapsed = time.time() - start
    return model, elapsed, avg_loss


def main():
    print("=" * 60)
    print("NFRA Lite vs Classical Transformer Comparison")
    print("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load tiny dataset
    dataset = get_tiny_dataset(tokenizer, max_samples=2000)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    print(f"\nDataset size: {len(dataset)} samples")
    
    # === Train NFRA Lite ===
    nfra_model, nfra_time, nfra_loss = train_nfra_lite(train_loader, epochs=2)
    
    # === Classical Transformer (for comparison) ===
    print("\n=== Training Classical Transformer ===")
    embed, transformer, lm_head = create_classical_transformer()
    optimizer = torch.optim.AdamW(list(embed.parameters()) + list(transformer.parameters()) + list(lm_head.parameters()), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    classical_loss = 0.0
    start = time.time()
    for epoch in range(2):
        total_loss = 0
        for batch in train_loader:
            input_ids, labels = batch
            optimizer.zero_grad()
            
            emb = embed(input_ids)
            out = transformer(emb)
            logits = lm_head(out)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        classical_loss = avg_loss
        print(f"Epoch {epoch+1}/2 | Loss: {avg_loss:.4f}")
    
    classical_time = time.time() - start
    
    # === Results ===
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    print(f"\nNFRA Lite:")
    print(f"  Training Time: {nfra_time:.1f} seconds")
    print(f"  Final Loss:    {nfra_loss:.4f}")
    print(f"  Parameters:    {sum(p.numel() for p in nfra_model.parameters()):,}")
    
    print(f"\nClassical Transformer:")
    print(f"  Training Time: {classical_time:.1f} seconds")
    print(f"  Final Loss:    {classical_loss:.4f}")
    
    print(f"\nAdvantage of NFRA Lite:")
    print(f"  - Built-in high sparsity (not present in classical model)")
    print(f"  - Energy-aware computation")
    print(f"  - Designed for old hardware from the ground up")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()