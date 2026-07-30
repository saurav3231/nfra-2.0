"""
Train NFRA Lite on your hardware

This script trains a small NFRA Lite model.

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
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from nfra.models import create_nfra_lite

def main():
    print("=" * 60)
    print("Training NFRA Lite")
    print("=" * 60)
    
    # Load tiny dataset
    print("\nLoading tiny dataset...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:3000]")
    
    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")
    
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    input_ids = torch.tensor(tokenized["input_ids"])
    labels = input_ids.clone()
    
    train_dataset = TensorDataset(input_ids, labels)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    print(f"Dataset loaded: {len(train_dataset)} samples")
    
    # Create model
    print("\nCreating NFRA Lite model...")
    model = create_nfra_lite()
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    print("\nStarting training...")
    start_time = time.time()
    
    for epoch in range(3):
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
        print(f"Epoch {epoch+1}/3 | Loss: {avg_loss:.4f}")
    
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time:.1f} seconds")
    print(f"Final Loss: {avg_loss:.4f}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Save model
    torch.save(model.state_dict(), "nfra_lite_trained.pth")
    print("\nModel saved as 'nfra_lite_trained.pth'")

if __name__ == "__main__":
    main()