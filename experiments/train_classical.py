"""
Train Classical Transformer on your hardware

This script trains a small classical Transformer for comparison.

Created by Saurav Bhandari
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from transformers import AutoTokenizer
import time

def create_classical_model(vocab_size=50257, hidden_size=384, num_layers=8):
    """Create a small classical Transformer"""
    embedding = nn.Embedding(vocab_size, hidden_size)
    transformer = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model=hidden_size, nhead=6, batch_first=True),
        num_layers=num_layers
    )
    lm_head = nn.Linear(hidden_size, vocab_size)
    lm_head.weight = embedding.weight  # Weight tying
    
    return embedding, transformer, lm_head

def main():
    print("=" * 60)
    print("Training Classical Transformer")
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
    print("\nCreating Classical Transformer...")
    embedding, transformer, lm_head = create_classical_model()
    embedding.train()
    transformer.train()
    lm_head.train()
    
    params = list(embedding.parameters()) + list(transformer.parameters()) + list(lm_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    print("\nStarting training...")
    start_time = time.time()
    
    for epoch in range(3):
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
        print(f"Epoch {epoch+1}/3 | Loss: {avg_loss:.4f}")
    
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time:.1f} seconds")
    print(f"Final Loss: {avg_loss:.4f}")
    print(f"Parameters: {sum(p.numel() for p in params):,}")
    
    # Save model
    torch.save({
        'embedding': embedding.state_dict(),
        'transformer': transformer.state_dict(),
        'lm_head': lm_head.state_dict()
    }, "classical_trained.pth")
    
    print("\nModel saved as 'classical_trained.pth'")

if __name__ == "__main__":
    main()