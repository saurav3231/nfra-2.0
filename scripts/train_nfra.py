#!/usr/bin/env python3
"""
NFRA 2.0 Training Script using Public Datasets

This script trains NFRA models on publicly available datasets
(WikiText-2, C4, etc.) in a professional and reproducible way.

Created by Saurav Bhandari
"""

import argparse
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from nfra.models import NFRAConfig, NFRAForCausalLM
from nfra.training import NFRACombinedLoss, NFRATrainer
from nfra.core.advanced_resonance import MixtureOfFractals, SelectiveResonanceScanner


def get_wikitext2(tokenizer, max_length=512, split="train"):
    """Load and tokenize WikiText-2 dataset."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
    
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Convert to tensors
    input_ids = torch.tensor(tokenized["input_ids"])
    labels = input_ids.clone()
    
    return torch.utils.data.TensorDataset(input_ids, labels)


def main():
    parser = argparse.ArgumentParser(description="Train NFRA 2.0")
    parser.add_argument("--model_size", type=str, default="small", 
                        choices=["tiny", "small", "medium"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--energy_budget", type=float, default=0.7)
    parser.add_argument("--dataset", type=str, default="wikitext2")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Model configuration (Advanced 2026 Edition)
    if args.model_size == "tiny":
        config = NFRAConfig(
            hidden_size=256, 
            num_layers=6, 
            vocab_size=50257,
            use_mixture_of_fractals=True,
            use_selective_scanning=True,
            num_fractal_experts=4,
            top_k_experts=2
        )
    elif args.model_size == "small":
        config = NFRAConfig(
            hidden_size=512, 
            num_layers=8, 
            vocab_size=50257,
            use_mixture_of_fractals=True,
            use_selective_scanning=True,
            num_fractal_experts=6,
            top_k_experts=2
        )
    else:
        config = NFRAConfig(
            hidden_size=768, 
            num_layers=12, 
            vocab_size=50257,
            use_mixture_of_fractals=True,
            use_selective_scanning=True,
            num_fractal_experts=8,
            top_k_experts=3
        )

    model = NFRAForCausalLM(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Tokenizer & Dataset
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading WikiText-2 dataset...")
    train_dataset = get_wikitext2(tokenizer, split="train")
    val_dataset = get_wikitext2(tokenizer, split="validation")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Loss and Trainer
    loss_fn = NFRACombinedLoss()
    trainer = NFRATrainer(
        model=model,
        loss_fn=loss_fn,
        learning_rate=args.lr,
        device=device
    )

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...\n")
    
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            loss_dict = trainer.train_step(
                input_ids, 
                labels, 
                energy_budget=args.energy_budget
            )
            epoch_losses.append(loss_dict["total_loss"])
            
            progress_bar.set_postfix({"loss": f"{loss_dict['total_loss']:.4f}"})
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch+1} | Average Loss: {avg_loss:.4f}")

        # Validation
        if epoch % 1 == 0:
            val_metrics = trainer.evaluate(val_loader)
            print(f"Validation → Loss: {val_metrics['eval_loss']:.4f} | Perplexity: {val_metrics['perplexity']:.2f}")

    # Save final model
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_checkpoint(f"{args.output_dir}/nfra_{args.model_size}_final.pt", epoch=args.epochs)
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()