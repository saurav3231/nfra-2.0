"""
NFRA Trainer class with support for real public datasets

Created by Saurav Bhandari
"""

import torch
from torch.optim import AdamW
from tqdm import tqdm
from typing import Optional, Dict, Any
from torch.utils.data import DataLoader


class NFRATrainer:
    """
    Professional trainer for NFRA 2.0 models.
    Supports real public datasets (WikiText, C4, etc.).
    """
    
    def __init__(
        self,
        model,
        loss_fn,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        device: str = "cpu",
        energy_aware: bool = True,
        max_grad_norm: float = 1.0,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device
        self.energy_aware = energy_aware
        self.max_grad_norm = max_grad_norm
        
        self.optimizer = AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        self.global_step = 0
        
    def train_step(
        self, 
        input_ids: torch.Tensor, 
        targets: torch.Tensor,
        energy_budget: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Single training step with real data.
        """
        self.model.train()
        input_ids = input_ids.to(self.device)
        targets = targets.to(self.device)
        
        # Forward pass
        outputs = self.model(input_ids, energy_budget=energy_budget)
        logits = outputs["logits"]
        
        # Collect auxiliary stats from FractalResonanceBlocks
        resonance_stats = None
        energy_used = None
        for module in self.model.modules():
            if hasattr(module, 'get_sparsity'):
                sparsity = module.get_sparsity()
                if sparsity > 0:
                    resonance_stats = {"sparsity": sparsity}
                    break
        
        if energy_budget is not None:
            energy_used = energy_budget
        
        # Compute NFRA loss
        loss, loss_dict = self.loss_fn(
            logits=logits,
            targets=targets,
            resonance_stats=resonance_stats,
            energy_used=energy_used,
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        self.global_step += 1
        
        return loss_dict
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on a validation set.
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            targets = batch["labels"].to(self.device)
            
            outputs = self.model(input_ids)
            logits = outputs["logits"]
            
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1),
                reduction="sum"
            )
            
            total_loss += loss.item()
            total_tokens += input_ids.numel()
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            "eval_loss": round(avg_loss, 4),
            "perplexity": round(perplexity, 2)
        }
    
    def save_checkpoint(self, path: str, epoch: int = 0):
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "global_step": self.global_step,
        }, path)
        print(f"Checkpoint saved to {path}")