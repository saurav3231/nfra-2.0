"""
Advanced Resonance Mechanisms for NFRA 2.0 (2026 Edition)

This version includes improved, more stable, and powerful components.

Created by Saurav Bhandari
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SelectiveResonanceScanner(nn.Module):
    """
    Improved Selective State Space Scanner (inspired by Mamba/SSM).
    
    This version is more stable and expressive than the original simplified version.
    It uses a linear state-space model with learned discretization.
    """
    
    def __init__(self, dim: int, state_dim: int = 32, dt_rank: int = 16):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        
        # Projection layers
        self.in_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        
        # State space parameters
        self.A_log = nn.Parameter(torch.randn(state_dim))
        self.D = nn.Parameter(torch.ones(dim))
        
        # Discretization
        self.dt_proj = nn.Linear(dim, dt_rank)
        self.dt_proj2 = nn.Linear(dt_rank, 1)
        
        # Selection
        self.selection = nn.Linear(dim, state_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with selective scanning.
        """
        batch, seq_len, dim = x.shape
        
        # Input projection
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        
        # Compute delta (discretization step)
        delta = torch.sigmoid(self.dt_proj2(F.silu(self.dt_proj(x))))
        
        # State space scan (simplified but stable)
        A = -torch.exp(self.A_log)  # Negative for stability
        
        h = torch.zeros(batch, self.state_dim, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            # Selective update
            h = h * torch.exp(A * delta[:, t]) + x[:, t].unsqueeze(-1) * self.selection(x[:, t])
            y = (h @ self.out_proj.weight.T[:, :self.state_dim].T) + self.D * x[:, t]
            outputs.append(y)
            
        y = torch.stack(outputs, dim=1)
        
        # Gating with z
        y = y * F.silu(z)
        
        return self.out_proj(y)


class MixtureOfFractals(nn.Module):
    """
    Efficient Mixture-of-Fractals with proper batched top-k routing.
    Note: This component is disabled in NFRA Lite for performance reasons.
    """
    
    def __init__(self, dim: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim)
            ) for _ in range(num_experts)
        ])
        
        self.gate = nn.Linear(dim, num_experts)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.shape
        
        router_logits = self.gate(x.mean(dim=1))
        scores = F.softmax(router_logits, dim=-1)
        
        topk_scores, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        topk_scores = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-8)
        
        output = torch.zeros_like(x)
        
        for i in range(self.top_k):
            expert_idx = topk_indices[:, i]
            weight = topk_scores[:, i].unsqueeze(-1).unsqueeze(-1)
            
            expert_outputs = []
            for b in range(batch):
                expert_out = self.experts[expert_idx[b]](x[b:b+1])
                expert_outputs.append(expert_out)
            
            expert_stack = torch.cat(expert_outputs, dim=0)
            output = output + weight * expert_stack
                
        return output


class DynamicPrecisionRouter(nn.Module):
    """
    Dynamically routes computation to different precision levels.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.importance = nn.Linear(dim, 3)
        
    def forward(self, x: torch.Tensor, hardware_capability: float = 1.0):
        scores = torch.sigmoid(self.importance(x.mean(dim=1)))
        return scores * hardware_capability