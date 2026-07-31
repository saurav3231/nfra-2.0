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
        self.state_proj = nn.Linear(dim, state_dim)
        self.state_out = nn.Linear(state_dim, dim, bias=False)
        
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
        Forward pass with selective scanning (parallel closed-form scan).
        """
        batch, seq_len, dim = x.shape
        
        # Input projection
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        
        # Compute delta (discretization step)
        delta = torch.sigmoid(self.dt_proj2(F.silu(self.dt_proj(x))))
        
        # Stable state-space decay: alpha_t = exp(-exp(A_log) * delta_t) in (0,1)
        A = -torch.exp(self.A_log.clamp(max=4.0))                    # [state_dim]
        alpha = torch.exp(A.view(1, 1, 1, self.state_dim)
                          * delta.view(batch, 1, seq_len, 1))        # [B, 1, S, state_dim]
        
        # Selection gates (sigmoid so gates stay in (0,1))
        gate = torch.sigmoid(self.selection(x))                      # [B, S, state_dim]
        value = self.state_proj(x)                                   # [B, S, state_dim]
        
        # Parallel scan h_t = alpha_t * h_{t-1} + gate_t * value_t
        from .resonance import parallel_scan_time_varying
        h = parallel_scan_time_varying(
            gate.view(batch, 1, seq_len, self.state_dim),
            value.view(batch, 1, seq_len, self.state_dim),
            alpha,
        )                                                            # [B, 1, S, state_dim]
        h = h.view(batch, seq_len, self.state_dim)
        
        y = self.state_out(h) + self.D.unsqueeze(0).unsqueeze(0) * x
        
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
        
        # Batched top-k routing: run each expert once on the samples routed to it,
        # instead of a per-batch Python loop.
        output = torch.zeros_like(x)
        flat_idx = torch.arange(batch * seq, device=x.device).view(batch, seq)
        
        for i in range(self.top_k):
            expert_idx = topk_indices[:, i]                 # [B]
            weight = topk_scores[:, i]                      # [B]
            
            for e in range(self.num_experts):
                mask = (expert_idx == e)                    # [B]
                if not mask.any():
                    continue
                rows = flat_idx[mask].reshape(-1)           # all tokens of routed samples
                expert_out = self.experts[e](x[mask]).view(-1, dim)   # [Bsel*S, D]
                w_flat = (weight[mask].view(-1, 1, 1, 1)
                          .expand(-1, seq, -1, -1).reshape(-1, 1))   # [Bsel*S, 1]
                output.view(-1, dim)[rows] += w_flat * expert_out
        
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