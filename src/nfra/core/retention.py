"""Shared retention math: the eager chunked form of the decayed-QK^T operator.

Standalone reference so both CortexMixer (eager fallback) and the fused Triton
kernel (retention_triton) call the exact same implementation — the Triton
forward and its checkpoint-recompute backward are validated against this.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def chunked_retention_eager(q, k, v, log_decay, chunk_size):
    """Exact chunked retention (standalone reference).

    The decayed causal attention y_h = ((Q_h . K_h^T / sqrt(Hd)) . D_h) @ V_h,
    D_h[i,j] = gamma_h^(i-j) for j <= i, computed as within-chunk quadratic
    attention + a cross-chunk linear state instead of the O(S^2) parallel form:

        y_local_i = sum_{j<=i in chunk} gamma^(i-j)  (Q_i.K_j^T / sqrt Hd) V_j
        y_cross_i = gamma^(i+1) (Q_i / sqrt Hd) R
        R_new     = gamma^C R + sum_{j in chunk} gamma^(C-1-j) V_j K_j^T

    Tail positions are zero-padded to a chunk multiple: zeros contribute
    nothing to the state and real positions never attend to them (causal local
    mask), so the result is unchanged for odd sequence lengths.

    Args:
        q, k, v: [B, H, S, Hd] input projections.
        log_decay: [H] per-head log decay.
        chunk_size: chunk length C > 0.
    Returns:
        y: [B, H, S, Hd], the same operator the parallel form computes.
    """
    B, H, S, Hd = q.shape
    C = chunk_size
    dtype = q.dtype
    nC = math.ceil(S / C)
    pad = nC * C - S
    if pad:
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
    q = q.view(B, H, nC, C, Hd)
    k = k.view(B, H, nC, C, Hd)
    v = v.view(B, H, nC, C, Hd)
    l = log_decay.to(dtype).view(1, H, 1, 1)  # [1,H,1,1]

    idx = torch.arange(C, device=q.device)
    rel = (idx.view(C, 1) - idx.view(1, C)).clamp(min=0).to(dtype)  # [C,C]
    causal = torch.triu(torch.ones(C, C, device=q.device, dtype=torch.bool), 1)
    decay = torch.exp(-torch.exp(l) * rel.view(1, 1, C, C)).masked_fill(
        causal.view(1, 1, C, C), 0.0
    )  # [1,H,C,C]

    qs = q * (Hd**-0.5)
    scores = torch.matmul(qs, k.transpose(-2, -1))  # [B,H,nC,C,C]
    y = torch.matmul(scores * decay.unsqueeze(2), v)  # [B,H,nC,C,Hd]

    pos_out = torch.arange(1, C + 1, device=q.device, dtype=dtype).view(1, 1, C, 1)
    dec_out = torch.exp(-torch.exp(l) * pos_out)  # [1,H,C,1]
    pos_carry = torch.arange(C - 1, -1, -1, device=q.device, dtype=dtype).view(
        1, 1, C, 1
    )
    dec_carry = torch.exp(-torch.exp(l) * pos_carry)  # [1,H,C,1]
    gC = torch.exp(-torch.exp(l) * C)  # [1,H,1,1]

    state = q.new_zeros(B, H, Hd, Hd)
    cross = []
    for c in range(nC):
        cross.append(torch.matmul(qs[:, :, c] * dec_out, state))
        state = state * gC + torch.matmul(
            (k[:, :, c] * dec_carry).transpose(-2, -1), v[:, :, c]
        )
    y = y + torch.stack(cross, dim=2)
    return y.view(B, H, nC * C, Hd)[:, :, :S, :]
