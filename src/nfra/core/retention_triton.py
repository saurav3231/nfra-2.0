"""Fused Triton chunked-retention kernel (Tier-1 speed/memory path).

The eager chunked path runs one Python loop per chunk with four small GEMMs
per block — 33 blocks x 4-5 chunks of tiny launches, the measured bottleneck.
This module fuses the WHOLE retention for one (batch, head) into a SINGLE
Triton kernel: one program per (b, h) walks the chunks sequentially, keeping
the linear state R [Hd, Hd] in registers across the loop.

Forward math is the exact chunked operator (see retention.chunked_retention_eager)
computed fully in fp32 in-kernel:

    within-chunk  scores = (Qc/sqrt(Hd)) Kc^T                       [C,C]
    local         y_local = scores . D_local @ Vc                   [C,Hd]
    cross         y_cross = (Qc/sqrt(Hd) . gamma^(i+1)) @ R
    state         R'      = gamma^C R + (Kc . gamma^(C-1-j))^T @ Vc

Backward uses checkpoint-recompute semantics: the custom autograd Function
saves only the inputs and recomputes the eager chunked retention inside
backward under autograd, so (a) gradients are exactly the validated eager
chunked gradients (forward/backward agree to fp32-kernel rounding), (b) peak
memory stays at the chunked level — no O(S^2) [B,H,S,S] activation anywhere —
and (c) no second custom Triton backward kernel is needed.
"""

from __future__ import annotations

import torch

from .retention import chunked_retention_eager

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - CUDA-less dev boxes
    triton = tl = None
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _chunked_ret_fwd_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        y_ptr,
        log_decay_ptr,
        n_heads,
        S: tl.constexpr,
        Hd: tl.constexpr,
        C: tl.constexpr,
        nC: tl.constexpr,
        scale: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid // n_heads
        h = pid % n_heads
        row = b * n_heads + h
        base = row * S * Hd

        iC = tl.arange(0, C)
        iHd = tl.arange(0, Hd)

        l = tl.load(log_decay_ptr + h)

        rel = iC[:, None] - iC[None, :]
        decay_c = tl.where(rel >= 0, tl.exp(-tl.exp(l) * rel), 0.0)  # [C,C]

        dec_out = tl.exp(-tl.exp(l) * (iC + 1))[:, None]  # [C,1]
        dec_carry = tl.exp(-tl.exp(l) * (C - 1 - iC))[:, None]  # [C,1]
        gC = tl.exp(-tl.exp(l) * C)  # scalar

        state = tl.zeros((Hd, Hd), dtype=tl.float32)
        for c in tl.static_range(nC):
            off = base + (c * C + iC)[:, None] * Hd + iHd[None, :]
            qc16 = tl.load(q_ptr + off)
            kc16 = tl.load(k_ptr + off)
            vc16 = tl.load(v_ptr + off)
            qc = qc16.to(tl.float32)
            kc = kc16.to(tl.float32)
            vc = vc16.to(tl.float32)

            scores = tl.dot(qc * scale, tl.trans(kc))  # [C,C]
            local = tl.dot(scores * decay_c, vc)  # [C,Hd]
            cross = tl.dot((qc * scale) * dec_out, state)  # [C,Hd]
            yc = local + cross

            tl.store(y_ptr + off, yc.to(qc16.dtype))
            state = state * gC + tl.dot(tl.trans(kc * dec_carry), vc)  # [Hd,Hd]


def _chunked_retention_triton(q, k, v, log_decay, chunk_size):
    """Triton fused chunked retention. q,k,v [B,H,S,Hd], log_decay [H]."""
    B, H, S, Hd = q.shape
    C = int(chunk_size)
    if (Hd & (Hd - 1)) or (C & (C - 1)):
        return chunked_retention_eager(q, k, v, log_decay, C)
    # The kernel does raw pointer arithmetic assuming contiguous [B,H,S,Hd];
    # model q/k/v are permuted views (non-contiguous) and MUST be materialized
    # before launch or the kernel reads wrong memory offsets.
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    nC = (S + C - 1) // C
    S_pad = nC * C
    if S_pad != S:
        q = torch.nn.functional.pad(q, (0, 0, 0, S_pad - S))
        k = torch.nn.functional.pad(k, (0, 0, 0, S_pad - S))
        v = torch.nn.functional.pad(v, (0, 0, 0, S_pad - S))
    y = torch.empty_like(q)
    _chunked_ret_fwd_kernel[(B * H,)](
        q,
        k,
        v,
        y,
        log_decay,
        H,
        S=S_pad,
        Hd=Hd,
        C=C,
        nC=nC,
        scale=Hd**-0.5,
        num_warps=4,
    )
    return y[:, :, :S, :]


class _ChunkedRetentionFn(torch.autograd.Function):
    """autograd wrapper: Triton fused forward (when available) + a
    checkpoint-recompute backward through the eager chunked reference, so
    forward/backward agree to fp32-kernel rounding and peak memory stays at
    the chunked level (no O(S^2) activation)."""

    @staticmethod
    def forward(ctx, q, k, v, log_decay, chunk_size, use_triton):
        ctx.save_for_backward(q, k, v, log_decay)
        ctx.chunk_size = int(chunk_size)
        if use_triton and q.is_cuda and _HAS_TRITON:
            return _chunked_retention_triton(q, k, v, log_decay, ctx.chunk_size)
        return chunked_retention_eager(q, k, v, log_decay, ctx.chunk_size)

    @staticmethod
    def backward(ctx, grad_y):
        q, k, v, log_decay = ctx.saved_tensors
        C = ctx.chunk_size
        with torch.enable_grad():
            qr = q.detach().requires_grad_(True)
            kr = k.detach().requires_grad_(True)
            vr = v.detach().requires_grad_(True)
            lr = log_decay.detach().requires_grad_(True)
            y = chunked_retention_eager(qr, kr, vr, lr, C)
            dq, dk, dv, dl = torch.autograd.grad(
                (y * grad_y).sum(), (qr, kr, vr, lr), allow_unused=True
            )
        return dq, dk, dv, dl, None, None


def chunked_retention(q, k, v, log_decay, chunk_size, use_triton=False):
    """Chunked retention with an optional fused Triton forward.

    q,k,v [B,H,S,Hd], log_decay [H]. Returns [B,H,S,Hd]. When use_triton is
    set and a CUDA device with Triton is available the forward runs as ONE
    fused kernel per (batch, head); otherwise it falls back to the eager
    chunked reference (same math, no correctness risk).
    """
    return _ChunkedRetentionFn.apply(q, k, v, log_decay, chunk_size, use_triton)
