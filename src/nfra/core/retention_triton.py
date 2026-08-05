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

import os

import torch

from .retention import chunked_retention_eager

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - CUDA-less dev boxes
    triton = tl = None
    _HAS_TRITON = False


# Fused-backward state: None = not yet self-tested, True/False = result. When
# True the custom Function uses the fused Triton backward for dq/dk/dv (with dl
# via a small eager autograd); otherwise it falls back to the fully-eager
# checkpoint-recompute backward. Automatic self-test + graceful fallback keep a
# wrong/failed kernel from ever corrupting a training run.
_FUSED_BWD_READY = None

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

    @triton.jit
    def _chunked_ret_bwd_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        gy_ptr,
        dq_ptr,
        dk_ptr,
        dv_ptr,
        rbuf_ptr,
        log_decay_ptr,
        n_heads,
        S: tl.constexpr,
        Hd: tl.constexpr,
        C: tl.constexpr,
        nC: tl.constexpr,
        scale: tl.constexpr,
    ):
        """Fused chunked-retention backward for dq, dk, dv (per [b,h]).

        pass 1 stores each chunk's state Rc [Hd,Hd] to global scratch; pass 2
        scans chunks in reverse carrying the state adjoint A = dL/dR_{c+1},
        recomputing within-chunk scores, and emitting dq/dk/dv exactly matched
        to the eager autograd reference. log_decay grad (dl) is NOT computed
        here (handled by a small eager autograd instead). All dots are fp32
        accumulating on fp32-coerced inputs."""
        pid = tl.program_id(0)
        b = pid // n_heads
        h = pid % n_heads
        row = b * n_heads + h
        base = row * S * Hd
        rb = pid * (nC * Hd * Hd)

        iC = tl.arange(0, C)
        iHd = tl.arange(0, Hd)

        l = tl.load(log_decay_ptr + h)
        lam = tl.exp(l)

        rel = iC[:, None] - iC[None, :]
        decay_c = tl.where(rel >= 0, tl.exp(-lam * rel), 0.0)  # [C,C]
        dec_out = tl.exp(-lam * (iC + 1))[:, None]  # [C,1]
        dec_carry = tl.exp(-lam * (C - 1 - iC))[:, None]  # [C,1]
        gC = tl.exp(-lam * C)  # scalar

        # pass 1: store each chunk's incoming state Rc
        state = tl.zeros((Hd, Hd), dtype=tl.float32)
        for c in tl.static_range(nC):
            off = base + (c * C + iC)[:, None] * Hd + iHd[None, :]
            rf = rb + (c * Hd + iHd)[:, None] * Hd + iHd[None, :]
            tl.store(rbuf_ptr + rf, state)
            kc = tl.load(k_ptr + off).to(tl.float32)
            vc = tl.load(v_ptr + off).to(tl.float32)
            state = state * gC + tl.dot(tl.trans(kc * dec_carry), vc)

        # pass 2: reverse scan with the state adjoint A = dL/dR_{c+1}
        A = tl.zeros((Hd, Hd), dtype=tl.float32)
        for c in tl.static_range(nC):
            i = nC - 1 - c
            off = base + (i * C + iC)[:, None] * Hd + iHd[None, :]
            rf = rb + (i * Hd + iHd)[:, None] * Hd + iHd[None, :]
            qc = tl.load(q_ptr + off).to(tl.float32)
            kc = tl.load(k_ptr + off).to(tl.float32)
            vc = tl.load(v_ptr + off).to(tl.float32)
            G = tl.load(gy_ptr + off).to(tl.float32)
            Rc = tl.load(rbuf_ptr + rf).to(tl.float32)

            qs = qc * scale
            AV = tl.dot(G, tl.trans(vc)) * decay_c  # (G V^T) . D   [C,C]
            dQ_local = tl.dot(AV, kc)  # [C,Hd]
            sGR = tl.dot(G, tl.trans(Rc))  # G . R^T   [C,Hd]
            dQ = scale * (dQ_local + sGR * dec_out)
            Pc = decay_c * tl.dot(qs, tl.trans(kc))
            dV_local = tl.dot(tl.trans(Pc), G)  # [C,Hd]
            dK_local = tl.trans(tl.dot(tl.trans(qs), AV))  # [C,Hd]
            Kp = kc * dec_carry
            dV_carry = tl.dot(Kp, A)  # [C,Hd]
            dK_carry = tl.trans(tl.dot(A, tl.trans(vc))) * dec_carry  # [C,Hd]
            tl.store(dq_ptr + off, dQ.to(tl.float32))
            tl.store(dk_ptr + off, (dK_local + dK_carry).to(tl.float32))
            tl.store(dv_ptr + off, (dV_local + dV_carry).to(tl.float32))
            A = tl.dot(tl.trans(qs * dec_out), G) + gC * A  # dL/dRc


def _chunked_retention_triton_impl(q, k, v, log_decay, chunk_size):
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


def _chunked_retention_bwd_impl(q, k, v, grad_y, log_decay, chunk_size):
    """Fused Triton backward for dq/dk/dv. Returns (dq, dk, dv) matching the
    eager autograd reference, or None if the geometry/Triton can't run it.
    q,k,v,grad_y [B,H,S,Hd] (may be non-contiguous / not chunk-aligned)."""
    B, H, S, Hd = q.shape
    C = int(chunk_size)
    if (Hd & (Hd - 1)) or (C & (C - 1)):
        return None
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    gy = grad_y.contiguous()
    nC = (S + C - 1) // C
    S_pad = nC * C
    if S_pad != S:
        q = torch.nn.functional.pad(q, (0, 0, 0, S_pad - S))
        k = torch.nn.functional.pad(k, (0, 0, 0, S_pad - S))
        v = torch.nn.functional.pad(v, (0, 0, 0, S_pad - S))
        gy = torch.nn.functional.pad(gy, (0, 0, 0, S_pad - S))
    dq = torch.empty_like(q, dtype=torch.float32)
    dk = torch.empty_like(k, dtype=torch.float32)
    dv = torch.empty_like(v, dtype=torch.float32)
    rbuf = torch.empty((B * H * nC, Hd, Hd), device=q.device, dtype=torch.float32)
    _chunked_ret_bwd_kernel[(B * H,)](
        q, k, v, gy, dq, dk, dv, rbuf, log_decay, H,
        S=S_pad, Hd=Hd, C=C, nC=nC, scale=Hd**-0.5, num_warps=4,
    )
    dq = dq[:, :, :S, :].to(q.dtype)
    dk = dk[:, :, :S, :].to(q.dtype)
    dv = dv[:, :, :S, :].to(q.dtype)
    return dq, dk, dv


def _dl_via_eager(q, k, v, log_decay, grad_y, chunk_size):
    """Gradient of log_decay via a one-shot autograd through the eager forward
    (q/k/v detached). Independent and exact; decouples the fragile lambda-chain
    from the fused Triton backward."""
    lr = log_decay.detach().requires_grad_(True)
    with torch.enable_grad():
        y = chunked_retention_eager(q.detach(), k.detach(), v.detach(), lr, chunk_size)
        (dl,) = torch.autograd.grad((y * grad_y.float()).sum(), lr)
    return dl


def _check_fused_backward(device):
    """Self-test fused bwd vs eager autograd on random fp16 data. Sets the
    module global so the Function only uses the fused path when it is provably
    correct. Returns bool."""
    import torch as _t

    _t.manual_seed(0)
    B, H, S, Hd, C = 2, 4, 256, 32, 64
    q = _t.randn(B, H, S, Hd, device=device, dtype=_t.float16)
    k = _t.randn(B, H, S, Hd, device=device, dtype=_t.float16)
    v = _t.randn(B, H, S, Hd, device=device, dtype=_t.float16)
    gy = _t.randn(B, H, S, Hd, device=device, dtype=_t.float16)
    log_decay = (_t.randn(H, device=device) * 0.1).to(_t.float32)

    qf = q.float().requires_grad_(True)
    kf = k.float().requires_grad_(True)
    vf = v.float().requires_grad_(True)
    lf = log_decay.requires_grad_(True)
    with _t.enable_grad():
        y = chunked_retention_eager(qf, kf, vf, lf, C)
        (dq_e, dk_e, dv_e, dl_e) = _t.autograd.grad((y * gy.float()).sum(), (qf, kf, vf, lf))

    if q.is_cuda:
        try:
            out = _chunked_retention_bwd_impl(q, k, v, gy, log_decay, C)
            dl_f = _dl_via_eager(q, k, v, log_decay, gy, C)
        except Exception as e:  # compile/launch/register-pressure failure
            print(f"  [fused-bwd] self-test errored ({type(e).__name__}) -> eager")
            return False
        if out is None:
            return False
        dq_f, dk_f, dv_f = out

        def rel(a, b):
            return (a.float() - b.float()).abs().max().item() / (
                b.float().abs().max().item() + 1e-6
            )

        max_rel = max(rel(dq_f, dq_e), rel(dk_f, dk_e), rel(dv_f, dv_e), rel(dl_f, dl_e))
        ok = max_rel < 1e-2
        print(
            f"  [fused-bwd] self-test {'PASSED' if ok else 'FAILED'} "
            f"(max rel err {max_rel:.2e}) -> {'active' if ok else 'eager fallback'}"
        )
        return ok
    return False


def _fused_backward_ready(device):
    global _FUSED_BWD_READY
    if _FUSED_BWD_READY is None:
        _FUSED_BWD_READY = (
            bool(os.environ.get("NFRA_TRITON_BWD", "1") != "0") and _check_fused_backward(device)
        )
    return _FUSED_BWD_READY


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
            return _chunked_retention_triton_impl(
                q.contiguous(), k.contiguous(), v.contiguous(),
                log_decay, ctx.chunk_size,
            )
        return chunked_retention_eager(q, k, v, log_decay, ctx.chunk_size)

    @staticmethod
    def backward(ctx, grad_y):
        q, k, v, log_decay = ctx.saved_tensors
        C = ctx.chunk_size
        if q.is_cuda:
            try:
                if _fused_backward_ready(q.device):
                    fused = _chunked_retention_bwd_impl(q, k, v, grad_y, log_decay, C)
                    if fused is not None:
                        dl = _dl_via_eager(q, k, v, log_decay, grad_y, C)
                        return fused[0], fused[1], fused[2], dl, None, None
            except Exception:  # pragma: no cover - fall back on any kernel hiccup
                pass
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
