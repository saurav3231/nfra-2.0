"""
Fused time-varying selective scan (the NFRA recurrence kernel).

Computes, for every timestep t and channel:

    h_t = alpha_t * h_{t-1} + gate_t * value_t          (h_{-1} = 0)

with a *per-timestep* decay alpha_t and optional input gate.  This is the
exact recurrence NFRA Brain's BrainMixer runs; Mamba's speed advantage over
pure-PyTorch recursions comes from fusing exactly this into one kernel.

PyTorch implementation (parallel_scan_time_varying) needs ~8 elementwise
passes + 2 cumsums over [B, H, S, Hd].  The Triton kernel does it in ONE
pass: fully parallel over (B*H) heads, each thread-block walks S sequentially
with fp32 accumulation (safe for the clamped decays in [alpha_min, alpha_max]).

Fallback: if Triton or CUDA is unavailable (CPU, dev box) the pure-torch
closed form is used, so behavior is identical everywhere.

Toggles:
  NFRA_SCAN_KERNEL   0 = always torch, 1 = auto (default), 2 = force triton
"""

import os
import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - depends on install
    triton = None
    tl = None
    _HAS_TRITON = False


# --------------------------------------------------------------------------
# reference implementations (for tests + the torch fallback)
# --------------------------------------------------------------------------

def scan_reference(gate, value, alpha, alpha_min=0.75, alpha_max=0.9995):
    """Exact sequential definition. Correct by construction (slow, O(S) ops)."""
    B, H, S, Hd = value.shape
    h = torch.zeros(B, H, 1, Hd, dtype=torch.float32, device=value.device)
    outs = []
    for t in range(S):
        a = alpha[:, :, t, :].clamp(alpha_min, alpha_max).unsqueeze(2)
        u = value[:, :, t, :].unsqueeze(2)
        if gate is not None:
            u = u * gate[:, :, t, :].unsqueeze(2)
        h = a * h + u
        outs.append(h)
    return torch.cat(outs, dim=2)


def _scan_torch(gate, value, alpha, alpha_min, alpha_max):
    """Closed-form via cumulative log-decay (two cumsums), fp32 in the scan."""
    u = value if gate is None else gate * value
    w = torch.log(alpha.clamp(min=alpha_min, max=alpha_max))
    cumlog = torch.cumsum(w, dim=2)
    return torch.exp(cumlog) * torch.cumsum(torch.exp(-cumlog) * u, dim=2)


def _reverse_cumsum(x, dim=2):
    return torch.flip(torch.cumsum(torch.flip(x, dims=[dim]), dim=dim), dims=[dim])


# --------------------------------------------------------------------------
# Triton kernels
# --------------------------------------------------------------------------

def _use_triton():
    mode = os.environ.get('NFRA_SCAN_KERNEL', '1')
    if mode == '0':
        return False
    if mode == '2':
        if not (_HAS_TRITON and torch.cuda.is_available()):
            raise RuntimeError(
                'NFRA_SCAN_KERNEL=2 but Triton/CUDA unavailable')
        return True
    return bool(_HAS_TRITON and torch.cuda.is_available())


def _scan_triton_unavailable(*args, **kwargs):
    raise RuntimeError('Triton not available; cannot run the CUDA scan kernel')


_scan_triton = _scan_triton_unavailable


if _HAS_TRITON:

    @triton.jit
    def _scan_fwd_kernel(
        value_ptr, gate_ptr, alpha_ptr, out_ptr,
        S,
        HEAD_DIM: tl.constexpr,
        HAS_GATE: tl.constexpr,
        ALPHA_MIN: tl.constexpr,
        ALPHA_MAX: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)          # index over (B * H)
        hd = tl.arange(0, HEAD_DIM)
        hd_mask = hd < HEAD_DIM
        base = pid * S * HEAD_DIM
        h = tl.zeros([HEAD_DIM], dtype=tl.float32)
        for t in range(S):                           # dynamic loop over sequence
            offs = base + t * HEAD_DIM + hd
            a = tl.load(alpha_ptr + offs, mask=hd_mask, other=1.0).to(tl.float32)
            a = tl.minimum(tl.maximum(a, ALPHA_MIN), ALPHA_MAX)
            v = tl.load(value_ptr + offs, mask=hd_mask, other=0.0).to(tl.float32)
            if HAS_GATE:
                g = tl.load(gate_ptr + offs, mask=hd_mask, other=0.0).to(tl.float32)
                v = v * g
            h = a * h + v
            tl.store(out_ptr + offs, h, mask=hd_mask)

    def _scan_triton(gate, value, alpha, alpha_min, alpha_max):
        B, H, S, Hd = value.shape
        out = torch.empty((B, H, S, Hd), dtype=torch.float32, device=value.device)
        head_dim = triton.next_power_of_2(Hd)
        gate_ptr = gate if gate is not None else value  # unused when HAS_GATE=0
        _scan_fwd_kernel[(B * H,)](
            value, gate_ptr, alpha, out, S,
            HEAD_DIM=head_dim,
            HAS_GATE=gate is not None,
            ALPHA_MIN=alpha_min,
            ALPHA_MAX=alpha_max,
            num_warps=1,
        )
        return out


class ScanFunction(torch.autograd.Function):
    """Fused forward (Triton) + closed-form backward (torch, exact)."""

    @staticmethod
    def forward(ctx, gate, value, alpha, alpha_min, alpha_max):
        if gate is not None:
            gate = gate.contiguous()
        value = value.contiguous()
        alpha = alpha.contiguous()
        if _use_triton() and value.is_cuda:
            out = _scan_triton(gate, value, alpha, alpha_min, alpha_max)
        else:
            out = _scan_torch(gate, value, alpha, alpha_min, alpha_max)
        ctx.save_for_backward(gate, value, alpha, out)
        ctx.alpha_min = alpha_min
        ctx.alpha_max = alpha_max
        return out

    @staticmethod
    def backward(ctx, dh):
        gate, value, alpha, h = ctx.saved_tensors
        alpha_min, alpha_max = ctx.alpha_min, ctx.alpha_max
        a = alpha.clamp(min=alpha_min, max=alpha_max)
        C = torch.cumsum(a.clamp(min=1e-6).log(), dim=2)
        eC = torch.exp(C)
        dh = dh.contiguous().to(torch.float32)

        # du_t = sum_{k>=t} dh_k * prod_{j=t+1..k} a_j  (reverse scan)
        #      = exp(-C_t) * reverse_cumsum(dh * exp(C))_t
        R = _reverse_cumsum(dh * eC, dim=2)
        du = torch.exp(-C) * R

        dvalue = du * gate if gate is not None else du
        dgate = du * value if gate is not None else None

        # da_t = h_{t-1} * du_t  (gradient of alpha through the recurrence)
        h_prev = torch.cat(
            [torch.zeros_like(h[:, :, :1, :]), h[:, :, :-1, :]], dim=2
        ).to(torch.float32)
        da = du * h_prev
        da = torch.where(alpha == a, da, torch.zeros_like(da))

        return dgate, dvalue, da, None, None


def selective_scan(gate, value, alpha, alpha_min=0.75, alpha_max=0.9995):
    """Fused time-varying selective scan with automatic fallback."""
    return ScanFunction.apply(gate, value, alpha, alpha_min, alpha_max)


def parallel_scan_time_varying(
    gate, value, alpha, alpha_min=0.75, alpha_max=0.9995,
):
    """Public entry point (kept for compatibility with resonance.core)."""
    return selective_scan(gate, value, alpha, alpha_min, alpha_max)


def benchmark(seq_lens=(128, 256, 512), heads=(8, 16, 32),
              head_dim=16, batch=8, repeats=20, warmup=3):
    """Time triton vs torch scan on CUDA. Returns dict + prints a table."""
    if not torch.cuda.is_available():
        raise RuntimeError('benchmark requires CUDA')
    import time

    print('scan kernel benchmark (triton vs torch closed-form), '
          'fp32 accumulation')
    rows = []
    for S in seq_lens:
        for H in heads:
            Hd = head_dim
            B = batch
            value = torch.randn(B, H, S, Hd, device='cuda')
            gate = torch.sigmoid(torch.randn(B, H, S, Hd, device='cuda'))
            alpha = torch.rand(B, H, S, Hd, device='cuda') * 0.2 + 0.75
            ref = scan_reference(gate, value, alpha)
            tri = _scan_triton(gate, value, alpha, 0.75, 0.9995)
            torch_to = _scan_torch(gate, value, alpha, 0.75, 0.9995)
            assert torch.allclose(tri, ref, atol=1e-3, rtol=1e-3), 'mismatch'
            assert torch.allclose(torch_to, ref, atol=1e-4, rtol=1e-4)

            for _ in range(warmup):
                _scan_triton(gate, value, alpha, 0.75, 0.9995)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(repeats):
                _scan_triton(gate, value, alpha, 0.75, 0.9995)
            torch.cuda.synchronize()
            t_tri = (time.perf_counter() - t0) / repeats * 1e6

            for _ in range(warmup):
                _scan_torch(gate, value, alpha, 0.75, 0.9995)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(repeats):
                _scan_torch(gate, value, alpha, 0.75, 0.9995)
            torch.cuda.synchronize()
            t_torch = (time.perf_counter() - t0) / repeats * 1e6

            speedup = t_torch / max(t_tri, 1e-6)
            rows.append((S, H, round(t_torch, 1), round(t_tri, 1),
                         round(speedup, 2)))
            print('S=%4d H=%2d Hd=%2d | torch %8.1fus | triton %7.1fus | '
                  'x%.2f' % (S, H, Hd, t_torch, t_tri, speedup))
    return rows


if __name__ == '__main__':
    benchmark()
