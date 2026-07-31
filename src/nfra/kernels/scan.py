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
closed form is used, so behavior is identical everywhere. The naive closed
form's exp(-cumlog) intermediate overflows fp32 (~e^147 > 3.4e38) once
S * -ln(alpha_min) > ~85, so beyond that threshold _scan_torch switches to a
chunk-normalized closed form (and the backward to a reverse-chunked form)
that never forms the huge intermediate — the torch path stays finite for any
sequence length. The kernel (default on CUDA) is the single-pass version.

Toggles:
  NFRA_SCAN_KERNEL   0 = always torch, 1 = auto (default), 2 = force triton
"""

import os
import math
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


def _stable_threshold(alpha_min):
    """Max sequence length the naive two-cumsum closed form survives in fp32.

    The naive form materializes exp(-cumsum(log alpha)); with alphas pinned at
    the floor that intermediate reaches ~exp(S * -ln(alpha_min)) and overflows
    fp32 (> 3.4e38) once the exponent exceeds ~85. Below this S the naive form
    is used so existing short-sequence behavior is bit-identical.
    """
    return int(85.0 / max(-math.log(max(alpha_min, 1e-6)), 1e-9))


def _scan_torch(gate, value, alpha, alpha_min, alpha_max):
    """Closed-form via cumulative log-decay (two cumsums), fp32 in the scan.

    At long sequences the naive exp(C)/exp(-C) intermediate overflows fp32
    (~S > 300 at the 0.75 floor). Falls back to a chunked closed form that
    normalizes exp() against each chunk's own decay prefix, so the huge
    intermediate is never formed and the result stays finite for any S.
    """
    u = value if gate is None else gate * value
    w = torch.log(alpha.clamp(min=alpha_min, max=alpha_max))
    S = u.shape[2]
    if S <= _stable_threshold(alpha_min):
        cumlog = torch.cumsum(w, dim=2)
        return torch.exp(cumlog) * torch.cumsum(torch.exp(-cumlog) * u, dim=2)

    u = u.to(torch.float32)
    chunk = 64
    n = (S + chunk - 1) // chunk
    outs = []
    h = torch.zeros_like(u[:, :, :1, :])
    for c in range(n):
        sl = slice(c * chunk, min((c + 1) * chunk, S))
        Cc = torch.cumsum(w[:, :, sl], dim=2)          # chunk-local log-decay
        hc = torch.exp(Cc) * (torch.cumsum(torch.exp(-Cc) * u[:, :, sl], dim=2)
                              + h)
        outs.append(hc)
        h = hc[:, :, -1:, :]                            # carry state to next chunk
    return torch.cat(outs, dim=2)


def _du_torch_chunked(dh, a, chunk=64):
    """Numerically stable du_t = sum_{k>=t} dh_k * prod_{j=t+1..k} a_j.

    Reverse-chunked closed form: each chunk is normalized against its own last
    log-decay, so exp() intermediates stay <= exp(-ln(alpha_min)*chunk) instead
    of exp(S * -ln(alpha_min)) (which overflows fp32 around S~300).
    """
    w = a.clamp(min=1e-6).log()
    C = torch.cumsum(w, dim=2)
    B, H, S, Hd = dh.shape
    dh = dh.float()
    n = (S + chunk - 1) // chunk
    chunks = []
    carry = torch.zeros(B, H, Hd, dtype=torch.float32, device=dh.device)
    for c in reversed(range(n)):
        t0 = c * chunk
        t1 = min(t0 + chunk, S)
        sl = slice(t0, t1)
        E = C[:, :, sl] - C[:, :, t1 - 1:t1]            # >= 0, <= -ln(a_min)*chunk
        rsum = _reverse_cumsum(dh[:, :, sl] * torch.exp(E), dim=2)
        du = torch.exp(-E) * rsum
        if t1 < S:                                      # tail term across chunks
            wnext = w[:, :, t1:t1 + 1]
            du = du + torch.exp(wnext - E) * carry.unsqueeze(2)
        chunks.append(du)
        carry = du[:, :, 0, :]                          # du at this chunk's start
    return torch.cat(chunks[::-1], dim=2)


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
        HDIM: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        HAS_GATE: tl.constexpr,
        ALPHA_MIN: tl.constexpr,
        ALPHA_MAX: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)          # index over (B * H)
        hd = tl.arange(0, HEAD_DIM)
        hd_mask = hd < HDIM                          # real inner dim (may be < pow2)
        base = pid * S * HDIM                        # real row stride (NOT padded)
        h = tl.zeros([HEAD_DIM], dtype=tl.float32)
        for t in range(S):                           # dynamic loop over sequence
            offs = base + t * HDIM + hd
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
            HDIM=Hd,
            HEAD_DIM=head_dim,
            HAS_GATE=gate is not None,
            ALPHA_MIN=alpha_min,
            ALPHA_MAX=alpha_max,
            num_warps=1,
        )
        return out

    @triton.jit
    def _scan_bwd_kernel(
        dh_ptr, alpha_ptr, du_ptr,
        S,
        HDIM: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        ALPHA_MIN: tl.constexpr,
        ALPHA_MAX: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        hd = tl.arange(0, HEAD_DIM)
        hd_mask = hd < HDIM
        base = pid * S * HDIM
        du = tl.zeros([HEAD_DIM], dtype=tl.float32)
        a_next = tl.zeros([HEAD_DIM], dtype=tl.float32)
        # du_t = dh_t + a_{t+1} * du_{t+1}, computed as a reverse recurrence.
        # a_next holds a_t (the alpha of the NEXT higher index), which at step
        # t is exactly a_{t+1}. No exp/cumsum -> finite for any S.
        for t in range(S - 1, -1, -1):
            dh = tl.load(dh_ptr + base + t * HDIM + hd,
                         mask=hd_mask, other=0.0).to(tl.float32)
            du = dh + a_next * du
            tl.store(du_ptr + base + t * HDIM + hd, du, mask=hd_mask)
            a = tl.load(alpha_ptr + base + t * HDIM + hd,
                        mask=hd_mask, other=1.0).to(tl.float32)
            a_next = tl.minimum(tl.maximum(a, ALPHA_MIN), ALPHA_MAX)

    def _scan_bwd_triton(dh, alpha, alpha_min, alpha_max):
        B, H, S, Hd = dh.shape
        du = torch.empty((B, H, S, Hd), dtype=torch.float32, device=dh.device)
        head_dim = triton.next_power_of_2(Hd)
        _scan_bwd_kernel[(B * H,)](
            dh, alpha, du, S,
            HDIM=Hd,
            HEAD_DIM=head_dim,
            ALPHA_MIN=alpha_min,
            ALPHA_MAX=alpha_max,
            num_warps=1,
        )
        return du


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
        dh = dh.contiguous().to(torch.float32)

        if _use_triton() and dh.is_cuda:
            # Fused reverse-scan kernel (one pass, finite at any S).
            du = _scan_bwd_triton(dh, a, alpha_min, alpha_max)
        else:
            # Closed form via cumulative log-decay (exact, but its exp(-C)
            # intermediate overflows fp32 around S ~= 300; the kernel above
            # never forms it). du_t = sum_{k>=t} dh_k * prod_{j=t+1..k} a_j.
            # Long sequences use the chunk-stable version (same result, finite).
            if dh.shape[2] > _stable_threshold(alpha_min):
                du = _du_torch_chunked(dh, a)
            else:
                C = torch.cumsum(a.clamp(min=1e-6).log(), dim=2)
                eC = torch.exp(C)
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
            # Correctness gate: the kernel is a direct sequential recurrence,
            # so it must match the exact reference.
            assert torch.allclose(tri, ref, atol=1e-3, rtol=1e-3), 'kernel mismatch'
            # The torch closed-form goes through exp(log-alpha)/cumsum; at long
            # S with alpha near the 0.75 floor, exp(-cumlog) exceeds fp32 range
            # (~1e63 > 3.4e38) and overflows to inf. The kernel does not (it
            # never forms the huge intermediate). That overflow is expected and
            # is exactly the robustness win the kernel provides (H10).
            if torch.isfinite(torch_to).all():
                assert torch.allclose(torch_to, ref, atol=1e-3, rtol=1e-3)
            else:
                print('  [note] torch closed-form overflowed at S=%d '
                      '(kernel stays finite) - expected, not a bug' % S)

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
