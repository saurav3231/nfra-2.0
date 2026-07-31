"""
Tests for the fused selective-scan kernel (nfra.kernels.scan).

CPU: exercises the torch fallback against the exact sequential definition.
CUDA: exercises the Triton kernel forward + closed-form backward against the
sequential reference and autograd of the torch path (skipped without CUDA).
"""

import os

import pytest
import torch

from nfra.kernels.scan import (
    scan_reference,
    _scan_torch,
    _scan_triton,
    selective_scan,
    ScanFunction,
)

HAS_CUDA = torch.cuda.is_available()


def _inputs(device, B=2, H=8, S=64, Hd=7, seed=0, gate=True):
    torch.manual_seed(seed)
    value = torch.randn(B, H, S, Hd, device=device)
    alpha = torch.rand(B, H, S, Hd, device=device) * 0.2 + 0.75
    if gate:
        g = torch.sigmoid(torch.randn(B, H, S, Hd, device=device))
    else:
        g = None
    return g, value, alpha


@pytest.mark.parametrize('Hd', [7, 8, 16])
@pytest.mark.parametrize('gate', [True, False])
def test_torch_fallback_matches_sequential(Hd, gate):
    g, value, alpha = _inputs('cpu', Hd=Hd, gate=gate)
    out = selective_scan(g, value, alpha, alpha_min=0.75, alpha_max=0.9995)
    ref = scan_reference(g, value, alpha, alpha_min=0.75, alpha_max=0.9995)
    assert out.dtype == torch.float32
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)


def test_torch_fallback_non_pow2_head_dim():
    # Hd=14 (BrainMixer: dim 224 / 16 heads) — exercises padding path on CUDA
    g, value, alpha = _inputs('cpu', Hd=14, gate=True)
    out = selective_scan(g, value, alpha, alpha_min=0.75, alpha_max=0.9995)
    ref = scan_reference(g, value, alpha, alpha_min=0.75, alpha_max=0.9995)
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)


def test_torch_fallback_is_old_behavior():
    # The public wrapper must reproduce the pre-kernel closed form exactly.
    g, value, alpha = _inputs('cpu', Hd=16, gate=True)
    out = selective_scan(g, value, alpha, alpha_min=0.85, alpha_max=0.9995)
    ref = _scan_torch(g, value, alpha, alpha_min=0.85, alpha_max=0.9995)
    assert torch.equal(out, ref)


@pytest.mark.skipif(not HAS_CUDA, reason='CUDA required')
@pytest.mark.parametrize('Hd', [7, 14, 16])
@pytest.mark.parametrize('gate', [True, False])
def test_kernel_forward_matches_reference(Hd, gate):
    g, value, alpha = _inputs('cuda', B=4, H=16, S=256, Hd=Hd, gate=gate)
    out = _scan_triton(g, value, alpha, 0.75, 0.9995)
    ref = scan_reference(g, value, alpha, 0.75, 0.9995)
    assert torch.allclose(out, ref, atol=2e-3, rtol=2e-3)
    torch_to = _scan_torch(g, value, alpha, 0.75, 0.9995)
    assert torch.allclose(out, torch_to, atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not HAS_CUDA, reason='CUDA required')
def test_kernel_backward_matches_autograd():
    g, value, alpha = _inputs('cuda', B=2, H=8, S=128, Hd=14, seed=1)
    for v in (g, value, alpha):
        v.requires_grad_(True)

    torch.manual_seed(0)
    w = torch.randn_like(torch.randn(2, 8, 128, 14, device='cuda'))
    tri = ScanFunction.apply(g, value, alpha, 0.75, 0.9995)
    ref = _scan_torch(g, value, alpha, 0.75, 0.9995)
    (dg_t, dv_t, da_t) = torch.autograd.grad((tri * w).sum(), (g, value, alpha))
    (dg_r, dv_r, da_r) = torch.autograd.grad((ref * w).sum(), (g, value, alpha))

    assert torch.allclose(tri, ref, atol=2e-3, rtol=2e-3)
    assert torch.allclose(dv_t, dv_r, atol=2e-3, rtol=2e-3)
    assert torch.allclose(dg_t, dg_r, atol=2e-3, rtol=2e-3)
    assert torch.allclose(da_t, da_r, atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not HAS_CUDA, reason='CUDA required')
def test_env_force_torch():
    os.environ['NFRA_SCAN_KERNEL'] = '0'
    try:
        g, value, alpha = _inputs('cuda', Hd=16)
        out = selective_scan(g, value, alpha)
        ref = _scan_torch(g, value, alpha, 0.75, 0.9995)
        assert torch.equal(out, ref)
    finally:
        os.environ.pop('NFRA_SCAN_KERNEL', None)
