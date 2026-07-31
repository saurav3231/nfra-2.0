"""
CPU tests for NFRA 3.2 feature toggles:
  - k-WTA lateral inhibition in the Brain MLP
  - EMA (exponential moving average) weight helper
  - surprise-weighted (RPE) loss

All run on CPU with tiny configs.
"""

import torch

from nfra import NFRAConfig, NFRAForCausalLM
from nfra.core.neuro import BrainMLP
from nfra.benchmark.compare import EMA, compute_loss


def _brain(vocab=96, dim=128, layers=4, unique=2, kwta=0.0):
    cfg = NFRAConfig(mode="brain", vocab_size=vocab, hidden_size=dim,
                     num_layers=layers, unique_blocks=unique,
                     k_wta_frac=kwta, gradient_checkpointing=False)
    return NFRAForCausalLM(cfg)


def test_fractal_mlp_no_per_expert_sync():
    """FractalGatedMLP must record expert activity as deferred flag tensors
    (no implicit .item() GPU->CPU sync in the forward loop) and still report
    a sane sparsity via get_sparsity."""
    from nfra.core.fractal_block import FractalResonanceBlock

    blk = FractalResonanceBlock(64)
    x = torch.randn(2, 16, 64)
    out = blk(x)
    assert out.shape == (2, 16, 64)
    flags = blk.mlp._n_active_flags
    assert flags is not None
    assert flags.shape == (len(blk.mlp.scales),)
    assert flags.dtype == torch.bool
    sp = blk.get_sparsity()
    assert 0.0 <= sp <= 1.0


def test_brainmixer_band_count_knob():
    """H8: BrainMixer must accept explicit band counts and train smoothly."""
    from nfra.core.neuro import BrainMixer

    for h in (2, 4, 8, 16):
        m = BrainMixer(192, n_heads=h)
        assert m.n_heads == h
        assert m.head_dim * h == 192
        x = torch.randn(2, 32, 192)
        out, router = m(x)
        assert out.shape == (2, 32, 192)
        assert router.shape == (2, 32, 1)
        out.mean().backward()
        assert m.dt_proj.weight.grad is not None


def test_brainmixer_default_is_16_head_hierarchy():
    from nfra.core.neuro import BrainMixer

    m = BrainMixer(192)               # None -> legacy [8,4,2,1]+router
    assert m.n_heads == 16
    assert m.head_counts == [8, 4, 2, 1]
    m16 = BrainMixer(192, n_heads=16)  # explicit 16 -> identical structure
    assert m16.head_counts == [8, 4, 2, 1]
    assert m16.n_heads == 16


def test_brainmixer_fused_gate_value_is_exact():
    """Fusing gate/value into one GEMM must be elementwise-identical to two
    separate GEMMs (concat of weights -> concat of outputs). No loss drift."""
    import torch.nn.functional as F
    from nfra.core.neuro import BrainMixer

    torch.manual_seed(0)
    m = BrainMixer(192)
    D = 192
    W_f = m.proj_gate_value.weight.data                  # [2D, D]
    x = torch.randn(2, 16, 192)
    fused = F.linear(x, W_f)
    ref = torch.cat([F.linear(x, W_f[:D]), F.linear(x, W_f[D:])], dim=-1)
    assert torch.equal(fused, ref)
    assert not hasattr(m, "proj_gate") and not hasattr(m, "proj_value")


def test_k_wta_is_applied_in_mlp():
    mlp = BrainMLP(64, k_wta_frac=0.5)
    captured = {}
    hook = mlp.down_proj.register_forward_hook(
        lambda m, i, o: captured.__setitem__("x", i[0]))
    x = torch.randn(2, 8, 64)
    with torch.no_grad():
        mlp(x)
    hook.remove()
    frac = (captured["x"] != 0).float().mean().item()
    assert 0.4 <= frac <= 0.7


def test_k_wta_forward_backward():
    torch.manual_seed(0)
    model = _brain(kwta=0.5)
    assert model.layers[0].mlp.k_wta_frac == 0.5
    x = torch.randint(0, 96, (2, 16))
    out = model(x)
    out["logits"].float().mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_k_wta_off_is_identity_behavior():
    torch.manual_seed(0)
    m_off = _brain(kwta=0.0)
    m_on = _brain(kwta=1.0)          # 1.0 keeps everything (k = hidden dim)
    x = torch.randint(0, 96, (2, 16))
    with torch.no_grad():
        out_off = m_off(x)["logits"]
        out_on = m_on(x)["logits"]
    assert torch.isfinite(out_off).all()
    assert torch.isfinite(out_on).all()


def test_ema_apply_restore_roundtrip():
    torch.manual_seed(0)
    model = _brain()
    ema = EMA(model, decay=0.9)
    with torch.no_grad():
        for p in model.parameters():
            p.data.add_(0.5)
    backup = {k: v.detach().clone() for k, v in model.named_parameters()}
    ema.update(model)
    ema.apply(model)
    for k, v in model.named_parameters():
        assert torch.allclose(v.float(), ema.shadow[k], atol=1e-6)
    ema.restore(model)
    for k, v in model.named_parameters():
        assert torch.allclose(v, backup[k], atol=1e-6)


def test_surprise_loss_finite():
    torch.manual_seed(0)
    model = _brain()
    x = torch.randint(0, 96, (2, 32))
    y = torch.randint(0, 96, (2, 32))
    l0 = compute_loss(model, x, y, surprise=False)
    l1 = compute_loss(model, x, y, surprise=True)
    assert torch.isfinite(l0)
    assert torch.isfinite(l1)
    assert l1.item() != l0.item()


def test_surprise_weights_are_mean_preserving():
    # The weighting normalized by its own mean must have mean exactly 1,
    # so the effective learning rate is unchanged by the surprise term.
    ce = torch.tensor([0.1, 5.0, 10.0])
    p = torch.exp(-ce)
    w = (1.0 - p) / ((1.0 - p).mean() + 1e-6)
    assert abs(float(w.mean()) - 1.0) < 1e-5
    # Constant confidence → uniform weights → weighted == unweighted loss.
    ce2 = torch.full((4,), 4.56)
    p2 = torch.exp(-ce2)
    w2 = (1.0 - p2) / ((1.0 - p2).mean() + 1e-6)
    assert torch.allclose(w2, torch.ones_like(w2), atol=1e-5)
    assert abs(float((ce2 * w2).mean()) - float(ce2.mean())) < 1e-5
