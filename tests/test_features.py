"""
CPU tests for NFRA 3.2 feature toggles:
  - k-WTA lateral inhibition in the Brain MLP
  - EMA (exponential moving average) weight helper
  - surprise-weighted (RPE) loss

All run on CPU with tiny configs.
"""

import math

import torch
import torch.nn as nn

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


def test_prefix_pool_is_causal():
    """prefix_pool_t must equal mean(x[0..t]) — no future leakage — and later
    positions must never affect earlier pooled rows."""
    from nfra.core.neuro import prefix_pool

    x = torch.randn(3, 5, 4)
    p = prefix_pool(x)
    for t in range(5):
        assert torch.allclose(p[:, t], x[:, : t + 1].mean(1), atol=1e-6)
    x2 = x.clone()
    x2[:, 3:, :] = 999.0
    p2 = prefix_pool(x2)
    assert torch.allclose(p[:, :3], p2[:, :3], atol=1e-6)


def test_fractal_router_uses_causal_prefix_pool(monkeypatch):
    """The gated/SwiGLU expert routers must pool causally (prefix_pool), not
    leak the future via a whole-sequence mean."""
    from nfra.core import fractal_block, neuro

    calls = []
    orig = neuro.prefix_pool

    def spy(x):
        calls.append(x)
        return orig(x)

    monkeypatch.setattr(neuro, "prefix_pool", spy)
    x = torch.randn(2, 16, 64)
    out = fractal_block.FractalGatedMLP(64)(x)
    assert out.shape == (2, 16, 64)
    assert len(calls) == 1 and calls[0].shape == (2, 16, 64)

    calls.clear()
    out = fractal_block.FractalSwiGLU(64)(x)
    assert out.shape == (2, 16, 64)
    assert len(calls) == 1 and calls[0].shape == (2, 16, 64)


def test_energy_budget_mean_normalized():
    """H3: budgets are mean-normalized so an energy_budget of 0.5 really means
    0.5 per block (uniform importance), never the old sum-collapse to the 0.1
    floor. Higher importance still gets proportionally more budget."""
    from nfra.core.energy import DynamicEnergyBudgetAllocator

    d = DynamicEnergyBudgetAllocator(num_blocks=4, default_budget=0.5,
                                     min_budget=0.1)
    b = d.forward(None, hardware_factor=0.5, power_remaining=1.0)
    assert b.shape == (4,)
    assert torch.allclose(b, torch.full((4,), 0.5), atol=1e-6)
    assert (b > 0.1).all()

    d2 = DynamicEnergyBudgetAllocator(num_blocks=4, default_budget=1.0,
                                      min_budget=0.05)
    b2 = d2.forward(torch.tensor([1.0, 2.0, 4.0, 8.0]),
                    hardware_factor=1.0, power_remaining=1.0)
    assert b2[3] > b2[0]


def test_trainer_eval_ignores_pad_and_empty():
    """H1: eval must ignore -100/pad tokens and report inf (not a bogus 0.0)
    when there is nothing valid to score."""
    from torch.utils.data import DataLoader, TensorDataset

    from nfra.training.trainer import NFRATrainer

    model = _brain()
    tr = NFRATrainer(model, lambda **kw: (torch.tensor(0.0), {}),
                     device="cpu")
    torch.manual_seed(0)
    x = torch.randint(0, 96, (4, 16))

    y_all = torch.full_like(x, -100)
    r_all = tr.evaluate(DataLoader(TensorDataset(x, y_all), batch_size=4))
    assert r_all["eval_loss"] == float("inf")
    assert r_all["perplexity"] == float("inf")

    y = torch.randint(0, 96, (4, 16))
    y[:, -3:] = -100
    r = tr.evaluate(DataLoader(TensorDataset(x, y), batch_size=4))
    assert math.isfinite(r["eval_loss"])
    logits = model(x)["logits"].view(-1, 96)
    t = y.view(-1)
    mask = t != -100
    ref = torch.nn.functional.cross_entropy(logits[mask], t[mask],
                                            reduction="mean")
    assert abs(r["eval_loss"] - ref.item()) < 1e-4


def test_loss_resonance_energy_are_report_only():
    """M5: detached resonance/energy terms must never inflate the total loss;
    only the (gradient-carrying) prediction-error tensor may add to it."""
    from nfra.training.losses import NFRACombinedLoss

    loss_fn = NFRACombinedLoss()
    torch.manual_seed(0)
    logits = torch.randn(2, 8, 96)
    targets = torch.randint(0, 96, (2, 8))

    total, d = loss_fn(logits, targets,
                       resonance_stats={"sparsity": 0.5}, energy_used=0.8)
    assert "resonance_loss" in d and "energy_loss" in d
    assert abs(float(d["energy_loss"]) - 0.2) < 1e-6
    assert torch.allclose(total, loss_fn.task_loss(
        logits.view(-1, 96), targets.view(-1)), atol=1e-6)

    total2, d2 = loss_fn(logits, targets,
                         resonance_stats={"sparsity": 0.5}, energy_used=0.8,
                         prediction_error=torch.tensor(0.5))
    assert d2["total_loss"] > d["total_loss"] > 0


def test_int8_linear_real_storage_and_roundtrip():
    """L4: Int8Linear stores weights as genuine int8 (+fp32 scale), forward
    matches the rounded grid, and quantize/dequantize round-trips."""
    from nfra.utils.quantization import (Int8Linear, apply_int8_to_model,
                                         dequantize, quantize_to_int8)

    torch.manual_seed(0)
    lin = nn.Linear(8, 8)
    fp = lin.weight.detach()
    scale = fp.abs().max() / 127.0
    q = Int8Linear(lin)
    assert q.qweight.dtype == torch.int8
    assert abs(q.scale.item() - scale.item()) < 1e-9
    w_deq = q.qweight.float() * q.scale
    assert (w_deq - fp).abs().max().item() <= q.scale.item() / 2 + 1e-6
    x = torch.randn(2, 8)
    assert torch.allclose(q(x), nn.functional.linear(x, w_deq, lin.bias),
                          atol=1e-6)

    net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
    apply_int8_to_model(net)
    assert isinstance(net[0], Int8Linear) and isinstance(net[2], Int8Linear)
    out = net(x)
    assert out.shape == (2, 4) and torch.isfinite(out).all()

    state = quantize_to_int8(net)
    deq = dequantize(state)
    for name, p in net.named_parameters():
        assert torch.allclose(deq[name], p, atol=q.scale.item() / 2 + 1e-6)


def test_evaluate_empty_loader_returns_inf():
    """M6: an empty eval loader must report inf, never a silent 'perfect' 0.0."""
    from torch.utils.data import DataLoader, TensorDataset

    from nfra.benchmark import compare

    model = _brain()
    empty = DataLoader(TensorDataset(torch.zeros(0, 16, dtype=torch.long),
                                     torch.zeros(0, 16, dtype=torch.long)),
                       batch_size=4)
    assert compare.evaluate(model, empty) == float("inf")
