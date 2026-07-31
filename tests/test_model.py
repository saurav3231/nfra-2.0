"""
Smoke tests for NFRA model construction, forward/backward, and IO.

Runs on CPU with tiny configs. Use: pytest tests/
"""

import torch

from nfra import NFRAConfig, NFRAForCausalLM
from nfra.models.nfra_lite import NFRALiteForCausalLM
from nfra.models.model_io import save_pretrained, from_pretrained
from nfra.core.energy import DynamicEnergyBudgetAllocator


def _brain(vocab=96, dim=128, layers=6, unique=2, ckpt=False):
    cfg = NFRAConfig(mode="brain", vocab_size=vocab, hidden_size=dim,
                     num_layers=layers, unique_blocks=unique,
                     gradient_checkpointing=ckpt)
    return NFRAForCausalLM(cfg)


def test_brain_forward_backward():
    torch.manual_seed(0)
    model = _brain()
    x = torch.randint(0, 96, (2, 32))
    out = model(x)
    assert out["logits"].shape == (2, 32, 96)
    out["logits"].float().mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_unique_blocks_scale_params():
    counts = [
        sum(p.numel() for p in _brain(dim=128, layers=12, unique=u).parameters())
        for u in (1, 2, 4)
    ]
    assert counts[1] > counts[0]
    assert counts[2] > counts[1]


def test_lite_build_and_forward():
    torch.manual_seed(0)
    cfg = NFRAConfig(mode="lite", vocab_size=96)
    model = NFRALiteForCausalLM(cfg)
    assert model.get_model_info()["mode"] == "lite"
    x = torch.randint(0, 96, (2, 16))
    out = model(x)
    assert out["logits"].shape == (2, 16, 96)


def test_save_load_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = _brain()
    save_pretrained(model, str(tmp_path))
    loaded = from_pretrained(str(tmp_path))
    model.eval()
    x = torch.randint(0, 96, (1, 16))
    with torch.no_grad():
        a = model(x)["logits"]
        b = loaded(x)["logits"]
    assert torch.allclose(a, b, atol=1e-5)


def test_energy_allocator_no_graph_leak():
    a = DynamicEnergyBudgetAllocator(4)
    b1 = a()
    b2 = a()
    assert not a.current_budget.requires_grad
    assert b2.requires_grad
    assert torch.isfinite(a.current_budget).all()
