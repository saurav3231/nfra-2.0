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


def test_brain_config_respects_band_knob():
    # The NFRA_BANDS ablation knob must reach the config: an explicit n_bands
    # is respected (was silently force-overridden to 16); only the dataclass
    # default (4) is promoted to the legacy 16-head hierarchy.
    assert NFRAConfig(mode="brain", n_bands=8).n_bands == 8
    assert NFRAConfig(mode="brain", n_bands=2).n_bands == 2
    assert NFRAConfig(mode="brain").n_bands == 16


def test_band_knob_reaches_brainmixer():
    cfg = NFRAConfig(mode="brain", vocab_size=96, hidden_size=128,
                     num_layers=2, n_bands=8)
    model = NFRAForCausalLM(cfg)
    assert model.layers[0].mixer.n_heads == 8
    default = NFRAForCausalLM(NFRAConfig(mode="brain", vocab_size=96,
                                         hidden_size=128, num_layers=2))
    assert default.layers[0].mixer.n_heads == 16


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


def test_brain_learns_short_recall():
    """Smoke test: NFRA Brain must LEARN a 2-step recall task (drop below the
    ln(16) floor) on a tiny CPU config. Guards against the block wiring
    starving the recurrence entirely. NOTE: this does NOT reproduce the H3
    flat-at-floor failure (dim 224 / 600 steps / k>=4), so it is not a
    regression test for that bug — that regime needs GPU (see
    nfra.benchmark.recall_diag)."""
    import math
    import numpy as np

    V, k, seq = 16, 2, 32
    torch.manual_seed(0)
    cfg = NFRAConfig(mode="brain", vocab_size=V, hidden_size=64,
                     num_layers=6, unique_blocks=2, dropout=0.0,
                     gradient_checkpointing=False)
    model = NFRAForCausalLM(cfg)
    from nfra.benchmark.compare import rescale_embed
    rescale_embed(model)

    rng = np.random.RandomState(0)
    keys = rng.randint(0, V, size=(8, seq + 1))
    toks = np.empty_like(keys)
    for t in range(seq + 1):
        src = t - k
        if src >= 0:
            toks[:, t] = (keys[:, src] + 1) % V
        else:
            toks[:, t] = rng.randint(0, V, size=8)
    x = torch.from_numpy(toks[:, :-1]).long()
    y = torch.from_numpy(toks[:, 1:]).long()

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    losses = []
    for _ in range(40):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(
            model(x)["logits"].reshape(-1, V), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

    floor = math.log(V)
    assert losses[-1] < floor - 0.03, f"stuck at floor: {losses[-1]:.3f}"
    assert losses[-1] < losses[0] - 0.1, f"no learning: {losses[0]:.3f}->{losses[-1]:.3f}"


def test_local_pool_matches_sliding_window():
    """Causal local pooling (cortical microcircuit routing context) must equal
    a manual sliding-window mean, including the sequence-start correction and
    the window edge (seq > local_win=64)."""
    from nfra.core.neuro import BrainMLP
    mlp = BrainMLP(dim=8, hidden_mult=2.0)
    torch.manual_seed(0)
    x = torch.randn(2, 70, 8)
    got = mlp._local_pool(x)
    w = 64
    ref = torch.zeros_like(x)
    for t in range(70):
        lo = max(0, t - w + 1)
        ref[:, t] = x[:, lo:t + 1].mean(1)
    assert torch.allclose(got, ref, atol=1e-6)


def test_brain_feature_toggles_forward_backward():
    """Each 'small-but-powerful' brain lever must build, forward, and produce
    finite gradients (CPU smoke)."""
    cfg = NFRAConfig(mode="brain", vocab_size=16, hidden_size=64,
                     num_layers=4, unique_blocks=2, dropout=0.0,
                     local_route=True, div_norm=True, astro=True)
    model = NFRAForCausalLM(cfg)
    assert model.layers[0].mlp.local_route is True
    assert model.layers[0].mlp.div_norm is True
    assert model.layers[0].mixer.astro_proj is not None
    torch.manual_seed(0)
    x = torch.randint(0, 16, (2, 32))
    out = model(x)
    assert out["logits"].shape == (2, 32, 16)
    out["logits"].float().mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_brain_feature_toggles_off_by_default():
    cfg = NFRAConfig(mode="brain", vocab_size=16, hidden_size=64,
                     num_layers=2, unique_blocks=2)
    model = NFRAForCausalLM(cfg)
    assert model.layers[0].mlp.local_route is False
    assert model.layers[0].mlp.div_norm is False
    assert model.layers[0].mixer.astro_proj is None


def test_energy_allocator_no_graph_leak():
    a = DynamicEnergyBudgetAllocator(4)
    b1 = a()
    b2 = a()
    assert not a.current_budget.requires_grad
    assert b2.requires_grad
    assert torch.isfinite(a.current_budget).all()
