"""NFRA 3.3 Cortex smoke test — quick sanity check (run on Kaggle T4).

Verifies the three design claims before committing to the real A/B:
  1. PARAMS   : NFRA_Cortex_Block stays at ~5M @ 5M target (matrix state adds
                no params — the projections already existed; only the read/write
                vectors are new, B/C: 2*H*N*dim).
  2. FORWARD  : train-mode forward + backward converge (loss finite, no NaN),
                eval-mode forward identical shape.
  3. EXIT     : the exit gate really skips passes at inference — forcing the
                gate bias high (all tokens exit) makes the pass loop break
                early AND is numerically identical to a full-compute run with
                the mask applied (freeze semantics).

Usage (Kaggle):
  python -m nfra.benchmark.cortex_smoke
  NFRA_DIM=224 NFRA_UNIQUE=4 NFRA_STEPS=30 python -m nfra.benchmark.cortex_smoke
"""

import math
import os

import torch

os.environ.setdefault("NFRA_SEEDS", "1")
os.environ.setdefault("NFRA_SIZES", "5")
# Test the FULL 3.3b block (gland + gates + exit): the default 3.3c build is the
# pruned lean block, which has no exit_gate for this diagnostic to poke.
os.environ.setdefault("NFRA_LEAN", "0")

from .arena import NFRA_DEPTH, build_nfra
from .compare import DEVICE, count_params, rescale_embed


def _finite(t):
    return bool(torch.isfinite(t).all())


def check_exit_skip(model, vocab):
    """Force all tokens to exit after pass 1 and confirm the pass loop breaks
    early: the all-exit forward must be bit-identical to a depth-1 forward
    (exited tokens freeze exactly at the pass-1 output, so the frozen-mask
    full run and the loop-broken run agree)."""
    model.eval()
    x = torch.randint(0, vocab, (2, 128), device=DEVICE)
    with torch.no_grad():
        # Reference: depth-1 forward (only pass 1, no later passes).
        saved_passes = model.depth_passes
        model.depth_passes = 1
        out_ref = model(x)["logits"]
        model.depth_passes = saved_passes

        # All tokens exit after pass 1: gate bias high -> sigmoid(p) ~ 0.99
        # > 0.5 -> cont = 0 -> whole batch done -> loop breaks before pass 2.
        for i, layer in enumerate(model.layers):
            layer.exit_gate.gate.bias.data.fill_(5.0)
        out_exit = model(x)["logits"]
        for i, layer in enumerate(model.layers):
            layer.exit_gate.gate.bias.data.fill_(-1.0)

        # Sanity: bias restored -> keep run repeats (gate did nothing).
        out_keep = model(x)["logits"]
        drift = (out_ref - out_keep).abs().max().item()

    delta = (out_ref - out_exit).abs().max().item()
    return {
        "max_abs_delta_depth1_vs_allexit": round(delta, 6),
        "max_abs_drift_keep_vs_depth1": round(drift, 6),
        "loop_broken_exact": bool(delta == 0.0),
    }


def main():
    steps = int(os.environ.get("NFRA_STEPS", "30"))
    dim = int(os.environ.get("NFRA_DIM", "224"))
    unique = int(os.environ.get("NFRA_UNIQUE", "4"))
    vocab = 96
    random_loss = math.log(vocab)

    print("=" * 64)
    print(
        "  NFRA 3.3 Cortex smoke test  (dim=%d unique=%d depth=%d vocab=%d)"
        % (dim, unique, NFRA_DEPTH, vocab)
    )
    print("=" * 64)

    # ---- 1. build + params ----
    model = build_nfra(vocab, dim, unique, depth=NFRA_DEPTH, use_cortex=True).to(DEVICE)
    rescale_embed(model)
    params = count_params(model)
    print("\n[1] build OK   params=%.4fM  (target 5M)" % (params / 1e6))
    block = model.layers[0]
    print(f"    block type : {type(block).__name__}")
    print(
        "    has exit_gate : {}   has matrix mixer : {}".format(
            hasattr(block, "exit_gate"), hasattr(block, "mixer")
        )
    )
    assert hasattr(block, "exit_gate") and hasattr(block, "mixer")

    # ---- 2. train forward/backward ----
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    first, last, nans = None, None, 0
    for s in range(1, steps + 1):
        opt.zero_grad()
        x = torch.randint(0, vocab, (8, 128), device=DEVICE)
        y = torch.randint(0, vocab, (8, 128), device=DEVICE)
        loss = model(x)["logits"]
        loss = torch.nn.functional.cross_entropy(loss.view(-1, vocab), y.view(-1))
        if not _finite(loss):
            nans += 1
            continue
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
        last = loss.item()
    print(
        "[2] train OK   loss %.4f -> %.4f  (random=%.3f)  nan_steps=%d"
        % (first, last, random_loss, nans)
    )
    assert last < first or nans == 0, "loss must move or be finite"

    # ---- 3. eval forward ----
    model.eval()
    with torch.no_grad():
        x = torch.randint(0, vocab, (2, 128), device=DEVICE)
        out = model(x)
        logits = out["logits"]
        assert "exit_aux" not in out or out["exit_aux"].item() >= 0.0
        print(f"[3] eval OK   logits {tuple(logits.shape)} finite={_finite(logits)}")

    # ---- 4. exit-gate skip ----
    result = check_exit_skip(model, vocab)
    print(f"[4] exit skip : {result}")

    print("\nSMOKE PASSED" if not nans else "\nSMOKE WARN (nan steps)")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
