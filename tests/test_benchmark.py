"""
Pure-logic tests for the benchmark scoring/scaling helpers.

These run instantly (no training, no GPU) and pin the math used by
nfra.benchmark.arena: scaling fits, learning-curve AUC, z-scores, and
the weighted composite score.
"""

from nfra.benchmark import arena


def _row(final, tok, mem):
    return {
        "final_eval": final,
        "sample_auc": final + 0.2,
        "param_eff": 0.5,
        "tok_s_train": tok,
        "peak_mem": mem,
        "gen_tok_s": tok * 0.1,
        "infer_mem": mem * 0.5,
        "extrap_delta": 0.1,
        "scaling_gain": 0.2,
    }


def test_fit_scaling_slope_negative():
    r = arena.fit_scaling([(5e6, 2.5), (20e6, 2.1)])
    assert r["slope"] < 0
    assert r["r2"] is not None


def test_fit_scaling_single_point():
    r = arena.fit_scaling([(5e6, 2.5)])
    assert r["n"] == 1


def test_sample_auc():
    h = [(50, 4.5), (100, 3.5), (150, 3.0)]
    auc = arena.sample_auc(h)
    assert auc is not None
    assert 0 < auc < 5


def test_zscores_direction():
    z = arena.zscores([1.0, 2.0, 3.0], direction=-1)
    assert z[0] > z[2]
    z2 = arena.zscores([1.0, 2.0, 3.0], direction=+1)
    assert z2[2] > z2[0]


def test_composite_prefers_quality_leader():
    rows = {
        "good": _row(final=2.0, tok=500.0, mem=5.0),
        "bad": _row(final=4.0, tok=9000.0, mem=1.0),
    }
    s = arena.composite_score(rows, arena.METRIC_SPEC)
    assert s["good"] > s["bad"]


def test_tune_nfra_matches_target():
    U, dim, params = arena.tune_nfra_size(5_000_000, 96, 12, [256, 192, 128])
    assert 12 % U == 0
    assert abs(params - 5e6) / 5e6 < 0.3


def test_recall_probe_sequential_builds_all_then_trains(monkeypatch):
    """recall_probe sequential mode must build every model under one RNG
    stream before training any (init parity with concurrent mode), and return
    the right per-(family, k) structure."""
    from nfra.benchmark import recall_probe

    calls = []

    def fake_train_one(model, vocab, steps, train_loader, eval_loader,
                       eval_gap, **kw):
        calls.append(id(model))
        return {"loss_hist": [1.0], "nan_steps": 0, "wall_s": 1.0,
                "ms_per_step": 1.0, "tok_s": 1.0, "eval_hist": []}

    def fake_metric(model, loader, k, V=16):
        return (0.0, 0.0, 0.0)

    monkeypatch.setattr(recall_probe, "train_one", fake_train_one)
    monkeypatch.setattr(recall_probe, "metric_by_span", fake_metric)
    rows = recall_probe._run_all([4, 16], steps=1, dim=128, seq_len=32,
                                 batch=2, unique=2, depth=4, concurrent=False)
    assert set(rows) == {"nfra", "mamba"}
    assert set(rows["nfra"]) == {4, 16}
    assert set(rows["mamba"]) == {4, 16}
    assert len(calls) == 4
