"""
GLOBAL ARENA — the almighty global-standard NFRA comparison (overnight run).

A phased, resumable, all-axes benchmark that answers, with evidence:
  WHO is better ON WHICH aspect — quality, sample-efficiency, scaling,
  speed, memory, latency, long-context extrapolation, recall — and whether
  NFRA's "small but powerful" brain-inspired levers actually close the gap.

Methodology (credibility controls, reused from arena.py):
  • Param-matched families (NFRA Brain / RWKV / RetNet / GPT-2, Mamba optional),
    identical data, identical optimizer + schedule, matched token budgets,
    multiple seeds.
  • Multiple sizes -> measured scaling slope (loss per doubling of params).
  • Full dashboard per run; JSON + Markdown report; NaN guard on every step.

Phases (env NFRA_GA_PHASES, default 'core,ablate,recall,perf'; resumable):
  core    head-to-head + scaling at sizes (default 5,20M) x seeds x families.
  ablate  NFRA toggle A/B at the primary size: ema / surprise / k-WTA /
          local cortical routing / divisive normalization / astrocytic
          homeostat / all — plus Mamba+ema, Mamba+surprise for fairness.
  recall  H3 root-cause diagnostic (recall_diag): fix / k1 / noshare variants
          run concurrently on the recall task (dim 224) — one run pins WHY
          NFRA was flat at the floor.
  perf    inference battery + 2x extrapolation on the primary-size models.

Env (all optional):
  NFRA_GA_PHASES      comma list of phases          (default all)
  NFRA_GA_SIZES       comma list of sizes in M      (default 5,20)
  NFRA_GA_SEEDS       seeds per (size,family)       (default 2)
  NFRA_GA_STEPS       train steps                    (default 600)
  NFRA_GA_DATA        wikitext2 | synthetic          (default wikitext2)
  NFRA_GA_OUTDIR      output dir                     (default CWD)
  NFRA_DIAG_*         recall-phase knobs (dim/steps/batch/seq)

Usage: python -m nfra.benchmark.global_arena     (Kaggle T4, leave running)
"""

import functools
import json
import math
import os
import time
import warnings

print = functools.partial(print, flush=True)
warnings.filterwarnings("ignore")

# ── global-arena config (read BEFORE importing arena so its env globals match)
GA_PHASES = [
    p.strip()
    for p in os.environ.get("NFRA_GA_PHASES", "core,ablate,recall,perf").split(",")
    if p.strip()
]
GA_SIZES = [
    int(x) for x in os.environ.get("NFRA_GA_SIZES", "5,20").split(",") if x.strip()
]
GA_SEEDS = int(os.environ.get("NFRA_GA_SEEDS", "2"))
GA_STEPS = int(os.environ.get("NFRA_GA_STEPS", "600"))
GA_DATA = os.environ.get("NFRA_GA_DATA", "wikitext2").lower()
GA_OUTDIR = os.environ.get("NFRA_GA_OUTDIR", os.getcwd())

os.environ.setdefault("NFRA_SIZES", ",".join(map(str, GA_SIZES)))
os.environ.setdefault("NFRA_SEEDS", str(GA_SEEDS))
os.environ.setdefault("NFRA_STEPS", str(GA_STEPS))
os.environ.setdefault("NFRA_DATA", GA_DATA)
os.environ.setdefault("NFRA_FAMILIES", "nfra,rwkv,retnet,gpt2")

import torch

from nfra.benchmark import arena
from nfra.benchmark.arena import (
    EVAL_GAP,
    FAMILIES,
    build_family_spec,
    build_gpt2,
    build_mamba,
    build_nfra,
    build_retnet,
    build_rwkv,
    composite_score,
    fit_scaling,
    generate_metrics,
    make_loaders,
    make_verdict,
    mean_std,
    prefill_tok_s,
    sample_auc,
    train_one,
)
from nfra.benchmark.compare import (
    CHAR_VOCAB,
    HAS_CUDA,
    SEQ_LEN,
    evaluate,
    rescale_embed,
)

# ─────────────────────────── variant tables ───────────────────────────
# name -> (build kwargs for NFRA, train_one kwargs, seeds)
ABLATE = [
    ("nfra_baseline", {}, {"ema_decay": 0.0}, 2),
    ("nfra_ema", {}, {"ema_decay": 0.99}, 2),
    ("nfra_surprise", {}, {"surprise": True}, 1),
    ("nfra_kwta", {"k_wta": 0.25}, {}, 1),
    ("nfra_local", {"local_route": True}, {}, 2),
    ("nfra_divnorm", {"div_norm": True}, {}, 1),
    ("nfra_astro", {"astro": True}, {}, 1),
    ("nfra_theta", {"theta": True}, {}, 1),
    ("nfra_achretain", {"ach_retain": True}, {}, 1),
    ("nfra_gainnov", {"gain_nov": True}, {}, 1),
    ("nfra_lora8", {"lora_rank": 8}, {}, 1),
    (
        "nfra_all",
        {"k_wta": 0.25, "local_route": True, "div_norm": True, "astro": True},
        {"ema_decay": 0.99, "surprise": True},
        2,
    ),
    ("mamba_ema", {"fam": "mamba"}, {"ema_decay": 0.99}, 1),
    ("mamba_surprise", {"fam": "mamba"}, {"surprise": True}, 1),
]

OUT_JSON = os.path.join(GA_OUTDIR, "global_arena_results.json")
OUT_MD = os.path.join(GA_OUTDIR, "global_arena_report.md")


# ─────────────────────────── helpers ───────────────────────────
def _run(model, vocab, steps, train_loader, eval_loader, eval_gap, **kw):
    """train_one with NaN-safe fallback so one bad run never kills the night."""
    try:
        return train_one(model, vocab, steps, train_loader, eval_loader, eval_gap, **kw)
    except Exception as e:  # pragma: no cover
        print(f"  [ERROR] run failed: {e!r}")
        return {
            "loss_hist": [],
            "eval_hist": [],
            "tok_s": 0.0,
            "ms_per_step": 0.0,
            "peak_mem": 0.0,
            "nan_steps": 0,
            "wall_s": 0.0,
        }


def _agg(recs):
    finals = [r["eval_hist"][-1][1] for r in recs if r["eval_hist"]]
    m_final, sd_final = mean_std(finals)
    m_auc, _ = mean_std([sample_auc(r["eval_hist"]) for r in recs if r["eval_hist"]])
    m_tok, _ = mean_std([r["tok_s"] for r in recs])
    m_ms, _ = mean_std([r["ms_per_step"] for r in recs])
    m_mem, _ = mean_std([r["peak_mem"] for r in recs])
    return {
        "final_eval": m_final,
        "final_eval_sd": sd_final,
        "ppl": math.exp(min(m_final, 30)) if m_final else None,
        "sample_auc": m_auc,
        "tok_s": m_tok,
        "ms_per_step": m_ms,
        "peak_mem": m_mem,
        "nan_steps": sum(r["nan_steps"] for r in recs),
        "wall_s": sum(r["wall_s"] for r in recs),
    }


def _build_family(fam, size, vocab, seed, **overrides):
    spec = build_family_spec(fam, size, vocab)
    torch.manual_seed(seed)
    if fam == "nfra":
        m = build_nfra(
            vocab,
            spec["dim"],
            spec["extra"]["unique_blocks"],
            depth=spec["depth"],
            **overrides,
        )
    elif fam == "mamba":
        m = build_mamba(vocab, spec["dim"], spec["extra"]["n_layers"])
    elif fam == "rwkv":
        m = build_rwkv(vocab, spec["dim"], spec["extra"]["n_layers"])
    elif fam == "retnet":
        m = build_retnet(
            vocab,
            spec["dim"],
            spec["extra"]["n_layers"],
            n_heads=spec["extra"].get("n_heads", 8),
        )
    else:
        m = build_gpt2(vocab, spec["dim"], spec["extra"]["n_layers"])
    rescale_embed(m)
    return m, spec["params"]


# ─────────────────────────── phase: core ───────────────────────────
def phase_core(data, vocab, random_loss):
    print("\n" + "=" * 72)
    print("PHASE 1/4  CORE — head-to-head + scaling")
    print("=" * 72)
    specs, runs, battery = {}, {}, {}
    for size in GA_SIZES:
        specs[size] = {f: build_family_spec(f, size, vocab) for f in FAMILIES}
        for f, s in specs[size].items():
            print(
                "  [build] %-6s @ %dM: dim %-4d %.2fM depth %d"
                % (f, size, s["dim"], s["params"] / 1e6, s["depth"])
            )
    for size in GA_SIZES:
        train_loaders, eval_loader, ext_loader = make_loaders(GA_SIZES.index(size))
        runs[size] = {}
        seeds = arena.SEED_LIST[:GA_SEEDS]
        for seed in seeds:
            runs[size][seed] = {}
            for fam in FAMILIES:
                t0 = time.perf_counter()
                m, _ = _build_family(fam, size, vocab, seed)
                rec = _run(
                    m,
                    vocab,
                    GA_STEPS,
                    train_loaders[seed],
                    eval_loader,
                    EVAL_GAP,
                    seed=seed,
                )
                runs[size][seed][fam] = rec
                print(
                    "  [train] %-6s @ %dM seed %-4d final %s  %.0f tok/s  "
                    "%.2f GB  (%.0fs)"
                    % (
                        fam,
                        size,
                        seed,
                        (
                            "{:.3f}".format(rec["eval_hist"][-1][1])
                            if rec["eval_hist"]
                            else "NA"
                        ),
                        rec["tok_s"],
                        rec["peak_mem"],
                        time.perf_counter() - t0,
                    )
                )
                if seed == seeds[-1]:
                    ext = evaluate(m, ext_loader, max_batches=6)
                    battery.setdefault(size, {})[fam] = {
                        "extrap_loss": ext,
                        "extrap_delta": (
                            ext - rec["eval_hist"][-1][1] if rec["eval_hist"] else None
                        ),
                    }
    metrics = {}
    for size in GA_SIZES:
        metrics[size] = {}
        for fam in FAMILIES:
            recs = [runs[size][s][fam] for s in arena.SEED_LIST[:GA_SEEDS]]
            row = _agg(recs)
            row["tok_s_train"] = row["tok_s"]
            row["params"] = specs[size][fam]["params"]
            row["depth"] = specs[size][fam]["depth"]
            row["param_eff"] = (
                ((random_loss - row["final_eval"]) / (row["params"] / 1e6))
                if row["final_eval"]
                else None
            )
            if fam in battery.get(size, {}):
                row.update(battery[size][fam])
            metrics[size][fam] = row
    scaling = {
        f: fit_scaling(
            [
                (metrics[s][f]["params"], metrics[s][f]["final_eval"])
                for s in GA_SIZES
                if metrics[s][f]["final_eval"]
            ]
        )
        for f in FAMILIES
    }
    for f in FAMILIES:
        for size in GA_SIZES:
            metrics[size][f]["scaling_gain"] = -scaling[f]["slope"]
    primary = max(GA_SIZES)
    scores = {s: composite_score(metrics[s], arena.METRIC_SPEC) for s in GA_SIZES}
    verdict = make_verdict(metrics, scores, scaling, primary, random_loss)
    return {
        "metrics": metrics,
        "scaling": scaling,
        "scores": scores,
        "verdict": verdict,
    }


# ─────────────────────────── phase: ablate ───────────────────────────
def phase_ablate(data, vocab, random_loss):
    print("\n" + "=" * 72)
    print('PHASE 2/4  ABLATE — NFRA "small but powerful" levers @ primary size')
    print("=" * 72)
    primary = max(GA_SIZES)
    train_loaders, eval_loader, _ext = make_loaders(GA_SIZES.index(primary))
    results = {}
    for name, build_kw, train_kw, n_seeds in ABLATE:
        seeds = arena.SEED_LIST[: min(n_seeds, GA_SEEDS)]
        recs, params = [], None
        build_kw = dict(build_kw)  # copy: never mutate the ABLATE table
        train_kw = dict(train_kw)
        for seed in seeds:
            t0 = time.perf_counter()
            fam = build_kw.pop("fam", "nfra")
            m, params = _build_family(fam, primary, vocab, seed, **build_kw)
            rec = _run(
                m,
                vocab,
                GA_STEPS,
                train_loaders[seed],
                eval_loader,
                EVAL_GAP,
                seed=seed,
                **train_kw,
            )
            recs.append(rec)
            print(
                "  [train] %-16s seed %-4d final %s  %.0f tok/s  (%.0fs)"
                % (
                    name,
                    seed,
                    (
                        "{:.3f}".format(rec["eval_hist"][-1][1])
                        if rec["eval_hist"]
                        else "NA"
                    ),
                    rec["tok_s"],
                    time.perf_counter() - t0,
                )
            )
        row = _agg(recs)
        row["params"] = params
        row["seeds"] = len(recs)
        results[name] = row
    return results


# ─────────────────────────── phase: recall ───────────────────────────
def phase_recall(data):
    print("\n" + "=" * 72)
    print("PHASE 3/4  RECALL — H3 root-cause diagnostic (concurrent)")
    print("=" * 72)
    from nfra.benchmark import recall_diag

    dim = int(os.environ.get("NFRA_DIAG_DIM", "224"))
    steps = int(os.environ.get("NFRA_DIAG_STEPS", "600"))
    batch = int(os.environ.get("NFRA_DIAG_BATCH", "8"))
    seq = int(os.environ.get("NFRA_DIAG_SEQ", "256"))
    results, wall, floor = recall_diag.run(dim, steps, seq, batch)
    return {"results": results, "wall_s": wall, "floor": floor, "dim": dim}


# ─────────────────────────── phase: perf ───────────────────────────
def phase_perf(data, vocab):
    print("\n" + "=" * 72)
    print("PHASE 4/4  PERF — inference battery + 2x extrapolation")
    print("=" * 72)
    primary = max(GA_SIZES)
    out = {}
    for fam in FAMILIES:
        torch.manual_seed(0)
        m, _ = _build_family(fam, primary, vocab, 0)
        pre = prefill_tok_s(m, max(arena.PRE_HEAD, 8), SEQ_LEN, vocab)
        gen = generate_metrics(m, vocab)
        out[fam] = {
            "prefill_tok_s": pre,
            "gen_tok_s": gen["gen_tok_s"],
            "ms_per_token": gen["ms_per_token"],
            "infer_mem": gen["infer_mem"],
        }
        print(
            "  [perf] %-6s prefill %5.0f tok/s | gen %5.1f tok/s | "
            "%5.2f ms/tok | %.2f GB"
            % (fam, pre, gen["gen_tok_s"], gen["ms_per_token"], gen["infer_mem"])
        )
    return out


# ─────────────────────────── report ───────────────────────────
def fmt(v, nd=3, suffix=""):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "-"
    return f"{v:.{nd}f}{suffix}"


def md_table(headers, rows):
    out = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def build_report(all_data, vocab, random_loss):
    L = []
    a = L.append
    a("# GLOBAL ARENA — all-axes NFRA comparison (overnight)\n")
    a(
        "**Families:** %s (param-matched)  |  "
        "**Data:** %s  |  **Vocab:** %d  |  **Steps:** %d  |  **Sizes:** %sM  |  "
        "**Seeds:** %d\n"
        % (" vs ".join(FAMILIES), GA_DATA, vocab, GA_STEPS, GA_SIZES, GA_SEEDS)
    )
    a(
        "**Optimizer:** AdamW 3e-4 (warmup+cosine)  **Device:** %s\n"
        % ("GPU " + torch.cuda.get_device_name(0) if HAS_CUDA else "CPU")
    )

    if "core" in all_data:
        c = all_data["core"]
        a("\n## 1. Core — head-to-head + scaling\n")
        a("\n### 1a. Eval loss by size (mean ± std over seeds)\n")
        rows = []
        for size in GA_SIZES:
            for fam in FAMILIES:
                m = c["metrics"][size][fam]
                rows.append(
                    [
                        f"{size}M",
                        fam,
                        fmt(m["final_eval"], 3),
                        fmt(m["final_eval_sd"], 3),
                        fmt(m["ppl"], 2),
                        fmt(m["sample_auc"], 3),
                        f"{m['tok_s']:.0f}",
                        fmt(m["peak_mem"], 2),
                        fmt(m["nan_steps"], 0),
                    ]
                )
        a(
            md_table(
                [
                    "size",
                    "family",
                    "eval",
                    "mean±std",
                    "ppl",
                    "AUC",
                    "train tok/s",
                    "peak GB",
                    "NaN steps",
                ],
                rows,
            )
        )
        a("")
        a("\n### 1b. Scaling (bits of loss per doubling of params; neg = better)\n")
        rows = [
            [
                f,
                fmt(c["scaling"][f]["slope"], 4),
                fmt(c["scaling"][f].get("r2"), 3),
                fmt(c["scaling"][f].get("loss_100m"), 3),
                fmt(c["scaling"][f].get("loss_1b"), 3),
            ]
            for f in FAMILIES
        ]
        a(md_table(["family", "slope", "R²", "extrap @100M", "@1B"], rows))
        a("")
        primary = max(GA_SIZES)
        a("\n### 1c. Composite scores @ %dM\n" % primary)
        rows = [[f, f"{c['scores'][primary][f]:.1f}"] for f in FAMILIES]
        a(md_table(["family", "score"], rows))
        a("")
        a("\n### 1d. Verdict\n")
        for cl in c["verdict"]["claims"]:
            ev = f" — {cl['evidence']}" if cl.get("evidence") else ""
            a(f"- **{cl['claim']}:** {cl['family']} ({cl['status']}){ev}")
        a("")

    if "ablate" in all_data:
        ab = all_data["ablate"]
        a("\n## 2. Ablate — NFRA levers @ %dM\n" % max(GA_SIZES))
        base = ab.get("nfra_baseline")
        rows = []
        for name, row in ab.items():
            if base and name != "nfra_baseline":
                dE = (
                    (row["final_eval"] - base["final_eval"])
                    if row["final_eval"] and base["final_eval"]
                    else None
                )
                dT = (
                    (row["tok_s"] / base["tok_s"] - 1.0) * 100
                    if base["tok_s"]
                    else None
                )
                rows.append(
                    [
                        name,
                        fmt(row["final_eval"], 3),
                        fmt(dE, 3, " Δ"),
                        f"{row['tok_s']:.0f}",
                        fmt(dT, 1, "% Δ"),
                        row["seeds"],
                    ]
                )
            else:
                rows.append(
                    [
                        name,
                        fmt(row["final_eval"], 3),
                        "—",
                        f"{row['tok_s']:.0f}",
                        "—",
                        row["seeds"],
                    ]
                )
        a(md_table(["variant", "eval", "Δ vs base", "tok/s", "Δ tok/s", "seeds"], rows))
        a("")
        if base:
            best = min(ab.items(), key=lambda kv: kv[1]["final_eval"] or 9e9)
            if (
                best[0] != "nfra_baseline"
                and best[1]["final_eval"]
                and base["final_eval"]
            ):
                d = base["final_eval"] - best[1]["final_eval"]
                a(
                    f"**Best variant:** {best[0]} (eval {fmt(best[1]['final_eval'],3)}, "
                    f"{d:+.3f} vs baseline)\n"
                )

    if "recall" in all_data:
        r = all_data["recall"]
        a(
            "\n## 3. Recall — H3 root-cause diagnostic (dim %d, floor %.3f)\n"
            % (r["dim"], r["floor"])
        )
        rows = []
        for name, rr in r["results"].items():
            rows.append(
                [
                    rr.get("label", name),
                    rr["train_first"],
                    rr["train_last"],
                    rr["span_ce"],
                    rr["span_acc"],
                    rr["pred_identity_dev"],
                ]
            )
        a(
            md_table(
                [
                    "variant",
                    "train first",
                    "train last",
                    "span CE",
                    "span acc",
                    "pred |W-I|",
                ],
                rows,
            )
        )
        a("")
        fix = r["results"].get("fix")
        nosh = r["results"].get("noshare")
        k1 = r["results"].get("k1")
        if fix and fix["span_ce"] < r["floor"] - 0.03:
            a(
                "**Recall verdict:** `fix` now LEARNS k=4 → the self-prediction "
                "residual was the root cause.\n"
            )
        elif (
            nosh
            and fix
            and nosh["span_ce"] < r["floor"] - 0.03
            and fix["span_ce"] >= r["floor"] - 0.03
        ):
            a(
                "**Recall verdict:** `fix` still floors but `noshare` learns → "
                "depth weight-sharing is the culprit.\n"
            )
        elif k1 and k1["span_ce"] < r["floor"] - 0.03:
            a(
                "**Recall verdict:** k=1 learns but k=4 does not → memory "
                "formability; bet on memory levers (AFC-α).\n"
            )
        else:
            a(
                "**Recall verdict:** nothing learns → capacity/optimization at "
                "dim 224; run the dim-512 probe.\n"
            )

    if "perf" in all_data:
        a("\n## 4. Perf — inference battery + extrapolation @ %dM\n" % max(GA_SIZES))
        rows = [
            [
                fam,
                f"{v['prefill_tok_s']:.0f}",
                f"{v['gen_tok_s']:.1f}",
                f"{v['ms_per_token']:.2f}",
                fmt(v["infer_mem"], 2),
            ]
            for fam, v in all_data["perf"].items()
        ]
        a(
            md_table(
                [
                    "family",
                    "prefill tok/s",
                    "gen tok/s (b=1)",
                    "ms/token",
                    "peak infer GB",
                ],
                rows,
            )
        )
        a("")

    a(
        "\n---\n*Methodology: param-matched families, identical data/optimizer/"
        "schedule/token budget, multiple seeds, multiple sizes; pure-PyTorch "
        "speed is a lower bound. Full per-run data in global_arena_results.json.*\n"
    )
    return "\n".join(L)


# ─────────────────────────── main ───────────────────────────
def main():
    if not HAS_CUDA:
        print(
            "[WARN] no CUDA — overnight run is meaningless on CPU. " "Run on Kaggle T4."
        )
    use_wiki = GA_DATA == "wikitext2"
    VOCAB = len(CHAR_VOCAB) if use_wiki else 4096
    RANDOM_LOSS = math.log(VOCAB)

    print("=" * 72)
    print("  GLOBAL ARENA — all-axes global-standard comparison")
    print("  NFRA Brain  vs  {}  (param-matched)".format(" vs ".join(FAMILIES)))
    print("=" * 72)
    print("  data   : %-12s vocab: %d   sizes: %s M" % (GA_DATA, VOCAB, GA_SIZES))
    print(
        "  steps  : %d    seeds: %d    phases: %s"
        % (GA_STEPS, GA_SEEDS, ",".join(GA_PHASES))
    )
    print("  device : %s" % (torch.cuda.get_device_name(0) if HAS_CUDA else "CPU"))
    print(f"  out    : {OUT_JSON}")
    print("=" * 72)

    data = {}
    t_all = time.perf_counter()
    for phase in GA_PHASES:
        try:
            if phase == "core":
                data["core"] = phase_core(data, VOCAB, RANDOM_LOSS)
            elif phase == "ablate":
                data["ablate"] = phase_ablate(data, VOCAB, RANDOM_LOSS)
            elif phase == "recall":
                data["recall"] = phase_recall(data)
            elif phase == "perf":
                data["perf"] = phase_perf(data, VOCAB)
            else:
                print(f"[skip] unknown phase {phase!r}")
                continue
            with open(OUT_JSON, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=float)
            print(f"[phase {phase} done] progress saved -> {OUT_JSON}")
        except Exception as e:
            print(f"[PHASE {phase} FAILED] {e!r} (continuing to next phase)")
            import traceback

            traceback.print_exc()

    report = build_report(data, VOCAB, RANDOM_LOSS)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + "=" * 72)
    print(
        f"  GLOBAL ARENA DONE in {time.perf_counter() - t_all:.0f}s  -> {OUT_MD}  -> {OUT_JSON}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
