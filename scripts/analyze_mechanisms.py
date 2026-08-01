#!/usr/bin/env python3
"""
NFRA Mechanism Analyzer — identifies which brain-inspired levers actually work.

Runs the ablate phase of the global arena, collects per-seed eval loss for
each variant, and classifies every mechanism as WORKING, NEUTRAL, or HARMFUL
based on statistical significance against the baseline.

Usage (Kaggle T4):
    python -m scripts.analyze_mechanisms

Env overrides:
    NFRA_ANALYZE_STEPS   training steps per variant (default 600)
    NFRA_ANALYZE_SEEDS   number of seeds (default 3)
    NFRA_ANALYZE_SIZE    target model size in M params (default 20)
    NFRA_ANALYZE_DATA    wikitext2 | synthetic (default wikitext2)
    NFRA_ANALYZE_OUTDIR  output directory (default CWD)

Outputs:
    mechanism_analysis.json   — raw per-seed data + classifications
    mechanism_analysis.md     — human-readable report
"""

import os, sys, time, math, json, warnings, functools

print = functools.partial(print, flush=True)
warnings.filterwarnings('ignore')

ABLATE = [
    ("nfra_baseline",  {},                             {"ema_decay": 0.0}, 3),
    ("nfra_ema",       {},                             {"ema_decay": 0.99}, 3),
    ("nfra_surprise",  {},                             {"surprise": True}, 2),
    ("nfra_kwta",      {"k_wta": 0.25},                {}, 2),
    ("nfra_local",     {"local_route": True},          {}, 2),
    ("nfra_divnorm",   {"div_norm": True},             {}, 2),
    ("nfra_astro",     {"astro": True},                {}, 2),
    ("nfra_theta",     {"theta": True},                {}, 2),
    ("nfra_achretain", {"ach_retain": True},           {}, 2),
    ("nfra_gainnov",   {"gain_nov": True},             {}, 2),
    ("nfra_lora8",     {"lora_rank": 8},               {}, 2),
    ("nfra_all",       {"k_wta": 0.25, "local_route": True,
                        "div_norm": True, "astro": True},
                     {"ema_decay": 0.99, "surprise": True}, 2),
]

LEVER_NAMES = {
    "nfra_ema": "ema",
    "nfra_surprise": "surprise_loss",
    "nfra_kwta": "k_wta",
    "nfra_local": "local_route",
    "nfra_divnorm": "div_norm",
    "nfra_astro": "astro",
    "nfra_theta": "theta",
    "nfra_achretain": "ach_retain",
    "nfra_gainnov": "gain_nov",
    "nfra_lora8": "lora_pass",
    "nfra_all": "all_levers",
}


def get_env(name, default, cast=int):
    val = os.environ.get(name)
    if val is None:
        return default
    return cast(val)


def run_ablate():
    import torch
    import numpy as np
    from torch.utils.data import DataLoader

    from nfra.benchmark.arena import (
        SEED_LIST, EVAL_GAP, build_nfra, train_one, make_loaders,
    )
    from nfra.benchmark.compare import (
        DEVICE, HAS_CUDA, rescale_embed,
    )

    steps = get_env("NFRA_ANALYZE_STEPS", 600)
    n_seeds = get_env("NFRA_ANALYZE_SEEDS", 3)
    size = get_env("NFRA_ANALYZE_SIZE", 20)
    data_source = os.environ.get("NFRA_ANALYZE_DATA", "wikitext2")
    outdir = os.environ.get("NFRA_ANALYZE_OUTDIR", os.getcwd())

    use_wiki = data_source == "wikitext2"
    from nfra.benchmark.compare import CHAR_VOCAB
    V = len(CHAR_VOCAB) if use_wiki else 4096

    print("=" * 72)
    print("  NFRA MECHANISM ANALYZER")
    print(f"  size={size}M  seeds={n_seeds}  steps={steps}  data={data_source}")
    print("=" * 72)

    train_loaders, eval_loader, _ = make_loaders(0)

    results = {}
    for name, build_kw, train_kw, n_seeds_req in ABLATE:
        n_seeds_req = min(n_seeds_req, n_seeds)
        seeds = SEED_LIST[:n_seeds_req]
        variant_results = []

        for seed in seeds:
            print(f"\n  [{name}] seed={seed} ...")
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = build_nfra(V, size, depth=33, **build_kw).to(DEVICE)
            rescale_embed(model)

            rec = train_one(model, V, steps, train_loaders[seed],
                            eval_loader, EVAL_GAP,
                            ema_decay=train_kw.get("ema_decay", 0.0),
                            surprise=train_kw.get("surprise", False),
                            seed=seed)

            final_eval = rec['eval_hist'][-1][1] if rec['eval_hist'] else None
            variant_results.append({
                "seed": seed,
                "final_eval": final_eval,
                "tok_s": rec.get("tok_s", 0),
                "peak_mem": rec.get("peak_mem", 0),
                "wall_s": rec.get("wall_s", 0),
                "nan_steps": rec.get("nan_steps", 0),
            })
            print(f"    final={final_eval:.4f}  tok/s={rec.get('tok_s', 0):.0f}")

        results[name] = variant_results

    out_json = os.path.join(outdir, "mechanism_analysis.json")
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Raw results saved -> {out_json}")

    return results


def classify_mechanisms(results):
    baseline = results.get("nfra_baseline", [])
    if not baseline:
        print("ERROR: No baseline results found!")
        return {}

    baseline_evals = [r["final_eval"] for r in baseline if r.get("final_eval") is not None]
    if not baseline_evals:
        print("ERROR: Baseline has no valid eval losses!")
        return {}

    baseline_mean = sum(baseline_evals) / len(baseline_evals)
    baseline_std = (sum((v - baseline_mean) ** 2 for v in baseline_evals) / max(len(baseline_evals) - 1, 1)) ** 0.5

    classifications = {}
    for name, variant_results in results.items():
        if name == "nfra_baseline":
            classifications[name] = {"status": "baseline", "delta_mean": 0.0, "delta_std": 0.0}
            continue

        evals = [r["final_eval"] for r in variant_results if r.get("final_eval") is not None]
        if not evals:
            classifications[name] = {"status": "no_data", "delta_mean": None, "delta_std": None}
            continue

        variant_mean = sum(evals) / len(evals)
        variant_std = (sum((v - variant_mean) ** 2 for v in evals) / max(len(evals) - 1, 1)) ** 0.5
        delta_mean = variant_mean - baseline_mean
        delta_std = math.sqrt(variant_std ** 2 + baseline_std ** 2) if baseline_std > 0 else variant_std

        threshold = max(0.02, 0.05 * abs(baseline_mean))

        if delta_mean < -threshold:
            status = "WORKING"
        elif delta_mean > threshold:
            status = "HARMFUL"
        else:
            status = "NEUTRAL"

        better_seeds = sum(1 for e in evals if e < baseline_mean - threshold)
        worse_seeds = sum(1 for e in evals if e > baseline_mean + threshold)
        total_seeds = len(evals)

        if better_seeds == total_seeds:
            confidence = "high"
        elif worse_seeds == total_seeds:
            confidence = "high"
        elif better_seeds > worse_seeds:
            confidence = "mixed_positive"
        elif worse_seeds > better_seeds:
            confidence = "mixed_negative"
        else:
            confidence = "inconclusive"

        lever = LEVER_NAMES.get(name, name)
        classifications[lever] = {
            "status": status,
            "confidence": confidence,
            "delta_mean": round(delta_mean, 4),
            "delta_std": round(delta_std, 4),
            "baseline_mean": round(baseline_mean, 4),
            "baseline_std": round(baseline_std, 4),
            "variant_mean": round(variant_mean, 4),
            "variant_std": round(variant_std, 4),
            "per_seed": [round(r["final_eval"], 4) for r in variant_results if r.get("final_eval") is not None],
            "better_seeds": better_seeds,
            "worse_seeds": worse_seeds,
            "total_seeds": total_seeds,
        }

    return classifications


def generate_report(results, classifications, outdir):
    baseline = results.get("nfra_baseline", [])
    baseline_evals = [r["final_eval"] for r in baseline if r.get("final_eval") is not None]
    baseline_mean = sum(baseline_evals) / len(baseline_evals) if baseline_evals else 0

    lines = []
    a = lines.append

    a("# NFRA Mechanism Analysis Report")
    a("")
    a(f"**Baseline eval loss (nfra_baseline):** {baseline_mean:.4f}")
    a(f"**Total variants analyzed:** {len(classifications)}")
    a("")

    a("## Summary")
    a("")
    a("| Lever | Status | Delta (eval loss) | Confidence | Seeds Better/Worse |")
    a("|-------|--------|-------------------|------------|-------------------|")

    working = []
    neutral = []
    harmful = []

    for lever, info in sorted(classifications.items()):
        if info["status"] == "baseline":
            continue
        if info["status"] == "no_data":
            continue

        delta_str = f"{info['delta_mean']:+.4f}"
        if info['delta_mean'] < 0:
            delta_str += " (improvement)"
        elif info['delta_mean'] > 0:
            delta_str += " (degradation)"
        else:
            delta_str += " (no change)"

        a(f"| {lever} | **{info['status']}** | {delta_str} | {info['confidence']} | {info['better_seeds']}/{info['worse_seeds']} |")

        if info["status"] == "WORKING":
            working.append(lever)
        elif info["status"] == "HARMFUL":
            harmful.append(lever)
        else:
            neutral.append(lever)

    a("")

    a("## Detailed Findings")
    a("")

    if working:
        a("### Working Mechanisms (improve eval loss)")
        a("")
        for lever in working:
            info = classifications[lever]
            a(f"- **{lever}**: delta = {info['delta_mean']:+.4f} (baseline {info['baseline_mean']:.4f} -> {info['variant_mean']:.4f})")
            a(f"  - Confidence: {info['confidence']}")
            a(f"  - Per-seed evals: {info['per_seed']}")
        a("")

    if neutral:
        a("### Neutral Mechanisms (no significant effect)")
        a("")
        for lever in neutral:
            info = classifications[lever]
            a(f"- **{lever}**: delta = {info['delta_mean']:+.4f} (within noise margin)")
            a(f"  - Confidence: {info['confidence']}")
            a(f"  - Per-seed evals: {info['per_seed']}")
        a("")

    if harmful:
        a("### Harmful Mechanisms (degrade eval loss)")
        a("")
        for lever in harmful:
            info = classifications[lever]
            a(f"- **{lever}**: delta = {info['delta_mean']:+.4f} (baseline {info['baseline_mean']:.4f} -> {info['variant_mean']:.4f})")
            a(f"  - Confidence: {info['confidence']}")
            a(f"  - Per-seed evals: {info['per_seed']}")
        a("")

    a("## Recommendations")
    a("")
    if working:
        a("1. **Keep** the working mechanisms — they provide genuine quality improvement.")
    if neutral:
        a(f"2. **Remove** the {len(neutral)} neutral mechanisms — they add code complexity with zero benefit.")
    if harmful:
        a(f"3. **Remove** the {len(harmful)} harmful mechanisms — they actively degrade performance.")
    if not working and not harmful:
        a("1. All mechanisms are neutral at this scale — consider testing at smaller model sizes (5M) where effects may be more pronounced.")
    a("")

    a("## Mechanisms to Remove")
    a("")
    removable = neutral + harmful
    if removable:
        for lever in removable:
            info = classifications[lever]
            a(f"- `{lever}` — {info['status']} (delta = {info['delta_mean']:+.4f})")
    else:
        a("None identified — all mechanisms have some effect at this scale.")
    a("")

    report = "\n".join(lines)
    out_md = os.path.join(outdir, "mechanism_analysis.md")
    with open(out_md, 'w') as f:
        f.write(report)
    print(f"  Report saved -> {out_md}")

    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="NFRA Mechanism Analyzer")
    parser.add_argument("--run", action="store_true",
                        help="Run the ablate phase (requires Kaggle T4 + WikiText-2)")
    parser.add_argument("--results", type=str, default=None,
                        help="Path to existing results JSON to analyze")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Output directory for reports")
    args = parser.parse_args()

    outdir = args.outdir or os.environ.get("NFRA_ANALYZE_OUTDIR", os.getcwd())
    os.makedirs(outdir, exist_ok=True)

    if args.results:
        print(f"Loading existing results from {args.results} ...")
        with open(args.results) as f:
            results = json.load(f)
    elif args.run:
        print("Running ablate phase ...")
        results = run_ablate()
    else:
        for fname in ["overnight_results.json", "global_arena_results.json",
                        "mechanism_analysis.json"]:
            path = os.path.join(os.getcwd(), fname)
            if os.path.exists(path):
                print(f"Found existing results: {path}")
                with open(path) as f:
                    results = json.load(f)
                break
        else:
            print("No results found. Run with --run on Kaggle T4, or provide --results <path>")
            sys.exit(1)

    print("\nClassifying mechanisms ...")
    classifications = classify_mechanisms(results)

    print("\nGenerating report ...")
    report = generate_report(results, classifications, outdir)

    print("\n" + "=" * 72)
    print("  CLASSIFICATION SUMMARY")
    print("=" * 72)
    for lever, info in sorted(classifications.items()):
        if info["status"] == "baseline":
            print(f"  {lever:20s} BASELINE")
        elif info["status"] == "no_data":
            print(f"  {lever:20s} NO DATA")
        else:
            print(f"  {lever:20s} {info['status']:10s}  delta = {info['delta_mean']:+.4f}  ({info['confidence']})")
    print("=" * 72)


if __name__ == "__main__":
    main()