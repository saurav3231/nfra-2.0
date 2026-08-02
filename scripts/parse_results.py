#!/usr/bin/env python3
"""
NFRA Results Parser — parses existing benchmark JSON files and
classifies mechanisms as WORKING, NEUTRAL, or HARMFUL.

Can parse:
  - global_arena_results.json (ablate phase results)
  - overnight_results.json (full overnight run results)
  - mechanism_analysis.json (output from analyze_mechanisms.py)

Usage:
    python -m scripts.parse_results --results global_arena_results.json
    python -m scripts.parse_results --results overnight_results.json
"""

import argparse
import json
import math
import os
import sys


def load_results(path):
    with open(path) as f:
        return json.load(f)


def extract_ablate(data):
    """Extract ablate phase results from global_arena_results.json."""
    ablate = data.get("ablate", {})
    if not ablate:
        return None

    variants = {}
    for name, metrics in ablate.items():
        variants[name] = {
            "final_eval": metrics.get("final_eval"),
            "final_eval_sd": metrics.get("final_eval_sd"),
            "tok_s": metrics.get("tok_s"),
            "peak_mem": metrics.get("peak_mem"),
            "params": metrics.get("params"),
            "seeds": metrics.get("seeds"),
        }
    return variants


def extract_overnight_ablate(data):
    """Extract ablate results from overnight_results.json format."""
    # overnight_results.json has a different structure
    # Look for ablate data in the metrics section
    metrics = data.get("metrics", {})
    if not metrics:
        return None

    # The ablate results are stored per-size
    for size_data in metrics.values():
        for fam in size_data:
            if fam.startswith("nfra_") or fam in ("nfra_baseline", "nfra_all"):
                return True  # Has ablate data
    return None


def classify_from_aggregated(variants):
    """Classify mechanisms from aggregated (mean/std) results."""
    baseline = variants.get("nfra_baseline")
    if not baseline or baseline.get("final_eval") is None:
        print("ERROR: No baseline found in results!")
        return {}

    baseline_eval = baseline["final_eval"]
    baseline_sd = baseline.get("final_eval_sd") or 0

    classifications = {}
    for name, metrics in variants.items():
        if name == "nfra_baseline":
            classifications[name] = {"status": "baseline", "delta": 0.0}
            continue

        variant_eval = metrics.get("final_eval")
        if variant_eval is None:
            classifications[name] = {"status": "no_data", "delta": None}
            continue

        delta = variant_eval - baseline_eval
        variant_sd = metrics.get("final_eval_sd") or 0
        (
            math.sqrt(baseline_sd**2 + variant_sd**2)
            if (baseline_sd + variant_sd) > 0
            else 0
        )

        threshold = max(0.02, 0.05 * abs(baseline_eval))

        if delta < -threshold:
            status = "WORKING"
        elif delta > threshold:
            status = "HARMFUL"
        else:
            status = "NEUTRAL"

        lever = name.replace("nfra_", "") if name.startswith("nfra_") else name
        classifications[lever] = {
            "status": status,
            "delta": round(delta, 4),
            "delta_pct": (
                round(delta / baseline_eval * 100, 2) if baseline_eval != 0 else 0
            ),
            "baseline_eval": baseline_eval,
            "variant_eval": variant_eval,
            "baseline_sd": baseline_sd,
            "variant_sd": variant_sd,
        }

    return classifications


def generate_report_from_aggregated(variants, classifications, outdir):
    baseline = variants.get("nfra_baseline", {})
    baseline_eval = baseline.get("final_eval", 0)

    lines = []
    a = lines.append

    a("# NFRA Mechanism Analysis Report (Aggregated)")
    a("")
    a(f"**Baseline eval loss (nfra_baseline):** {baseline_eval:.4f}")
    a(f"**Total variants analyzed:** {len(classifications)}")
    a("")

    a("## Summary")
    a("")
    a("| Lever | Status | Delta (eval loss) | Delta (%) |")
    a("|-------|--------|-------------------|-----------|")

    working = []
    neutral = []
    harmful = []

    for lever, info in sorted(classifications.items()):
        if info["status"] == "baseline":
            continue
        if info["status"] == "no_data":
            continue

        delta_str = f"{info['delta']:+.4f}"
        pct_str = f"{info['delta_pct']:+.1f}%"

        a(f"| {lever} | **{info['status']}** | {delta_str} | {pct_str} |")

        if info["status"] == "WORKING":
            working.append(lever)
        elif info["status"] == "HARMFUL":
            harmful.append(lever)
        else:
            neutral.append(lever)

    a("")

    if working:
        a("### Working Mechanisms")
        a("")
        for lever in working:
            info = classifications[lever]
            a(
                f"- **{lever}**: delta = {info['delta']:+.4f} ({info['delta_pct']:+.1f}%)"
            )
        a("")

    if neutral:
        a("### Neutral Mechanisms")
        a("")
        for lever in neutral:
            info = classifications[lever]
            a(
                f"- **{lever}**: delta = {info['delta']:+.4f} ({info['delta_pct']:+.1f}%)"
            )
        a("")

    if harmful:
        a("### Harmful Mechanisms")
        a("")
        for lever in harmful:
            info = classifications[lever]
            a(
                f"- **{lever}**: delta = {info['delta']:+.4f} ({info['delta_pct']:+.1f}%)"
            )
        a("")

    a("## Recommendations")
    a("")
    if working:
        a("1. **Keep** the working mechanisms.")
    if neutral:
        a(f"2. **Remove** the {len(neutral)} neutral mechanisms.")
    if harmful:
        a(f"3. **Remove** the {len(harmful)} harmful mechanisms.")
    a("")

    report = "\n".join(lines)
    out_md = os.path.join(outdir, "mechanism_analysis_aggregated.md")
    with open(out_md, "w") as f:
        f.write(report)
    print(f"  Report saved -> {out_md}")

    return report


def main():
    parser = argparse.ArgumentParser(description="NFRA Results Parser")
    parser.add_argument(
        "--results", type=str, default=None, help="Path to results JSON file"
    )
    parser.add_argument(
        "--outdir", type=str, default=None, help="Output directory for reports"
    )
    args = parser.parse_args()

    outdir = args.outdir or os.getcwd()
    os.makedirs(outdir, exist_ok=True)

    if args.results:
        path = args.results
    else:
        for fname in [
            "global_arena_results.json",
            "overnight_results.json",
            "mechanism_analysis.json",
        ]:
            p = os.path.join(os.getcwd(), fname)
            if os.path.exists(p):
                path = p
                break
        else:
            print("No results file found. Specify --results <path>")
            sys.exit(1)

    print(f"Loading results from {path} ...")
    data = load_results(path)

    # Try to extract ablate results
    variants = extract_ablate(data)
    if variants is None:
        # Try overnight format
        variants = extract_overnight_ablate(data)
        if variants is None:
            print("ERROR: Could not find ablate results in the file!")
            print(f"Top-level keys: {list(data.keys())}")
            sys.exit(1)

    if isinstance(variants, bool):
        print("Found ablate data but in a format that requires per-seed analysis.")
        print("Use analyze_mechanisms.py --run for per-seed classification.")
        sys.exit(0)

    print(f"\nFound {len(variants)} variants in ablate results.")

    print("\nClassifying mechanisms ...")
    classifications = classify_from_aggregated(variants)

    print("\nGenerating report ...")
    generate_report_from_aggregated(variants, classifications, outdir)

    print("\n" + "=" * 72)
    print("  CLASSIFICATION SUMMARY")
    print("=" * 72)
    for lever, info in sorted(classifications.items()):
        if info["status"] == "baseline":
            print(f"  {lever:20s} BASELINE")
        elif info["status"] == "no_data":
            print(f"  {lever:20s} NO DATA")
        else:
            print(
                f"  {lever:20s} {info['status']:10s}  delta = {info['delta']:+.4f} ({info['delta_pct']:+.1f}%)"
            )
    print("=" * 72)


if __name__ == "__main__":
    main()
