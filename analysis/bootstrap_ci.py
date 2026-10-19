#!/usr/bin/env python3
"""
bootstrap_ci.py — paired bootstrap 95% CI of case-averaged differences.

Cases of two evaluation summaries (nwp_comparison_summary.json) are matched by
case_label; the matched cases are resampled with replacement B times, and the
2.5th / 97.5th percentiles of the resampled case-averaged difference (A - B)
give the CI. A CI excluding 0 is significant at alpha = 0.05 (two-sided).
Same procedure as the manuscript (B = 10000, seed 42, case-averaged metrics).

Each side is given as PATH or PATH:BRANCH, where BRANCH is `ours` (PA-MCDNet,
default) or `paper` (UNet-VDA, evaluated in the same run). PA-MCDNet vs.
UNet-VDA on the locked real-observation set:

    python analysis/bootstrap_ci.py \
        --a eval_out/realobs_locked200/nwp_comparison_summary.json:ours \
        --b eval_out/realobs_locked200/nwp_comparison_summary.json:paper

Note: the manuscript ran several comparisons from one random stream, so the
last digit of a CI bound can differ when a single pair is run on its own.
"""
import argparse
import json

import numpy as np

METRICS = [
    ("SR (pp)", "correction_success_rate", 100.0),
    ("FPR (pp)", "false_positive_rate", 100.0),
    ("RMSE (m/s)", "rmse", 1.0),
]


def load_cases(spec):
    path, _, branch = spec.rpartition(":") if spec.endswith((":ours", ":paper")) else (spec, "", "ours")
    with open(path, encoding="utf-8") as f:
        per_case = json.load(f)["per_case"]
    out = {}
    for c in per_case:
        m = c[f"{branch}_metrics"]
        out[c["case_label"]] = {key: float(m[key]) for _, key, _ in METRICS}
    return out, f"{path}:{branch}"


def bootstrap_ci(a, b, rng, n_boot):
    n = len(a)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return a.mean() - b.mean(), np.percentile(diffs, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="model A: PATH[:ours|:paper]")
    ap.add_argument("--b", required=True, help="model B: PATH[:ours|:paper]")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    A, name_a = load_cases(args.a)
    B, name_b = load_cases(args.b)
    common = sorted(set(A) & set(B))
    rng = np.random.default_rng(args.seed)

    print(f"A = {name_a}\nB = {name_b}\nmatched cases: {len(common)}, B = {args.n_boot}\n")
    for label, key, scale in METRICS:
        a = np.array([A[c][key] for c in common]) * scale
        b = np.array([B[c][key] for c in common]) * scale
        d, (lo, hi) = bootstrap_ci(a, b, rng, args.n_boot)
        sig = "significant" if (lo > 0 or hi < 0) else "not significant"
        print(f"  delta {label:<11} = {d:+8.3f}   95% CI [{lo:+8.3f}, {hi:+8.3f}]   {sig}")


if __name__ == "__main__":
    main()
