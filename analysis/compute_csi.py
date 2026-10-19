#!/usr/bin/env python3
"""
compute_csi.py — case-averaged CSI from an evaluation summary.

Reads the nwp_comparison_summary.json written by test_nwp_comparison.py and
reports the case-averaged CSI used in the manuscript:

    CSI = hits / (hits + misses + false_alarms), per case, then simple mean
      hits   = SR  x need_correction_pixels
      misses = (1 - SR) x need_correction_pixels
      FA     = FPR x clean_pixels
    =>  CSI  = hits / (need_correction_pixels + FA)

Cases with no aliased pixels and no false alarms count as CSI = 1, consistent
with the vacuous handling of SR. Case-averaged SR, FPR and RMSE are printed
alongside for reference.

Usage:
    python analysis/compute_csi.py eval_out/realobs_locked200/nwp_comparison_summary.json
"""
import argparse
import json

import numpy as np

BRANCHES = {"ours": "PA-MCDNet", "paper": "UNet-VDA"}


def case_csi(m):
    need = float(m["need_correction_pixels"])
    clean = float(m["clean_pixels"])
    hits = float(m["correction_success_rate"]) * need
    fa = float(m["false_positive_rate"]) * clean
    denom = need + fa
    return hits / denom if denom > 0 else 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("summary", nargs="+", help="nwp_comparison_summary.json file(s)")
    args = ap.parse_args()

    for path in args.summary:
        with open(path, encoding="utf-8") as f:
            cases = json.load(f)["per_case"]
        print(f"{path}  (n = {len(cases)} cases)")
        print(f"  {'model':<10} {'SR %':>7} {'FPR %':>7} {'RMSE':>7} {'CSI %':>7}")
        for key, name in BRANCHES.items():
            ms = [c.get(f"{key}_metrics") for c in cases]
            if any(m is None for m in ms):
                continue
            sr = np.mean([float(m["correction_success_rate"]) for m in ms]) * 100
            fpr = np.mean([float(m["false_positive_rate"]) for m in ms]) * 100
            rmse = np.mean([float(m["rmse"]) for m in ms])
            csi = np.mean([case_csi(m) for m in ms]) * 100
            print(f"  {name:<10} {sr:7.2f} {fpr:7.2f} {rmse:7.2f} {csi:7.2f}")


if __name__ == "__main__":
    main()
