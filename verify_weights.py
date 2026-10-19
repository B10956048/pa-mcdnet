"""
verify_weights.py — functional check of the released PA-MCDNet weights.

Runs WITHOUT any CWA data. It
  1. verifies the MD5 checksum of the weight file against the archived value,
  2. builds the physics-aware model and loads the weights,
  3. dealiases a synthetic, analytically folded vortex (a Rankine vortex plus a
     uniform background wind, projected onto the radar beam and folded at a
     chosen Nyquist velocity), using the same inference path as
     test_nwp_comparison.py (model forward pass + physics constraint),
  4. reports how many folded gates were restored to the true velocity.

This confirms the archive is complete and the inference pipeline runs end to
end. It is NOT a reproduction of the manuscript metrics, which are computed on
CWA radar data (see REPRODUCE.md, Level 3).

Usage:
    python verify_weights.py --model_path weights/pa_mcdnet_M6_headline.h5
"""
import fix_typing  # noqa: F401  (must be imported first, as in the other entry points)

import argparse
import hashlib
import os
import sys

import numpy as np

EXPECTED_MD5 = "1b1685ec72c7a765a6f957afb518a551"  # pa_mcdnet_M6_headline.h5 (Zenodo)


def md5sum(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def synthetic_folded_vortex(naz=360, ngates=400, gate_km=0.25, nyquist=21.16,
                            vmax=50.0, rmw_km=15.0, cx_km=40.0, cy_km=40.0,
                            u_bg=5.0, v_bg=3.0):
    """Return (true_vr, folded_vr) on a (naz, ngates) polar grid."""
    az = np.deg2rad(np.arange(naz) * 360.0 / naz)[:, None]   # 0 = north, clockwise
    r = (np.arange(ngates) + 0.5)[None, :] * gate_km
    x, y = r * np.sin(az), r * np.cos(az)                      # east, north (km)

    dx, dy = x - cx_km, y - cy_km
    d = np.hypot(dx, dy) + 1e-6
    vt = np.where(d <= rmw_km, vmax * d / rmw_km, vmax * rmw_km / d)
    u = -vt * dy / d + u_bg                                    # cyclonic rotation
    v = vt * dx / d + v_bg

    true_vr = u * np.sin(az) + v * np.cos(az)                  # radial component
    true_vr = true_vr.astype(np.float32)
    true_vr[:, :8] = np.nan                                    # no data near the radar

    folded = (true_vr + nyquist) % (2.0 * nyquist) - nyquist
    return true_vr, folded.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_path", required=True,
                    help="released weight file, e.g. weights/pa_mcdnet_M6_headline.h5")
    ap.add_argument("--nyquist", type=float, default=21.16)
    ap.add_argument("--skip_md5", action="store_true",
                    help="skip the checksum test (e.g. for a self-trained weight)")
    args = ap.parse_args()

    if not os.path.isfile(args.model_path):
        sys.exit(f"[FAIL] weight file not found: {args.model_path}")

    if not args.skip_md5:
        got = md5sum(args.model_path)
        if got != EXPECTED_MD5:
            sys.exit(f"[FAIL] MD5 mismatch: {got} (expected {EXPECTED_MD5}). "
                     "Re-download the weight from the Zenodo record.")
        print(f"[OK] MD5 matches the archived weight ({got})")

    from mixed_patch_test_model import build_mixed_patch_model_for_inference, auto_zero_pad
    from test_nwp_comparison import run_our_model_on_cwa, apply_physics_constraint

    nyq = float(args.nyquist)
    truth, raw = synthetic_folded_vortex(nyquist=nyq)

    model = build_mixed_patch_model_for_inference(use_physics_model=True)
    pad, _, _ = auto_zero_pad(raw, layers=4, fill_value=0)
    _ = model({"vel": pad[None, None, :, :, None].astype(np.float32),
               "nyq": np.array([[nyq]], dtype=np.float32)}, training=False)
    model.load_weights(args.model_path)
    print(f"[OK] model built and weights loaded: {args.model_path}")

    pred, _ = run_our_model_on_cwa(model, raw, nyq)
    ours = apply_physics_constraint(pred, raw, nyq)

    valid = ~np.isnan(truth) & ~np.isnan(ours)
    folded = valid & (np.abs(truth - raw) > nyq)               # gates that were aliased
    correct = valid & (np.abs(ours - truth) < 0.5 * nyq)       # right Nyquist interval
    rmse = float(np.sqrt(np.nanmean((ours[valid] - truth[valid]) ** 2)))

    n_f = int(folded.sum())
    print(f"\nSynthetic vortex, V_nyq = {nyq:.2f} m/s, {int(valid.sum()):,} valid gates")
    print(f"  folded gates             : {n_f:,} ({n_f / valid.sum():.1%})")
    print(f"  folded gates restored    : {int((folded & correct).sum()):,} "
          f"({(folded & correct).sum() / max(n_f, 1):.1%})")
    print(f"  all gates in right fold  : {correct[valid].mean():.1%}")
    print(f"  RMSE vs. true velocity   : {rmse:.2f} m/s")
    print("\n[DONE] Functional check finished. These numbers describe one synthetic "
          "field only and are not the manuscript metrics.")


if __name__ == "__main__":
    main()
