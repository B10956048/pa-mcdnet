# Reproducing the PA-MCDNet results

This guide accompanies the manuscript *Physics-Aware Deep Learning for Radar
Doppler Velocity Dealiasing Using NWP Simulations and Observations*
(Lin, Zhang, Fang, and Chang; submitted to *Artificial Intelligence for the
Earth Systems*).

## Which archive to use

| Identifier | Use |
|---|---|
| **Version DOI [10.5281/zenodo.23479449](https://doi.org/10.5281/zenodo.23479449)** | The fixed archive for this manuscript. Use and cite this one to reproduce the reported results. |
| Concept DOI [10.5281/zenodo.22167560](https://doi.org/10.5281/zenodo.22167560) | Resolves to the latest version of PA-MCDNet (all versions). |
| GitHub tag `v1.0.1` | The same source code as the version DOI. |

Software release: **v1.0.1**. The "Version" field shown on the Zenodo page
numbers archive records and is not the software version.

## What can be reproduced without CWA data

The radar observations, operational reference products and NWP-simulated fields
belong to Taiwan's Central Weather Administration (CWA) and are not
redistributed (see the manuscript's Data Availability Statement). The
reproduction is therefore organised in levels:

| Level | Needs | Shows |
|---|---|---|
| 1. Archive check | Public files only | The weights are intact and the model loads and runs end to end |
| 2. Inference | Your own radar PPI files | Dealiased velocity fields for any CWA-format sweep |
| 3. Locked evaluation | CWA data + UNet-VDA | The manuscript metrics (SR, FPR, RMSE, CSI) |
| 4. Retraining | CWA data + GPU | The two-stage training (see README) |

All commands are run from the repository root.

## 0. Environment

```bash
conda create -n pa-mcdnet python=3.10 -y
conda activate pa-mcdnet
pip install tensorflow==2.10.0 \
            numpy==1.26.4 scipy==1.15.3 pandas==2.3.2 h5py==3.14.0 \
            pyarrow==22.0.0 matplotlib==3.8.4 seaborn==0.13.2 tqdm==4.67.1 \
            scikit-image==0.25.1 scikit-learn==1.6.1 opencv-python==4.11.0.86
```

Optional geographic plots additionally need
`pip install arm_pyart basemap basemap-data pyproj==3.6.1 pillow`.

## 1. Archive check (public files only)

Get the code (either clone the tag or unzip `pa-mcdnet-code-v1.0.1.zip` from the
Zenodo record), then place the downloaded Zenodo files as follows:

| Zenodo file | Place at | MD5 |
|---|---|---|
| `pa_mcdnet_M6_headline.h5` | `weights/pa_mcdnet_M6_headline.h5` | `1b1685ec72c7a765a6f957afb518a551` |
| `locked_realobs_test_set.json` | `data/` (already in the code) | `3c40202652d3fc80ecaed0d183c98f4d` |
| `locked_nwp_test_set.json` | `data/` (already in the code) | `88ad9747ca652860cb4db08074742077` |

```bash
git clone --branch v1.0.1 https://github.com/B10956048/pa-mcdnet.git
cd pa-mcdnet
mkdir -p weights        # put pa_mcdnet_M6_headline.h5 here
python verify_weights.py --model_path weights/pa_mcdnet_M6_headline.h5
```

`verify_weights.py` checks the MD5 of the weight file, builds the model, loads
the weights and dealiases a synthetic, analytically folded vortex. It needs no
CWA data. Expected output with the archived weight (CPU, ~1 min):

```
[OK] MD5 matches the archived weight (1b1685ec72c7a765a6f957afb518a551)
[OK] model built and weights loaded: weights/pa_mcdnet_M6_headline.h5

Synthetic vortex, V_nyq = 21.16 m/s, 141,120 valid gates
  folded gates             : 9,822 (7.0%)
  folded gates restored    : 9,822 (100.0%)
  all gates in right fold  : 100.0%
  RMSE vs. true velocity   : 0.00 m/s
```

This output describes one synthetic field only; it is a functional
check, not a reproduction of the manuscript metrics.

## 2. Inference on your own radar files

```bash
python test_nwp_comparison.py \
  --model_path weights/pa_mcdnet_M6_headline.h5 \
  --inference_input path/to/ppi_folder \
  --output_dir infer_out \
  --use_physics_model --downsample_720 --save_fields
```

`--inference_input` accepts one CWA-format `.gz` sweep or a folder of them; the
Nyquist velocity is read from the file header. Dealiased fields are written to
`infer_out/_fields/*.npz`. Add
`--enable_geo_viz --shape_path mapdata201805310314/COUNTY_MOI_1070516` for maps.

## 3. Locked evaluation (manuscript metrics)

This level requires the CWA data, obtainable by formal request to the CWA, and
the third-party comparison model UNet-VDA (MIT License, MIT Lincoln Laboratory):

```bash
git clone https://github.com/mit-ll/unet-vda.git unet-vda-main
```

The evaluation script loads UNet-VDA from `unet-vda-main/` by default and stops
if it is missing.

**Real observations (200 locked cases).** Each entry of
`data/locked_realobs_test_set.json` gives the raw sweep and its reference
(`bvel_sfda` or `bvel_da`) as paths beginning with the placeholder
`/path/to/typhoonnew`, for example

```
/path/to/typhoonnew/test/20190418_SQUALLLINE/polar/binary/RCCG/RCCG.20190418.2338.bvel_raw.09.gz
```

Arrange the CWA files under a local root with the same relative structure
(`<root>/test/<event>/...`) and pass that root as `--realobs_root`; the script
substitutes it for the placeholder:

```bash
python test_nwp_comparison.py \
  --model_path weights/pa_mcdnet_M6_headline.h5 \
  --realobs_force_cases_file data/locked_realobs_test_set.json \
  --realobs_root path/to/typhoonnew \
  --use_physics_model --downsample_720 \
  --output_dir eval_out/realobs_locked200
```

Cases whose files are missing are skipped; the log reports how many, and the
first missing path. Results are
written to `eval_out/realobs_locked200/nwp_comparison_summary.json`; the
command line that produced them is recorded in its `run_info.command` field.

Expected `overall` values with all 200 cases present (the run reported in the
manuscript):

| Field | PA-MCDNet (`ours_*`) | UNet-VDA (`paper_*`) |
|---|---|---|
| `*_success_rate` | 0.8822 | 0.8939 |
| `*_false_positive_rate` | 0.0039 | 0.0081 |
| `*_rmse` (m/s) | 2.750 | 3.671 |

**CSI and confidence intervals.** The manuscript's CSI and paired-bootstrap
95% confidence intervals are computed from that summary file:

```bash
S=eval_out/realobs_locked200/nwp_comparison_summary.json
python analysis/compute_csi.py $S
python analysis/bootstrap_ci.py --a $S:ours --b $S:paper
```

Expected output (case-averaged over the 200 cases):

| | SR % | FPR % | RMSE (m/s) | CSI % |
|---|---|---|---|---|
| PA-MCDNet | 88.22 | 0.39 | 2.75 | 84.62 |
| UNet-VDA | 89.39 | 0.81 | 3.67 | 82.22 |

| PA-MCDNet − UNet-VDA | Difference | 95% CI |
|---|---|---|
| SR (pp) | −1.17 | [−3.30, +0.99], not significant |
| FPR (pp) | −0.42 | [−0.70, −0.16], significant |
| RMSE (m/s) | −0.92 | [−1.62, −0.27], significant |

Only the final PA-MCDNet weight is released, so comparisons that involve the
ablation variants or earlier models require retraining them (Level 4).

**NWP simulations (100 locked cases).** `data/locked_nwp_test_set.json` lists
each case by NWP run directory (`nwp_dir`) and file stem. The script also needs
the NWP patch file produced by `nwp_to_patches.py`, which it uses only to
confirm that no test case was seen in training:

```bash
python test_nwp_comparison.py \
  --model_path weights/pa_mcdnet_M6_headline.h5 \
  --nwp_root path/to/nwp_parquet_root \
  --h5_path data/nwp_patches.h5 \
  --force_cases_file data/locked_nwp_test_set.json \
  --use_physics_model \
  --output_dir eval_out/nwp_locked100
```

## 4. Retraining

The preprocessing and two-stage training commands, with the loss weights used in
the manuscript, are in the README ("Preprocessing" and "Training"); every
parameter is listed in `docs/USAGE.md`.

## Notes

- The `.json` files on the Zenodo record of the previous version were stored
  with Windows line endings, so their MD5 differs from the GitHub copies; the
  content is identical. The files in this version use Unix line endings and
  match the table above.
- `gradient_checker` (an internal diagnostic) is not part of the public release.
  It is imported optionally and is not used by `test_nwp_comparison.py` or by
  any reported metric.
