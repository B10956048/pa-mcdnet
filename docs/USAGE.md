# Program Parameters

Full command-line parameters for each program. All commands are run from the
repository root. Loss-weight defaults differ from the paper (e.g.
`--lambda_spatial` defaults to 50, the paper uses 10) — always pass them
explicitly for real experiments.

---

## `transfer_learning_complete.py` — training (Stage 1 & Stage 2)

One program handles both stages. Without `--pretrained_model` it trains from
scratch (Stage 1); with `--pretrained_model` + `--freeze_ratio` it fine-tunes
(Stage 2). It reads `--nwp_h5` (an `.h5` file or a tfrecord directory, auto-
detected) and writes `best_model.h5`, `config.json`, `training_history.csv` to
`--output_dir`.

**Common**

| Parameter | Default | Notes | Stage 1 | Stage 2 |
|---|---|---|---|---|
| `--nwp_h5` (required) | | Main training data (h5 or tfrecord dir). Despite the name, Stage 2 puts the real-obs data here | NWP patches | real-obs patches |
| `--pretrained_model` | None | Stage-1 weights; omit to train from scratch | omit | Stage-1 `best_model.h5` |
| `--freeze_ratio` | 0.8 | Fraction of layers frozen (only with pretrained). **Set 0.0 for Stage 1** | 0.0 | 0.3 |
| `--learning_rate` | 5e-5 | | 5e-5 | 5e-5 |
| `--batch_size` | 4 | Up to ~2048 depending on GPU memory | 2048 | 2048 |
| `--epochs` / `--patience` | 50 / 5 | Max epochs and early-stopping patience | 200 / 5 | 100 / 5 |
| `--lambda_cls / reg / soft / spatial` | 2.0 / 0.1 / 0.5 / 50 | Loss weights — pass explicitly (paper spatial = 10) | 2.0 / 0.1 / 0.5 / 10.0 | same |
| `--lambda_physics / --lambda_confidence` | 0.1 / 0.1 | Set 0 to disable | 0.0 / 0.0 | 0.0 / 0.0 |
| `--typhoon_h5` | None | Optional second validation set (h5 or tfrecord) | optional | optional |
| `--steps_per_epoch` | None | Override steps per epoch (default = whole train set); use a small value for a quick smoke test | | |
| `--output_dir` | results/transfer_learning | Output directory | | |

**Advanced** (usually left at defaults)

| Parameter | Default | Notes |
|---|---|---|
| `--resume` | None | Resume from a previous result directory |
| `--lr_scheduling / --lr_factor / --lr_patience / --lr_min` | True / 0.5 / 5 / 1e-6 | Learning-rate schedule |
| `--baseline_typhoon_success` | 84.0 | Baseline typhoon success used for early stopping |
| `--disable_spatial_smoothness` / `--smoothness_type` | on / v_corrected | Spatial-smoothness (PA) switch and type |
| `--class_weight` | None | Per-class weights `[cat0..cat5]` for class imbalance |
| `--aliased_weight / --clean_weight` | 0.8 / 0.2 | Sample weights for aliased vs clean patches |
| `--chunk_size / --shuffle_buffer` | None | h5 read chunk / shuffle buffer (lower if memory-tight) |
| `--mask_strategy / --filter_patch_type` | v1 / all | Mask strategy and patch-type filter |
| `--focal_gamma / --lambda_fpr / --lambda_fnr / --weight_decay` | 0 | Advanced loss/regularization terms (unused in the paper) |

> Multi-GPU training must use TFRecord (a single h5 cannot be file-sharded across
> GPUs). Convert with `convert_h5_to_tfrecord.py`, then point `--nwp_h5` at the
> tfrecord directory.

---

## `test_nwp_comparison.py` — inference & evaluation

The evaluation mode also loads the comparison model (UNet-VDA); pure inference
does not. One of four modes is selected by the arguments:

| Trigger | Mode | Ground truth needed |
|---|---|---|
| `--inference_input` (file or folder) | Pure inference — reads raw, Nyquist from header, outputs dealiased field + geo images | No |
| `--realobs_force_cases_file` (json) | Locked evaluation — the JSON lists raw + gt (fully reproducible) | Yes |
| `--realobs_root` (folder) | Scan a folder, auto-pair raw + gt | Yes |
| none | NWP mode (`--nwp_root` + `--h5_path`) | Yes |

**Common**

| Parameter | Notes |
|---|---|
| `--model_path` (required) | Model weight — prefer `best_model_manual.h5` |
| `--use_physics_model` | Use the physics-constrained model; pass it to match training |
| `--paper_model_path` | UNet-VDA SavedModel dir (evaluation mode loads it and aborts if missing); not needed for pure inference |
| `--enable_geo_viz` + `--shape_path` | Geo visualization (needs pyart + basemap); `--shape_path mapdata201805310314/COUNTY_MOI_1070516` |
| `--downsample_720` | 720-ray stations are split for inference; add for real observations |
| `--save_fields` | Also save `_fields/*.npz` (raw, ours, paper, gt) |
| `--infer_nyquist` | Fallback Nyquist for pure-inference when the header lacks it |
| `--output_dir` | Output directory (writes `nwp_comparison_summary.json`) |

**By mode / advanced**

| Parameter | Default | Notes |
|---|---|---|
| `--realobs_elevation` | all | Elevation for `--realobs_root` scan (01/02/… or all) |
| `--realobs_sample_n` | 100 | Max sweeps sampled per station/case in `--realobs_root` mode (0 = all) |
| `--exclude_stations` | None | Exclude stations (comma-separated), e.g. `RCCG,RCGI` |
| `--save_confidence` | off | Per-case softmax classification-confidence statistics |
| `--enable_vcorrected_pp` / `--vcorrected_threshold` | off / 1.0 | V_corrected physical post-processing and its jump threshold |
| `--max_cases` / `--random_seed` / `--clean_ratio` | 100 / 42 / 0.0 | NWP mode: case count, seed, clean-case ratio |
| `--conf_threshold` | 0.0 | Classification-confidence threshold |

---

## `nwp_to_patches.py` — preprocessing (NWP → patches)

Scans NWP parquet per station, splits by event (to avoid leakage), crops
128×128 patches centered on folding, balances aliased/clean, augments by azimuth
rotation, and writes `--output_h5`.

| Parameter | Default | Notes |
|---|---|---|
| `--nwp_root` (required) | | NWP parquet root |
| `--output_h5` (required) | | Output h5 |
| `--target_stations` | RCCG | Station list |
| `--train_ratio / --val_ratio / --test_ratio` | 0.6 / 0.2 / 0.2 | Event-level split |
| `--aliased_ratio` | 1.0 | Fraction of aliased patches (paper: 0.9) |
| `--num_patches_per_file` | 5 | Patches sampled per file |
| `--patch_size` | 128 | Patch side length |
| `--min_alias_pixels` | 10 | Min folded pixels for an aliased patch |
| `--skip_first_n` | 6 | Skip the first N files per directory (NWP spin-up) |
| `--max_dirs` / `--max_cases_per_station` | None | Limit directories / cases (for quick tests) |
| `--enable_balanced_sampling` / `--target_distribution` | off / None | Bucketed resampling to balance the alias-ratio distribution |
| `--enable_azimuth_rotation` / `--max_azimuth_variations` / `--augment_variations` | on / 2 / 1 | Augmentation settings |
| `--seed` / `--num_workers` | fixed / None | Random seed / parallel workers |

---

## `realobs_to_patches.py` — preprocessing (real-obs → patches)

Scans raw + reference (`bvel_sfda` / `bvel_vdaqc`) pairs in `--train_cases`,
carves a validation split from the training events, crops patches, augments, and
writes `--output_h5`. Append events with `realobs_append_patches.py`.

| Parameter | Default | Notes |
|---|---|---|
| `--realobs_root` (required) | | typhoonnew root |
| `--output_h5` (required) | | Output h5 |
| `--train_cases` (required) | | Whitelist of training events (only events named here are collected) |
| `--test_cases` | None | Events saved as a separate test split |
| `--val_ratio` | 0.15 | Validation fraction carved from train_cases |
| `--elevation` | all | Elevation (01/02/… or all) |
| `--num_patches_per_sweep` | 8 | Aliased patches cropped per sweep |
| `--patch_size` / `--min_alias_pixels` | 128 / 10 | Patch side / min folded pixels |
| `--minority_multiplier` | 5 | Augmentation multiplier for the minority class (squall lines) |
| `--augment_rotations` / `--augment_matrix` | 3 / 1 | Rotation / matrix augmentation counts |
| `--include_clean` / `--no_clean` / `--clean_ratio_threshold` | include / 0.7 | Whether to include clean patches, and their clean-pixel threshold |
| `--seed` | 46 | Random seed |

---

## `convert_h5_to_tfrecord.py` — h5 → tfrecord (required for multi-GPU)

```bash
python convert_h5_to_tfrecord.py \
  --h5_path data/nwp_patches.h5 --output_dir data/nwp_tfrecord \
  --splits train,val --num_shards 32 --compression GZIP
# then point training at the directory: --nwp_h5 data/nwp_tfrecord
```

| Parameter | Default | Notes |
|---|---|---|
| `--h5_path` (required) | | Source h5 |
| `--output_dir` (required) | | Output directory (created automatically) |
| `--splits` | train,val | Splits to convert (comma-separated) |
| `--num_shards` | 32 | Shards per split |
| `--compression` | GZIP | GZIP (≈ h5 size) or NONE (faster, ~3.5× larger) |
| `--chunk_size` / `--h5_cache_mb` | 512 / 512 | h5 read chunk / cache (rarely changed) |
| `--mode` | sequential | `sequential` (fast, recommended) or `shuffle` (slower) |

---

## `manual_weight_transfer.py` — produce `best_model_manual.h5`

Run after Stage 2. Rebuilds the model, loads the fine-tuned `best_model.h5`, and
re-saves it as `best_model_manual.h5` so newer TensorFlow can load it (avoids the
subclassed-model layer-count mismatch). Stage-1 `best_model.h5` loads directly
and does not need this.

```bash
python manual_weight_transfer.py \
  --input results/stage2/best_model.h5 \
  --output results/stage2/best_model_manual.h5
```

| Parameter | Default | Notes |
|---|---|---|
| `--input` (required) | | Source `best_model.h5` (Stage-2 output) |
| `--output` | `<input dir>/best_model_manual.h5` | Output path |
