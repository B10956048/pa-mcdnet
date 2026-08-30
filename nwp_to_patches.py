#!/usr/bin/env python3
"""
NWP 數據轉 H5 Patch 工具 (v3.0)

用於遷移學習：將 NWP Parquet 數據轉換為與真實雷達訓練相同的 H5 Patch 格式

v3.0 重要改進：避免資料洩漏 + 支援 Clean Patches
  - 按 source file (station + timestamp) 分割，確保無資料洩漏
  - 支援 aliased_ratio 參數控制 aliased/clean patches 比例
  - Clean patches (fold=0) 用於訓練模型識別無需修正的像素
  - 自動驗證資料洩漏並報告結果

v2.0 特性（保留）：
  - 記錄來源的 NWP 預報目錄 (nwp_forecast_dir)
  - 支援方位角旋轉增強和矩陣增強
  - 輸出格式與 dealiasing_pre_patch_v2.py 完全一致

使用方式：
    # 全量處理所有站點（推薦）
    python nwp_to_patches.py \
        --nwp_root /path/to/nwp_data_csv \
        --output_h5 data/nwp_patches_v3.h5 \
        --target_stations RCCG RCWF RCHL RCKT RCGI \
        --aliased_ratio 0.9 \
        --skip_first_n 6

    # 包含更多 clean patches（提高 fold=0 識別能力）
    python nwp_to_patches.py \
        --nwp_root /path/to/nwp_data_csv \
        --output_h5 data/nwp_patches_with_clean.h5 \
        --target_stations RCCG RCWF \
        --aliased_ratio 0.8 \
        --clean_ratio_threshold 0.7

    # 小規模測試（限制處理的目錄數）
    python nwp_to_patches.py \
        --nwp_root /path/to/nwp_data_csv \
        --output_h5 data/nwp_patches_test.h5 \
        --target_stations RCCG RCWF \
        --max_dirs 10

重要：
- 使用 DA 作為 pseudo-GT
- 按 source file 分割確保無資料洩漏
- aliased_ratio=1.0 表示全部 aliased patches（原始行為）
- aliased_ratio=0.8 表示 80% aliased + 20% clean patches
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import h5py
import random
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import argparse
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

# 抑制警告和無用訊息
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 抑制 TensorFlow 訊息

# 導入 Parquet 載入函數（抑制其輸出）
sys.path.append(str(Path(__file__).parent))

# 暫時重定向 stdout 來抑制 "地理可視化模塊已載入" 訊息
import io
from contextlib import redirect_stdout

_original_stdout = sys.stdout
sys.stdout = io.StringIO()
try:
    from batch_test_nwp import load_parquet_as_matrix, find_nwp_triplets
finally:
    sys.stdout = _original_stdout

# 導入 patch 切割相關函數
from dealiasing_pre_patch_v2 import (
    auto_zero_pad,
    compute_alias_label_swapped,
    augment_sample,
)

# 設置隨機種子
SEED = 46
np.random.seed(SEED)
random.seed(SEED)


def save_parquet_matrix_as_temp_csv(matrix, nyquist, template_parquet_path, output_csv_path):
    """
    將 Parquet 矩陣保存為臨時 CSV（用於方位角旋轉增強）

    Parameters:
    -----------
    matrix : np.ndarray
        速度矩陣 (azimuth, range)
    nyquist : float
        Nyquist 速度
    template_parquet_path : str
        模板 Parquet 檔案路徑（用於獲取坐標信息）
    output_csv_path : str
        輸出 CSV 路徑
    """
    import pyarrow.parquet as pq

    # 讀取模板獲取坐標信息
    table = pq.read_table(template_parquet_path)
    df_template = table.to_pandas()

    az_col = 'Azimuth' if 'Azimuth' in df_template.columns else 'azimuth'
    r_col = 'Range' if 'Range' in df_template.columns else 'range'

    # 獲取唯一的方位角和距離
    azimuths_raw = df_template[az_col].unique()
    ranges = sorted(df_template[r_col].unique())

    # 使用矩陣的實際形狀
    azimuths = azimuths_raw[:matrix.shape[0]]
    ranges = ranges[:matrix.shape[1]]

    # 構建輸出數據
    data = []
    for i, az in enumerate(azimuths):
        for j, r in enumerate(ranges):
            value = matrix[i, j]
            if np.isnan(value):
                value = -999.0
            data.append({
                'Azimuth': az,
                'Range': r,
                'Nyquist': nyquist,
                'Value': value
            })

    df_out = pd.DataFrame(data)
    df_out.to_csv(output_csv_path, index=False)


def load_csv_as_matrix_from_temp(csv_path):
    """從臨時 CSV 載入矩陣（與 dealiasing_pre_patch_v2.py 一致）"""
    data = pd.read_csv(csv_path)
    data['Value'] = data['Value'].replace(-999.0, np.nan)
    data = data.sort_values(by=['Azimuth', 'Range']).reset_index(drop=True)

    azimuths = data['Azimuth'].unique()
    ranges = data['Range'].unique()
    n_az = len(azimuths)
    n_rad = len(ranges)

    velocity_grid = np.full((n_az, n_rad), np.nan)

    # 創建索引字典
    az_indices = {az: i for i, az in enumerate(azimuths)}
    rad_indices = {r: i for i, r in enumerate(ranges)}

    # 填充矩陣
    az_idx = np.array([az_indices[az] for az in data['Azimuth']])
    rad_idx = np.array([rad_indices[r] for r in data['Range']])

    velocity_grid[az_idx, rad_idx] = data['Value'].values
    nyquist = data['Nyquist'].unique()

    return velocity_grid, nyquist


def extract_patches_from_nwp_triplet(triplet, temp_dir, layers=4, patch_size=128,
                                     num_patches_per_file=5, min_alias_pixels=10,
                                     augment_variations=1, enable_azimuth_rotation=True,
                                     max_azimuth_variations=2, patch_type='aliased',
                                     clean_ratio_threshold=0.7,
                                     enable_targeted_patches=False):
    """
    從 NWP 三元組提取 patches

    Parameters:
    -----------
    triplet : dict
        {'raw': path, 'da': path, 'station': ..., 'timestamp': ...}
    temp_dir : str
        臨時 CSV 目錄（用於方位角旋轉）

    Returns:
    --------
    list of dict : patches
    """
    # 抑制子進程中的所有輸出
    import sys
    import os
    import io
    import warnings

    warnings.filterwarnings('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # 重定向 stderr 和 stdout（抑制 typing 模組修復訊息）
    sys.stderr = io.StringIO()

    try:
        # 載入 RAW 和 DA 矩陣
        raw_matrix, nyquist, _ = load_parquet_as_matrix(triplet['raw'])
        da_matrix, _, _ = load_parquet_as_matrix(triplet['da'])

        nyq_val = nyquist if isinstance(nyquist, (int, float)) else np.nanmean(nyquist)

        # 記錄來源信息（包含 NWP 預報目錄以避免資料洩漏）
        source_info = {
            'raw_file': os.path.basename(triplet['raw']),
            'da_file': os.path.basename(triplet['da']),
            'nwp_forecast_dir': triplet.get('nwp_dir', os.path.basename(os.path.dirname(triplet['raw']))),
            'station': triplet['station'],
            'timestamp': triplet['timestamp']
        }

        # 準備資料版本清單（原始 + 方位角旋轉）
        data_versions = []

        # 添加原始版本（直接使用矩陣，不需要CSV）
        data_versions.append(('original', raw_matrix, da_matrix, nyq_val))

        # 添加方位角旋轉版本（純 numpy np.roll，無磁碟 I/O）
        # 方位角旋轉 = row 循環位移：angle° 對應 shift = round(n_az * angle / 360)
        if enable_azimuth_rotation and max_azimuth_variations > 0:
            n_az = raw_matrix.shape[0]
            rotation_angles = [45, 90, 135, 180, 225, 270, 315]
            selected_angles = np.random.choice(rotation_angles,
                                              min(max_azimuth_variations, len(rotation_angles)),
                                              replace=False)

            for angle in selected_angles:
                try:
                    shift = round(n_az * angle / 360)
                    rotated_raw = np.roll(raw_matrix, shift, axis=0)
                    rotated_da = np.roll(da_matrix, shift, axis=0)
                    data_versions.append((f'azimuth_rot_{angle}deg', rotated_raw, rotated_da, nyq_val))
                except Exception as e:
                    print(f"⚠️ 方位角旋轉失敗 {angle}°: {e}")

        all_patches = []

        # 處理每個資料版本
        for version_name, data_vel, gt_vel, nyq in data_versions:
            try:
                # 計算 alias label
                label_2d = compute_alias_label_swapped(data_vel, gt_vel, nyq)
                if np.all(label_2d == 0):
                    print(f"⚠️ {version_name}: 全部為無效像素，跳過")
                    continue

                # Zero-pad
                data_vel_pad, _, _ = auto_zero_pad(data_vel, layers=layers, fill_value=np.nan)
                gt_vel_pad, _, _ = auto_zero_pad(gt_vel, layers=layers, fill_value=np.nan)
                label_pad, _, _ = auto_zero_pad(label_2d, layers=layers, fill_value=0)
                H_pad, W_pad = data_vel_pad.shape

                version_patches = []
                extracted_count = 0
                attempts = 0
                max_attempts = num_patches_per_file * 10

                while extracted_count < num_patches_per_file and attempts < max_attempts:
                    attempts += 1

                    if H_pad < patch_size or W_pad < patch_size:
                        top, left = 0, 0
                        ph, pw = H_pad, W_pad
                    else:
                        top = np.random.randint(0, H_pad - patch_size + 1)
                        left = np.random.randint(0, W_pad - patch_size + 1)
                        ph, pw = patch_size, patch_size

                    patch_vel = data_vel_pad[top:top+ph, left:left+pw]
                    patch_gtvel = gt_vel_pad[top:top+ph, left:left+pw]
                    patch_label = label_pad[top:top+ph, left:left+pw]

                    # 計算 alias 和 clean 像素數
                    alias_mask = np.isin(patch_label, [1,2,4,5])
                    alias_count = np.sum(alias_mask)
                    clean_mask = (patch_label == 3)  # 標籤3為無摺錯
                    clean_count = np.sum(clean_mask)
                    total_valid_pixels = np.sum(patch_label > 0)  # 排除標籤0(無效像素)

                    # 根據 patch_type 進行不同的篩選策略
                    if patch_type == 'aliased':
                        # 對於摺錯 patch，需要足夠的 alias pixels
                        if alias_count < min_alias_pixels:
                            continue
                    elif patch_type == 'clean':
                        # 對於乾淨 patch，需要大部分像素是乾淨的
                        if total_valid_pixels == 0:
                            continue
                        clean_ratio = clean_count / total_valid_pixels
                        if clean_ratio < clean_ratio_threshold:  # 至少70%是乾淨像素
                            continue

                    # 根據 patch_type 設定不同的目標
                    if patch_type == 'aliased':
                        # 摺錯 patch：目標是 DA 結果（pseudo-GT）
                        target_vel = patch_gtvel[..., None].astype(np.float32)
                    elif patch_type == 'clean':
                        # 乾淨 patch：目標是保持原始 RAW (identity mapping)
                        target_vel = patch_vel[..., None].astype(np.float32)
                    else:
                        # 預設行為
                        target_vel = patch_gtvel[..., None].astype(np.float32)

                    # 計算 alias_ratio
                    patch_alias_ratio = alias_count / total_valid_pixels if total_valid_pixels > 0 else 0.0

                    version_source_info = source_info.copy()
                    version_source_info['augment_type'] = version_name

                    # 原始 patch
                    patch_dict = {
                        'vel': patch_vel[..., None].astype(np.float32),
                        'nyq': np.array([nyq], dtype=np.float32),
                        'alias_label': patch_label.astype(np.int32),
                        'gt_vel': patch_gtvel[..., None].astype(np.float32),
                        'target_vel': target_vel,
                        'patch_type': patch_type,  # 使用傳入的 patch_type
                        'source_info': version_source_info,
                        'alias_ratio': patch_alias_ratio
                    }
                    version_patches.append(patch_dict)

                    # 矩陣增強（只對原始版本）
                    if version_name == 'original' and augment_variations > 0:
                        for aug_idx in range(augment_variations):
                            aug_vel, aug_gtvel, aug_label = augment_sample(
                                patch_vel.copy(), patch_gtvel.copy(), patch_label.copy()
                            )

                            # 根據 patch_type 設定增強資料的目標
                            if patch_type == 'aliased':
                                aug_target_vel = aug_gtvel[..., None].astype(np.float32)
                            elif patch_type == 'clean':
                                aug_target_vel = aug_vel[..., None].astype(np.float32)
                            else:
                                aug_target_vel = aug_gtvel[..., None].astype(np.float32)

                            # 計算增強後的 alias_ratio
                            aug_valid = np.sum(aug_label > 0)
                            aug_alias = np.sum(np.isin(aug_label, [1, 2, 4, 5]))
                            aug_alias_ratio = aug_alias / aug_valid if aug_valid > 0 else 0.0

                            matrix_source_info = source_info.copy()
                            matrix_source_info['augment_type'] = f'matrix_aug_{aug_idx}'

                            aug_patch = {
                                'vel': aug_vel[..., None].astype(np.float32),
                                'nyq': np.array([nyq], dtype=np.float32),
                                'alias_label': aug_label.astype(np.int32),
                                'gt_vel': aug_gtvel[..., None].astype(np.float32),
                                'target_vel': aug_target_vel,
                                'patch_type': patch_type,  # 使用傳入的 patch_type
                                'source_info': matrix_source_info,
                                'alias_ratio': aug_alias_ratio
                            }
                            version_patches.append(aug_patch)

                    extracted_count += 1

                # 定向切割高 aliasing 區域 patches
                if enable_targeted_patches and patch_type == 'aliased':
                    targeted = extract_targeted_high_alias_patches(
                        data_vel_pad, gt_vel_pad, label_pad, nyq,
                        patch_size=patch_size,
                        min_alias_ratio=0.3,
                        max_patches=num_patches_per_file,
                        source_info=source_info.copy(),
                        patch_type=patch_type
                    )
                    # 標記 augment_type 以包含版本資訊
                    for tp in targeted:
                        tp['source_info']['augment_type'] = f'{version_name}_targeted'
                    version_patches.extend(targeted)

                all_patches.extend(version_patches)

            except Exception as e:
                print(f"⚠️ 處理版本 {version_name} 失敗: {e}")
                continue

        return all_patches

    except Exception as e:
        import traceback
        print(f"處理 NWP 三元組時發生錯誤: {triplet.get('raw', 'unknown')}")
        print(f"   錯誤類型: {type(e).__name__}")
        print(f"   錯誤訊息: {e}")
        traceback.print_exc()
        return []


def extract_targeted_high_alias_patches(data_vel_pad, gt_vel_pad, label_pad, nyq,
                                        patch_size, min_alias_ratio=0.3,
                                        max_patches=None, source_info=None,
                                        patch_type='aliased'):
    """
    定向切割高 aliasing 區域的 patches

    使用 sliding window 掃描 padded alias_label，找出 alias_ratio > min_alias_ratio
    的候選位置，按 alias_ratio 降序排列後取 top-N（相鄰至少隔 patch_size//2）。

    Parameters:
    -----------
    data_vel_pad : np.ndarray
        Padded RAW velocity (H, W)
    gt_vel_pad : np.ndarray
        Padded DA velocity (H, W)
    label_pad : np.ndarray
        Padded alias label (H, W)
    nyq : float
        Nyquist velocity
    patch_size : int
        Patch size
    min_alias_ratio : float
        Minimum alias ratio threshold (default 0.3)
    max_patches : int
        Maximum number of targeted patches to extract
    source_info : dict
        Source information for the patch
    patch_type : str
        Patch type ('aliased' or 'clean')

    Returns:
    --------
    list of dict : targeted patches
    """
    H_pad, W_pad = label_pad.shape
    if H_pad < patch_size or W_pad < patch_size:
        return []

    stride = patch_size // 2

    # 計算所有 sliding window 位置的 alias_ratio（向量化）
    # 使用 stride_tricks 建立 view
    from numpy.lib.stride_tricks import sliding_window_view

    # 取所有 stride 步長的起始位置
    top_positions = np.arange(0, H_pad - patch_size + 1, stride)
    left_positions = np.arange(0, W_pad - patch_size + 1, stride)

    candidates = []
    for top in top_positions:
        for left in left_positions:
            window = label_pad[top:top+patch_size, left:left+patch_size]
            valid_pixels = np.sum(window > 0)
            if valid_pixels == 0:
                continue
            alias_pixels = np.sum(np.isin(window, [1, 2, 4, 5]))
            alias_ratio = alias_pixels / valid_pixels
            if alias_ratio > min_alias_ratio:
                candidates.append((top, left, alias_ratio))

    if not candidates:
        return []

    # 按 alias_ratio 降序排列
    candidates.sort(key=lambda x: x[2], reverse=True)

    # 選取 top-N，相鄰位置至少隔 patch_size//2
    selected = []
    for top, left, ratio in candidates:
        too_close = False
        for sel_top, sel_left, _ in selected:
            if abs(top - sel_top) < stride and abs(left - sel_left) < stride:
                too_close = True
                break
        if not too_close:
            selected.append((top, left, ratio))
            if max_patches and len(selected) >= max_patches:
                break

    # 從選中的位置切出 patches
    targeted_patches = []
    for top, left, ratio in selected:
        patch_vel = data_vel_pad[top:top+patch_size, left:left+patch_size]
        patch_gtvel = gt_vel_pad[top:top+patch_size, left:left+patch_size]
        patch_label = label_pad[top:top+patch_size, left:left+patch_size]

        # 計算此 patch 的 alias_ratio
        valid_px = np.sum(patch_label > 0)
        alias_px = np.sum(np.isin(patch_label, [1, 2, 4, 5]))
        patch_alias_ratio = alias_px / valid_px if valid_px > 0 else 0.0

        if patch_type == 'aliased':
            target_vel = patch_gtvel[..., None].astype(np.float32)
        elif patch_type == 'clean':
            target_vel = patch_vel[..., None].astype(np.float32)
        else:
            target_vel = patch_gtvel[..., None].astype(np.float32)

        targeted_source = source_info.copy() if source_info else {}
        targeted_source['augment_type'] = targeted_source.get('augment_type', 'original') + '_targeted'

        patch_dict = {
            'vel': patch_vel[..., None].astype(np.float32),
            'nyq': np.array([nyq], dtype=np.float32),
            'alias_label': patch_label.astype(np.int32),
            'gt_vel': patch_gtvel[..., None].astype(np.float32),
            'target_vel': target_vel,
            'patch_type': patch_type,
            'source_info': targeted_source,
            'alias_ratio': patch_alias_ratio
        }
        targeted_patches.append(patch_dict)

    return targeted_patches


def process_nwp_triplet_batch(batch_triplets, temp_dir, **kwargs):
    """並行處理一批 NWP 三元組（支援 patch_type）"""
    all_patches = []
    for triplet in batch_triplets:
        # triplet 可能包含 patch_type 資訊
        patch_type = triplet.get('patch_type', 'aliased')
        patches = extract_patches_from_nwp_triplet(triplet, temp_dir, patch_type=patch_type, **kwargs)
        all_patches.extend(patches)
    return all_patches


def print_alias_distribution(alias_ratios, label=""):
    """
    印出 alias_ratio 的 5-bin 分佈統計

    Parameters:
    -----------
    alias_ratios : list of float
        每個 patch 的 alias_ratio
    label : str
        標題標籤
    """
    bins = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]
    bin_labels = ['0-10%', '10-30%', '30-50%', '50-70%', '70-100%']
    total = len(alias_ratios)
    if total == 0:
        print(f"[BALANCE] {label}: 無 patches")
        return

    ratios_arr = np.array(alias_ratios)
    print(f"[BALANCE] {label}:")
    for (lo, hi), bl in zip(bins, bin_labels):
        count = np.sum((ratios_arr >= lo) & (ratios_arr < hi))
        pct = count / total * 100
        print(f"  {bl:>8s}: {count:>8,d} ({pct:5.1f}%)")
    print(f"  {'Total':>8s}: {total:>8,d}")


def resample_h5(src_h5_path, dst_h5_path, target_dist, seed=46):
    """
    從臨時 H5 讀取 train patches 的 alias_ratio，分桶重採樣後寫入最終 H5。
    val/test split 原封複製。

    **Clean patches (alias_ratio < 0.01) 不參與重採樣，全數保留。**
    target_dist 只作用在 aliased patches 內部的 alias_ratio 分佈。

    支援大 array 格式（v4.0）和舊版 per-group 格式（向後兼容）。
    """
    rng = np.random.RandomState(seed)

    CLEAN_THRESHOLD = 0.01
    bins = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]
    bin_labels = ['0-10%', '10-30%', '30-50%', '50-70%', '70-100%']

    with h5py.File(src_h5_path, 'r') as src_h5:
        if 'train' not in src_h5:
            print("[WARN] src H5 中沒有 train split，跳過重採樣")
            return

        train = src_h5['train']

        # 偵測格式：大 array（'vel' 是直屬 dataset）或 per-group（'patch_0' 是 group）
        is_big_array = ('vel' in train and isinstance(train['vel'], h5py.Dataset)
                        and train['vel'].ndim >= 3)

        if is_big_array:
            _resample_big_array(src_h5, dst_h5_path, target_dist, rng, bins, bin_labels,
                                CLEAN_THRESHOLD)
        else:
            _resample_per_group(src_h5, dst_h5_path, target_dist, rng, bins, bin_labels,
                                CLEAN_THRESHOLD)


def _resample_big_array(src_h5, dst_h5_path, target_dist, rng, bins, bin_labels,
                         CLEAN_THRESHOLD):
    """大 array 格式的重採樣（v4.0）"""
    train = src_h5['train']
    alias_ratios = train['alias_ratio'][:]
    N = len(alias_ratios)

    clean_indices = np.where(alias_ratios < CLEAN_THRESHOLD)[0]
    aliased_indices = np.where(alias_ratios >= CLEAN_THRESHOLD)[0]
    aliased_ratios_vals = alias_ratios[aliased_indices]

    print(f"\n[BALANCE] 重採樣前: {N} patches "
          f"(clean={len(clean_indices)}, aliased={len(aliased_indices)})")
    print_alias_distribution(alias_ratios.tolist(), "重採樣前 train 分佈")

    # 對 aliased patches 分桶（按原始 index）
    bin_idx_lists = {i: [] for i in range(len(bins))}
    for orig_idx, ar in zip(aliased_indices, aliased_ratios_vals):
        assigned = False
        for i, (lo, hi) in enumerate(bins):
            if lo <= ar < hi:
                bin_idx_lists[i].append(orig_idx)
                assigned = True
                break
        if not assigned:
            bin_idx_lists[0].append(orig_idx)

    total_aliased = len(aliased_indices)
    target_counts = [int(total_aliased * t) for t in target_dist]
    diff = total_aliased - sum(target_counts)
    target_counts[int(np.argmax(target_counts))] += diff

    print(f"\n[BALANCE] Aliased patches 重採樣計畫 (clean {len(clean_indices)} 全數保留):")
    for i, bl in enumerate(bin_labels):
        current = len(bin_idx_lists[i])
        target = target_counts[i]
        action = "下採樣" if current > target else ("上採樣" if current < target else "不變")
        print(f"  {bl:>8s}: {current:>6d} -> {target:>6d} ({action})")

    # 重採樣 aliased indices
    selected_aliased = []
    for i in range(len(bins)):
        current_list = bin_idx_lists[i]
        target = target_counts[i]
        if len(current_list) == 0:
            print(f"  [WARN] Bin {bin_labels[i]} 為空，無法採樣")
            continue
        if len(current_list) >= target:
            chosen = rng.choice(current_list, size=target, replace=False).tolist()
        else:
            chosen = list(current_list)
            extra = rng.choice(current_list, size=target - len(current_list), replace=True).tolist()
            chosen.extend(extra)
        selected_aliased.extend(chosen)

    # 合併索引（排序以利 sequential read）
    selected_indices = np.sort(np.concatenate([
        clean_indices, np.array(selected_aliased, dtype=np.int64)
    ]))
    M = len(selected_indices)

    print(f"\n[BALANCE] 重採樣後 train patches: {M} "
          f"(clean={len(clean_indices)}, aliased={len(selected_aliased)})")

    # 寫入目標 H5（分批讀寫，避免 OOM）
    with h5py.File(dst_h5_path, 'w') as dst_h5:
        dst_train = dst_h5.create_group('train')
        chunk_n = min(32, M)

        # 數值型 datasets —— 先建立可擴展的空 dataset，再分批寫入
        num_keys = ['vel', 'nyq', 'alias_label', 'gt_vel', 'target_vel', 'alias_ratio']
        dst_datasets = {}
        for key in num_keys:
            if key not in train:
                continue
            src_shape = train[key].shape[1:]  # 去掉第一維
            dst_shape = (M,) + src_shape
            kw = {}
            if len(src_shape) >= 2:
                cs = (chunk_n,) + src_shape
                kw = {'chunks': cs, 'compression': 'lzf'}
            dst_datasets[key] = dst_train.create_dataset(
                key, shape=dst_shape, dtype=train[key].dtype, **kw)

        # 分批寫入：每批處理 1024 個 dst 位置
        copy_batch = 1024
        for key, dst_ds in dst_datasets.items():
            for dst_start in range(0, M, copy_batch):
                dst_end = min(dst_start + copy_batch, M)
                needed_src = selected_indices[dst_start:dst_end]
                # 去重 + 排序，符合 h5py 嚴格遞增要求
                needed_unique = np.unique(needed_src)
                src_data = train[key][list(needed_unique)]
                # 建立 source_index → local_position 的映射
                uq_to_local = {idx: i for i, idx in enumerate(needed_unique)}
                local_indices = np.array([uq_to_local[s] for s in needed_src])
                dst_ds[dst_start:dst_end] = src_data[local_indices]

        # 字串型 datasets（字串較小，可一次讀取）
        dt = h5py.string_dtype()
        for key in ['patch_type', 'source_file', 'station', 'nwp_dir', 'timestamp', 'augment_type']:
            if key not in train:
                continue
            all_vals = train[key][:]
            dst_train.create_dataset(key, data=[all_vals[i] for i in selected_indices], dtype=dt)

        dst_train.attrs['format'] = 'big_array'
        dst_train.attrs['num_patches'] = M

        # 複製 val/test 原封不動
        for split in ['val', 'test']:
            if split in src_h5:
                src_h5.copy(src_h5[split], dst_h5, name=split)
                ns = src_h5[split]['vel'].shape[0] if 'vel' in src_h5[split] else len(src_h5[split])
                print(f"[BALANCE] {split} patches 原封複製: {ns}")

        # 複製 metadata 並更新
        if 'metadata' in src_h5:
            src_h5.copy(src_h5['metadata'], dst_h5, name='metadata')
            dst_h5['metadata'].attrs['balanced_sampling'] = True
            dst_h5['metadata'].attrs['target_distribution'] = str(target_dist)
            dst_h5['metadata'].attrs['resample_clean_preserved'] = int(len(clean_indices))
            dst_h5['metadata'].attrs['resample_aliased_after'] = len(selected_aliased)
            dst_h5['metadata'].attrs['train_patches_before_resample'] = N
            dst_h5['metadata'].attrs['train_patches_after_resample'] = M

    # 驗證
    with h5py.File(dst_h5_path, 'r') as dst_h5:
        if 'train' in dst_h5 and 'alias_ratio' in dst_h5['train']:
            print_alias_distribution(dst_h5['train']['alias_ratio'][:].tolist(),
                                     "重採樣後 train 分佈")


def _resample_per_group(src_h5, dst_h5_path, target_dist, rng, bins, bin_labels,
                         CLEAN_THRESHOLD):
    """舊版 per-group 格式的重採樣（向後兼容）"""
    train_group = src_h5['train']

    clean_patches = []
    aliased_patches_info = []
    for patch_name in train_group:
        ar = float(train_group[patch_name].attrs.get('alias_ratio', -1.0))
        if ar < CLEAN_THRESHOLD:
            clean_patches.append(patch_name)
        else:
            aliased_patches_info.append((patch_name, ar))

    total_train_before = len(clean_patches) + len(aliased_patches_info)
    print(f"\n[BALANCE] 重採樣前: {total_train_before} patches "
          f"(clean={len(clean_patches)}, aliased={len(aliased_patches_info)})")
    all_ratios = [0.0] * len(clean_patches) + [ar for _, ar in aliased_patches_info]
    print_alias_distribution(all_ratios, "重採樣前 train 分佈")

    bin_patches = {i: [] for i in range(len(bins))}
    for patch_name, ar in aliased_patches_info:
        assigned = False
        for i, (lo, hi) in enumerate(bins):
            if lo <= ar < hi:
                bin_patches[i].append(patch_name)
                assigned = True
                break
        if not assigned:
            bin_patches[0].append(patch_name)

    total_aliased = len(aliased_patches_info)
    target_counts = [int(total_aliased * t) for t in target_dist]
    diff = total_aliased - sum(target_counts)
    target_counts[int(np.argmax(target_counts))] += diff

    print(f"\n[BALANCE] Aliased patches 重採樣計畫 (clean {len(clean_patches)} 全數保留):")
    for i, bl in enumerate(bin_labels):
        current = len(bin_patches[i])
        target = target_counts[i]
        action = "下採樣" if current > target else ("上採樣" if current < target else "不變")
        print(f"  {bl:>8s}: {current:>6d} -> {target:>6d} ({action})")

    selected_aliased = []
    for i in range(len(bins)):
        current_list = bin_patches[i]
        target = target_counts[i]
        if len(current_list) == 0:
            print(f"  [WARN] Bin {bin_labels[i]} 為空，無法採樣")
            continue
        if len(current_list) >= target:
            chosen = rng.choice(current_list, size=target, replace=False).tolist()
        else:
            chosen = list(current_list)
            extra = rng.choice(current_list, size=target - len(current_list), replace=True).tolist()
            chosen.extend(extra)
        selected_aliased.extend(chosen)

    selected_patches = clean_patches + selected_aliased

    with h5py.File(dst_h5_path, 'w') as dst_h5:
        dst_train = dst_h5.create_group('train')
        count = 0
        for patch_name in selected_patches:
            src_patch = src_h5['train'][patch_name]
            src_h5['train'].copy(src_patch, dst_train, name=f'patch_{count}')
            count += 1

        print(f"\n[BALANCE] 重採樣後 train patches: {count} "
              f"(clean={len(clean_patches)}, aliased={len(selected_aliased)})")

        for split in ['val', 'test']:
            if split in src_h5:
                src_h5.copy(src_h5[split], dst_h5, name=split)
                print(f"[BALANCE] {split} patches 原封複製: {len(src_h5[split])}")

        if 'metadata' in src_h5:
            src_h5.copy(src_h5['metadata'], dst_h5, name='metadata')
            dst_h5['metadata'].attrs['balanced_sampling'] = True
            dst_h5['metadata'].attrs['target_distribution'] = str(target_dist)
            dst_h5['metadata'].attrs['resample_clean_preserved'] = len(clean_patches)
            dst_h5['metadata'].attrs['resample_aliased_after'] = len(selected_aliased)
            dst_h5['metadata'].attrs['train_patches_before_resample'] = total_train_before
            dst_h5['metadata'].attrs['train_patches_after_resample'] = count

    resampled_ratios = []
    with h5py.File(dst_h5_path, 'r') as dst_h5:
        if 'train' in dst_h5:
            for patch_name in dst_h5['train']:
                ar = dst_h5['train'][patch_name].attrs.get('alias_ratio', -1.0)
                resampled_ratios.append(float(ar))
    print_alias_distribution(resampled_ratios, "重採樣後 train 分佈")


def convert_nwp_to_patches(nwp_root, output_h5_path,
                           target_stations=['RCCG', 'RCWF'],
                           skip_first_n=6,
                           max_dirs=None,
                           max_cases_per_station=None,
                           train_ratio=0.6,
                           val_ratio=0.2,
                           test_ratio=0.2,
                           patch_size=128,
                           num_patches_per_file=5,
                           min_alias_pixels=10,
                           augment_variations=1,
                           enable_azimuth_rotation=True,
                           max_azimuth_variations=2,
                           num_workers=None,
                           random_seed=SEED,
                           aliased_ratio=1.0,
                           clean_ratio_threshold=0.7,
                           force_test_cases=None,
                           enable_targeted_patches=False,
                           enable_balanced_sampling=False,
                           target_distribution=None,
                           save_sweeps=False,
                           sweep_only=False):
    """
    將 NWP Parquet 數據轉換為 H5 Patches

    Parameters:
    -----------
    nwp_root : str
        NWP 數據根目錄
    output_h5_path : str
        輸出 H5 檔案路徑
    target_stations : list
        目標雷達站列表
    skip_first_n : int
        每個目錄跳過前N個檔案（NWP spin-up）
    max_dirs : int
        最多處理幾個目錄（None=全部）
    max_cases_per_station : int
        每個站點最多處理幾個案例（None=全部）
    train_ratio : float
        訓練集比例（預設 0.6）
    val_ratio : float
        驗證集比例（預設 0.2）
    test_ratio : float
        測試集比例（預設 0.2）
    """

    random.seed(random_seed)
    np.random.seed(random_seed)

    if num_workers is None:
        num_workers = min(8, max(1, multiprocessing.cpu_count() - 1))

    print("🚀 NWP 轉 H5 Patch 轉換工具")
    print(f"   目標站點: {target_stations}")
    print(f"   輸出檔案: {output_h5_path}")

    # 創建臨時目錄
    temp_dir = "temp_nwp_patches"
    os.makedirs(temp_dir, exist_ok=True)

    # 收集所有 NWP 三元組
    all_triplets = []
    nwp_root_path = Path(nwp_root)
    subdirs = sorted([d for d in nwp_root_path.iterdir() if d.is_dir()])

    if max_dirs:
        subdirs = subdirs[:max_dirs]
        print(f"   限制處理前 {max_dirs} 個目錄")

    print(f"   總共 {len(subdirs)} 個目錄")

    # 從每個目錄載入triplets（記錄 NWP 預報目錄以避免資料洩漏）
    for subdir in subdirs:
        triplets = find_nwp_triplets(str(subdir), skip_first_n=skip_first_n)
        filtered = [t for t in triplets if t['station'] in target_stations]

        # 添加 NWP 預報目錄資訊
        for t in filtered:
            t['nwp_dir'] = subdir.name  # 記錄來源目錄

        all_triplets.extend(filtered)

        if filtered:
            print(f"   {subdir.name}: {len(filtered)} 個三元組")

    print(f"\n✅ 總共載入 {len(all_triplets)} 個三元組")

    # 按站點統計和限制
    station_triplets = defaultdict(list)
    for t in all_triplets:
        station_triplets[t['station']].append(t)

    print("\n站點分佈:")
    for station in target_stations:
        count = len(station_triplets[station])
        print(f"   {station}: {count} 個案例")

        # 限制每站案例數（如果指定）
        if max_cases_per_station and count > max_cases_per_station:
            random.shuffle(station_triplets[station])
            station_triplets[station] = station_triplets[station][:max_cases_per_station]
            print(f"      → 限制為 {max_cases_per_station} 個")
            print(f"      提示: 建議使用 --max_dirs 限制資料量以保持所有站點平衡")

    # ========================================
    # v3.0: 按 source file 分割以避免資料洩漏
    # 同一個 source file 的所有 patches 只會出現在同一個 split
    # source file = station + timestamp (e.g., RCCG.20241001.1350)
    # ========================================

    train_triplets = []
    val_triplets = []
    test_triplets = []

    # 解析強制 test 案例（支援三種格式）
    # 格式: 目錄 | 目錄:站點 | 目錄:站點:時間
    force_test_rules = []
    if force_test_cases:
        for case in force_test_cases:
            parts = case.split(':')
            if len(parts) == 1:
                # 格式 1: 只有目錄
                force_test_rules.append({'nwp_dir': parts[0]})
            elif len(parts) == 2:
                # 格式 2: 目錄:站點
                force_test_rules.append({'nwp_dir': parts[0], 'station': parts[1]})
            elif len(parts) >= 3:
                # 格式 3: 目錄:站點:時間
                force_test_rules.append({'nwp_dir': parts[0], 'station': parts[1], 'timestamp': parts[2]})
        print(f"\n   強制 test 規則: {len(force_test_rules)} 條")
        for rule in force_test_rules:
            print(f"      {rule}")

    def matches_force_test_rule(triplet, rules):
        """檢查 triplet 是否匹配任何強制 test 規則"""
        for rule in rules:
            match = True
            if 'nwp_dir' in rule and triplet.get('nwp_dir', '') != rule['nwp_dir']:
                match = False
            if 'station' in rule and triplet.get('station', '') != rule['station']:
                match = False
            if 'timestamp' in rule and triplet.get('timestamp', '') != rule['timestamp']:
                match = False
            if match:
                return True
        return False

    # 追蹤強制 test 匹配情況
    total_forced_matched = 0

    # 按站點分層、按 source file 分割
    for station in target_stations:
        triplets = station_triplets[station]

        # 將 triplets 按 source file 分組
        # source_key = (station, timestamp) 代表同一個原始檔案
        source_groups = defaultdict(list)
        for t in triplets:
            source_key = (t['station'], t['timestamp'])
            source_groups[source_key].append(t)

        # 分離強制 test 的 sources 和其他 sources
        forced_test_sources = []
        remaining_sources = []

        for source_key, group_triplets in source_groups.items():
            # 檢查該 source group 中是否有 ANY triplet 匹配強制 test 規則
            # (同一個 timestamp 可能來自不同的 nwp_dir，需要全部檢查)
            is_forced = False
            if force_test_rules:
                for t in group_triplets:
                    if matches_force_test_rule(t, force_test_rules):
                        is_forced = True
                        break

            if is_forced:
                forced_test_sources.append(source_key)
            else:
                remaining_sources.append(source_key)

        # 隨機打亂剩餘的 sources
        random.shuffle(remaining_sources)

        # 按 source 分割剩餘的（不是按 triplet）
        # 先分配正常的 test_ratio，再加入 forced_test_sources
        n_remaining = len(remaining_sources)

        # 正常三分：train / val / test
        n_train_sources = int(n_remaining * train_ratio)
        n_val_sources = int(n_remaining * val_ratio)
        # 剩餘的給 test（確保總數正確）

        train_sources = remaining_sources[:n_train_sources]
        val_sources = remaining_sources[n_train_sources:n_train_sources + n_val_sources]
        normal_test_sources = remaining_sources[n_train_sources + n_val_sources:]

        # 測試集 = 正常分配 + 強制 test 案例
        test_sources = normal_test_sources + forced_test_sources

        # 將 source 分組的 triplets 加入對應的 split
        for sk in train_sources:
            train_triplets.extend(source_groups[sk])
        for sk in val_sources:
            val_triplets.extend(source_groups[sk])
        for sk in test_sources:
            test_triplets.extend(source_groups[sk])

        forced_count = len(forced_test_sources)
        normal_test_count = len(normal_test_sources)
        total_forced_matched += forced_count
        print(f"   {station}: {len(source_groups)} 個 source files -> "
              f"Train:{len(train_sources)}, Val:{len(val_sources)}, Test:{len(test_sources)}"
              + (f" (正常:{normal_test_count}+強制:{forced_count})" if forced_count > 0 else ""))

    # 顯示強制 test 匹配摘要
    if force_test_rules:
        print(f"\n   強制 test 匹配結果: {total_forced_matched}/{len(force_test_rules)} 條規則已匹配")
        if total_forced_matched < len(force_test_rules):
            print(f"   ⚠️ 有 {len(force_test_rules) - total_forced_matched} 條規則未匹配！")
            print(f"   提示: timestamp 格式應為 YYYYMMDDHHMM_EL (例如 202407242020_01)")

            # 除錯：顯示每條規則的匹配狀態
            print(f"\n   === 強制 test 規則除錯 ===")
            all_triplet_keys = set()
            for t in all_triplets:
                key = (t.get('nwp_dir', ''), t.get('station', ''), t.get('timestamp', ''))
                all_triplet_keys.add(key)

            for rule in force_test_rules:
                rule_nwp = rule.get('nwp_dir', '*')
                rule_station = rule.get('station', '*')
                rule_ts = rule.get('timestamp', '*')

                # 檢查是否有匹配
                matched = False
                partial_matches = []
                for key in all_triplet_keys:
                    nwp, sta, ts = key
                    nwp_match = ('nwp_dir' not in rule) or (nwp == rule_nwp)
                    sta_match = ('station' not in rule) or (sta == rule_station)
                    ts_match = ('timestamp' not in rule) or (ts == rule_ts)

                    if nwp_match and sta_match and ts_match:
                        matched = True
                        break
                    # 收集部分匹配（nwp_dir 和 station 匹配但 timestamp 不匹配）
                    if nwp_match and sta_match and not ts_match:
                        partial_matches.append(ts)

                status = "✅ 已匹配" if matched else "❌ 未匹配"
                print(f"   規則: nwp_dir={rule_nwp}, station={rule_station}, timestamp={rule_ts} -> {status}")

                if not matched and partial_matches:
                    # 顯示該 nwp_dir + station 實際存在的 timestamps
                    print(f"      該目錄+站點實際存在的 timestamps (前5個):")
                    for ts in sorted(partial_matches)[:5]:
                        print(f"         {ts}")
                    if len(partial_matches) > 5:
                        print(f"         ... 共 {len(partial_matches)} 個")
            print(f"   === 除錯結束 ===\n")

    # 根據 aliased_ratio 為每個 triplet 分配 patch_type
    # aliased_ratio=1.0 表示全部是 aliased patches
    # aliased_ratio=0.8 表示 80% aliased + 20% clean patches
    print(f"\n   Aliased ratio: {aliased_ratio:.1%}")

    def assign_patch_types(triplets_list, aliased_ratio):
        """為 triplets 分配 patch_type"""
        n_total = len(triplets_list)
        n_aliased = int(n_total * aliased_ratio)

        # 隨機打亂後分配
        indices = list(range(n_total))
        random.shuffle(indices)

        for i, idx in enumerate(indices):
            if i < n_aliased:
                triplets_list[idx]['patch_type'] = 'aliased'
            else:
                triplets_list[idx]['patch_type'] = 'clean'

        return triplets_list

    train_triplets = assign_patch_types(train_triplets, aliased_ratio)
    val_triplets = assign_patch_types(val_triplets, aliased_ratio)
    test_triplets = assign_patch_types(test_triplets, aliased_ratio)

    # 統計 patch type 分佈
    for split_name, triplets in [('Train', train_triplets), ('Val', val_triplets), ('Test', test_triplets)]:
        n_aliased = sum(1 for t in triplets if t.get('patch_type', 'aliased') == 'aliased')
        n_clean = len(triplets) - n_aliased
        print(f"   {split_name}: {n_aliased} aliased + {n_clean} clean")

    # 統計每個集合涵蓋的 NWP 目錄（用於驗證）
    train_dirs = set(t['nwp_dir'] for t in train_triplets)
    val_dirs = set(t['nwp_dir'] for t in val_triplets)
    test_dirs = set(t['nwp_dir'] for t in test_triplets)

    print(f"\n目錄分布統計:")
    print(f"   訓練集: {len(train_dirs)} 個目錄")
    print(f"   驗證集: {len(val_dirs)} 個目錄")
    print(f"   測試集: {len(test_dirs)} 個目錄")
    print(f"   目錄重疊: 訓練/測試={len(train_dirs & test_dirs)} 個 (目錄重疊不算洩漏)")

    # 驗證資料洩漏：檢查 source file 是否重疊
    train_sources = set((t['station'], t['timestamp']) for t in train_triplets)
    val_sources = set((t['station'], t['timestamp']) for t in val_triplets)
    test_sources = set((t['station'], t['timestamp']) for t in test_triplets)

    train_val_leak = train_sources & val_sources
    train_test_leak = train_sources & test_sources
    val_test_leak = val_sources & test_sources

    if train_val_leak or train_test_leak or val_test_leak:
        print(f"\n   [ERROR] 發現資料洩漏!")
        print(f"   Train-Val: {len(train_val_leak)} 個 source files")
        print(f"   Train-Test: {len(train_test_leak)} 個 source files")
        print(f"   Val-Test: {len(val_test_leak)} 個 source files")
    else:
        print(f"\n   [OK] 無資料洩漏 (source file level)")

    print(f"\n分割結果:")
    print(f"   訓練集: {len(train_triplets)} 個 triplets")
    print(f"   驗證集: {len(val_triplets)} 個 triplets")
    print(f"   測試集: {len(test_triplets)} 個 triplets")

    # 警告：如果驗證集為空
    if len(val_triplets) == 0:
        print(f"\n   警告: 驗證集為空！總樣本數 ({len(all_triplets)}) 太少")
        print(f"   建議: 增加 --max_cases_per_station 或移除 --max_dirs 限制")

    # 提取參數
    extract_kwargs = {
        'layers': 4,
        'patch_size': patch_size,
        'num_patches_per_file': num_patches_per_file,
        'min_alias_pixels': min_alias_pixels,
        'augment_variations': augment_variations,
        'enable_azimuth_rotation': enable_azimuth_rotation,
        'max_azimuth_variations': max_azimuth_variations,
        'clean_ratio_threshold': clean_ratio_threshold,
        'enable_targeted_patches': enable_targeted_patches
    }

    if enable_targeted_patches:
        print(f"   定向高 aliasing 切割: 啟用")
    if enable_balanced_sampling:
        if target_distribution is None:
            target_distribution = [0.15, 0.20, 0.25, 0.25, 0.15]
        print(f"   分桶重採樣: 啟用")
        print(f"   目標分佈: {target_distribution}")

    # ================================================================
    # 儲存 full sweeps（供未來 Online Random Crop / Full Sweep 訓練��用）
    # ================================================================
    if sweep_only:
        save_sweeps = True  # sweep_only 模式自動啟用
    if save_sweeps:
        sweep_h5_path = output_h5_path.replace('.h5', '_sweeps.h5')
        print(f"\n💾 儲存 full sweeps 到 {sweep_h5_path}...")
        os.makedirs(os.path.dirname(os.path.abspath(sweep_h5_path)), exist_ok=True)

        # 收集所有不重複的 triplets（按 nwp_dir + station + timestamp 去重）
        all_sweep_triplets = []
        for split_name, triplets in [('train', train_triplets),
                                     ('val', val_triplets),
                                     ('test', test_triplets)]:
            seen = set()
            for t in triplets:
                key = (t.get('nwp_dir', ''), t['station'], t['timestamp'])
                if key not in seen:
                    seen.add(key)
                    all_sweep_triplets.append((split_name, t))

        # 統計各 split 各站數量
        split_station_counts = defaultdict(lambda: defaultdict(int))
        for split_name, t in all_sweep_triplets:
            split_station_counts[split_name][t['station']] += 1

        print(f"\n   Sweep 分佈:")
        for sn in ['train', 'val', 'test']:
            counts = split_station_counts.get(sn, {})
            total_s = sum(counts.values())
            detail = ', '.join(f"{st}:{c}" for st, c in sorted(counts.items()))
            print(f"   {sn}: {total_s} sweeps ({detail})")

        with h5py.File(sweep_h5_path, 'w') as swf:
            failed_count = 0
            for i, (split_name, triplet) in enumerate(
                    tqdm(all_sweep_triplets, desc="儲存 sweeps")):
                try:
                    # 只讀 DA（真值），用於 random Nyquist 訓練
                    # raw_vel 也存，供對比/除錯用
                    da_matrix, da_nyquist, _ = load_parquet_as_matrix(triplet['da'])
                    raw_matrix, raw_nyquist, _ = load_parquet_as_matrix(triplet['raw'])
                    nyq_val = da_nyquist if isinstance(da_nyquist, (int, float)) else np.nanmean(da_nyquist)

                    # 基本驗證
                    if da_matrix.shape != raw_matrix.shape:
                        print(f"⚠️ sweep {i}: DA shape {da_matrix.shape} != RAW shape {raw_matrix.shape}，跳過")
                        failed_count += 1
                        continue

                    sg = swf.create_group(f'sweep_{i}')
                    sg.create_dataset('da_vel', data=da_matrix.astype(np.float32),
                                      compression='lzf')
                    sg.create_dataset('raw_vel', data=raw_matrix.astype(np.float32),
                                      compression='lzf')
                    sg.attrs['nyq'] = float(nyq_val)
                    sg.attrs['station'] = triplet['station']
                    sg.attrs['timestamp'] = triplet['timestamp']
                    sg.attrs['nwp_dir'] = triplet.get('nwp_dir', '')
                    sg.attrs['split'] = split_name
                    sg.attrs['shape_az'] = da_matrix.shape[0]
                    sg.attrs['shape_range'] = da_matrix.shape[1]
                except Exception as e:
                    print(f"⚠️ 儲存 sweep {i} 失敗: {e}")
                    failed_count += 1

            swf.attrs['num_sweeps'] = len(all_sweep_triplets) - failed_count
            swf.attrs['num_failed'] = failed_count
            swf.attrs['created_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
            swf.attrs['target_stations'] = ','.join(target_stations)
            swf.attrs['skip_first_n'] = skip_first_n
            swf.attrs['train_ratio'] = train_ratio
            swf.attrs['val_ratio'] = val_ratio
            swf.attrs['test_ratio'] = test_ratio
            swf.attrs['seed'] = random_seed
            swf.attrs['purpose'] = 'full_sweep_training_with_random_nyquist'

        print(f"✅ 已儲存 {len(all_sweep_triplets)} 個 full sweeps 到 {sweep_h5_path}")

    # sweep_only 模式：只存 sweep，跳過 patch 切割
    if sweep_only:
        print(f"\n🎉 sweep_only 模式完成！")
        print(f"💾 Full sweeps 已保存到 {sweep_h5_path}")
        return

    # ================================================================
    # 並行提取 patches，邊收邊寫入 H5（大 array + lzf）
    # ================================================================
    os.makedirs(os.path.dirname(os.path.abspath(output_h5_path)), exist_ok=True)

    if enable_balanced_sampling:
        write_h5_path = output_h5_path + '.tmp'
        print(f"\n   [兩階段模式] 先寫入臨時 H5: {write_h5_path}")
    else:
        write_h5_path = output_h5_path

    print(f"\n🔧 使用 {num_workers} 個工作進程並行處理...")

    ps = patch_size
    chunk_n = 32  # chunk 對齊 batch_size

    with h5py.File(write_h5_path, 'w') as h5f:

        for split_name, triplets in [('train', train_triplets),
                                     ('val', val_triplets),
                                     ('test', test_triplets)]:
            if not triplets:
                print(f"⚠️  {split_name}集沒有數據，跳過")
                continue

            # Val/Test 不做資料增強
            if split_name in ('val', 'test'):
                split_kwargs = {**extract_kwargs,
                                'augment_variations': 0,
                                'enable_azimuth_rotation': False}
            else:
                split_kwargs = extract_kwargs

            print(f"\n🔄 處理{split_name}集 ({len(triplets)}個三元組)...")

            group = h5f.create_group(split_name)

            # 建立可動態擴展的 datasets
            group.create_dataset('vel', shape=(0, ps, ps, 1), maxshape=(None, ps, ps, 1),
                                 dtype='float32', chunks=(chunk_n, ps, ps, 1), compression='lzf')
            group.create_dataset('nyq', shape=(0, 1), maxshape=(None, 1), dtype='float32')
            group.create_dataset('alias_label', shape=(0, ps, ps), maxshape=(None, ps, ps),
                                 dtype='int32', chunks=(chunk_n, ps, ps), compression='lzf')
            group.create_dataset('gt_vel', shape=(0, ps, ps, 1), maxshape=(None, ps, ps, 1),
                                 dtype='float32', chunks=(chunk_n, ps, ps, 1), compression='lzf')
            group.create_dataset('target_vel', shape=(0, ps, ps, 1), maxshape=(None, ps, ps, 1),
                                 dtype='float32', chunks=(chunk_n, ps, ps, 1), compression='lzf')
            group.create_dataset('alias_ratio', shape=(0,), maxshape=(None,), dtype='float32')

            # 字串型用 list 累積（h5py string dataset 不支援 resize，最後一次寫入）
            str_lists = {'patch_type': [], 'source_file': [], 'station': [],
                         'nwp_dir': [], 'timestamp': [], 'augment_type': []}

            buffer = []
            pos = 0  # 已寫入的 patch 總數

            def flush_buffer():
                """將 buffer 中的 patches 寫入 H5"""
                nonlocal pos
                if not buffer:
                    return
                n = len(buffer)

                # 擴展 datasets
                new_size = pos + n
                group['vel'].resize(new_size, axis=0)
                group['nyq'].resize(new_size, axis=0)
                group['alias_label'].resize(new_size, axis=0)
                group['gt_vel'].resize(new_size, axis=0)
                group['target_vel'].resize(new_size, axis=0)
                group['alias_ratio'].resize(new_size, axis=0)

                # 寫入數值型
                group['vel'][pos:new_size] = np.stack([p['vel'] for p in buffer])
                group['nyq'][pos:new_size] = np.stack([p['nyq'] for p in buffer])
                group['alias_label'][pos:new_size] = np.stack([p['alias_label'] for p in buffer])
                group['gt_vel'][pos:new_size] = np.stack([p['gt_vel'] for p in buffer])
                group['target_vel'][pos:new_size] = np.stack([p['target_vel'] for p in buffer])
                group['alias_ratio'][pos:new_size] = np.array(
                    [p.get('alias_ratio', -1.0) for p in buffer], dtype=np.float32)

                # 累積字串型
                for p in buffer:
                    str_lists['patch_type'].append(p['patch_type'])
                    str_lists['source_file'].append(p['source_info'].get('raw_file', ''))
                    str_lists['station'].append(p['source_info'].get('station', ''))
                    str_lists['nwp_dir'].append(p['source_info'].get('nwp_forecast_dir', ''))
                    str_lists['timestamp'].append(p['source_info'].get('timestamp', ''))
                    str_lists['augment_type'].append(p['source_info'].get('augment_type', ''))

                pos = new_size
                buffer.clear()

            # 多進程並行處理，每個 task = 1 triplet
            # 滑動窗口提交，避免一次排隊全部 triplet 佔記憶體
            effective_workers = min(num_workers, 4)
            with ProcessPoolExecutor(max_workers=effective_workers) as executor:
                pbar = tqdm(total=len(triplets), desc=f"處理{split_name}")

                # 分批提交，每批 = effective_workers * 3
                batch_window = effective_workers * 3
                for start in range(0, len(triplets), batch_window):
                    window = triplets[start:start + batch_window]
                    futures = {
                        executor.submit(
                            process_nwp_triplet_batch, [item], temp_dir, **split_kwargs
                        ): ti
                        for ti, item in enumerate(window, start=start)
                    }

                    for fut in as_completed(futures):
                        ti = futures[fut]
                        try:
                            patches = fut.result()
                            buffer.extend(patches)

                            while len(buffer) >= chunk_n:
                                to_write = buffer[:chunk_n]
                                buffer = buffer[chunk_n:]
                                buffer_bak = buffer
                                buffer = to_write
                                flush_buffer()
                                buffer = buffer_bak

                        except Exception as e:
                            print(f"❌ 處理 triplet {ti} 時發生錯誤: {e}")

                        pbar.update(1)

                pbar.close()

            # 寫入剩餘的 buffer
            flush_buffer()

            # 寫入字串型 datasets
            dt = h5py.string_dtype()
            for key, vals in str_lists.items():
                if vals:
                    group.create_dataset(key, data=vals, dtype=dt)

            group.attrs['format'] = 'big_array'
            group.attrs['num_patches'] = pos

            print(f"💾 寫入 {pos} 個 {split_name} patches（大 array + lzf）")

        # --- 保存元數據 ---
        metadata = h5f.create_group('metadata')
        metadata.attrs['patch_size'] = patch_size
        metadata.attrs['min_alias_pixels'] = min_alias_pixels
        metadata.attrs['created_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
        metadata.attrs['data_source'] = 'NWP_parquet'
        metadata.attrs['target_stations'] = ','.join(target_stations)
        metadata.attrs['skip_first_n'] = skip_first_n
        metadata.attrs['nwp_patch_version'] = 'v4.0'
        metadata.attrs['h5_format'] = 'big_array_lzf'
        metadata.attrs['split_strategy'] = 'source_file_based'
        metadata.attrs['train_dirs_count'] = len(train_dirs)
        metadata.attrs['val_dirs_count'] = len(val_dirs)
        metadata.attrs['test_dirs_count'] = len(test_dirs)
        metadata.attrs['dir_overlap_train_test'] = len(train_dirs & test_dirs)
        metadata.attrs['augment_variations'] = augment_variations
        metadata.attrs['max_azimuth_variations'] = max_azimuth_variations
        metadata.attrs['aliased_ratio'] = aliased_ratio
        metadata.attrs['clean_ratio_threshold'] = clean_ratio_threshold
        metadata.attrs['train_sources_count'] = len(train_sources)
        metadata.attrs['val_sources_count'] = len(val_sources)
        metadata.attrs['test_sources_count'] = len(test_sources)
        metadata.attrs['source_leak_train_val'] = len(train_val_leak)
        metadata.attrs['source_leak_train_test'] = len(train_test_leak)
        metadata.attrs['source_leak_val_test'] = len(val_test_leak)

        if force_test_cases:
            metadata.attrs['force_test_cases'] = ','.join(force_test_cases)
            metadata.attrs['force_test_cases_count'] = len(force_test_cases)
        if len(train_dirs) > 0:
            metadata.attrs['train_dirs_sample'] = ','.join(sorted(train_dirs)[:10])
        if len(test_dirs) > 0:
            metadata.attrs['test_dirs_sample'] = ','.join(sorted(test_dirs)[:10])

    # 階段二：分桶重採樣（僅 train split）
    if enable_balanced_sampling:
        print(f"\n🔄 階段二：分桶重採樣...")
        resample_h5(write_h5_path, output_h5_path, target_distribution, seed=random_seed)

        try:
            os.remove(write_h5_path)
            print(f"   已刪除臨時 H5: {write_h5_path}")
        except Exception as e:
            print(f"   ⚠️ 刪除臨時 H5 失敗: {e}")

    # 清理臨時目錄
    try:
        import shutil
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"⚠️ 清理臨時目錄失敗: {e}")

    print(f"\n🎉 NWP Patch 轉換完成！")
    print(f"💾 數據已保存到 {output_h5_path}")

    return output_h5_path


def main():
    parser = argparse.ArgumentParser(description='NWP Parquet 轉 H5 Patch 工具')

    parser.add_argument('--nwp_root', type=str, required=True,
                       help='NWP 數據根目錄')
    parser.add_argument('--output_h5', type=str, required=True,
                       help='輸出 H5 檔案路徑')
    parser.add_argument('--target_stations', nargs='+', default=['RCCG', 'RCWF'],
                       help='目標雷達站列表')
    parser.add_argument('--skip_first_n', type=int, default=6,
                       help='跳過每個目錄的前N個檔案（NWP spin-up）')
    parser.add_argument('--max_dirs', type=int, default=None,
                       help='最多處理幾個目錄（None=全部）')
    parser.add_argument('--max_cases_per_station', type=int, default=None,
                       help='每個站點最多處理幾個案例（None=全部）')
    parser.add_argument('--train_ratio', type=float, default=0.6,
                       help='訓練集比例')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='驗證集比例')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                       help='測試集比例')
    parser.add_argument('--patch_size', type=int, default=128,
                       help='Patch 大小')
    parser.add_argument('--num_patches_per_file', type=int, default=5,
                       help='每個檔案提取的 patch 數量')
    parser.add_argument('--min_alias_pixels', type=int, default=10,
                       help='最小 alias 像素數')
    parser.add_argument('--augment_variations', type=int, default=1,
                       help='矩陣增強變體數量')
    parser.add_argument('--enable_azimuth_rotation', action='store_true', default=True,
                       help='啟用方位角旋轉增強')
    parser.add_argument('--max_azimuth_variations', type=int, default=2,
                       help='最大方位角旋轉變體數')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='並行工作進程數')
    parser.add_argument('--seed', type=int, default=SEED,
                       help='隨機種子')
    parser.add_argument('--aliased_ratio', type=float, default=1.0,
                       help='Aliased patches 的比例 (1.0=全部aliased, 0.8=80%% aliased + 20%% clean)')
    parser.add_argument('--clean_ratio_threshold', type=float, default=0.7,
                       help='Clean patch 的最小乾淨像素比例 (預設 0.7)')
    parser.add_argument('--force_test_cases', nargs='+', default=None,
                       help='強制放入 test 集的案例，支援三種格式：'
                            '(1) 目錄: 24072417 '
                            '(2) 目錄:站點: 24072417:RCWF '
                            '(3) 目錄:站點:時間: 24072417:RCWF:202407241700')
    parser.add_argument('--enable_targeted_patches', action='store_true', default=False,
                       help='啟用定向高 aliasing 區域切割（增加高 alias ratio patches 來源）')
    parser.add_argument('--enable_balanced_sampling', action='store_true', default=False,
                       help='啟用分桶重採樣以平衡 train split 的 alias ratio 分佈')
    parser.add_argument('--target_distribution', nargs=5, type=float, default=None,
                       help='5 個 bin 的目標比例 [0-10%%, 10-30%%, 30-50%%, 50-70%%, 70-100%%] '
                            '(預設: 0.15 0.20 0.25 0.25 0.15)')
    parser.add_argument('--save_sweeps', action='store_true', default=False,
                       help='儲存 full sweeps 到獨立 H5（供未來 Online Random Crop 使用）')
    parser.add_argument('--sweep_only', action='store_true', default=False,
                       help='只儲存 full sweeps，跳過 patch 切割（供 full sweep 訓練用）')

    args = parser.parse_args()

    convert_nwp_to_patches(
        nwp_root=args.nwp_root,
        output_h5_path=args.output_h5,
        target_stations=args.target_stations,
        skip_first_n=args.skip_first_n,
        max_dirs=args.max_dirs,
        max_cases_per_station=args.max_cases_per_station,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        patch_size=args.patch_size,
        num_patches_per_file=args.num_patches_per_file,
        min_alias_pixels=args.min_alias_pixels,
        augment_variations=args.augment_variations,
        enable_azimuth_rotation=args.enable_azimuth_rotation,
        max_azimuth_variations=args.max_azimuth_variations,
        num_workers=args.num_workers,
        random_seed=args.seed,
        aliased_ratio=args.aliased_ratio,
        clean_ratio_threshold=args.clean_ratio_threshold,
        force_test_cases=args.force_test_cases,
        enable_targeted_patches=args.enable_targeted_patches,
        enable_balanced_sampling=args.enable_balanced_sampling,
        target_distribution=args.target_distribution,
        save_sweeps=args.save_sweeps,
        sweep_only=args.sweep_only
    )


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\n總執行時間: {(end_time - start_time)/60:.2f} 分鐘")
