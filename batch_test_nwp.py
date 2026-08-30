#!/usr/bin/env python3
"""
NWP資料批量測試系統 - 比較CNN vs DA vs RAW

特點：
1. 支援Parquet格式（NWP資料）
2. 使用bref_qc作為100%正確的GT
3. 比較三種方法：RAW vs DA（物理算法）vs CNN模型
4. 使用fold-based成功率計算（無容忍度）
5. 完整的可視化功能：
   - 矩陣視圖比較圖（RAW/DA/CNN/QC）
   - Fold判斷分析圖（DA vs CNN準確度）
   - 地理位置圖（自動從nwp_data尋找對應.gz檔案）

資料結構：
/path/to/nwp_data_csv/
├── 24072411/
│   ├── RCCG.20240724.1200.bvel_raw.01.parquet    # RAW速度
│   ├── RCCG.20240724.1200.bvel_da.01.parquet     # DA算法結果
│   └── RCCG.20240724.1200.bref_qc.01.parquet     # GT（100%正確）
└── 24072412/
    └── ...

使用範例：
    # 測試單一目錄
    python batch_test_nwp.py --model_path model.h5 --nwp_dir /path/to/nwp_data_csv/24072411 --max_cases 5

    # 分層隨機抽樣（確保每個資料夾和雷達站都涵蓋）[*]推薦
    python batch_test_nwp.py --model_path model.h5 --nwp_root /path/to/nwp_data_csv --max_cases 100 --stratified_sample --max_dirs 20

    # 啟用可視化（生成比較圖、fold分析圖、地理圖）
    python batch_test_nwp.py --model_path model.h5 --nwp_root /path/to/nwp_data_csv --max_cases 20 --stratified_sample --enable_viz

    # 指定雷達站 + 分層抽樣 + 可視化 + 台灣地圖
    python batch_test_nwp.py --model_path model.h5 --nwp_root /path/to/nwp_data_csv --stations RCCG RCMK --max_cases 50 --stratified_sample --enable_viz --shape_path mapdata201805310314/COUNTY_MOI_1070516

    # 可重現的隨機抽樣（使用random_seed）
    python batch_test_nwp.py --model_path model.h5 --nwp_root /path/to/nwp_data_csv --max_cases 100 --stratified_sample --random_seed 123
"""

import pandas as pd
import numpy as np
import os
import json
import argparse
from datetime import datetime
from collections import defaultdict
import traceback
import sys
from pathlib import Path
import pyarrow.parquet as pq
import random
import matplotlib
matplotlib.use('Agg')  # 非GUI後端
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import h5py
import tqdm

# 氣象署標準速度色階（與 mixed_patch_test_model.py 一致）
CWA_VEL_LEVELS = [-80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80]
CWA_VEL_COLORS = [
    "#000078", "#0000ce", "#0063ff", "#31ffff",
    "#31ff63", "#00ce00", "#009c00", "#cecece",
    "#636363",
    "#ffff31", "#ffce31", "#ff9c00", "#ff6363",
    "#ff0000", "#ce0000", "#8d005c"
]
CWA_VEL_CMAP = mcolors.LinearSegmentedColormap.from_list("cwa_vel", CWA_VEL_COLORS)
CWA_VEL_NORM = mcolors.BoundaryNorm(CWA_VEL_LEVELS, CWA_VEL_CMAP.N)
# 導入測試函數和模型構建函數
try:
    from mixed_patch_test_model import build_mixed_patch_model_for_inference
    import tensorflow as tf
    TEST_FUNCTION_AVAILABLE = True
except ImportError as e:
    print(f"警告: 無法導入測試函數 - {e}")
    print("這通常是因為 tensorflow 等依賴套件未安裝")
    TEST_FUNCTION_AVAILABLE = False

# 導入修正後的成功率計算
try:
    from dealiasing_success_metrics import compute_dealiasing_success_metrics
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    print("警告: 無法導入成功率指標函數")

# 導入地理可視化相關模組
try:
    import pyart
    from pyart.graph import RadarMapDisplayBasemap
    from mixed_patch_test_model import (read_cwb_radar_sweep, plot_taiwan_basemap, get_metadata,
                                       save_visualizations, save_geographical_visualizations)
    GEO_VIZ_AVAILABLE = True
except ImportError as e:
    GEO_VIZ_AVAILABLE = False
    print(f"警告: 地理可視化模組不可用 - {e}")


def crf_spatial_smoothing(fold_probs, unary_weight=1.0, pairwise_weight=1.5,
                          max_iterations=10, threshold=1.5, confidence_threshold=None):
    """
    使用簡化版 CRF 進行空間平滑後處理

    修正版本（2025-12-24）：
    - 移除信心度過濾（處理所有像素）
    - 使用對數機率作為 unary energy
    - 調整 pairwise_weight 預設值（50 → 1.5）
    - 調整 threshold（0.5 → 1.5，只懲罰大跳躍）

    Args:
        fold_probs: (H, W, 5) fold number 機率分布 [-2,-1,0,+1,+2]
        unary_weight: 一元勢能權重（保留原始預測）
        pairwise_weight: 二元勢能權重（空間一致性）
        max_iterations: 最大迭代次數
        threshold: fold 差異閾值（預設 1.5：只懲罰 diff >= 2 的跳躍）
        confidence_threshold: 已廢棄，保留向後相容性

    Returns:
        smoothed_fold_labels: (H, W) 平滑後的 fold labels (index: 0-4)
    """
    H, W, _ = fold_probs.shape
    fold_values = np.array([-2, -1, 0, 1, 2])

    # 初始化：使用原始預測
    current_fold = np.argmax(fold_probs, axis=-1)  # (H, W)

    # 信心度過濾已移除（處理所有像素）
    if confidence_threshold is not None:
        print(f"     [WARNING]  警告：confidence_threshold 參數已廢棄，CRF 將處理所有像素")

    for iteration in range(max_iterations):
        new_fold = current_fold.copy()

        for i in range(H):
            for j in range(W):
                # 一元勢能：使用對數機率（修正量級問題）
                unary_energy = -unary_weight * np.log(fold_probs[i, j] + 1e-10)  # (5,)

                # 二元勢能：與鄰居的一致性
                pairwise_energy = np.zeros(5)
                neighbor_count = 0

                # 檢查 4-鄰域
                for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W:
                        neighbor_fold_idx = current_fold[ni, nj]
                        neighbor_fold_value = fold_values[neighbor_fold_idx]

                        # 計算每個候選 fold 與鄰居的差異
                        for k, candidate_fold_value in enumerate(fold_values):
                            diff = abs(candidate_fold_value - neighbor_fold_value)
                            # 懲罰大跳躍
                            if diff > threshold:
                                pairwise_energy[k] += pairwise_weight * (diff - threshold) ** 2

                        neighbor_count += 1

                if neighbor_count > 0:
                    pairwise_energy /= neighbor_count

                # 總能量 = 一元 + 二元
                total_energy = unary_energy + pairwise_energy

                # 選擇能量最小的 fold
                new_fold[i, j] = np.argmin(total_energy)

        # 檢查收斂
        if np.array_equal(new_fold, current_fold):
            break

        current_fold = new_fold

    return current_fold


def find_corresponding_gz_file(parquet_path):
    """
    從Parquet路徑找到對應的.gz檔案

    路徑映射：
    /path/to/nwp_data_csv/24072411/RCMK.20240724.1200.bvel_raw.01.parquet
    → /path/to/nwp_data/24072411/polar/binary/RCMK/RCMK.20240724.1200.bvel_raw.01.gz

    Args:
        parquet_path: Parquet檔案路徑

    Returns:
        gz_path: 對應的.gz檔案路徑，如果不存在則返回None
    """
    parquet_path = Path(parquet_path)

    # 提取檔名和日期目錄
    filename = parquet_path.name.replace('.parquet', '.gz')
    date_dir = parquet_path.parent.name  # 例如 24072411

    # 提取雷達站名稱（檔名前4個字符，例如 RCMK）
    station = filename.split('.')[0]

    # 構建 .gz 檔案路徑；NWP 反射率資料根目錄以環境變數 NWP_DATA_ROOT 指定
    gz_path = Path(os.environ.get('NWP_DATA_ROOT', 'nwp_data')) / date_dir / 'polar' / 'binary' / station / filename

    # 調試信息
    print(f"  [SEARCH] 尋找GZ檔案: {gz_path}")

    if os.path.exists(str(gz_path)):
        print(f"  [OK] 找到GZ檔案")
        return str(gz_path)
    else:
        print(f"  [ERROR] GZ檔案不存在")
        return None


def sort_azimuths_circular(azimuths):
    """
    對方位角進行周期性排序，處理跨越0度邊界的情況

    雷達掃描可能從任意角度開始，例如:
    - 從350度開始: [350, 351, ..., 359, 0, 1, ..., 349]
    - 從-10度開始: [-10, -9, ..., 0, 1, ..., 349] (等同於 [350, 351, ..., 359, 0, ..., 349])

    Args:
        azimuths: 方位角陣列

    Returns:
        sorted_azimuths: 按掃描順序排序的方位角（保持周期連續性）
    """
    azimuths = np.array(azimuths)

    # 正規化到 [0, 360) 範圍
    azimuths_normalized = azimuths % 360.0

    # 找到最小方位角作為起始點
    min_az = azimuths_normalized.min()

    # 檢查是否跨越0度邊界
    # 如果最大值和最小值差距接近360度，表示跨越邊界
    az_span = azimuths_normalized.max() - min_az

    if az_span > 180:
        # 跨越0度邊界的情況
        # 找到間隔最大的位置（掃描起點）
        sorted_az = np.sort(azimuths_normalized)
        gaps = np.diff(sorted_az)

        # 找到最大間隔（這是掃描的起點/終點斷裂處）
        if len(gaps) > 0:
            max_gap_idx = np.argmax(gaps)
            # 將陣列從斷裂處重新排列
            sorted_azimuths = np.concatenate([sorted_az[max_gap_idx+1:], sorted_az[:max_gap_idx+1]])
        else:
            sorted_azimuths = sorted_az
    else:
        # 未跨越邊界，直接排序
        sorted_azimuths = np.sort(azimuths_normalized)

    return sorted_azimuths


def load_parquet_as_matrix(parquet_path, debug=False):
    """
    載入Parquet並轉換為極坐標矩陣

    Returns:
        velocity_matrix: (H, W) 速度矩陣
        nyquist: Nyquist速度
        metadata: 額外資訊（站名、時間等）
    """
    # 讀取Parquet
    table = pq.read_table(parquet_path)
    df = table.to_pandas()

    if debug:
        print(f"    DEBUG - Parquet 欄位: {list(df.columns)}")
        print(f"    DEBUG - 資料筆數: {len(df)}")
        value_col = 'Value' if 'Value' in df.columns else 'value'
        print(f"    DEBUG - Value 範圍: [{df[value_col].min():.2f}, {df[value_col].max():.2f}]")
        print(f"    DEBUG - Value 有效值數: {df[value_col].notna().sum()} / {len(df)}")

    # 提取Nyquist速度（假設在Nyquist欄位或metadata中）
    if 'Nyquist' in df.columns:
        nyquist = df['Nyquist'].iloc[0]
    elif 'nyquist' in df.columns:
        nyquist = df['nyquist'].iloc[0]
    else:
        # 從檔名或metadata推斷
        nyquist = 33.62  # 預設值，需根據實際情況調整

    # 獲取方位角和距離的唯一值
    az_col = 'Azimuth' if 'Azimuth' in df.columns else 'azimuth'
    r_col = 'Range' if 'Range' in df.columns else 'range'

    azimuths_raw = df[az_col].unique()
    ranges = sorted(df[r_col].unique())

    # [FIX] 使用原始方位角順序（不排序），保持與 .gz 檔案一致
    # 這樣地理可視化時，方位角和速度值才能正確對應
    azimuths = azimuths_raw  # 保持 Parquet 中的原始順序

    n_az, n_range = len(azimuths), len(ranges)
    velocity_matrix = np.full((n_az, n_range), np.nan)

    # 建立方位角查找字典（考慮周期性：原始值 → 正規化值 → 索引）
    # 因為 df 中的方位角可能是負數或 >360，需要映射到正規化的 azimuths
    value_col = 'Value' if 'Value' in df.columns else 'value'

    # 建立方位角到索引的映射（使用正規化值）
    azimuth_to_idx = {az: idx for idx, az in enumerate(azimuths)}

    # 向量化填充矩陣（取代 iterrows，快 50-100x）
    az_vals = (df[az_col].values % 360.0)
    r_vals = df[r_col].values
    v_vals = df[value_col].values

    # 方位角索引：先用 dict 查找，找不到的用最近鄰
    az_indices = np.array([azimuth_to_idx.get(az, -1) for az in az_vals], dtype=np.intp)
    missing_mask = az_indices == -1
    if missing_mask.any():
        azimuths_arr = np.asarray(azimuths, dtype=np.float64)
        for i in np.where(missing_mask)[0]:
            az_indices[i] = np.argmin(np.abs(azimuths_arr - az_vals[i]))

    # 距離索引：用 dict 查找
    range_to_idx = {r: idx for idx, r in enumerate(ranges)}
    r_indices = np.array([range_to_idx[r] for r in r_vals], dtype=np.intp)

    # 篩選有效值
    valid = (v_vals != -999.0) & ~np.isnan(v_vals)
    velocity_matrix[az_indices[valid], r_indices[valid]] = v_vals[valid]

    if debug:
        valid_count = np.sum(~np.isnan(velocity_matrix))
        print(f"    DEBUG - Matrix 形狀: {velocity_matrix.shape}")
        print(f"    DEBUG - Matrix 有效值: {valid_count} / {velocity_matrix.size}")
        print(f"    DEBUG - Matrix 範圍: [{np.nanmin(velocity_matrix):.2f}, {np.nanmax(velocity_matrix):.2f}]")

    # 提取metadata
    metadata = {
        'shape': velocity_matrix.shape,
        'nyquist': nyquist,
        'n_azimuths': n_az,
        'n_ranges': n_range
    }

    return velocity_matrix, nyquist, metadata


def save_matrix_as_csv(velocity_matrix, nyquist, template_parquet_path, output_csv_path):
    """將速度矩陣保存為CSV（用於模型輸入）"""
    # 讀取模板獲取坐標信息
    table = pq.read_table(template_parquet_path)
    df_template = table.to_pandas()

    az_col = 'Azimuth' if 'Azimuth' in df_template.columns else 'azimuth'
    r_col = 'Range' if 'Range' in df_template.columns else 'range'

    # 使用周期性排序處理方位角（與 load_parquet_as_matrix 保持一致）
    azimuths_raw = df_template[az_col].unique()
    azimuths = sort_azimuths_circular(azimuths_raw)[:velocity_matrix.shape[0]]
    ranges = sorted(df_template[r_col].unique())[:velocity_matrix.shape[1]]

    # 構建輸出數據
    data = []
    for i, az in enumerate(azimuths):
        for j, r in enumerate(ranges):
            value = velocity_matrix[i, j]
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


def find_nwp_triplets(nwp_dir, skip_first_n=6):
    """
    尋找NWP資料三元組（raw, da, qc）

    根據資料提供者建議：每個資料夾內的資料請略過前6筆，即第1小時的資料不要使用
    原因：NWP模式在初始化後第一小時有 spin-up 問題，品質較差

    Args:
        nwp_dir: NWP資料目錄
        skip_first_n: 每個 station+elevation 組合要跳過的前N個檔案（預設6 = 第1小時）

    Returns:
        List of dicts: [{'raw': path, 'da': path, 'qc': path, 'station': ..., ...}, ...]
    """
    triplets = []
    nwp_path = Path(nwp_dir)

    if not nwp_path.exists():
        print(f"[ERROR] 目錄不存在: {nwp_dir}")
        return triplets

    # 找所有raw檔案
    raw_files = list(nwp_path.glob("*.bvel_raw.*.parquet"))

    # 按 station + elevation 分組，然後按時間排序
    from collections import defaultdict
    grouped_files = defaultdict(list)

    for raw_file in raw_files:
        # 從檔名提取資訊
        # 例如：RCCG.20240724.1200.bvel_raw.01.parquet
        parts = raw_file.name.split('.')
        station = parts[0]
        date = parts[1]
        time = parts[2]
        elevation = parts[-2]

        # 分組 key: station + elevation
        group_key = f"{station}_{elevation}"
        grouped_files[group_key].append({
            'file': raw_file,
            'station': station,
            'date': date,
            'time': time,
            'elevation': elevation,
            'timestamp': f"{date}{time}_{elevation}"  # 包含 elevation 以便精確匹配
        })

    # 對每組按時間排序，跳過前N個
    skipped_count = 0
    used_count = 0

    for group_key, files in grouped_files.items():
        # 按時間戳排序
        files_sorted = sorted(files, key=lambda x: x['timestamp'])

        # 跳過前N個
        files_to_use = files_sorted[skip_first_n:]
        skipped_count += min(skip_first_n, len(files_sorted))
        used_count += len(files_to_use)

        # 構建三元組
        for file_info in files_to_use:
            raw_file = file_info['file']

            # 構建對應的da和qc檔案路徑
            da_file = Path(str(raw_file).replace('bvel_raw', 'bvel_da'))
            qc_file = Path(str(raw_file).replace('bvel_raw', 'bref_qc'))

            # 檢查三個檔案都存在
            if da_file.exists() and qc_file.exists():
                triplets.append({
                    'raw': str(raw_file),
                    'da': str(da_file),
                    'qc': str(qc_file),
                    'station': file_info['station'],
                    'date': file_info['date'],
                    'time': file_info['time'],
                    'elevation': file_info['elevation'],
                    'timestamp': file_info['timestamp'],
                    'filename': raw_file.name
                })

    if skip_first_n > 0:
        print(f"  [SKIP]  跳過前 {skip_first_n} 筆/組: 已略過 {skipped_count} 個檔案，使用 {used_count} 個檔案")

    return triplets


def scan_nwp_root(nwp_root, max_dirs=None, stations=None):
    """
    掃描NWP根目錄下的所有子目錄

    Args:
        nwp_root: NWP資料根目錄（例如 /path/to/nwp_data_csv）
        max_dirs: 最多掃描幾個子目錄
        stations: 指定雷達站列表（例如 ['RCCG', 'RCMK']）

    Returns:
        List of triplets
    """
    root_path = Path(nwp_root)

    if not root_path.exists():
        print(f"[ERROR] 根目錄不存在: {nwp_root}")
        return []

    # 獲取所有子目錄
    subdirs = sorted([d for d in root_path.iterdir() if d.is_dir()])

    if max_dirs:
        subdirs = subdirs[:max_dirs]

    print(f"[DIR] 掃描 {len(subdirs)} 個子目錄...")

    all_triplets = []
    for subdir in subdirs:
        triplets = find_nwp_triplets(subdir)

        # 篩選雷達站
        if stations:
            triplets = [t for t in triplets if t['station'] in stations]

        all_triplets.extend(triplets)
        print(f"  {subdir.name}: 找到 {len(triplets)} 組資料")

    return all_triplets


def check_triplet_has_aliasing(triplet):
    """
    檢查一個三元組是否有疊加問題（使用與 compute_metrics_triplet 相同的 fold 邏輯）

    Returns:
        (has_aliasing, need_correction_pixels, total_pixels): 是否有疊加, 需修正像素數, 總有效像素數
    """
    try:
        # 讀取 RAW 和 DA
        raw_matrix, nyquist, _ = load_nwp_parquet(triplet['raw'])
        da_matrix, _, _ = load_nwp_parquet(triplet['da'])

        # 對齊尺寸
        min_h = min(raw_matrix.shape[0], da_matrix.shape[0])
        min_w = min(raw_matrix.shape[1], da_matrix.shape[1])
        raw_matrix = raw_matrix[:min_h, :min_w]
        da_matrix = da_matrix[:min_h, :min_w]

        # 有效像素遮罩
        valid_mask = ~np.isnan(raw_matrix) & ~np.isnan(da_matrix)

        if not valid_mask.any():
            return False, 0, 0

        # 使用與 compute_metrics_triplet 相同的 fold 計算邏輯
        raw_fold = np.round(raw_matrix[valid_mask] / (2 * nyquist))
        da_fold = np.round((da_matrix[valid_mask] - raw_matrix[valid_mask]) / (2 * nyquist))

        # 需要修正的像素（RAW 與 DA 的 fold 不同）
        need_correction_mask = (raw_fold != da_fold)
        need_correction_pixels = int(np.sum(need_correction_mask))
        total_pixels = int(np.sum(valid_mask))

        # 如果有任何像素需要修正，就視為有疊加問題
        has_aliasing = need_correction_pixels > 0

        return has_aliasing, need_correction_pixels, total_pixels

    except Exception as e:
        print(f"  [WARNING]  檢查失敗 {triplet['filename']}: {e}")
        return False, 0, 0


def filter_triplets_with_aliasing(all_triplets, show_progress=True):
    """
    過濾出有疊加問題的三元組（排除 RAW == DA 的案例）

    Args:
        all_triplets: 所有三元組列表
        show_progress: 是否顯示進度

    Returns:
        filtered_triplets: 有疊加問題的三元組列表
    """
    print(f"\n[SEARCH] 正在過濾有疊加問題的案例...")
    print(f"   總案例數: {len(all_triplets)}")

    filtered = []
    total_need_correction = 0
    total_pixels = 0

    for i, triplet in enumerate(all_triplets, 1):
        if show_progress and i % 100 == 0:
            print(f"   進度: {i}/{len(all_triplets)} ({i/len(all_triplets)*100:.1f}%)")

        has_aliasing, need_correction_pixels, case_total_pixels = check_triplet_has_aliasing(triplet)

        total_pixels += case_total_pixels

        if has_aliasing:
            total_need_correction += need_correction_pixels
            triplet['need_correction_pixels'] = need_correction_pixels
            triplet['total_pixels'] = case_total_pixels
            triplet['aliasing_ratio'] = need_correction_pixels / case_total_pixels if case_total_pixels > 0 else 0
            filtered.append(triplet)

    print(f"\n[OK] 過濾完成:")
    print(f"   • 原始案例數: {len(all_triplets)}")
    print(f"   • 有疊加問題: {len(filtered)} ({len(filtered)/len(all_triplets)*100:.1f}%)")
    print(f"   • 已排除 (無疊加): {len(all_triplets)-len(filtered)}")
    print(f"   • 總需修正像素: {total_need_correction:,} / {total_pixels:,} ({total_need_correction/total_pixels*100:.1f}%)" if total_pixels > 0 else "")

    return filtered


def stratified_sample_triplets(all_triplets, max_cases, random_seed=42, min_per_station=None):
    """
    分層隨機抽樣 - 確保每個雷達站都有足夠數量的案例

    策略：
    1. 按雷達站分組
    2. 確保每個雷達站至少有 min_per_station 個案例
    3. 剩餘配額按雷達站案例數比例分配

    Args:
        all_triplets: 所有三元組列表
        max_cases: 最大案例數
        random_seed: 隨機種子（可重現）
        min_per_station: 每個雷達站最少案例數（預設: max_cases // n_stations）

    Returns:
        sampled_triplets: 抽樣後的三元組列表
    """
    if len(all_triplets) <= max_cases:
        print(f"[INFO] 總案例數 {len(all_triplets)} ≤ {max_cases}，使用全部資料")
        return all_triplets

    random.seed(random_seed)

    # 1. 按雷達站統計分布
    df = pd.DataFrame(all_triplets)
    stations = sorted(df['station'].unique())
    n_stations = len(stations)

    print(f"\n[INFO] 資料分布:")
    print(f"  • 雷達站數: {n_stations}")
    print(f"  • 總案例數: {len(all_triplets)}")

    # 2. 計算每個雷達站的配額
    if min_per_station is None:
        min_per_station = max(1, max_cases // n_stations)

    min_total = n_stations * min_per_station

    if max_cases < min_total:
        print(f"[WARNING]  警告: max_cases ({max_cases}) < 最小需求 ({min_total})")
        print(f"  調整為確保每站至少 {min_per_station} 個案例")
        max_cases = min_total

    print(f"  • 配額策略: 每站至少 {min_per_station} 個案例")

    # 3. 按雷達站分層抽樣
    sampled = []
    station_stats = []

    for station in stations:
        # 獲取此雷達站的所有案例
        station_df = df[df['station'] == station]
        available = len(station_df)

        if available == 0:
            continue

        # 基礎配額
        n_sample = min(min_per_station, available)

        # 抽樣
        sampled_indices = random.sample(list(station_df.index), n_sample)
        sampled_group = [all_triplets[i] for i in sampled_indices]

        sampled.extend(sampled_group)

        station_stats.append({
            'station': station,
            'available': available,
            'sampled': n_sample
        })

    # 4. 如果有剩餘配額，按比例追加
    current_total = len(sampled)
    if current_total < max_cases:
        remainder = max_cases - current_total
        print(f"  • 剩餘配額: {remainder} 個案例（按雷達站比例分配）")

        # 計算每站可追加的案例數（按可用案例數比例）
        total_available = sum(stat['available'] - stat['sampled'] for stat in station_stats)

        for stat in station_stats:
            station = stat['station']
            available_more = stat['available'] - stat['sampled']

            if available_more > 0 and total_available > 0:
                # 按比例分配剩餘配額
                additional = int(remainder * (available_more / total_available))
                additional = min(additional, available_more)

                if additional > 0:
                    station_df = df[df['station'] == station]
                    already_sampled_indices = [i for i, t in enumerate(all_triplets) if t in sampled and t['station'] == station]
                    remaining_indices = [i for i in station_df.index if i not in already_sampled_indices]

                    if remaining_indices:
                        n_add = min(additional, len(remaining_indices))
                        additional_indices = random.sample(remaining_indices, n_add)
                        sampled.extend([all_triplets[i] for i in additional_indices])
                        stat['sampled'] += n_add

    # 5. 打印抽樣統計
    print(f"\n[OK] 分層抽樣完成:")
    print(f"  • 抽樣後案例數: {len(sampled)}")

    # 按雷達站統計
    sampled_df = pd.DataFrame(sampled)
    print(f"\n  [SITE] 按雷達站分布:")
    print(f"  {'雷達站':<10} {'可用案例':<12} {'抽樣數量':<12} {'抽樣率':<12}")
    print(f"  {'-'*50}")
    for stat in sorted(station_stats, key=lambda x: x['sampled'], reverse=True):
        station = stat['station']
        available = stat['available']
        sampled_count = len(sampled_df[sampled_df['station'] == station])
        rate = sampled_count / available * 100 if available > 0 else 0
        print(f"  {station:<10} {available:<12,} {sampled_count:<12} {rate:<12.2f}%")

    return sampled


def auto_zero_pad(data_2d, layers=4, fill_value=0):
    """
    自動padding以符合U-Net要求（維度必須能被2^layers整除）

    Args:
        data_2d: 2D輸入數據
        layers: U-Net層數（預設4層，需要能被16整除）
        fill_value: padding值

    Returns:
        padded: padding後的數據
        pad_h: 垂直方向padding量
        pad_w: 水平方向padding量
    """
    factor = 2 ** layers
    H, W = data_2d.shape
    pad_h = (factor - (H % factor)) % factor  # 修正：使用 % factor 避免已經整除時多padding
    pad_w = (factor - (W % factor)) % factor
    padded = np.pad(data_2d, ((0, pad_h), (0, pad_w)),
                    mode='constant', constant_values=fill_value)
    return padded, pad_h, pad_w


def run_model_inference(model, raw_csv, use_physics_model, enable_crf=False,
                        crf_unary_weight=1.0, crf_pairwise_weight=1.5,
                        crf_threshold=1.5, crf_max_iterations=10, crf_confidence_threshold=None,
                        temperature=1.0, fold0_bias=0.0):
    """
    運行模型推理（模型權重已在外部載入）

    Args:
        model: 已載入權重的模型
        raw_csv: RAW CSV路徑（臨時轉換的）
        use_physics_model: 是否使用物理模型
        enable_crf: 是否啟用CRF空間平滑後處理
        crf_unary_weight: CRF一元勢能權重
        crf_pairwise_weight: CRF二元勢能權重
        crf_threshold: CRF fold差異閾值
        crf_max_iterations: CRF最大迭代次數
        temperature: Temperature Scaling 溫度參數（>1.0 降低信心度）
        fold0_bias: Fold=0 的 logit 偏移量（負值降低 fold=0 機率）

    Returns:
        dealiased_cls_optimized: 軟分類訓練優化後的速度矩陣
        dealiased_cls: 傳統硬分類速度矩陣（可能經過CRF平滑）
        crf_info: CRF 修改信息字典（如果啟用CRF）或 None
    """
    # 讀取CSV
    df = pd.read_csv(raw_csv)
    nyquist = df['Nyquist'].iloc[0]

    # 轉換為矩陣
    azimuths = sorted(df['Azimuth'].unique())
    ranges = sorted(df['Range'].unique())
    n_az, n_range = len(azimuths), len(ranges)

    raw_matrix = np.full((n_az, n_range), np.nan)
    for _, row in df.iterrows():
        az_idx = azimuths.index(row['Azimuth'])
        r_idx = ranges.index(row['Range'])
        if row['Value'] != -999.0:
            raw_matrix[az_idx, r_idx] = row['Value']

    # Padding（確保維度能被16整除）- 注意：不要歸一化！模型訓練時用的是原始速度值
    raw_padded, pad_h, pad_w = auto_zero_pad(raw_matrix, layers=4, fill_value=0)
    print(f"  [SHAPE] 原始尺寸: {raw_matrix.shape}, Padding後: {raw_padded.shape}")

    # 推理（模型權重已經載入）
    if use_physics_model:
        # 物理模型需要字典輸入 {'vel': (1,1,H,W,1), 'nyq': (1,1)}
        vel_5d = raw_padded[None, None, :, :, None].astype(np.float32)  # (1, 1, H_pad, W_pad, 1) - 原始速度值
        nyq_2d = np.array([[nyquist]], dtype=np.float32)  # (1, 1)
        predictions = model({'vel': vel_5d, 'nyq': nyq_2d}, training=False)
        # 物理模型返回字典：{'alias_mask': cls_logits, 'dealiased_vel': optimized_vel, ...}
        cls_logits = predictions['alias_mask']  # (1, H_pad, W_pad, 6)
        optimized_vel_padded = predictions['dealiased_vel'][0, :, :, 0]  # (H_pad, W_pad) - 軟分類訓練優化（已經是原始速度）
    else:
        # 標準模型使用普通 tensor 輸入
        raw_input = np.expand_dims(raw_padded, axis=(0, -1))  # (1, H_pad, W_pad, 1)
        predictions = model.predict(raw_input, verbose=0)
        cls_logits = predictions  # (1, H_pad, W_pad, 6)
        optimized_vel_padded = None  # 標準模型沒有優化分支

    # [TEMP] Temperature Scaling（校準信心度）
    if temperature != 1.0:
        print(f"  [TEMP]  應用 Temperature Scaling (T={temperature:.2f})")
        cls_logits = cls_logits / temperature

    # [FOLD0] Fold=0 Bias（調整 fold=0 傾向）
    # 類別對應：index 3 是 fold=0（cls_logits 的 6 個類別：[mask?, -2, -1, 0, +1, +2]）
    if fold0_bias != 0.0:
        print(f"  [FOLD0] 應用 Fold0 Bias = {fold0_bias:.2f}（負值降低 fold=0 機率）")
        cls_logits = cls_logits.numpy() if hasattr(cls_logits, 'numpy') else cls_logits
        cls_logits[:, :, :, 3] = cls_logits[:, :, :, 3] + fold0_bias

    # 1. 硬分類：argmax 或 CRF
    if enable_crf:
        print(f"  [FIX] 啟用 CRF 後處理 (unary={crf_unary_weight}, pairwise={crf_pairwise_weight}, threshold={crf_threshold})")
        # CRF 後處理：使用空間平滑優化
        # 提取 fold 機率（類別 1-5 對應 [-2, -1, 0, +1, +2]）
        fold_probs_padded = tf.nn.softmax(cls_logits[0, :, :, 1:6], axis=-1).numpy()  # (H_pad, W_pad, 5)

        # 先計算原始 argmax 結果（用於比較）
        original_fold_idx = np.argmax(fold_probs_padded, axis=-1)

        # 應用 CRF 空間平滑
        smoothed_fold_idx = crf_spatial_smoothing(
            fold_probs_padded,
            unary_weight=crf_unary_weight,
            pairwise_weight=crf_pairwise_weight,
            threshold=crf_threshold,
            max_iterations=crf_max_iterations,
            confidence_threshold=crf_confidence_threshold
        )  # (H_pad, W_pad), 值為 0-4 對應 [-2, -1, 0, +1, +2]

        # Unpad 回原始尺寸（先 unpad，再統計有效區域的修改）
        if pad_h > 0:
            original_fold_idx = original_fold_idx[:-pad_h, :]
            smoothed_fold_idx = smoothed_fold_idx[:-pad_h, :]
        if pad_w > 0:
            original_fold_idx = original_fold_idx[:, :-pad_w]
            smoothed_fold_idx = smoothed_fold_idx[:, :-pad_w]

        # 統計 CRF 在有效區域修改了多少像素
        num_changed = np.sum(original_fold_idx != smoothed_fold_idx)
        total_pixels = original_fold_idx.size
        change_rate = num_changed / total_pixels * 100

        # 計算修改後的 fold 值差異
        fold_values = np.array([-2, -1, 0, 1, 2])
        original_fold_values = fold_values[original_fold_idx]
        smoothed_fold_values = fold_values[smoothed_fold_idx]
        fold_diff = smoothed_fold_values - original_fold_values

        print(f"     [+] CRF 修改了 {num_changed}/{total_pixels} 個像素 ({change_rate:.2f}%)")
        if num_changed > 0:
            print(f"       - 修改範圍: {fold_diff.min():.0f} 到 {fold_diff.max():.0f} fold")
            print(f"       - 平均修改: {np.mean(np.abs(fold_diff[fold_diff != 0])):.2f} fold")
            # 統計修改方向
            increased = np.sum(fold_diff > 0)
            decreased = np.sum(fold_diff < 0)
            print(f"       - 增加 fold: {increased} 像素, 減少 fold: {decreased} 像素")

            # 統計修改位置是否在有效數據區域
            valid_data_mask = ~np.isnan(raw_matrix)
            changes_in_valid = np.sum((fold_diff != 0) & valid_data_mask)
            print(f"       - 在有效數據區域的修改: {changes_in_valid}/{num_changed} 個")

            # 儲存 CRF 修改信息供後續分析
            crf_change_mask = (fold_diff != 0) & valid_data_mask
            crf_fold_diff = fold_diff.copy()

        # 轉換為 fold 值
        fold_matrix = smoothed_fold_values  # (H, W)

    else:
        print(f"  [INFO] 使用傳統 argmax（無後處理）")
        # 傳統硬分類：argmax
        cls_predictions = np.argmax(cls_logits[0], axis=-1)  # (H_pad, W_pad)

        # Unpad回原始尺寸
        if pad_h > 0:
            cls_predictions = cls_predictions[:-pad_h, :]
        if pad_w > 0:
            cls_predictions = cls_predictions[:, :-pad_w]

        # 轉換為fold（-2, -1, 0, +1, +2）
        fold_map = {0: 0, 1: -2, 2: -1, 3: 0, 4: +1, 5: +2}
        fold_matrix = np.vectorize(fold_map.get)(cls_predictions)

    # Unpad optimized_vel_padded (如果有的話)
    if optimized_vel_padded is not None:
        if pad_h > 0:
            optimized_vel_padded = optimized_vel_padded[:-pad_h, :]
        if pad_w > 0:
            optimized_vel_padded = optimized_vel_padded[:, :-pad_w]

    # 計算傳統硬分類的解混疊速度
    correction = fold_matrix * (2 * nyquist)
    dealiased_cls = raw_matrix + correction

    # 調試：對比 CRF 前後的結果
    if enable_crf and num_changed > 0:
        # 計算如果不用 CRF 會是什麼結果
        original_correction = original_fold_values * (2 * nyquist)
        dealiased_no_crf = raw_matrix + original_correction
        velocity_diff = dealiased_cls - dealiased_no_crf

        # 只統計有效像素（非 nan）
        valid_mask = ~np.isnan(velocity_diff)
        if np.any(valid_mask):
            max_velocity_change = np.max(np.abs(velocity_diff[valid_mask]))
            mean_velocity_change = np.mean(np.abs(velocity_diff[valid_mask]))
            num_velocity_changed = np.sum(np.abs(velocity_diff[valid_mask]) > 0.01)
            print(f"       - 最大速度變化: {max_velocity_change:.2f} m/s")
            print(f"       - 平均速度變化: {mean_velocity_change:.2f} m/s")
            print(f"       - 速度變化像素: {num_velocity_changed} 個 (有效像素中)")
        else:
            print(f"       - [WARNING] 所有像素都是 nan，無法計算速度變化")

    # 2. 軟分類訓練優化後的速度（如果有的話）
    if optimized_vel_padded is not None:
        # 模型輸出已經是原始速度值（m/s），不需要反歸一化
        dealiased_cls_optimized = optimized_vel_padded
    else:
        # 如果沒有優化分支，兩者相同
        dealiased_cls_optimized = dealiased_cls.copy()

    return dealiased_cls_optimized, dealiased_cls


def compute_metrics_triplet(raw_matrix, da_matrix, cnn_matrix, qc_matrix, nyquist):
    """
    計算 RAW vs DA vs CNN 的比較指標

    注意：QC (bref_qc) 是 reflectivity 不是 velocity，所以不使用
    比較三種速度場：RAW（原始）, DA（NWP物理算法）, CNN（深度學習模型）
    """
    metrics = {}

    # 有效像素mask
    valid_mask = ~np.isnan(da_matrix) & ~np.isnan(cnn_matrix) & ~np.isnan(raw_matrix)

    # 1. RAW 的統計資訊
    raw_valid = raw_matrix[valid_mask]
    metrics['raw'] = {
        'mean': float(np.mean(raw_valid)),
        'std': float(np.std(raw_valid)),
        'min': float(np.min(raw_valid)),
        'max': float(np.max(raw_valid)),
        'total_pixels': int(np.sum(valid_mask))
    }

    # 2. DA 的統計資訊（相對於 RAW）
    da_diff = da_matrix[valid_mask] - raw_matrix[valid_mask]
    metrics['da'] = {
        'mean': float(np.mean(da_matrix[valid_mask])),
        'std': float(np.std(da_matrix[valid_mask])),
        'mean_correction': float(np.mean(da_diff)),
        'std_correction': float(np.std(da_diff)),
        'max_correction': float(np.max(np.abs(da_diff))),
        'corrected_pixels': int(np.sum(np.abs(da_diff) > 0.1))  # 有明顯校正的像素
    }

    # 3. CNN 的統計資訊（相對於 RAW）
    cnn_diff = cnn_matrix[valid_mask] - raw_matrix[valid_mask]
    metrics['cnn'] = {
        'mean': float(np.mean(cnn_matrix[valid_mask])),
        'std': float(np.std(cnn_matrix[valid_mask])),
        'mean_correction': float(np.mean(cnn_diff)),
        'std_correction': float(np.std(cnn_diff)),
        'max_correction': float(np.max(np.abs(cnn_diff))),
        'corrected_pixels': int(np.sum(np.abs(cnn_diff) > 0.1))
    }

    # 4. RAW vs DA 的比較
    raw_da_diff = raw_matrix[valid_mask] - da_matrix[valid_mask]
    metrics['raw_vs_da'] = {
        'rmse': float(np.sqrt(np.mean(raw_da_diff**2))),
        'mae': float(np.mean(np.abs(raw_da_diff))),
        'max_diff': float(np.max(np.abs(raw_da_diff))),
        'correlation': float(np.corrcoef(raw_matrix[valid_mask], da_matrix[valid_mask])[0, 1])
    }

    # 5. RAW vs CNN 的比較
    raw_cnn_diff = raw_matrix[valid_mask] - cnn_matrix[valid_mask]
    metrics['raw_vs_cnn'] = {
        'rmse': float(np.sqrt(np.mean(raw_cnn_diff**2))),
        'mae': float(np.mean(np.abs(raw_cnn_diff))),
        'max_diff': float(np.max(np.abs(raw_cnn_diff))),
        'correlation': float(np.corrcoef(raw_matrix[valid_mask], cnn_matrix[valid_mask])[0, 1])
    }

    # 6. DA vs CNN 的直接比較
    da_cnn_diff = da_matrix[valid_mask] - cnn_matrix[valid_mask]
    metrics['da_vs_cnn'] = {
        'rmse': float(np.sqrt(np.mean(da_cnn_diff**2))),
        'mae': float(np.mean(np.abs(da_cnn_diff))),
        'max_diff': float(np.max(np.abs(da_cnn_diff))),
        'correlation': float(np.corrcoef(da_matrix[valid_mask], cnn_matrix[valid_mask])[0, 1]),
        'agreement_rate': float(np.mean(np.abs(da_cnn_diff) < 2.0))  # 差異<2 m/s視為一致
    }

    # 7. Fold-based 比較
    raw_fold = np.round(raw_matrix[valid_mask] / (2 * nyquist))
    da_fold = np.round((da_matrix[valid_mask] - raw_matrix[valid_mask]) / (2 * nyquist))
    cnn_fold = np.round((cnn_matrix[valid_mask] - raw_matrix[valid_mask]) / (2 * nyquist))

    metrics['fold_comparison'] = {
        'da_cnn_fold_agreement': float(np.mean(da_fold == cnn_fold)),
        'da_non_zero_folds': int(np.sum(da_fold != 0)),
        'cnn_non_zero_folds': int(np.sum(cnn_fold != 0)),
        'fold_agreement_rate': float(np.mean(da_fold == cnn_fold))  # 與 DA 一致率（作為參考標準）
    }

    # 8. 以 DA 為參考的修正成功率（因為沒有真實 GT，使用 DA 作為 pseudo-GT）
    # 使用與 batch_test_system 相同的基於 fold 的嚴格成功率計算

    # 計算需要修正的像素（RAW 與 DA 不一致，即 fold 不同）
    need_correction_mask = (raw_fold != da_fold)
    need_correction_pixels = int(np.sum(need_correction_mask))

    # 計算 CNN 成功修正的像素（CNN 的 fold 與 DA 一致）
    cnn_corrected_mask = need_correction_mask & (cnn_fold == da_fold)
    cnn_corrected_pixels = int(np.sum(cnn_corrected_mask))

    # 計算修正成功率
    if need_correction_pixels > 0:
        cnn_correction_success_rate = float(cnn_corrected_pixels / need_correction_pixels)
    else:
        cnn_correction_success_rate = 1.0  # 沒有需要修正的像素

    # CNN 相對於 DA 的性能指標
    metrics['cnn_performance'] = {
        'need_correction_pixels': need_correction_pixels,
        'corrected_pixels': cnn_corrected_pixels,
        'correction_success_rate': cnn_correction_success_rate,
        'fold_agreement': float(np.mean(da_fold == cnn_fold)),  # 整體一致率
        'relative_to_da': {
            'rmse_ratio': float(metrics['raw_vs_cnn']['rmse'] / metrics['raw_vs_da']['rmse']) if metrics['raw_vs_da']['rmse'] > 0 else 1.0,
            'mae_ratio': float(metrics['raw_vs_cnn']['mae'] / metrics['raw_vs_da']['mae']) if metrics['raw_vs_da']['mae'] > 0 else 1.0
        }
    }

    # 9. 校正行為分析
    metrics['correction_analysis'] = {
        # DA 和 CNN 都校正的像素（一致行為）
        'both_corrected': int(np.sum((np.abs(da_diff) > 0.1) & (np.abs(cnn_diff) > 0.1))),
        # 只有 DA 校正
        'only_da_corrected': int(np.sum((np.abs(da_diff) > 0.1) & (np.abs(cnn_diff) <= 0.1))),
        # 只有 CNN 校正
        'only_cnn_corrected': int(np.sum((np.abs(da_diff) <= 0.1) & (np.abs(cnn_diff) > 0.1))),
        # 兩者都沒校正
        'neither_corrected': int(np.sum((np.abs(da_diff) <= 0.1) & (np.abs(cnn_diff) <= 0.1)))
    }

    return metrics


def save_nwp_visualizations(raw_vel, da_vel, cnn_vel, qc_vel, filename, result_dir, nyquist=None, triplet=None):
    """
    保存NWP測試的可視化圖像（RAW vs DA vs CNN vs QC）

    Args:
        raw_vel: RAW速度場
        da_vel: DA算法結果
        cnn_vel: CNN模型結果
        qc_vel: QC數據（Ground Truth）
        filename: 檔案名稱
        result_dir: 結果目錄
        nyquist: Nyquist速度
        triplet: 三元組資料（包含檔案路徑，用於提取目錄名稱）
    """
    os.makedirs(result_dir, exist_ok=True)
    basename = Path(filename).stem

    # 從triplet提取目錄名稱（NWP forecast initialization time）
    dir_name = ""
    if triplet and 'raw' in triplet:
        raw_path = Path(triplet['raw'])
        dir_name = raw_path.parent.name + "_"  # 例如 "24103105_"

    # 更新basename包含目錄名稱
    basename_with_dir = f"{dir_name}{basename}"

    # 設置colormap和normalization（不使用 QC）
    cmap = 'RdYlBu_r'
    vmin = np.nanmin([raw_vel, da_vel, cnn_vel])
    vmax = np.nanmax([raw_vel, da_vel, cnn_vel])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # 1. 創建 2x3 比較圖
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'NWP Data Comparison (DA vs CNN): {basename_with_dir}', fontsize=16)

    # 第一行：RAW, DA, CNN
    im1 = axes[0,0].imshow(raw_vel, cmap=cmap, norm=norm, aspect='auto')
    axes[0,0].set_title('RAW Velocity (Aliased)')
    axes[0,0].set_xlabel('Range'); axes[0,0].set_ylabel('Azimuth')
    plt.colorbar(im1, ax=axes[0,0], shrink=0.8)

    im2 = axes[0,1].imshow(da_vel, cmap=cmap, norm=norm, aspect='auto')
    axes[0,1].set_title('DA Algorithm (NWP Physical)')
    axes[0,1].set_xlabel('Range'); axes[0,1].set_ylabel('Azimuth')
    plt.colorbar(im2, ax=axes[0,1], shrink=0.8)

    im3 = axes[0,2].imshow(cnn_vel, cmap=cmap, norm=norm, aspect='auto')
    axes[0,2].set_title('CNN Model (Optimized Classification)')
    axes[0,2].set_xlabel('Range'); axes[0,2].set_ylabel('Azimuth')
    plt.colorbar(im3, ax=axes[0,2], shrink=0.8)

    # 第二行：RAW vs DA校正, RAW vs CNN校正, DA vs CNN差異
    # RAW → DA 的校正
    da_correction = da_vel - raw_vel
    im4 = axes[1,0].imshow(da_correction, cmap='RdBu_r', vmin=-40, vmax=40, aspect='auto')
    axes[1,0].set_title('DA Correction (DA - RAW)')
    axes[1,0].set_xlabel('Range'); axes[1,0].set_ylabel('Azimuth')
    plt.colorbar(im4, ax=axes[1,0], shrink=0.8)

    # RAW → CNN 的校正
    cnn_correction = cnn_vel - raw_vel
    im5 = axes[1,1].imshow(cnn_correction, cmap='RdBu_r', vmin=-40, vmax=40, aspect='auto')
    axes[1,1].set_title('CNN Correction (CNN - RAW)')
    axes[1,1].set_xlabel('Range'); axes[1,1].set_ylabel('Azimuth')
    plt.colorbar(im5, ax=axes[1,1], shrink=0.8)

    # DA vs CNN 差異
    da_cnn_diff = da_vel - cnn_vel
    im6 = axes[1,2].imshow(da_cnn_diff, cmap='RdBu_r', vmin=-20, vmax=20, aspect='auto')
    axes[1,2].set_title('DA - CNN Difference')
    axes[1,2].set_xlabel('Range'); axes[1,2].set_ylabel('Azimuth')
    plt.colorbar(im6, ax=axes[1,2], shrink=0.8)

    plt.tight_layout()

    # 保存比較圖
    comparison_path = os.path.join(result_dir, f"{basename_with_dir}_nwp_comparison.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [INFO] 比較圖已保存: {comparison_path}")

    # 2. 創建 fold 判斷可視化（如果提供了 Nyquist）
    if nyquist is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Fold Number Analysis: {basename_with_dir}', fontsize=16)

        # 計算fold number
        raw_fold = np.round(raw_vel / (2 * nyquist))
        da_fold = np.round((da_vel - raw_vel) / (2 * nyquist))
        cnn_fold = np.round((cnn_vel - raw_vel) / (2 * nyquist))
        qc_fold = np.round((qc_vel - raw_vel) / (2 * nyquist))

        # DA fold判斷
        da_fold_correct = (da_fold == qc_fold).astype(float)
        da_fold_correct[np.isnan(qc_vel)] = np.nan
        im1 = axes[0].imshow(da_fold_correct, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        axes[0].set_title(f'DA Fold Correctness\n(Accuracy: {np.nanmean(da_fold_correct):.2%})')
        axes[0].set_xlabel('Range'); axes[0].set_ylabel('Azimuth')
        plt.colorbar(im1, ax=axes[0], shrink=0.8)

        # CNN fold判斷
        cnn_fold_correct = (cnn_fold == qc_fold).astype(float)
        cnn_fold_correct[np.isnan(qc_vel)] = np.nan
        im2 = axes[1].imshow(cnn_fold_correct, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        axes[1].set_title(f'CNN Fold Correctness\n(Accuracy: {np.nanmean(cnn_fold_correct):.2%})')
        axes[1].set_xlabel('Range'); axes[1].set_ylabel('Azimuth')
        plt.colorbar(im2, ax=axes[1], shrink=0.8)

        # 比較：CNN是否優於DA
        improvement = cnn_fold_correct - da_fold_correct
        im3 = axes[2].imshow(improvement, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
        axes[2].set_title('CNN vs DA Improvement\n(Green = CNN better, Red = DA better)')
        axes[2].set_xlabel('Range'); axes[2].set_ylabel('Azimuth')
        plt.colorbar(im3, ax=axes[2], shrink=0.8)

        plt.tight_layout()

        fold_path = os.path.join(result_dir, f"{basename_with_dir}_fold_analysis.png")
        plt.savefig(fold_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  [TARGET] Fold分析圖已保存: {fold_path}")


def save_single_geo_visualization(velocity_matrix, gz_path, field_name, title, filename, result_dir,
                                 shape_path=None, vmin=-80, vmax=80, dir_prefix="",
                                 cmap_name='velocity'):
    """
    保存單個速度場的地理位置可視化圖像

    與 batch_test_system.py 和 mixed_patch_test_model.py 的方法保持一致

    Args:
        velocity_matrix: 速度場矩陣
        gz_path: 對應的.gz雷達檔案路徑
        field_name: 場名稱（用於內部識別）
        title: 圖表標題
        filename: 檔案名稱
        result_dir: 結果目錄
        shape_path: 台灣地圖shapefile路徑（可選）
        vmin: 色階最小值（預設-80 m/s，與原始方法一致）
        vmax: 色階最大值（預設+80 m/s，與原始方法一致）
        dir_prefix: 目錄名稱前綴（例如 "24103105_"）

    Returns:
        output_path: 輸出圖片路徑，失敗則返回None
    """
    if not GEO_VIZ_AVAILABLE:
        return None

    try:
        # 讀取雷達資訊
        radar_data, radius_km, nyq = read_cwb_radar_sweep(gz_path)

        # 處理 720-ray radar + 360-ray velocity 形狀錯位
        # velocity 已 azimuth-aware 對齊到 GT 360 標準（0°起始 1° 間隔），
        # 因此 radar 物件的 azimuth 也要重新生成為 GT 標準（不能用 [::2]，
        # 那樣會留下 raw 掃描起始角度，與 vel 不一致）。
        n_vel = velocity_matrix.shape[0]
        n_radar = radar_data.azimuth['data'].shape[0]
        if n_radar == 2 * n_vel:
            import numpy as _np
            new_azm_sp = 360.0 / n_vel
            radar_data.azimuth['data']   = _np.arange(n_vel, dtype='float32') * new_azm_sp
            radar_data.elevation['data'] = radar_data.elevation['data'][:n_vel].copy()
            radar_data.time['data']      = radar_data.time['data'][:n_vel].copy()
            radar_data.sweep_end_ray_index['data'] = _np.array([n_vel - 1], dtype='int32')
            try:
                radar_data.nrays = n_vel
            except AttributeError:
                pass
            for k in list(radar_data.fields.keys()):
                f = radar_data.fields[k]
                if hasattr(f.get('data', None), 'shape') and f['data'].shape[0] == n_radar:
                    f['data'] = f['data'][:n_vel].copy()
        elif n_radar != n_vel:
            print(f"     [WARNING] {title}: shape mismatch radar={n_radar} vs vel={n_vel}")

        lat = radar_data.latitude['data'][0]
        lon = radar_data.longitude['data'][0]

        # 將場資料添加到 radar 物件（velocity 或 reflectivity）
        vel_field = get_metadata('velocity')
        vel_field['data'] = velocity_matrix
        radar_data.fields[field_name] = vel_field

        # 色階選擇：velocity (CWA 標準藍紅) 或 reflectivity (NWS 標準離散 dBZ 13 帶)
        # 反射率使用跟 plot_reflectivity_geo.py 一致的色階，每 5 dBZ 一帶
        if cmap_name == 'reflectivity':
            _dbz_bounds = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
            _dbz_colors = [
                '#00FFFF',  # 0-5    cyan
                '#00AAFF',  # 5-10   light blue
                '#0000FF',  # 10-15  blue
                '#00CC44',  # 15-20  green
                '#00FF00',  # 20-25  bright green
                '#AAFF00',  # 25-30  yellow-green
                '#FFFF00',  # 30-35  yellow
                '#FFCC00',  # 35-40  yellow-orange
                '#FF8800',  # 40-45  orange
                '#FF4400',  # 45-50  orange-red
                '#FF0000',  # 50-55  red
                '#FF00CC',  # 55-60  pink
                '#FF00FF',  # 60-65  magenta
            ]
            use_cmap = mcolors.ListedColormap(_dbz_colors, name='cwb_dbz')
            use_cmap.set_under((1,1,1,0))   # < 0   透明
            use_cmap.set_over('#FF00FF')    # > 65  封頂 magenta
            use_cmap.set_bad((1,1,1,0))     # NaN   白色透明
            use_norm = mcolors.BoundaryNorm(_dbz_bounds, use_cmap.N)
        else:
            use_cmap = CWA_VEL_CMAP
            use_norm = CWA_VEL_NORM

        # 繪製地理圖
        m = plot_taiwan_basemap(lon, lat, radius_km, shape_path)
        display = RadarMapDisplayBasemap(radar_data)
        display.plot_ppi_map(
            field_name, sweep=0, resolution='h', vmin=vmin, vmax=vmax,
            cmap=use_cmap, norm=use_norm,
            min_lon=lon-1, max_lon=lon+1,
            min_lat=lat-1, max_lat=lat+1,
            mask_outside=True, projection='aeqd', basemap=m
        )
        display.plot_range_rings([radius_km[-1]])
        plt.title(title)

        # ⚠️ 修正：用 str(filename) 不用 .stem，因為 case_label 含 dot（station.date.time）
        # 會把最後一段 (.time) 當 extension 砍掉 → 同站同日不同時段互相覆蓋
        basename = str(filename)
        basename_with_dir = f"{dir_prefix}{basename}"
        output_path = os.path.join(result_dir, f"{basename_with_dir}_geo_{field_name}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    except Exception as e:
        print(f"     [WARNING]  {title} 地理圖失敗: {e}")
        return None


def plot_single_velocity_field_geo(vel_matrix, gz_path, field_name, title, filename, result_dir, shape_path=None):
    """
    完全複製 mixed_patch_test_model.py 中 save_geographical_visualizations 的繪圖邏輯
    但用於單一速度場，使用其對應的 .gz 檔案

    這樣既保持 batch_test_system.py 的可視化風格，又讓每個場使用自己的 .gz 座標
    """
    if not GEO_VIZ_AVAILABLE:
        return None

    try:
        # 1) 從 .gz 檔拿雷達資訊 (完全複製原始邏輯)
        radar_data, radius_km, nyq = read_cwb_radar_sweep(gz_path)
        lat = radar_data.latitude['data'][0]
        lon = radar_data.longitude['data'][0]

        # 2) Radar 物件更新 (與原始函數相同)
        vel_field = get_metadata('velocity')
        vel_field['data'] = vel_matrix
        radar_data.fields['velocity'] = vel_field

        # 3) 畫圖（使用氣象署標準色階）
        m = plot_taiwan_basemap(lon, lat, radius_km, shape_path)
        display = RadarMapDisplayBasemap(radar_data)
        display.plot_ppi_map(
            'velocity', sweep=0, resolution='h', vmin=-80, vmax=80,
            cmap=CWA_VEL_CMAP, norm=CWA_VEL_NORM,
            min_lon=lon-1, max_lon=lon+1,
            min_lat=lat-1, max_lat=lat+1,
            mask_outside=True, projection='aeqd', basemap=m
        )
        display.plot_range_rings([radius_km[-1]])
        plt.title(title)

        # 5) 保存圖片
        basename = Path(filename).stem
        output_path = os.path.join(result_dir, f"{basename}_geo_{field_name}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    except Exception as e:
        print(f"     [WARNING]  {title} 地理圖失敗: {e}")
        return None


def save_nwp_geo_visualizations_multi(triplet, raw_vel, da_vel, cnn_vel, qc_vel, filename, result_dir, shape_path=None):
    """
    保存NWP資料的地理位置可視化圖像（使用各自對應的.gz檔案）

    Args:
        triplet: 包含raw/da/qc路徑的字典
        raw_vel: RAW速度場
        da_vel: DA算法結果
        cnn_vel: CNN模型結果
        qc_vel: QC數據（Ground Truth）
        filename: 檔案名稱
        result_dir: 結果目錄
        shape_path: 台灣地圖shapefile路徑（可選）
    """
    if not GEO_VIZ_AVAILABLE:
        print("  [WARNING]  地理可視化模組不可用，跳過地理圖")
        return

    os.makedirs(result_dir, exist_ok=True)
    basename = Path(filename).stem

    # 從triplet提取目錄名稱前綴
    dir_prefix = ""
    if triplet and 'raw' in triplet:
        raw_path = Path(triplet['raw'])
        dir_prefix = raw_path.parent.name + "_"  # 例如 "24103105_"

    try:
        print("  [GEO] 繪製地理可視化圖（各自使用對應的.gz檔案）...")

        # 找到每個速度場對應的.gz檔案
        raw_gz = find_corresponding_gz_file(triplet['raw'])
        da_gz = find_corresponding_gz_file(triplet['da'])
        qc_gz = find_corresponding_gz_file(triplet['qc'])

        # 1. 繪製RAW地理圖
        if raw_gz:
            print("     繪製 RAW Velocity...")
            path = save_single_geo_visualization(
                raw_vel, raw_gz, 'raw_velocity',
                f'RAW Velocity - {basename}',
                filename, result_dir, shape_path, dir_prefix=dir_prefix
            )
            if path:
                print(f"     [OK] {path}")
        else:
            print("     [WARNING]  RAW .gz檔案不存在，跳過")

        # 2. 繪製DA地理圖
        if da_gz:
            print("     繪製 DA Velocity...")
            path = save_single_geo_visualization(
                da_vel, da_gz, 'da_velocity',
                f'DA Algorithm - {basename}',
                filename, result_dir, shape_path, dir_prefix=dir_prefix
            )
            if path:
                print(f"     [OK] {path}")
        else:
            print("     [WARNING]  DA .gz檔案不存在，跳過")

        # 3. 繪製CNN地理圖（使用RAW的.gz檔案，因為CNN是從RAW生成的）
        if raw_gz:
            print("     繪製 CNN Velocity...")
            path = save_single_geo_visualization(
                cnn_vel, raw_gz, 'cnn_velocity',
                f'CNN Model - {basename}',
                filename, result_dir, shape_path, dir_prefix=dir_prefix
            )
            if path:
                print(f"     [OK] {path}")

        # 4. 繪製QC地理圖
        if qc_gz:
            print("     繪製 QC Ground Truth...")
            path = save_single_geo_visualization(
                qc_vel, qc_gz, 'qc_velocity',
                f'QC Ground Truth - {basename}',
                filename, result_dir, shape_path, dir_prefix=dir_prefix
            )
            if path:
                print(f"     [OK] {path}")
        else:
            print("     [WARNING]  QC .gz檔案不存在，跳過")

        print(f"  [GEO] 地理圖已全部保存")

    except Exception as e:
        print(f"  [WARNING]  地理可視化失敗: {e}")
        import traceback
        traceback.print_exc()


def align_matrices(raw_matrix, da_matrix, qc_matrix):
    """
    對齊三個矩陣的尺寸（以最小尺寸為準）

    Args:
        raw_matrix, da_matrix, qc_matrix: 輸入矩陣

    Returns:
        aligned_raw, aligned_da, aligned_qc: 對齊後的矩陣
    """
    # 獲取每個矩陣的尺寸
    shapes = [raw_matrix.shape, da_matrix.shape, qc_matrix.shape]

    # 找到最小的共同尺寸
    min_h = min(s[0] for s in shapes)
    min_w = min(s[1] for s in shapes)

    # 如果尺寸不一致，打印警告
    if not all(s == (min_h, min_w) for s in shapes):
        print(f"  [WARNING]  尺寸不一致: RAW{raw_matrix.shape}, DA{da_matrix.shape}, QC{qc_matrix.shape}")
        print(f"  [SIZE] 對齊到: ({min_h}, {min_w})")

    # 裁剪到共同尺寸
    aligned_raw = raw_matrix[:min_h, :min_w]
    aligned_da = da_matrix[:min_h, :min_w]
    aligned_qc = qc_matrix[:min_h, :min_w]

    return aligned_raw, aligned_da, aligned_qc


def run_single_nwp_test(triplet, model, use_physics_model, temp_dir, enable_viz=True, viz_dir=None, shape_path=None,
                        enable_crf=False, crf_unary_weight=1.0, crf_pairwise_weight=50.0,
                        crf_threshold=0.5, crf_max_iterations=10, crf_confidence_threshold=None,
                        temperature=1.0, fold0_bias=0.0):
    """
    運行單個NWP測試案例

    Args:
        triplet: {'raw': path, 'da': path, 'qc': path, ...}
        model: 已載入權重的模型
        use_physics_model: 是否使用物理模型
        temp_dir: 臨時目錄（用於存放轉換的CSV）
        enable_viz: 是否啟用可視化
        viz_dir: 可視化輸出目錄
        enable_crf: 是否啟用CRF空間平滑後處理
        crf_unary_weight: CRF一元勢能權重
        crf_pairwise_weight: CRF二元勢能權重
        crf_threshold: CRF fold差異閾值
        crf_max_iterations: CRF最大迭代次數

    Returns:
        (results, error)
    """
    try:
        filename = triplet['filename']
        station = triplet['station']

        # 提取完整路徑（目錄 + 文件名）用於結果儲存
        raw_path = Path(triplet['raw'])
        directory = raw_path.parent.name  # 例如: 24103100
        full_filename = f"{directory}/{filename}"  # 例如: 24103100/RCCG.20241031.1640.bvel_raw.01.parquet

        print(f"[TEST] 測試 {full_filename} ({station})")

        # 1. 載入三種速度場
        raw_matrix, nyquist, _ = load_parquet_as_matrix(triplet['raw'], debug=False)
        da_matrix, _, _ = load_parquet_as_matrix(triplet['da'], debug=False)

        # NWP 沒有 QC，如果 qc 路徑為空則創建 NaN 矩陣
        if triplet['qc'] and os.path.exists(triplet['qc']):
            qc_matrix, _, _ = load_parquet_as_matrix(triplet['qc'], debug=False)
        else:
            qc_matrix = np.full_like(raw_matrix, np.nan)

        # Debug: 檢查 RAW 和 DA 是否相同
        valid_check = ~np.isnan(raw_matrix) & ~np.isnan(da_matrix)
        if np.any(valid_check):
            diff_count = np.sum(np.abs(raw_matrix[valid_check] - da_matrix[valid_check]) > 0.01)
            diff_ratio = diff_count / np.sum(valid_check)
            print(f"  [SEARCH] RAW vs DA 差異: {diff_count} / {np.sum(valid_check)} ({diff_ratio*100:.1f}%)")

            if diff_count == 0:
                print(f"  [WARNING]  跳過: RAW 和 DA 完全相同（此案例無折錯或DA未執行）")
                return None, "RAW and DA are identical - no aliasing to correct"
            elif diff_ratio < 0.01:  # 差異小於1%
                print(f"  [WARNING]  跳過: RAW 和 DA 幾乎相同（差異率 {diff_ratio*100:.2f}%）")
                return None, f"RAW and DA too similar - diff ratio {diff_ratio*100:.2f}%"
            else:
                # 顯示一些差異範例
                diff_mask = np.abs(raw_matrix - da_matrix) > 0.01
                diff_indices = np.where(diff_mask)
                if len(diff_indices[0]) > 0:
                    sample_idx = min(3, len(diff_indices[0]))
                    print(f"  [LIST] 差異範例 (前{sample_idx}個):")
                    for i in range(sample_idx):
                        r, c = diff_indices[0][i], diff_indices[1][i]
                        print(f"      位置({r},{c}): RAW={raw_matrix[r,c]:.2f}, DA={da_matrix[r,c]:.2f}, Δ={da_matrix[r,c]-raw_matrix[r,c]:.2f}")
        else:
            print(f"  [WARNING]  跳過: 沒有有效像素可比較")
            return None, "No valid pixels to compare"

        # 2. 運行CNN模型（在對齊之前，使用原始尺寸）
        # 先將原始RAW轉換為臨時CSV（模型需要CSV格式）
        os.makedirs(temp_dir, exist_ok=True)
        temp_csv = os.path.join(temp_dir, f"{Path(filename).stem}.csv")
        save_matrix_as_csv(raw_matrix, nyquist, triplet['raw'], temp_csv)

        # 運行模型推理（使用原始尺寸的RAW），返回兩個分支
        cnn_optimized, cnn_traditional = run_model_inference(
            model, temp_csv, use_physics_model,
            enable_crf=enable_crf,
            crf_unary_weight=crf_unary_weight,
            crf_pairwise_weight=crf_pairwise_weight,
            crf_threshold=crf_threshold,
            crf_max_iterations=crf_max_iterations,
            crf_confidence_threshold=crf_confidence_threshold,
            temperature=temperature,
            fold0_bias=fold0_bias
        )

        # 3. 保存原始尺寸的所有矩陣（用於地理可視化）
        raw_matrix_orig = raw_matrix.copy()
        da_matrix_orig = da_matrix.copy()
        qc_matrix_orig = qc_matrix.copy()

        # 確保 CNN 輸出是 NumPy array（如果是 Tensor 則轉換）
        if hasattr(cnn_optimized, 'numpy'):
            cnn_optimized = cnn_optimized.numpy()
        if hasattr(cnn_traditional, 'numpy'):
            cnn_traditional = cnn_traditional.numpy()

        cnn_optimized_orig = np.array(cnn_optimized)  # CNN軟分類訓練優化
        cnn_traditional_orig = np.array(cnn_traditional)  # CNN傳統硬分類

        # 使用優化分支作為主要結果
        cnn_matrix = cnn_optimized_orig

        # 4. 對齊矩陣尺寸（處理尺寸不一致的情況，用於公平的指標計算）
        raw_matrix, da_matrix, qc_matrix = align_matrices(raw_matrix, da_matrix, qc_matrix)
        # 同時對齊CNN矩陣到相同的尺寸
        cnn_matrix = cnn_matrix[:raw_matrix.shape[0], :raw_matrix.shape[1]]

        # 4.5 強制物理約束（消除數值精度誤差）
        # 同時重新計算 DA 和 CNN，確保使用相同的 nyquist 值和計算方式

        # [FIX] DA 的 valid mask（RAW 和 DA 都有效）
        valid_da_mask = ~np.isnan(raw_matrix) & ~np.isnan(da_matrix)

        # [FIX] CNN 的 valid mask（RAW 和 CNN 都有效，不受 DA 限制）
        # 實務上使用時沒有 DA 參考，CNN 仍要給出所有 RAW 有效點的預測
        valid_cnn_mask = ~np.isnan(raw_matrix) & ~np.isnan(cnn_matrix)

        # 計算 DA 的 fold（只在 DA 有效的地方）
        da_fold = np.zeros_like(da_matrix)
        da_fold[valid_da_mask] = np.round((da_matrix[valid_da_mask] - raw_matrix[valid_da_mask]) / (2 * nyquist))

        # 計算 CNN 的 fold（在所有 CNN 有效的地方）
        cnn_fold = np.zeros_like(cnn_matrix)
        cnn_fold[valid_cnn_mask] = np.round((cnn_matrix[valid_cnn_mask] - raw_matrix[valid_cnn_mask]) / (2 * nyquist))

        # 使用相同的方式重新計算，確保數值完全一致
        da_matrix_constrained = np.full_like(da_matrix, np.nan)
        da_matrix_constrained[valid_da_mask] = raw_matrix[valid_da_mask] + da_fold[valid_da_mask] * (2.0 * nyquist)

        cnn_matrix_constrained = np.full_like(cnn_matrix, np.nan)
        cnn_matrix_constrained[valid_cnn_mask] = raw_matrix[valid_cnn_mask] + cnn_fold[valid_cnn_mask] * (2.0 * nyquist)

        # 5. 計算指標（使用物理約束後的矩陣，確保公平比較）
        metrics = compute_metrics_triplet(raw_matrix, da_matrix_constrained, cnn_matrix_constrained, qc_matrix, nyquist)

        # 6. 生成可視化（如果啟用）
        if enable_viz and viz_dir is not None:
            try:
                # 加入資料夾名稱以區分同名檔案
                dir_name = Path(triplet['raw']).parent.name
                case_viz_dir = os.path.join(viz_dir, f"{station}_{triplet['elevation']}", dir_name)
                os.makedirs(case_viz_dir, exist_ok=True)

                # 6.1 矩陣視圖可視化（使用 mixed_patch_test_model.py 的函數）
                # 注意：這些函數期望特定的參數結構
                # save_visualizations 需要：raw_vel, dealiased_vel, classification_vel, gt_vel, residual等
                # 為了簡化，我們使用自訂的NWP可視化（因為參數結構不同）
                save_nwp_visualizations(raw_matrix, da_matrix, cnn_matrix, qc_matrix,
                                       filename, case_viz_dir, nyquist, triplet=triplet)

                # 6.2 地理位置可視化（使用 batch_test_system 的方法）
                gz_path = find_corresponding_gz_file(triplet['raw'])
                da_gz_path = find_corresponding_gz_file(triplet['da'])  # [FIX] 找到 DA 的 .gz 檔案

                if gz_path and GEO_VIZ_AVAILABLE:
                    print(f"  [MAP]  使用 batch_test_system 方法生成地理圖")
                    print(f"     CNN使用: alias_mask (優化分類分支) 硬分類結果")
                    print(f"     DA 作為參考標準（NWP物理算法）")

                    # 先保存臨時CSV（原始函數需要）
                    temp_csv = os.path.join(temp_dir, f"{Path(filename).stem}_raw_for_geo.csv")
                    save_matrix_as_csv(raw_matrix_orig, nyquist, triplet['raw'], temp_csv)

                    # [FIX] 保存 DA 的臨時 CSV
                    temp_da_csv = os.path.join(temp_dir, f"{Path(filename).stem}_da_for_geo.csv")
                    save_matrix_as_csv(da_matrix_orig, nyquist, triplet['da'], temp_da_csv)

                    # 使用 save_geographical_visualizations 函數（完全一致的方式）
                    # 映射:
                    #   RAW → raw_vel
                    #   CNN軟分類訓練優化 → dealiased_cls_optimized
                    #   CNN傳統硬分類 → dealiased_cls
                    #   DA物理算法 → gt_vel（作為Ground Truth參考標準）

                    save_geographical_visualizations(
                        raw_vel=raw_matrix_orig,                # RAW觀測速度
                        dealiased_cls_optimized=cnn_optimized_orig,  # CNN模型 (軟分類訓練優化)
                        dealiased_cls=cnn_traditional_orig,     # CNN模型 (傳統硬分類)
                        gt_vel=da_matrix_orig,                  # DA物理算法（作為Ground Truth）
                        csv_path=temp_csv,
                        gz_path=gz_path,
                        result_dir=case_viz_dir,
                        shape_path=shape_path,
                        gt_csv_path=temp_da_csv,  # [FIX] 傳入 DA CSV
                        gt_gz_path=da_gz_path     # [FIX] 傳入 DA .gz
                    )

                    print(f"     [OK] 地理圖生成完成")
                else:
                    print(f"  [WARNING]  未找到 .gz 檔案或地理可視化不可用")

            except Exception as viz_error:
                print(f"  [WARNING]  可視化失敗: {viz_error}")
                import traceback
                traceback.print_exc()

        # 5. 組裝結果
        results = {
            'metadata': {
                'filename': full_filename,  # 完整路徑: 目錄/文件名
                'directory': directory,      # 目錄名稱
                'basename': filename,        # 只有文件名（向後兼容）
                'station': station,
                'elevation': triplet['elevation'],
                'date': triplet['date'],
                'time': triplet['time'],
                'timestamp': triplet['timestamp']
            },
            'metrics': metrics,
            'nyquist_velocity': float(nyquist)
        }

        # 6. 打印每個個案的關鍵指標
        print(f"\n  [INFO] 個案指標 - {filename}")
        print(f"  ├─ Nyquist速度: {nyquist:.2f} m/s")
        print(f"  ├─ 有效像素: {metrics['raw']['total_pixels']:,}")

        # RAW vs DA 比較
        print(f"  ├─ RAW vs DA:")
        print(f"  │   ├─ RMSE: {metrics['raw_vs_da']['rmse']:.3f} m/s")
        print(f"  │   ├─ MAE: {metrics['raw_vs_da']['mae']:.3f} m/s")
        print(f"  │   └─ 相關性: {metrics['raw_vs_da']['correlation']:.3f}")

        # RAW vs CNN 比較
        print(f"  ├─ RAW vs CNN:")
        print(f"  │   ├─ RMSE: {metrics['raw_vs_cnn']['rmse']:.3f} m/s")
        print(f"  │   ├─ MAE: {metrics['raw_vs_cnn']['mae']:.3f} m/s")
        print(f"  │   └─ 相關性: {metrics['raw_vs_cnn']['correlation']:.3f}")

        # DA vs CNN 比較
        print(f"  ├─ DA vs CNN:")
        print(f"  │   ├─ RMSE: {metrics['da_vs_cnn']['rmse']:.3f} m/s")
        print(f"  │   ├─ MAE: {metrics['da_vs_cnn']['mae']:.3f} m/s")
        print(f"  │   └─ 相關性: {metrics['da_vs_cnn']['correlation']:.3f}")

        # 修正成功率（最重要！）
        cnn_perf = metrics['cnn_performance']
        print(f"  └─ CNN 修正成功率 (以DA為參考):")
        print(f"      ├─ 需要修正像素: {cnn_perf['need_correction_pixels']:,}")
        print(f"      ├─ 成功修正像素: {cnn_perf['corrected_pixels']:,}")
        print(f"      ├─ 修正成功率: {cnn_perf['correction_success_rate']:.1%}")
        print(f"      └─ Fold整體一致率: {cnn_perf['fold_agreement']:.1%}")

        # 7. 保存每個案例的指標到 JSON
        if viz_dir is not None:
            # 提取目錄名稱
            dir_name = Path(triplet['raw']).parent.name
            basename = Path(filename).stem

            # JSON 也放到資料夾子目錄中，與可視化圖片對應
            case_metrics_path = os.path.join(viz_dir, f"{station}_{triplet['elevation']}", dir_name, f"{basename}_metrics.json")
            os.makedirs(os.path.dirname(case_metrics_path), exist_ok=True)
            with open(case_metrics_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n  [SAVE] 個案指標已保存: {dir_name}/{Path(case_metrics_path).name}")

        # 清理臨時文件
        if os.path.exists(temp_csv):
            os.remove(temp_csv)

        return results, None

    except Exception as e:
        import traceback as tb
        error_msg = f"測試 {triplet['filename']} 失敗: {str(e)}\n{tb.format_exc()}"
        print(f"[ERROR] {error_msg}")
        return None, error_msg


def aggregate_nwp_results(all_results):
    """聚合NWP測試結果"""

    # 按雷達站統計
    station_stats = defaultdict(list)

    successful_results = []

    for result, error in all_results:
        if result is None:
            continue

        successful_results.append(result)
        metadata = result['metadata']
        station = metadata['station']

        # 提取指標
        metrics = result['metrics']

        # 新的指標結構使用 raw_vs_da, da_vs_cnn 等
        # 我們為每個方法構建記錄

        # RAW 方法記錄（與 DA 比較）
        if 'raw_vs_da' in metrics:
            record_raw = {
                'filename': metadata['filename'],
                'station': station,
                'elevation': metadata['elevation'],
                'timestamp': metadata['timestamp'],
                'method': 'raw',
                'rmse': metrics['raw_vs_da']['rmse'],
                'mae': metrics['raw_vs_da']['mae'],
                'corr': metrics['raw_vs_da']['correlation'],
                'correction_success_rate': 0.0,  # RAW 沒有修正
                'fold_accuracy': 0.0
            }
            station_stats[station].append(record_raw)

        # DA 方法記錄（作為參考，與自己比較結果為0）
        record_da = {
            'filename': metadata['filename'],
            'station': station,
            'elevation': metadata['elevation'],
            'timestamp': metadata['timestamp'],
            'method': 'da',
            'rmse': 0.0,  # DA 與自己比較
            'mae': 0.0,
            'corr': 1.0,
            'correction_success_rate': 1.0,  # DA 作為參考標準
            'fold_accuracy': 1.0
        }
        station_stats[station].append(record_da)

        # CNN 方法記錄（與 DA 比較）
        if 'da_vs_cnn' in metrics:
            cnn_perf = metrics.get('cnn_performance', {})
            record_cnn = {
                'filename': metadata['filename'],
                'station': station,
                'elevation': metadata['elevation'],
                'timestamp': metadata['timestamp'],
                'method': 'cnn',
                'rmse': metrics['da_vs_cnn']['rmse'],
                'mae': metrics['da_vs_cnn']['mae'],
                'corr': metrics['da_vs_cnn']['correlation'],
                'correction_success_rate': cnn_perf.get('correction_success_rate', 0.0),
                'fold_accuracy': cnn_perf.get('fold_agreement', 0.0)
            }
            station_stats[station].append(record_cnn)

    print(f"[OK] 成功處理 {len(successful_results)} 個案例")

    return station_stats, successful_results


def compute_summary_stats_nwp(records, group_name=""):
    """計算NWP統計摘要"""
    if not records:
        return {}

    df = pd.DataFrame(records)
    summary = {}

    # 按方法統計
    for method in ['raw', 'da', 'cnn']:
        method_data = df[df['method'] == method]

        if len(method_data) > 0:
            summary[method] = {
                'count': len(method_data),
                'rmse_mean': method_data['rmse'].mean(),
                'rmse_std': method_data['rmse'].std(),
                'mae_mean': method_data['mae'].mean(),
                'mae_std': method_data['mae'].std(),
                'corr_mean': method_data['corr'].mean(),
                'corr_std': method_data['corr'].std(),
                'correction_success_rate_mean': method_data['correction_success_rate'].mean(),
                'fold_accuracy_mean': method_data['fold_accuracy'].mean()
            }

    return summary


def save_nwp_results(station_stats, successful_results, output_dir, test_config=None):
    """保存NWP測試結果"""

    os.makedirs(output_dir, exist_ok=True)

    # 1. 保存詳細結果
    detailed_results_path = os.path.join(output_dir, "nwp_detailed_results.json")
    with open(detailed_results_path, 'w') as f:
        json.dump(successful_results, f, indent=2, default=str)

    # 2. 計算統計摘要
    analysis = {
        'test_summary': {
            'total_cases': len(successful_results),
            'test_timestamp': datetime.now().isoformat(),
            'data_source': 'NWP (bref_qc as GT)',
            'command': " ".join(sys.argv),
            'python_version': sys.version.split()[0],
        },
        'by_station': {}
    }

    # 添加測試配置（包含 CRF 參數）
    if test_config is not None:
        analysis['test_config'] = test_config

    for station, records in station_stats.items():
        analysis['by_station'][station] = compute_summary_stats_nwp(records, station)

    # 保存分析結果
    analysis_path = os.path.join(output_dir, "nwp_analysis_summary.json")
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)

    # 3. 生成CSV報告
    all_records = []
    for records in station_stats.values():
        all_records.extend(records)

    df = pd.DataFrame(all_records)
    csv_path = os.path.join(output_dir, "nwp_test_results.csv")
    df.to_csv(csv_path, index=False)

    print(f"\n[INFO] NWP測試結果已保存:")
    print(f"  • 詳細結果: {detailed_results_path}")
    print(f"  • 統計摘要: {analysis_path}")
    print(f"  • CSV報告: {csv_path}")

    return analysis


def print_nwp_summary(analysis):
    """打印NWP測試總結報告"""

    print("\n" + "="*90)
    print("[LIST] NWP資料批量測試總結報告 (GT = bref_qc, 100%正確)")
    print("="*90)

    total_cases = analysis['test_summary']['total_cases']
    print(f"\n[OK] 總測試案例: {total_cases}")

    # 按雷達站和方法統計
    print(f"\n[SITE] 按雷達站統計 (比較 RAW vs DA vs CNN):")
    print("-" * 90)
    print(f"{'雷達站':<8} {'方法':<6} {'RMSE':<10} {'MAE':<10} {'相關性':<10} {'Fold準確率':<12}")
    print("-" * 90)

    for station, stats in analysis['by_station'].items():
        for method in ['raw', 'da', 'cnn']:
            if method in stats:
                method_stats = stats[method]
                method_label = {'raw': 'RAW', 'da': 'DA', 'cnn': 'CNN'}[method]
                marker = '[TARGET]' if method == 'cnn' else '  '

                print(f"{marker}{station:<8} {method_label:<6} "
                      f"{method_stats['rmse_mean']:<10.3f} "
                      f"{method_stats['mae_mean']:<10.3f} "
                      f"{method_stats['corr_mean']:<10.3f} "
                      f"{method_stats.get('fold_accuracy_mean', 0):<12.1%}")

    # 整體比較
    print(f"\n[INFO] 整體性能比較:")
    print("-" * 90)

    # 計算全局平均
    all_stats = defaultdict(list)
    for station_stats in analysis['by_station'].values():
        for method, stats in station_stats.items():
            all_stats[method].append(stats)

    print(f"{'方法':<10} {'平均RMSE':<12} {'平均Fold準確率':<18} {'CNN改善幅度':<15}")
    print("-" * 90)

    baseline_rmse = None
    for method in ['raw', 'da', 'cnn']:
        if method in all_stats:
            avg_rmse = np.mean([s['rmse_mean'] for s in all_stats[method]])
            avg_fold_acc = np.mean([s.get('fold_accuracy_mean', 0) for s in all_stats[method]])

            if method == 'raw':
                baseline_rmse = avg_rmse

            improvement = ""
            if baseline_rmse and method != 'raw':
                improve_pct = (baseline_rmse - avg_rmse) / baseline_rmse * 100
                improvement = f"↓{improve_pct:.1f}%" if improve_pct > 0 else f"↑{abs(improve_pct):.1f}%"

            method_label = {'raw': 'RAW', 'da': 'DA演算法', 'cnn': 'CNN模型'}[method]
            marker = '[BEST]' if method == 'cnn' else '  '

            print(f"{marker}{method_label:<10} {avg_rmse:<12.3f} {avg_fold_acc:<18.1%} {improvement:<15}")

    print("\n" + "="*90)


def get_test_files_from_h5(h5_path, nwp_root):
    """
    從 H5 test set 中提取完整的測試文件列表，並構建 triplets

    重要：同一個文件名可能出現在多個時間段目錄中（滾動預報），全部都會被測試

    Returns:
    --------
    list of dict : triplets 格式，與 scan_nwp_root 返回格式一致
    """
    print(f"\n[DIR] 從 H5 test set 提取測試文件列表...")
    print(f"   H5 文件: {h5_path}")
    print(f"   NWP 根目錄: {nwp_root}")
    print("   模式: 測試所有時間段目錄中的匹配文件")

    from collections import defaultdict
    test_files = defaultdict(set)  # 使用 set 避免重複
    nwp_root_path = Path(nwp_root)

    with h5py.File(h5_path, 'r') as f:
        test_group = f['test']

        for patch_id in tqdm.tqdm(test_group.keys(), desc="掃描 test patches"):
            patch = test_group[patch_id]

            # 從 source 讀取 RAW 文件名
            if 'source' in patch:
                raw_filename = patch['source'][()]

                # 處理可能的 bytes 類型
                if isinstance(raw_filename, bytes):
                    raw_filename = raw_filename.decode('utf-8')

                # 嘗試從 source_detail 讀取 nwp_forecast_dir（精確目錄）
                nwp_forecast_dir = None
                if 'source_detail' in patch:
                    source_detail = patch['source_detail']
                    if 'nwp_forecast_dir' in source_detail.attrs:
                        nwp_forecast_dir = source_detail.attrs['nwp_forecast_dir']
                        if isinstance(nwp_forecast_dir, bytes):
                            nwp_forecast_dir = nwp_forecast_dir.decode('utf-8')

                # raw_filename 格式: RCCG.20241031.1850.bvel_raw.01.parquet
                # 解析信息
                parts = raw_filename.split('.')
                if len(parts) >= 5:
                    station = parts[0]  # RCCG
                    date = parts[1]     # 20241031
                    time = parts[2]     # 1850
                    sweep = parts[4]    # 01

                    timestamp = f"{date}{time}"
                    da_filename = f"{station}.{date}.{time}.bvel_da.{sweep}.parquet"
                    qc_filename = f"{station}.{date}.{time}.bvel_qc.{sweep}.parquet"

                    # 優先使用 H5 記錄的精確目錄
                    if nwp_forecast_dir:
                        candidate_dirs = [nwp_root_path / nwp_forecast_dir]
                    else:
                        # 回退：搜索所有可能的目錄
                        date_short = date[2:]  # 20241031 -> 241031
                        try:
                            candidate_dirs = [d for d in nwp_root_path.iterdir()
                                             if d.is_dir() and d.name.startswith(date_short)]
                        except (PermissionError, OSError) as e:
                            # Windows 外接硬碟權限問題處理
                            continue

                        if not candidate_dirs:
                            try:
                                candidate_dirs = [d for d in nwp_root_path.iterdir()
                                                 if d.is_dir() and date in d.name]
                            except (PermissionError, OSError) as e:
                                continue

                    # 搜索所有包含該文件的目錄（不 break，全部收集）
                    for d in sorted(candidate_dirs):
                        raw_path = d / raw_filename
                        da_path = d / da_filename
                        qc_path = d / qc_filename

                        try:
                            raw_exists = raw_path.exists()
                            da_exists = da_path.exists()
                        except (PermissionError, OSError) as e:
                            # 跳過無法訪問的目錄
                            continue

                        if raw_exists and da_exists:
                            # NWP 沒有 QC，使用空字符串
                            try:
                                qc_exists = qc_path.exists()
                            except (PermissionError, OSError):
                                qc_exists = False
                            qc_path_str = str(qc_path) if qc_exists else ""
                            # 包含目錄名稱以確保唯一性
                            unique_id = f"{timestamp}_{sweep}_{d.name}"
                            test_files[station].add((str(raw_path), str(da_path), qc_path_str, unique_id, station, sweep, date, time))
                            # 不 break - 繼續搜索其他目錄

    # 轉換為 triplets 格式
    triplets = []
    for station, files in test_files.items():
        for raw_path, da_path, qc_path, unique_id, station_name, sweep, date, time in sorted(files):
            triplet = {
                'raw': raw_path,
                'da': da_path,
                'qc': qc_path,
                'station': station_name,
                'filename': os.path.basename(raw_path),
                'timestamp': unique_id.rsplit('_', 1)[0],  # 去掉目錄名部分
                'elevation': sweep,
                'unique_id': unique_id,
                'date': date,
                'time': time
            }
            triplets.append(triplet)

    print(f"\n[OK] 提取完成: {len(triplets)} 個測試案例")
    station_counts = defaultdict(int)
    for t in triplets:
        station_counts[t['station']] += 1
    for station, count in sorted(station_counts.items()):
        print(f"   {station}: {count} 個完整掃描（包含多個時間段目錄）")

    return triplets


def main():
    parser = argparse.ArgumentParser(description='NWP資料批量測試系統')
    parser.add_argument('--model_path', type=str, required=True, help='模型檔案路徑')
    parser.add_argument('--nwp_dir', type=str, help='單一NWP資料目錄（例如 24072411）')
    parser.add_argument('--nwp_root', type=str, help='NWP資料根目錄（掃描所有子目錄）')
    parser.add_argument('--use_h5_test_set', action='store_true', help='使用 H5 test set 來選擇測試文件（需同時提供 --nwp_h5 和 --nwp_root）')
    parser.add_argument('--nwp_h5', type=str, help='NWP H5 文件路徑（用於提取 test set，需搭配 --use_h5_test_set）')
    parser.add_argument('--no_physics_model', dest='use_physics_model', action='store_false', help='不使用物理約束模型')
    parser.set_defaults(use_physics_model=True)
    parser.add_argument('--output_dir', type=str, default=None, help='輸出目錄')
    parser.add_argument('--max_cases', type=int, default=None, help='最大測試案例數')
    parser.add_argument('--max_dirs', type=int, default=None, help='最多掃描幾個子目錄（用於nwp_root）')
    parser.add_argument('--stations', type=str, nargs='+', default=None, help='指定測試的雷達站')
    parser.add_argument('--specific_file', type=str, default=None, help='指定測試單一檔案（例如: RCCK.20241031.0600.bvel_raw.01）')
    parser.add_argument('--temp_dir', type=str, default='temp_nwp_csv', help='臨時CSV目錄')
    parser.add_argument('--min_per_station', type=int, default=None, help='每個雷達站最少測試案例數（預設: max_cases除以雷達站數）')
    parser.add_argument('--random_seed', type=int, default=42, help='隨機種子（用於可重現的抽樣）')
    parser.add_argument('--enable_viz', action='store_true', help='啟用可視化（生成比較圖和地理圖）')
    parser.add_argument('--viz_dir', type=str, default=None, help='可視化輸出目錄（預設為output_dir/visualizations）')
    parser.add_argument('--shape_path', type=str, default=None, help='台灣地圖shapefile路徑（用於地理圖）')

    # CRF 後處理參數（修正版 2025-12-24）
    parser.add_argument('--enable_crf', action='store_true', help='啟用CRF空間平滑後處理（預設關閉，使用傳統argmax）')
    parser.add_argument('--crf_unary_weight', type=float, default=1.0, help='CRF一元勢能權重（保留原始預測）')
    parser.add_argument('--crf_pairwise_weight', type=float, default=1.5, help='CRF二元勢能權重（修正：50.0→1.5，與unary同量級）')
    parser.add_argument('--crf_threshold', type=float, default=1.5, help='CRF fold差異閾值（修正：0.5→1.5，只懲罰大跳躍）')
    parser.add_argument('--crf_max_iterations', type=int, default=10, help='CRF最大迭代次數')
    parser.add_argument('--crf_confidence_threshold', type=float, default=None, help='已廢棄：信心度閾值（現在處理所有像素）')

    # Temperature Scaling 參數（校準信心度）
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature Scaling 溫度參數（>1.0 降低信心度，用於校準過度自信的預測）')

    # Fold0 Bias 參數（調整 fold=0 的傾向）
    parser.add_argument('--fold0_bias', type=float, default=0.0, help='Fold=0 的 logit 偏移量（負值降低 fold=0 機率，讓模型更傾向修正）')

    # 強制測試案例清單
    parser.add_argument('--force_cases_file', type=str, default=None,
                        help='強制測試案例 JSON 檔案路徑（格式: [{"nwp_dir":"24072417","file":"RCWF.20240724.2020.bvel_raw.01"}, ...]，需搭配 --nwp_root）')

    args = parser.parse_args()

    if not TEST_FUNCTION_AVAILABLE:
        print("[ERROR] 錯誤: 測試函數不可用，需要tensorflow等依賴")
        return

    # 檢查參數
    if args.force_cases_file:
        if not args.nwp_root:
            print("[ERROR] 錯誤: 使用 --force_cases_file 時必須同時提供 --nwp_root")
            return
    elif args.use_h5_test_set:
        if not args.nwp_h5 or not args.nwp_root:
            print("[ERROR] 錯誤: 使用 --use_h5_test_set 時必須同時提供 --nwp_h5 和 --nwp_root")
            return
    elif not args.nwp_dir and not args.nwp_root:
        print("[ERROR] 錯誤: 必須提供 --nwp_dir 或 --nwp_root 參數（或使用 --use_h5_test_set 或 --force_cases_file）")
        return

    # 設定輸出目錄
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"results/nwp_test_{timestamp}"

    # 設定可視化目錄
    if args.enable_viz:
        if args.viz_dir is None:
            args.viz_dir = os.path.join(args.output_dir, "visualizations")
        os.makedirs(args.viz_dir, exist_ok=True)

    print(f"[START] 開始NWP資料批量測試...")
    print(f"  • 模型: {args.model_path}")
    print(f"  • 輸出目錄: {args.output_dir}")
    print(f"  • GT來源: DA (資料同化物理算法)")
    print(f"  • 成功率計算: fold-based (無容忍度)")
    print(f"  • 測試策略: 動態收集需要退疊的案例（自動跳過 RAW==DA）")
    if args.enable_viz:
        print(f"  • 可視化: 啟用 (輸出至 {args.viz_dir})")
    if args.enable_crf:
        print(f"  • CRF後處理: 啟用 (unary={args.crf_unary_weight}, pairwise={args.crf_pairwise_weight}, threshold={args.crf_threshold})")
    else:
        print(f"  • CRF後處理: 停用 (使用傳統argmax)")

    # 掃描NWP資料
    if args.force_cases_file:
        import json as _json
        print(f"  • 測試集來源: 強制案例清單 ({args.force_cases_file})")
        with open(args.force_cases_file, 'r') as f:
            forced_cases = _json.load(f)
        triplets = []
        missing_count = 0
        for case in forced_cases:
            nwp_dir_path = Path(args.nwp_root) / case['nwp_dir']
            raw_file = nwp_dir_path / f"{case['file']}.parquet"
            da_file = Path(str(raw_file).replace('bvel_raw', 'bvel_da'))
            qc_file = Path(str(raw_file).replace('bvel_raw', 'bref_qc'))
            if raw_file.exists() and da_file.exists():
                parts = case['file'].split('.')
                triplets.append({
                    'raw': str(raw_file),
                    'da': str(da_file),
                    'qc': str(qc_file) if qc_file.exists() else '',
                    'station': parts[0],
                    'date': parts[1],
                    'time': parts[2],
                    'elevation': parts[-1],
                    'timestamp': f"{parts[1]}{parts[2]}_{parts[-1]}",
                    'filename': f"{case['file']}.parquet"
                })
            else:
                missing_count += 1
                print(f"  [MISS] 找不到: {case['nwp_dir']}/{case['file']}")
        print(f"[FORCE] 從清單載入 {len(triplets)} 個案例 (缺失: {missing_count})")
    elif args.use_h5_test_set:
        print(f"  • 測試集來源: H5 Test Set (無數據洩漏)")
        triplets = get_test_files_from_h5(args.nwp_h5, args.nwp_root)
    elif args.nwp_dir:
        triplets = find_nwp_triplets(args.nwp_dir)
        print(f"[DIR] 單一目錄: {args.nwp_dir}")
    else:
        triplets = scan_nwp_root(args.nwp_root, max_dirs=args.max_dirs, stations=args.stations)
        print(f"[DIR] 根目錄掃描: {args.nwp_root}")

    # 按雷達站分組
    print(f"\n[INFO] 掃描到 {len(triplets)} 個三元組")

    # 如果指定了特定檔案，只測試該檔案
    if args.specific_file:
        print(f"\n[TARGET] 指定測試單一檔案: {args.specific_file}")
        filtered_triplets = [t for t in triplets if args.specific_file in t['filename']]
        if len(filtered_triplets) == 0:
            print(f"[ERROR] 找不到檔案: {args.specific_file}")
            return
        triplets = filtered_triplets
        print(f"[OK] 找到 {len(triplets)} 個匹配的檔案")

    df = pd.DataFrame(triplets)
    stations = sorted(df['station'].unique())

    print(f"[RADAR] 雷達站: {', '.join(stations)} (共 {len(stations)} 站)")

    # 按站分組並隨機排序
    station_triplets = {}
    for station in stations:
        station_df = df[df['station'] == station]
        station_list = [triplets[i] for i in station_df.index]
        random.seed(args.random_seed)
        random.shuffle(station_list)
        station_triplets[station] = station_list
        print(f"  • {station}: {len(station_list)} 個案例")

    # 確定每站目標數量
    if args.force_cases_file:
        # 強制案例模式：測試所有案例，不限制數量
        target_per_station = 999999
        print(f"\n[TARGET] 強制案例模式: 測試清單中所有案例 (共 {len(triplets)} 個)")
    elif args.min_per_station:
        target_per_station = args.min_per_station
    elif args.max_cases:
        target_per_station = max(1, args.max_cases // len(stations))
    else:
        target_per_station = 10  # 預設值

    if not args.force_cases_file:
        print(f"\n[TARGET] 目標: 每站收集 {target_per_station} 個有效案例（需要退疊的案例）")
        print(f"   註: 不需要退疊的案例會被跳過，不計入抽樣次數")

    if len(triplets) == 0:
        print("[ERROR] 沒有找到測試資料")
        return

    # 構建模型
    print(f"[FIX] 構建模型...")
    try:
        model = build_mixed_patch_model_for_inference(use_physics_model=args.use_physics_model)
        print("[OK] 模型構建完成")

        # 用一個小的dummy input初始化模型變數
        print(f"[FIX] 初始化模型變數...")
        if args.use_physics_model:
            # 物理模型需要字典輸入 {'vel': ..., 'nyq': ...}
            # [WARNING] 重要：形狀必須與訓練時一致 (128x128，不是 64x64)
            dummy_vel = np.random.randn(1, 1, 128, 128, 1).astype(np.float32)  # (batch, time, H, W, channel)
            dummy_nyq = np.array([[33.62]], dtype=np.float32)  # (batch, 1)
            _ = model({'vel': dummy_vel, 'nyq': dummy_nyq}, training=False)
        else:
            # 標準模型使用普通 tensor 輸入
            dummy_input = np.random.randn(1, 128, 128, 1).astype(np.float32)  # 修正為 128x128
            _ = model(dummy_input, training=False)

        # 載入權重
        print(f"[LOAD] 載入模型權重...")
        try:
            model.load_weights(args.model_path)
            print("[OK] 權重載入完成")
        except ValueError as e:
            if "axes don't match array" in str(e):
                print(f"[WARNING]  權重格式不匹配，嘗試使用 by_name + skip_mismatch 模式...")
                model.load_weights(args.model_path, by_name=True, skip_mismatch=True)
                print("[OK] 權重載入完成 (skip_mismatch 模式)")
            else:
                raise

    except Exception as e:
        print(f"[ERROR] 模型初始化失敗: {e}")
        traceback.print_exc()
        return

    # 運行批量測試 - 按站點動態收集有效案例
    all_results = []
    station_stats_detailed = {}

    print(f"\n{'='*90}")
    print(f"[START] 開始測試...")
    print(f"{'='*90}\n")

    for station in stations:
        print(f"\n{'='*70}")
        print(f"[RADAR] 測試雷達站: {station}")
        print(f"{'='*70}")

        valid_results = []
        tested_count = 0
        skipped_count = 0
        failed_count = 0

        available_triplets = station_triplets[station]

        for triplet in available_triplets:
            if len(valid_results) >= target_per_station:
                print(f"  [OK] {station}: 已達目標 {target_per_station} 個有效案例")
                break

            tested_count += 1
            print(f"\n  [{tested_count}] {triplet['filename']}")

            result, error = run_single_nwp_test(
                triplet,
                model,
                args.use_physics_model,
                args.temp_dir,
                enable_viz=args.enable_viz,
                viz_dir=args.viz_dir if args.enable_viz else None,
                shape_path=args.shape_path,
                enable_crf=args.enable_crf,
                crf_unary_weight=args.crf_unary_weight,
                crf_pairwise_weight=args.crf_pairwise_weight,
                crf_threshold=args.crf_threshold,
                crf_max_iterations=args.crf_max_iterations,
                crf_confidence_threshold=args.crf_confidence_threshold,
                temperature=args.temperature,
                fold0_bias=args.fold0_bias
            )

            if error:
                failed_count += 1
                print(f"     [ERROR] 測試失敗: {error}")
                continue

            if result is None:
                failed_count += 1
                print(f"     [ERROR] 測試失敗")
                continue

            # 檢查是否需要退疊
            need_correction = result['metrics']['cnn_performance']['need_correction_pixels']

            if need_correction > 0:
                # 有效案例
                valid_results.append((result, None))
                all_results.append((result, None))
                print(f"     [OK] 有效案例 {len(valid_results)}/{target_per_station} (需修正: {need_correction:,} 像素)")
            else:
                # 不需要退疊，跳過
                skipped_count += 1
                print(f"     [SKIP]  跳過（無需退疊，RAW 已正確）")

        station_stats_detailed[station] = {
            'tested': tested_count,
            'valid': len(valid_results),
            'skipped': skipped_count,
            'failed': failed_count,
            'available': len(available_triplets)
        }

        print(f"\n  [INFO] {station} 統計:")
        print(f"     • 測試案例: {tested_count}")
        print(f"     • 有效案例: {len(valid_results)}")
        print(f"     • 跳過（無需退疊）: {skipped_count}")
        print(f"     • 失敗: {failed_count}")

    # 聚合結果
    print(f"\n{'='*90}")
    print(f"[STATS] 聚合測試結果...")
    station_stats, successful_results = aggregate_nwp_results(all_results)

    # 準備測試配置
    test_config = {
        'model_path': args.model_path,
        'use_physics_model': args.use_physics_model,
        'crf_postprocessing': {
            'enabled': args.enable_crf,
            'unary_weight': args.crf_unary_weight,
            'pairwise_weight': args.crf_pairwise_weight,
            'threshold': args.crf_threshold,
            'max_iterations': args.crf_max_iterations,
            'confidence_threshold': args.crf_confidence_threshold
        }
    }

    # 保存結果
    analysis = save_nwp_results(station_stats, successful_results, args.output_dir, test_config=test_config)

    # 保存詳細測試統計
    stats_path = os.path.join(args.output_dir, "test_statistics.json")
    with open(stats_path, 'w') as f:
        json.dump(station_stats_detailed, f, indent=2)
    print(f"  • 測試統計: {stats_path}")

    # 打印總結報告
    print_nwp_summary(analysis)

    # 打印詳細測試統計
    print(f"\n{'='*90}")
    print(f"[INFO] 測試統計摘要")
    print(f"{'='*90}\n")

    total_tested = sum(s['tested'] for s in station_stats_detailed.values())
    total_valid = sum(s['valid'] for s in station_stats_detailed.values())
    total_skipped = sum(s['skipped'] for s in station_stats_detailed.values())
    total_failed = sum(s['failed'] for s in station_stats_detailed.values())

    print(f"{'雷達站':<10} {'可用':<10} {'測試':<10} {'有效':<10} {'跳過':<10} {'失敗':<10} {'達標率':<10}")
    print(f"{'-'*90}")
    for station in sorted(station_stats_detailed.keys()):
        s = station_stats_detailed[station]
        completion = f"{s['valid']}/{target_per_station}"
        print(f"{station:<10} {s['available']:<10} {s['tested']:<10} {s['valid']:<10} {s['skipped']:<10} {s['failed']:<10} {completion:<10}")

    print(f"{'-'*90}")
    print(f"{'總計':<10} {'-':<10} {total_tested:<10} {total_valid:<10} {total_skipped:<10} {total_failed:<10}")

    print(f"\n[SUCCESS] NWP批量測試完成!")
    print(f"  • 有效案例: {total_valid} (需要退疊的案例)")
    print(f"  • 跳過案例: {total_skipped} (RAW==DA，無需退疊)")
    print(f"  • 失敗案例: {total_failed}")
    print(f"  • 總測試數: {total_tested}")
    print(f"  • 結果目錄: {args.output_dir}")

    # 清理臨時目錄
    if os.path.exists(args.temp_dir) and len(os.listdir(args.temp_dir)) == 0:
        os.rmdir(args.temp_dir)

if __name__ == "__main__":
    main()
