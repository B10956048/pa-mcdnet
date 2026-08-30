#!/usr/bin/env python3
"""
NWP 個案上 Ours vs UNet-VDA 跨模型比較腳本

在 CWA NWP 資料上比較兩個模型：
- Ours: 物理約束 CNN (遷移學習後)
- Paper: UNet-VDA (原始論文模型)

使用 H5 test split 進行資料洩漏防護，確保所有測試案例不在訓練集中。

使用範例:
    python test_nwp_comparison.py \
        --model_path results/transfer_clean_patches_impove_20260213_101154/best_model_manual.h5 \
        --nwp_root /path/to/nwp_data_csv \
        --output_dir Transfer_result/nwp_comparison \
        --enable_viz \
        --use_physics_model
"""

# 必須在所有其他導入之前執行修復
import fix_typing

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import re
import sys
import json
import math
import argparse
import importlib.util
import types
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
from matplotlib import cm
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import h5py
import pyarrow.parquet as pq

# 複用現有模型構建與指標函數
from mixed_patch_test_model import (
    build_mixed_patch_model_for_inference,
    auto_zero_pad,
    compute_metrics,
)
from dealiasing_success_metrics import compute_dealiasing_success_metrics

# 地理可視化（lazy import，避免 pyart/basemap 提前載入影響 TF）
GEO_VIZ_AVAILABLE = None  # None = 尚未檢查

def _ensure_geo_imports():
    """Lazy import，只在首次需要 geo viz 時才載入 batch_test_nwp"""
    global GEO_VIZ_AVAILABLE, find_corresponding_gz_file, save_single_geo_visualization
    if GEO_VIZ_AVAILABLE is not None:
        return GEO_VIZ_AVAILABLE
    try:
        from batch_test_nwp import find_corresponding_gz_file as _find, save_single_geo_visualization as _save
        find_corresponding_gz_file = _find
        save_single_geo_visualization = _save
        GEO_VIZ_AVAILABLE = True
    except ImportError:
        GEO_VIZ_AVAILABLE = False
    return GEO_VIZ_AVAILABLE

# ---------------------------------------------------------------------------
# 常數
# ---------------------------------------------------------------------------
SEED = 46
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 論文模型參數 (從 test_nexrad.py 複用)
PAPER_PAD_DEG = 12        # 方位角 periodic padding 度數
PAPER_SN = 16             # start_neurons
PAPER_NRAD_DIVISOR = 64   # nrad 必須整除 64
PAPER_NAZ_DIVISOR = 32    # naz 必須整除 32
DEFAULT_PAPER_MODEL = 'unet-vda-main/models/dealias_sn16_csi9764.SavedModel'

# H5 資料路徑
H5_PATH = 'data/nwp_all_patches_clean.h5'

# 5 個指定必測案例
REQUIRED_CASES = [
    {'nwp_dir': '24103104', 'file': 'RCWF.20241031.1240.bvel_raw.01'},
    {'nwp_dir': '24103113', 'file': 'RCHL.20241031.1750.bvel_raw.01'},
    {'nwp_dir': '24093003', 'file': 'RCCG.20240930.0640.bvel_raw.01'},
    {'nwp_dir': '24103100', 'file': 'RCWF.20241031.0940.bvel_raw.01'},
    {'nwp_dir': '24072417', 'file': 'RCWF.20240724.2020.bvel_raw.01'},
]

# ---------------------------------------------------------------------------
# 1. 資料載入
# ---------------------------------------------------------------------------

def read_gz_radar(gz_path: str):
    """
    讀取 CWA 雷達 .gz 二進位檔（支援 bvel_raw / bvel / bvel_sfda 格式）。
    Returns: vel (H,W) float32 NaN=invalid, nyquist float, meta dict
    """
    import gzip, struct
    with gzip.open(gz_path, 'rb') as gf:
        f = gf.read()
    header_end = 160
    header   = struct.unpack('<16s36i', f[0:header_end])
    info     = np.array(header[1:-1])
    h_scale  = info[0]
    info_flt = info / h_scale
    nray  = int(info_flt[14]);  ngate = int(info_flt[15])
    nyq   = float(info_flt[10])
    try:
        name = header[0][:4].decode('utf-8').strip('\x00').strip()
    except Exception:
        name = 'UNKN'
    yyyy = int(info_flt[4]); mm = int(info_flt[5]); dd = int(info_flt[6])
    hh   = int(info_flt[7]); mn = int(info_flt[8])
    n        = nray * ngate
    raw_int  = np.array(struct.unpack('<' + str(n) + 'i', f[header_end:]))
    # 實測確認：CWA .gz 實際 fill value 為 -990（decoded: -99.0）和 -9990（decoded: -999.0，RCWF）
    # header 的 var_miss(-9900) 與 rf_miss(-44400) 從未出現在資料中
    valid    = (raw_int != -990) & (raw_int != -9990)
    vel = (raw_int / info_flt[20]).reshape(nray, ngate).astype(np.float32)
    vel[~valid.reshape(nray, ngate)] = np.nan
    # azm_start (header info_flt[16]) 為雷達實際掃描起始方位
    # azm_sp = 360.0 / nray (720-ray 是 0.5°, 360-ray 是 1°)
    # 用於 720 raw + 360 GT 對齊（避免單純 raw[::2] 造成 azimuth 錯位）
    azm_start = float(info_flt[16])
    azm_sp = 360.0 / nray if nray > 0 else 1.0
    meta = {
        'station': name,
        'date':    f"{yyyy:04d}{mm:02d}{dd:02d}",
        'time':    f"{hh:02d}{mn:02d}",
        'nray': nray, 'ngate': ngate,
        'rlat': float(info_flt[2]), 'rlon': float(info_flt[3]),
        'azm_start': azm_start,
        'azm_sp':    azm_sp,
    }
    return vel, nyq, meta


def _find_reflectivity_gz(vel_gz_path):
    """
    從速度 .gz 路徑反查對應的反射率 .gz（同站同時段）。
    優先 bref_qc，其次 bref_raw。

    Returns:
        str | None: 反射率檔路徑（若存在）；找不到回傳 None。
    """
    if vel_gz_path is None:
        return None
    from pathlib import Path
    p = Path(vel_gz_path)
    name = p.name
    for vel_token in ['bvel_raw', 'bvel_sfda', 'bvel_da', 'bvel_vdaqc', 'bvel']:
        if vel_token in name:
            for refl_token in ['bref_qc', 'bref_raw']:
                cand = p.parent / name.replace(vel_token, refl_token)
                if cand.exists():
                    return str(cand)
            break  # 找到 vel_token 就只試一次（不要連 bvel_sfda 都替換）
    return None


def align_raw_to_gt_by_azimuth(raw_vel, raw_azm_start, raw_azm_sp,
                                gt_azm_start, gt_azm_sp, gt_naz):
    """
    依方位角將 raw 重新採樣到 GT 的 ray 順序。
    對每個 GT ray，找 raw 中環形距離最近的 ray，取該 ray 速度資料。

    解決 RCWF 720-ray + GT 360-ray 時，舊版 raw[::2] 直接砍奇數 ray 會造成
    azimuth 錯位（例：raw 從 69.71° 開始、GT 從 0° 開始，[::2] 後 raw[i] 跟
    gt[i] 仍對應完全不同的物理方位，label/metric 全錯）。
    """
    raw_naz = raw_vel.shape[0]
    raw_az = (raw_azm_start + np.arange(raw_naz) * raw_azm_sp) % 360.0
    gt_az  = (gt_azm_start  + np.arange(gt_naz)  * gt_azm_sp ) % 360.0
    diff = np.abs(gt_az[:, None] - raw_az[None, :])
    circ_diff = np.minimum(diff, 360.0 - diff)
    nearest_idx = np.argmin(circ_diff, axis=1)
    return raw_vel[nearest_idx, :]


def load_velocity_data(path: str, debug=False):
    """
    統一資料載入：自動偵測 .gz（實測觀測）或 .parquet（NWP）。
    Returns: (velocity_matrix, nyquist, metadata)  — 與 load_parquet_as_matrix 相同簽名
    """
    if str(path).endswith('.gz'):
        return read_gz_radar(path)
    else:
        return load_parquet_as_matrix(path, debug=debug)


def build_realobs_cases(realobs_root: str, elevation: str = 'all',
                        sample_n: int = 100, exclude_stations=None):
    """
    從 typhoonnew/ 目錄建立 triplet list（格式對齊 NWP triplets）。

    支援新舊兩種命名格式：
      新: {station}.{date}.{time}.bvel_raw.{elev}.gz + bvel_sfda.{elev}.gz
      舊: {station}.{date}.{time}.bvel.{elev}.gz     + bvel_sfda.{elev}.gz
    """
    exclude_set = set(exclude_stations or [])
    cases = []
    root  = Path(realobs_root)

    # GT 候選 suffix（優先序：vdaqc → sfda → da）
    GT_SUFFIXES = ['bvel_vdaqc', 'bvel_sfda', 'bvel_da']
    SKIP_TOKENS = ('vdaqc', 'sfda', 'bvel_da')  # 不當 raw 處理的檔名標記

    def find_gt_for_raw(raw_path_str: str, raw_token: str):
        """嘗試各 GT suffix，回傳第一個存在的；無則 None。"""
        for suf in GT_SUFFIXES:
            gt = Path(raw_path_str.replace(raw_token, suf))
            if gt.exists():
                return gt
        return None

    def collect_pairs_in_dir(scan_dir, station, elev_pats):
        """在指定目錄掃 raw + GT 配對"""
        pairs = []
        for pat in elev_pats:
            for raw_gz in sorted(scan_dir.glob(pat)):
                nm = raw_gz.name
                # 跳過 GT 檔
                if any(tok in nm for tok in SKIP_TOKENS):
                    continue
                # 找對應 GT
                if 'bvel_raw' in nm:
                    gt = find_gt_for_raw(str(raw_gz), 'bvel_raw')
                elif '.bvel.' in nm:
                    gt = None
                    for suf in GT_SUFFIXES:
                        cand = Path(str(raw_gz).replace('.bvel.', f'.{suf}.'))
                        if cand.exists():
                            gt = cand; break
                else:
                    continue
                if gt is not None:
                    pairs.append((raw_gz, gt))
        return pairs

    for case_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        case_name = case_dir.name
        elev_pats = ['*.gz'] if (elevation == 'all' or elevation is None) \
                    else [f'*{elevation}.gz']

        # 判斷目錄結構
        binary_root = case_dir / 'polar' / 'binary'
        station_dirs = []
        if binary_root.exists():
            # 深層格式：polar/binary/STATION/
            station_dirs = [(d.name, d) for d in sorted(binary_root.iterdir()) if d.is_dir()]
        else:
            # 平面格式：直接在 case_dir/*.gz，station 從檔名解析
            # 把同站檔案 group 起來
            flat_files = list(case_dir.glob('*.gz'))
            if flat_files:
                from collections import defaultdict
                by_station = defaultdict(list)
                for f in flat_files:
                    st = f.name.split('.')[0]
                    by_station[st].append(f)
                # 用「整個 case_dir」當 scan_dir，但每個 station 分開處理
                station_dirs = [(st, case_dir) for st in sorted(by_station.keys())]

        if not station_dirs:
            continue

        for station, scan_dir in station_dirs:
            if station in exclude_set:
                continue

            # 對平面結構，要把 glob 限定於該 station
            if scan_dir == case_dir:
                # 修改 elev_pats 加上 station 前綴
                st_elev_pats = [f"{station}*{pat[1:]}" for pat in elev_pats]
            else:
                st_elev_pats = elev_pats

            pairs = collect_pairs_in_dir(scan_dir, station, st_elev_pats)
            if not pairs:
                continue

            # 均勻取樣
            if sample_n and len(pairs) > sample_n:
                idx     = np.round(np.linspace(0, len(pairs)-1, sample_n)).astype(int)
                sampled = [pairs[i] for i in idx]
            else:
                sampled = pairs

            for raw_gz, gt_gz in sampled:
                stem  = raw_gz.stem
                elev  = stem.rsplit('.', 1)[-1] if '.' in stem else '01'
                parts = stem.split('.')
                st    = parts[0] if parts else station
                date  = parts[1] if len(parts) > 1 else ''
                tstr  = parts[2] if len(parts) > 2 else ''
                cases.append({
                    'raw':       str(raw_gz),
                    'da':        str(gt_gz),
                    'station':   st,
                    'date':      date,
                    'time':      tstr,
                    'filename':  stem,
                    'nwp_dir':   case_name,
                    'case_type': 'squall_line' if 'SQUALLLINE' in case_name.upper()
                                 else 'typhoon',
                })

    print(f"[REALOBS] Found {len(cases)} raw/sfda pairs from {realobs_root}")
    return cases


def build_realobs_cases_from_force_file(force_cases_file):
    """
    從 locked realobs JSON 直接讀 cases，繞過 realobs_root 掃描。

    JSON schema 來自 build_locked_test_sets.py：
        [{
            "event": "2023_KOINU",
            "event_type": "typhoon" | "squall",
            "station": "RCKT",
            "date": "20231003",
            "time": "0332",
            "elevation": "01",
            "raw": "/path/to/typhoonnew/test\\.../bvel_raw.01.gz",
            "gt":  "/path/to/typhoonnew/test\\.../bvel_sfda.01.gz",
            "filename": "RCKT.20231003.0332.bvel_raw.01",
            "alias_ratio": 0.0234,
            "alias_bin": "aliased",
            ...
        }, ...]

    優點：完全 deterministic（locked test set），跨模型 100% 一致。
    """
    if not os.path.exists(force_cases_file):
        print(f"[ERROR] realobs force cases file 不存在: {force_cases_file}")
        return []

    with open(force_cases_file, 'r', encoding='utf-8') as f:
        forced = json.load(f)
    print(f"\n[REALOBS FORCE] 載入 {len(forced)} 個 locked cases from {force_cases_file}")

    cases = []
    missing = 0
    for entry in forced:
        raw_path = entry['raw']
        gt_path  = entry['gt']

        # 驗證檔案存在（locked JSON 內 path 是當時生成時的環境）
        if not os.path.exists(raw_path):
            missing += 1
            continue
        if not os.path.exists(gt_path):
            missing += 1
            continue

        # 轉成 test loop 預期的 case dict 格式
        event_type = entry.get('event_type', 'typhoon')
        case_type = 'squall_line' if event_type == 'squall' else 'typhoon'

        cases.append({
            'raw':       raw_path,
            'da':        gt_path,
            'station':   entry['station'],
            'date':      entry.get('date', ''),
            'time':      entry.get('time', ''),
            'filename':  entry.get('filename', ''),
            'nwp_dir':   entry.get('event', ''),  # 用 event 當 nwp_dir 組織 output
            'case_type': case_type,
            # nyquist override（僅供檔頭無 Nyquist 之舊資料，如 2007 bvel；
            # 2019+ 檔頭 Nyquist 有效時此值不會被使用，見 test_single_case）
            'nyquist': entry.get('nyquist', None),
            # 額外 metadata（供下游 per-case analysis）
            '_locked_alias_ratio': entry.get('alias_ratio', None),
            '_locked_alias_bin':   entry.get('alias_bin', None),
            '_locked_elevation':   entry.get('elevation', ''),
        })

    if missing > 0:
        print(f"   ⚠️  跳過 {missing} 個檔案缺失的 case (path 在 locked JSON 但本機沒檔)")
    print(f"   ✅ 有效 cases: {len(cases)}")
    return cases


def load_parquet_as_matrix(parquet_path, debug=False):
    """
    載入 Parquet 並轉換為極坐標矩陣

    Returns:
        velocity_matrix: (H, W) 速度矩陣
        nyquist: Nyquist 速度
        metadata: 額外資訊
    """
    table = pq.read_table(parquet_path)
    df = table.to_pandas()

    # 提取 Nyquist 速度
    if 'Nyquist' in df.columns:
        nyquist = df['Nyquist'].iloc[0]
    elif 'nyquist' in df.columns:
        nyquist = df['nyquist'].iloc[0]
    else:
        nyquist = 33.62

    az_col = 'Azimuth' if 'Azimuth' in df.columns else 'azimuth'
    r_col = 'Range' if 'Range' in df.columns else 'range'
    value_col = 'Value' if 'Value' in df.columns else 'value'

    azimuths = df[az_col].unique()
    ranges = sorted(df[r_col].unique())

    n_az, n_range = len(azimuths), len(ranges)
    velocity_matrix = np.full((n_az, n_range), np.nan)

    azimuth_to_idx = {az: idx for idx, az in enumerate(azimuths)}

    for _, row in df.iterrows():
        az_normalized = row[az_col] % 360.0
        az_idx = azimuth_to_idx.get(az_normalized)
        if az_idx is None:
            az_idx = np.argmin(np.abs(azimuths - az_normalized))
        r_idx = ranges.index(row[r_col])
        val = row[value_col]
        if val != -999.0 and not np.isnan(val):
            velocity_matrix[az_idx, r_idx] = val

    metadata = {
        'shape': velocity_matrix.shape,
        'nyquist': nyquist,
        'n_azimuths': n_az,
        'n_ranges': n_range,
    }

    if debug:
        valid_count = np.sum(~np.isnan(velocity_matrix))
        print(f"    DEBUG - shape: {velocity_matrix.shape}, valid: {valid_count}/{velocity_matrix.size}, "
              f"range: [{np.nanmin(velocity_matrix):.2f}, {np.nanmax(velocity_matrix):.2f}]")

    return velocity_matrix, nyquist, metadata


# ---------------------------------------------------------------------------
# 2. H5 test split 提取 & NWP 掃描
# ---------------------------------------------------------------------------

def extract_test_sources(h5_path):
    """
    從 H5 test split 提取所有 test source 檔名。

    支援兩種 H5 格式：
      1. 新版（array 格式 v4.0）：test/source_file 是 1D dataset
      2. 舊版（per-group）：test/{pname}/source per patch
    """
    test_sources = set()
    with h5py.File(h5_path, 'r') as f:
        test_grp = f['test']
        # 偵測格式：array 版的 source_file 是 dataset
        if 'source_file' in test_grp and isinstance(test_grp['source_file'], h5py.Dataset):
            raw = test_grp['source_file'][:]
            # 向量化 decode
            if raw.dtype.kind == 'S':
                test_sources = set(np.char.decode(raw, 'utf-8'))
            else:
                test_sources = set(
                    (s.decode('utf-8') if isinstance(s, bytes) else str(s))
                    for s in raw
                )
        else:
            # 舊版 per-group 格式
            for pname in test_grp.keys():
                src = test_grp[pname]['source'][()]
                if isinstance(src, bytes):
                    src = src.decode('utf-8')
                test_sources.add(src)
    print(f"[INFO] H5 test split: {len(test_sources)} unique sources")
    return test_sources


def find_nwp_triplets(nwp_dir, skip_first_n=6):
    """
    尋找 NWP 資料三元組 (raw, da, qc)
    跳過每個 station+elevation 的前 N 個 (spin-up 問題)
    """
    triplets = []
    nwp_path = Path(nwp_dir)
    if not nwp_path.exists():
        return triplets

    raw_files = list(nwp_path.glob("*.bvel_raw.*.parquet"))
    grouped_files = defaultdict(list)

    for raw_file in raw_files:
        parts = raw_file.name.split('.')
        station = parts[0]
        date = parts[1]
        time_str = parts[2]
        elevation = parts[-2]
        group_key = f"{station}_{elevation}"
        grouped_files[group_key].append({
            'file': raw_file,
            'station': station,
            'date': date,
            'time': time_str,
            'elevation': elevation,
            'timestamp': f"{date}{time_str}_{elevation}",
        })

    for group_key, files in grouped_files.items():
        files_sorted = sorted(files, key=lambda x: x['timestamp'])
        files_to_use = files_sorted[skip_first_n:]

        for file_info in files_to_use:
            raw_file = file_info['file']
            da_file = Path(str(raw_file).replace('bvel_raw', 'bvel_da'))

            if da_file.exists():
                triplets.append({
                    'raw': str(raw_file),
                    'da': str(da_file),
                    'station': file_info['station'],
                    'date': file_info['date'],
                    'time': file_info['time'],
                    'elevation': file_info['elevation'],
                    'filename': raw_file.name,
                    'nwp_dir': nwp_path.name,
                })

    return triplets


def _quick_aliasing_check(raw_path, da_path):
    """
    快速檢查是否有 aliasing (只讀 parquet Value 欄，不建完整矩陣)

    Returns:
        (has_aliasing, aliased_ratio, valid_count, nyquist) or None on failure
    """
    try:
        raw_table = pq.read_table(raw_path, columns=['Value', 'Nyquist'])
        da_table = pq.read_table(da_path, columns=['Value'])
        raw_df = raw_table.to_pandas()
        da_df = da_table.to_pandas()

        raw_vals = raw_df['Value'].values.astype(np.float64)
        da_vals = da_df['Value'].values.astype(np.float64)
        nyquist = float(raw_df['Nyquist'].iloc[0])

        # Mask invalid
        valid = (raw_vals != -999.0) & (da_vals != -999.0) & np.isfinite(raw_vals) & np.isfinite(da_vals)
        valid_count = int(np.sum(valid))
        if valid_count < 100:
            return None

        diff = np.abs(raw_vals[valid] - da_vals[valid])
        aliased_pixels = int(np.sum(diff > nyquist * 0.5))
        aliased_ratio = aliased_pixels / valid_count

        return (aliased_ratio > 0.01, float(aliased_ratio), valid_count, nyquist)
    except Exception:
        return None


def build_forced_cases(nwp_root, force_cases_file, test_sources):
    """
    從 JSON 檔案載入指定案例，直接構建 triplets（跳過隨機選取）。

    JSON 格式: [{"nwp_dir": "24072417", "file": "RCWF.20240724.2020.bvel_raw.01"}, ...]

    Returns:
        test_cases: list of triplet dicts (與 select_test_cases 相同格式)
    """
    with open(force_cases_file, 'r') as f:
        forced = json.load(f)

    print(f"\n[FORCE] 載入 {len(forced)} 個指定案例 from {force_cases_file}")

    test_cases = []
    missing = []
    leaked = []

    for entry in forced:
        nwp_dir_name = entry['nwp_dir']
        file_stem = entry['file']  # e.g. RCWF.20240724.2020.bvel_raw.01
        filename_parquet = file_stem + '.parquet'

        # 解析 station/date/time
        parts = file_stem.split('.')
        station = parts[0]
        date = parts[1]
        time_str = parts[2]
        elevation = parts[-1]  # e.g. "01"

        raw_path = Path(nwp_root) / nwp_dir_name / filename_parquet
        da_filename = file_stem.replace('bvel_raw', 'bvel_da') + '.parquet'
        da_path = Path(nwp_root) / nwp_dir_name / da_filename

        if not raw_path.exists() or not da_path.exists():
            missing.append(f"{nwp_dir_name}/{filename_parquet}")
            continue

        # 洩漏檢查
        if filename_parquet not in test_sources:
            leaked.append(f"{nwp_dir_name}/{filename_parquet}")
            continue

        # 快速 aliasing 檢查
        result = _quick_aliasing_check(str(raw_path), str(da_path))
        aliased_ratio = result[1] if result else 0.0
        valid_count = result[2] if result else 0
        nyquist = result[3] if result else 33.62

        # 判斷是否為 REQUIRED_CASES
        required_keys = {(c['nwp_dir'], c['file'] + '.parquet') for c in REQUIRED_CASES}
        is_required = (nwp_dir_name, filename_parquet) in required_keys

        test_cases.append({
            'raw': str(raw_path),
            'da': str(da_path),
            'station': station,
            'date': date,
            'time': time_str,
            'elevation': elevation,
            'filename': filename_parquet,
            'nwp_dir': nwp_dir_name,
            'aliased_ratio': aliased_ratio,
            'valid_pixels': valid_count,
            'nyquist': nyquist,
        })

    if missing:
        print(f"[WARNING] {len(missing)} 個案例檔案不存在:")
        for m in missing[:5]:
            print(f"   {m}")
        if len(missing) > 5:
            print(f"   ... 共 {len(missing)} 個")

    if leaked:
        print(f"[ERROR] {len(leaked)} 個案例不在 test split 中 (洩漏):")
        for l in leaked:
            print(f"   {l}")
        return []

    print(f"[OK] 成功構建 {len(test_cases)} 個指定案例 (0 洩漏, {len(missing)} 缺失)")

    # 印出站分佈
    station_counts = Counter(t['station'] for t in test_cases)
    for st, cnt in sorted(station_counts.items()):
        print(f"   {st}: {cnt}")

    return test_cases


def select_test_cases(nwp_root, test_sources, max_cases=100, random_seed=42, clean_ratio=0.0):
    """
    掃描 NWP 目錄，與 H5 test split 交叉比對，選取測試案例

    策略 (效能優化):
    1. 掃描所有 NWP 目錄取得 triplets
    2. 只保留在 H5 test split 中的 source
    3. 先按站分組 + 隨機打亂
    4. 每站取候選 (per_station × 3 倍餘裕)
    5. 只對候選做 aliasing 快速檢查 (避免掃描全部 15000+)
    6. 確保 5 個指定案例
    """
    print(f"\n[STEP 1] 掃描 NWP 目錄: {nwp_root}")
    all_triplets = []
    nwp_dirs = sorted([d for d in Path(nwp_root).iterdir() if d.is_dir()])
    print(f"   找到 {len(nwp_dirs)} 個子目錄")

    for nwp_d in nwp_dirs:
        triplets = find_nwp_triplets(str(nwp_d), skip_first_n=6)
        all_triplets.extend(triplets)

    print(f"   共 {len(all_triplets)} 個 triplets (raw+da)")

    # 與 test split 交叉比對
    test_triplets = []
    for t in all_triplets:
        if t['filename'] in test_sources:
            test_triplets.append(t)

    print(f"   在 test split 中: {len(test_triplets)} 個")

    rng = np.random.RandomState(random_seed)

    # 分離指定案例 — 用 (nwp_dir, filename) 精確匹配
    required_keys = {(c['nwp_dir'], c['file'] + '.parquet') for c in REQUIRED_CASES}
    required_candidates = [t for t in test_triplets
                           if (t['nwp_dir'], t['filename']) in required_keys]
    other_triplets = [t for t in test_triplets
                      if (t['nwp_dir'], t['filename']) not in required_keys]

    # 按站分組
    station_triplets = defaultdict(list)
    for t in other_triplets:
        station_triplets[t['station']].append(t)

    n_stations = len(station_triplets)
    budget = max_cases - len(REQUIRED_CASES)
    per_station_target = max(1, budget // max(n_stations, 1))
    # 取 3 倍餘裕候選 (因為有些可能無 aliasing)
    per_station_candidates = per_station_target * 3

    print(f"\n[STEP 2] 每站抽樣候選 (target={per_station_target}/station, "
          f"candidates={per_station_candidates}/station)...")

    # 打亂並取候選
    candidates_by_station = {}
    for station, trips in sorted(station_triplets.items()):
        rng.shuffle(trips)
        candidates_by_station[station] = trips[:per_station_candidates]
        print(f"   {station}: {len(trips)} available -> {len(candidates_by_station[station])} candidates")

    # 合併所有候選 + 指定案例
    all_candidates = required_candidates.copy()
    for station, cands in sorted(candidates_by_station.items()):
        all_candidates.extend(cands)

    # 只對候選做 aliasing 快速檢查
    print(f"\n[STEP 3] 快速 aliasing 檢查 ({len(all_candidates)} candidates)...")
    checked = 0
    aliased_required = []
    aliased_by_station = defaultdict(list)
    clean_by_station = defaultdict(list)

    for t in all_candidates:
        result = _quick_aliasing_check(t['raw'], t['da'])
        checked += 1
        if checked % 100 == 0:
            print(f"   ... checked {checked}/{len(all_candidates)}")

        if result is None:
            continue

        has_aliasing, aliased_ratio, valid_count, nyquist = result
        t['aliased_ratio'] = aliased_ratio
        t['valid_pixels'] = valid_count
        t['nyquist'] = nyquist

        if (t['nwp_dir'], t['filename']) in required_keys:
            # 指定案例: 不論 aliasing 都加入
            aliased_required.append(t)
        elif has_aliasing:
            aliased_by_station[t['station']].append(t)
        else:
            # clean/low-alias case — 用於 FPR 評估
            clean_by_station[t['station']].append(t)

    print(f"   檢查完成: {checked} candidates")
    print(f"   指定案例: {len(aliased_required)}/{len(REQUIRED_CASES)}")
    for rf in aliased_required:
        print(f"      {rf['nwp_dir']}/{rf['filename']} (aliased_ratio={rf.get('aliased_ratio', 'N/A'):.3f})")

    # 指定案例如果沒在 candidates 找到 (不該發生), 強制加入
    found_req_keys = {(t['nwp_dir'], t['filename']) for t in aliased_required}
    for rc in REQUIRED_CASES:
        key = (rc['nwp_dir'], rc['file'] + '.parquet')
        if key not in found_req_keys:
            for t in test_triplets:
                if t['nwp_dir'] == rc['nwp_dir'] and t['filename'] == key[1]:
                    t['aliased_ratio'] = -1.0
                    t['valid_pixels'] = 0
                    t['nyquist'] = 33.62
                    aliased_required.append(t)
                    print(f"      [FORCE] {key[0]}/{key[1]}")
                    break

    # 分配 budget: aliased vs clean
    n_clean_target = int(budget * clean_ratio)
    n_aliased_target = budget - n_clean_target

    # 每站平衡選取 (aliased)
    n_aliased_stations = max(len(aliased_by_station), 1)
    per_station_aliased = max(1, n_aliased_target // n_aliased_stations)

    selected_aliased = []
    for station in sorted(aliased_by_station.keys()):
        cases = aliased_by_station[station]
        # 按 aliased_ratio 降序 (優先選 aliasing 多的)
        cases.sort(key=lambda x: x['aliased_ratio'], reverse=True)
        selected_aliased.extend(cases[:per_station_aliased])

    # 如果還有 aliased budget, 從剩餘中補
    if len(selected_aliased) < n_aliased_target:
        extra = []
        for station in sorted(aliased_by_station.keys()):
            cases = aliased_by_station[station]
            extra.extend(cases[per_station_aliased:])
        rng.shuffle(extra)
        selected_aliased.extend(extra[:n_aliased_target - len(selected_aliased)])

    selected_aliased = selected_aliased[:n_aliased_target]

    # 每站平衡選取 (clean) — 用於 FPR 評估
    selected_clean = []
    if n_clean_target > 0 and clean_by_station:
        n_clean_stations = max(len(clean_by_station), 1)
        per_station_clean = max(1, n_clean_target // n_clean_stations)
        for station in sorted(clean_by_station.keys()):
            cases = clean_by_station[station]
            rng.shuffle(cases)
            selected_clean.extend(cases[:per_station_clean])
        # 補足
        if len(selected_clean) < n_clean_target:
            extra_clean = []
            for station in sorted(clean_by_station.keys()):
                extra_clean.extend(clean_by_station[station][per_station_clean:])
            rng.shuffle(extra_clean)
            selected_clean.extend(extra_clean[:n_clean_target - len(selected_clean)])
        selected_clean = selected_clean[:n_clean_target]

    final_cases = aliased_required + selected_aliased + selected_clean

    print(f"\n[STEP 4] 最終選取: {len(final_cases)} 案例 "
          f"(required={len(aliased_required)}, aliased={len(selected_aliased)}, clean={len(selected_clean)})")
    station_counts = Counter(t['station'] for t in final_cases)
    for st, cnt in sorted(station_counts.items()):
        print(f"   {st}: {cnt}")

    return final_cases


# ---------------------------------------------------------------------------
# 3. 論文模型載入 & CWA 適配 (從 test_nexrad.py 複用 + 修改)
# ---------------------------------------------------------------------------

def _import_module_from_file(name, filepath):
    """動態載入模組"""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_paper_model(model_dir: str, dummy_naz=384, dummy_nrad=1216):
    """
    建立並載入論文的 UNet-VDA 模型
    (從 test_nexrad.py 複用，Keras 3 相容)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(script_dir, 'unet-vda-main', 'src')

    paper_fe = _import_module_from_file(
        'paper_feature_extraction',
        os.path.join(src_dir, 'feature_extraction.py'))

    dealias_path = os.path.join(src_dir, 'dealias.py')
    with open(dealias_path, 'r') as f:
        src_code = f.read()
    src_code = src_code.replace(
        'tf.keras.optimizers.legacy.Adam()',
        'tf.keras.optimizers.Adam()')
    paper_dealias = types.ModuleType('paper_dealias')
    paper_dealias.__file__ = dealias_path
    exec(compile(src_code, dealias_path, 'exec'), paper_dealias.__dict__)

    vel_input = tf.keras.Input(shape=(None, None, 1))
    down = paper_fe.create_downsampler(
        inp=vel_input, start_neurons=PAPER_SN, input_channels=1)
    up = paper_fe.create_upsampler(
        n_inputs=1, start_neurons=PAPER_SN, n_outputs=6)

    paper_model = paper_dealias.VelocityDealiaser(down, up)

    # Dummy forward 初始化
    dummy_vel = np.zeros((1, 1, dummy_naz, dummy_nrad, 1), dtype=np.float32)
    dummy_nyq = np.array([[[24.0]]], dtype=np.float32)
    _ = paper_model({'vel': dummy_vel, 'nyq': dummy_nyq}, training=False)

    # 從 checkpoint 載入變數
    ckpt_prefix = os.path.join(model_dir, 'variables', 'variables')
    all_ckpt_vars = tf.train.list_variables(ckpt_prefix)
    model_vars_ckpt = [(n, s) for n, s in all_ckpt_vars
                       if 'OPTIMIZER_SLOT' not in n and 'CHECKPOINTABLE' not in n]

    def _extract_var_idx(name):
        m = re.search(r'variables/(\d+)/', name)
        return int(m.group(1)) if m else -1

    model_vars_ckpt.sort(key=lambda x: _extract_var_idx(x[0]))

    assert len(model_vars_ckpt) == len(paper_model.variables), \
        f"Variable count mismatch: ckpt={len(model_vars_ckpt)} vs model={len(paper_model.variables)}"

    for (ckpt_name, _), model_var in zip(model_vars_ckpt, paper_model.variables):
        val = tf.train.load_variable(ckpt_prefix, ckpt_name)
        model_var.assign(val)

    print(f"[OK] 論文模型載入完成 (sn={PAPER_SN}, {len(model_vars_ckpt)} vars)")
    return paper_model


def run_paper_model_on_cwa(paper_model, raw_vel, nyquist):
    """
    用論文模型推理 CWA NWP 資料

    CWA 已經是 ~360 rays，不需要 720→360 split。
    流程:
    1. pad nrad to multiple of 64
    2. periodic azimuth padding +12 rays each side
    3. pad naz to multiple of 32 if needed
    4. reshape to (1, 1, naz_padded, nrad_padded, 1)
    5. paper_model inference
    6. strip padding → (naz, N_gates)

    Args:
        paper_model: 論文 VelocityDealiaser
        raw_vel: (naz, N_gates) 原始 CWA 速度
        nyquist: Nyquist 速度

    Returns:
        dealiased: (naz, N_gates) 解折疊結果
    """
    naz, n_gates = raw_vel.shape
    pad_deg = PAPER_PAD_DEG

    # 複製一份工作用
    work = raw_vel.copy()

    # Pad nrad to multiple of 64
    nrad_padded = math.ceil(n_gates / PAPER_NRAD_DIVISOR) * PAPER_NRAD_DIVISOR
    if nrad_padded > n_gates:
        pad_r = nrad_padded - n_gates
        work = np.pad(work, ((0, 0), (0, pad_r)),
                      mode='constant', constant_values=np.nan)

    # Periodic azimuth padding (+12 rays each side)
    work = np.concatenate([
        work[-pad_deg:, :],
        work,
        work[:pad_deg, :],
    ], axis=0)

    naz_with_pad = work.shape[0]  # naz + 2*pad_deg

    # Pad naz to multiple of 32 if needed
    naz_target = math.ceil(naz_with_pad / PAPER_NAZ_DIVISOR) * PAPER_NAZ_DIVISOR
    if naz_target > naz_with_pad:
        pad_az = naz_target - naz_with_pad
        work = np.pad(work, ((0, pad_az), (0, 0)),
                      mode='constant', constant_values=np.nan)

    # Replace bad values
    work[work <= -64] = np.nan

    # Reshape to (1, 1, naz_padded, nrad_padded, 1)
    vel_5d = work[None, None, :, :, None].astype(np.float32)
    nyq_arr = np.array([[[nyquist]]], dtype=np.float32)

    print(f"   [Paper] Input shape: {vel_5d.shape}, nyq: {nyq_arr.shape}")

    # 推理
    out = paper_model({'vel': vel_5d, 'nyq': nyq_arr}, training=False)
    dealiased = out['dealiased_vel'].numpy().copy()  # (1, naz_total, nrad_padded, 1)

    # Strip to (1, naz_with_pad, nrad_padded, 1)
    if naz_target > naz_with_pad:
        dealiased = dealiased[:, :naz_with_pad, :, :]

    # Strip azimuth padding → (1, naz, nrad_padded, 1)
    dealiased = dealiased[:, pad_deg:pad_deg + naz, :, :]

    # Strip range padding → (naz, n_gates)
    result = dealiased[0, :, :n_gates, 0]

    print(f"   [Paper] Output: {result.shape}, "
          f"range: [{np.nanmin(result):.1f}, {np.nanmax(result):.1f}]")

    return result


# ---------------------------------------------------------------------------
# 3b. 720-ray interleaved split（RCWF 等 720-ray 站專用）
# ---------------------------------------------------------------------------

_720_THRESHOLD = 500  # naz > 500 視為 720-ray 站


def split_720_to_halves(vel_720):
    """
    720-ray → 偶數半 (360) + 奇數半 (360)

    與論文 (UNet-VDA) 的 interleaved split 一致：
    偶數 ray (0,2,4,...,718) 和 奇數 ray (1,3,5,...,719) 分別推理。

    Args:
        vel_720: (720, N_gates)
    Returns:
        vel_even: (360, N_gates)  — ray 0, 2, 4, ...
        vel_odd:  (360, N_gates)  — ray 1, 3, 5, ...
    """
    return vel_720[0::2, :], vel_720[1::2, :]


def merge_halves_to_720(dealiased_even, dealiased_odd, raw_720, nyquist):
    """
    將兩組 360-ray 推理結果合併回 720-ray（基於 fold）

    策略：
    1. 各自算出 fold = round((pred - raw) / 2Nyq)
    2. 偶數 fold → ray 0,2,4,...
    3. 奇數 fold → ray 1,3,5,...
    4. 套用 fold 到原始 720-ray raw velocity

    Args:
        dealiased_even: (360, N_gates) 偶數半推理結果（已 constrained）
        dealiased_odd:  (360, N_gates) 奇數半推理結果（已 constrained）
        raw_720: (720, N_gates) 原始 720-ray 速度
        nyquist: float
    Returns:
        dealiased_720: (720, N_gates)
    """
    naz_720, n_gates = raw_720.shape
    dealiased_720 = np.full_like(raw_720, np.nan)

    for half_pred, raw_half, start_idx in [
        (dealiased_even, raw_720[0::2, :], 0),
        (dealiased_odd,  raw_720[1::2, :], 1),
    ]:
        valid = ~np.isnan(half_pred) & ~np.isnan(raw_half)
        fold = np.full_like(raw_half, np.nan)
        fold[valid] = np.round((half_pred[valid] - raw_half[valid]) / (2 * nyquist))

        # 套用 fold 到對應的 720-ray 位置
        valid_fold = ~np.isnan(raw_half) & ~np.isnan(fold)
        result_half = np.full_like(raw_half, np.nan)
        result_half[valid_fold] = raw_half[valid_fold] + fold[valid_fold] * 2.0 * nyquist
        dealiased_720[start_idx::2, :] = result_half

    return dealiased_720


# ---------------------------------------------------------------------------
# 4. 我們的模型推理
# ---------------------------------------------------------------------------

# ── Softmax 信心統計 helper（口委#8 confidence score）────────────────────
LOW_CONF_MAXPROB = 0.5          # 逐像素 max softmax prob < 此值 → 低信心
_ALIAS_CLASSES = (1, 2, 4, 5)   # 折疊類（排除 cat0 無效、cat3 乾淨）


def _softmax_conf_accum(probs, valid_mask):
    """從 softmax 機率 (H,W,6) 算逐 case 信心累加量（供跨半掃描相加後平均）。
    只統計 valid_mask（raw 非 NaN）像素。回傳可相加的 sum/count dict。"""
    m = valid_mask & np.isfinite(probs).all(-1)
    if not np.any(m):
        return None
    p = probs[m]                                    # (Nvalid, 6)
    max_prob = p.max(-1)
    part = np.sort(p, axis=-1)
    margin = part[:, -1] - part[:, -2]
    ent = -(p * np.log(p + 1e-12)).sum(-1) / np.log(p.shape[-1])   # 正規化 0..1
    alias_conf = p[:, 1] + p[:, 2] + p[:, 4] + p[:, 5]
    is_corr = np.isin(p.argmax(-1), _ALIAS_CLASSES)  # 模型決定要折的像素
    return dict(
        n_valid=int(m.sum()),
        sum_max_prob=float(max_prob.sum()),
        sum_entropy=float(ent.sum()),
        sum_margin=float(margin.sum()),
        sum_alias_conf=float(alias_conf.sum()),
        n_low_conf=int((max_prob < LOW_CONF_MAXPROB).sum()),
        n_corrected=int(is_corr.sum()),
        sum_max_prob_corr=float(max_prob[is_corr].sum()) if is_corr.any() else 0.0,
        sum_entropy_corr=float(ent[is_corr].sum()) if is_corr.any() else 0.0,
    )


def finalize_confidence(accums):
    """合併多個（半掃描）累加量 → 逐 case 信心 scalar dict。
    case_uncertainty（平均熵，越高越該人工複檢）為主要 triage 量。"""
    accums = [a for a in accums if a]
    if not accums:
        return None
    S = {k: sum(a[k] for a in accums) for k in accums[0]}
    nv = max(S['n_valid'], 1)
    nc = max(S['n_corrected'], 1)
    return dict(
        n_valid=S['n_valid'],
        n_corrected=S['n_corrected'],
        frac_corrected=S['n_corrected'] / nv,
        mean_max_prob=S['sum_max_prob'] / nv,
        mean_entropy=S['sum_entropy'] / nv,
        mean_margin=S['sum_margin'] / nv,
        mean_alias_conf=S['sum_alias_conf'] / nv,
        frac_low_conf=S['n_low_conf'] / nv,
        mean_max_prob_on_corrected=S['sum_max_prob_corr'] / nc,
        mean_entropy_on_corrected=S['sum_entropy_corr'] / nc,
        case_uncertainty=S['sum_entropy'] / nv,
    )


def run_our_model_on_cwa(model, raw_vel, nyquist, conf_threshold=0.0,
                         return_confidence=False):
    """
    用我們的模型推理 CWA NWP 資料

    Args:
        model: 我們的 VelocityDealiaser
        raw_vel: (naz, N_gates) 原始速度
        nyquist: Nyquist 速度
        conf_threshold: float (0~1)
            > 0 時啟用 confidence gate：只有 alias confidence ≥ threshold 的像素才套用修正，
            其餘保留 raw velocity。可降低 FPR（過度修正）。
            0.0 = 關閉（預設行為）。

    Returns:
        dealiased: (naz, N_gates) 解折疊結果
    """
    H_orig, W_orig = raw_vel.shape
    padded, pad_h, pad_w = auto_zero_pad(raw_vel, layers=4, fill_value=np.nan)
    vel_5d = padded[None, None, :, :, None].astype(np.float32)
    nyq_2d = np.array([[nyquist]], dtype=np.float32)

    out = model({'vel': vel_5d, 'nyq': nyq_2d}, training=False)

    result = out['dealiased_vel'][0, :, :, 0].numpy()[:H_orig, :W_orig]

    conf_accum = None
    need_logits = (conf_threshold > 0.0 or return_confidence) and 'alias_mask' in out
    if need_logits:
        logits = out['alias_mask'][0, :H_orig, :W_orig, :].numpy()  # (H, W, 6)
        # Softmax
        logits_shifted = logits - logits.max(-1, keepdims=True)
        probs = np.exp(logits_shifted)
        probs /= probs.sum(-1, keepdims=True)
        # alias confidence = sum of aliased class probs (cat 1,2,4,5; exclude cat0 invalid, cat3 clean)
        alias_conf = probs[:, :, 1] + probs[:, :, 2] + probs[:, :, 4] + probs[:, :, 5]

        # ── Confidence Gate ──────────────────────────────────────────────────
        if conf_threshold > 0.0:
            # 低 confidence 的像素保留 raw velocity（不修正）
            low_conf = alias_conf < conf_threshold
            result = np.where(low_conf, raw_vel, result)
            n_gated = int(np.sum(low_conf & ~np.isnan(raw_vel)))
            n_valid = int(np.sum(~np.isnan(raw_vel)))
            print(f"   [Conf Gate thr={conf_threshold:.2f}] gated {n_gated:,}/{n_valid:,} pixels "
                  f"({n_gated/max(n_valid,1)*100:.1f}%) → kept raw")

        # ── 逐 case softmax 信心統計（口委#8 confidence score）──
        if return_confidence:
            conf_accum = _softmax_conf_accum(probs, ~np.isnan(raw_vel))

    return result, conf_accum


# ---------------------------------------------------------------------------
# 5. 物理約束
# ---------------------------------------------------------------------------

def apply_physics_constraint(pred_vel, raw_vel, nyquist):
    """強制 V = Vraw + 2n*Vnyq"""
    valid = ~np.isnan(raw_vel) & ~np.isnan(pred_vel)
    constrained = np.full_like(pred_vel, np.nan)
    if np.any(valid):
        n = np.round((pred_vel[valid] - raw_vel[valid]) / (2 * nyquist))
        constrained[valid] = raw_vel[valid] + n * (2.0 * nyquist)
    return constrained


def apply_vcorrected_postprocessing(pred_vel, raw_vel, nyquist,
                                    threshold=1.0, max_iter=3):
    """
    V_corrected 物理後處理：基於修正後速度的空間連續性

    物理原理：真實大氣風場是空間連續的，因此：
    - 正確的 dealiasing → V_corrected 與鄰居平滑 → 保留
    - FP（誤修 clean pixel）→ V_corrected 跳 ~2Vn → 回退為 fold=0
    - FN（漏修 aliased pixel）→ V_raw 跳 ~2Vn → 嘗試 fold=±1,±2 修正

    Parameters:
    -----------
    pred_vel : (H, W) 經 physics_constraint 後的預測速度
    raw_vel  : (H, W) 原始觀測速度
    nyquist  : float  Nyquist 速度
    threshold: float  跳躍閾值（Nyquist 倍數），預設 1.0
    max_iter : int    最大迭代次數

    Returns:
    --------
    result : (H, W) 後處理後的速度
    stats  : dict 統計資訊
    """
    valid = ~np.isnan(raw_vel) & ~np.isnan(pred_vel)
    result = pred_vel.copy()

    total_fp_reverted = 0
    total_fn_recovered = 0

    for it in range(max_iter):
        # 當前 fold
        fold = np.zeros_like(raw_vel)
        fold[valid] = np.round((result[valid] - raw_vel[valid]) / (2 * nyquist))

        v_corr = result.copy()

        # 4-鄰居：方位角循環 (axis=0)，距離門不循環 (axis=1)
        n_az_prev = np.roll(v_corr, 1, axis=0)
        n_az_next = np.roll(v_corr, -1, axis=0)
        n_rng_near = np.full_like(v_corr, np.nan)
        n_rng_near[:, 1:] = v_corr[:, :-1]
        n_rng_far = np.full_like(v_corr, np.nan)
        n_rng_far[:, :-1] = v_corr[:, 1:]

        neighbors = np.stack([n_az_prev, n_az_next, n_rng_near, n_rng_far],
                             axis=0)  # (4, H, W)

        # 當前 jump 統計（歸一化 by Nyquist）
        diffs = np.abs(v_corr[np.newaxis] - neighbors) / nyquist
        # min_jump: 至少有一個有效鄰居的最小跳躍（孤立 FP 時所有鄰居都大 → min 也大）
        # 邊界 TP 至少有一個同 fold 鄰居跳躍小 → min 小 → 不會被誤 revert
        min_jump = np.nanmin(diffs, axis=0)
        # 若所有鄰居均為 NaN（無效），設為 0 避免誤觸發
        no_valid_neighbors = np.all(np.isnan(diffs), axis=0)
        min_jump = np.where(no_valid_neighbors, 0.0, min_jump)

        iter_fp = 0
        iter_fn = 0

        # ── Phase 1: FP removal（僅 revert 孤立錯誤校正）──
        # 條件：fold≠0 且所有有效鄰居的 v_corr 跳躍均 > threshold（孤立像素）
        # 邊界 TP 至少一個鄰居是同 fold → min_jump 小 → 不被 revert
        has_fold = (fold != 0) & valid
        if np.any(has_fold):
            diffs_0 = np.abs(raw_vel[np.newaxis] - neighbors) / nyquist
            min_jump_0 = np.nanmin(diffs_0, axis=0)
            min_jump_0 = np.where(no_valid_neighbors, 0.0, min_jump_0)

            revert = has_fold & (min_jump > threshold) & (min_jump_0 < min_jump)
            iter_fp = int(np.sum(revert))
            if iter_fp > 0:
                result[revert] = raw_vel[revert]
                total_fp_reverted += iter_fp

        # ── Phase 2: FN recovery ──
        # Phase 1 可能已修改 result，重新計算 fold 和 neighbors
        fold2 = np.zeros_like(raw_vel)
        fold2[valid] = np.round((result[valid] - raw_vel[valid]) / (2 * nyquist))

        v_corr2 = result.copy()
        n2_az_prev = np.roll(v_corr2, 1, axis=0)
        n2_az_next = np.roll(v_corr2, -1, axis=0)
        n2_rng_near = np.full_like(v_corr2, np.nan)
        n2_rng_near[:, 1:] = v_corr2[:, :-1]
        n2_rng_far = np.full_like(v_corr2, np.nan)
        n2_rng_far[:, :-1] = v_corr2[:, 1:]
        neighbors2 = np.stack([n2_az_prev, n2_az_next, n2_rng_near, n2_rng_far],
                              axis=0)

        diffs2 = np.abs(v_corr2[np.newaxis] - neighbors2) / nyquist
        max_jump2 = np.nanmax(diffs2, axis=0)
        max_jump2 = np.where(np.isnan(max_jump2), 0.0, max_jump2)

        no_fold = (fold2 == 0) & valid & (max_jump2 > threshold)
        if np.any(no_fold):
            best_jump = max_jump2.copy()
            best_vel = result.copy()

            for try_fold in [1, -1, 2, -2]:
                v_try = raw_vel + try_fold * 2.0 * nyquist
                diffs_try = np.abs(v_try[np.newaxis] - neighbors2) / nyquist
                max_jump_try = np.nanmax(diffs_try, axis=0)
                max_jump_try = np.where(np.isnan(max_jump_try), 999.0,
                                        max_jump_try)

                better = no_fold & (max_jump_try < best_jump)
                best_jump = np.where(better, max_jump_try, best_jump)
                best_vel = np.where(better, v_try, best_vel)

            # 至少改善 0.1 Nyquist 才接受
            recover = no_fold & (best_jump < max_jump2 - 0.1)
            iter_fn = int(np.sum(recover))
            if iter_fn > 0:
                result[recover] = best_vel[recover]
                total_fn_recovered += iter_fn

        print(f"      [V_corr iter {it+1}] FP reverted: {iter_fp:,}, "
              f"FN recovered: {iter_fn:,}")

        if iter_fp + iter_fn == 0:
            break

    stats = {
        'fp_reverted': total_fp_reverted,
        'fn_recovered': total_fn_recovered,
        'iterations': it + 1,
    }
    return result, stats


# ---------------------------------------------------------------------------
# 6. 指標計算
# ---------------------------------------------------------------------------

def compute_raw_metrics(raw_vel, gt_vel):
    """RAW vs GT 基準指標"""
    mask = ~np.isnan(raw_vel) & ~np.isnan(gt_vel)
    if not np.any(mask):
        return {"rmse": np.nan, "mae": np.nan, "corr": np.nan, "valid_pixels": 0}
    r, g = raw_vel[mask], gt_vel[mask]
    rmse = np.sqrt(np.mean((r - g) ** 2))
    mae = np.mean(np.abs(r - g))
    corr = np.corrcoef(r, g)[0, 1] if len(r) > 1 else 0.0
    return {"rmse": float(rmse), "mae": float(mae), "corr": float(corr),
            "valid_pixels": int(np.sum(mask))}


# ---------------------------------------------------------------------------
# 7. 視覺化
# ---------------------------------------------------------------------------

def create_polar_plot(ax, img, cmap, norm, gate_spacing=250, start_range=2125, elev=0.5):
    """
    極座標雷達圖
    img: (naz, nrad)
    """
    rvec = np.linspace(start_range, start_range + gate_spacing * (img.shape[1] - 1), img.shape[1])
    azmax = 360 + 360 / img.shape[0]
    azvec = np.pi / 180 * np.linspace(0, azmax, img.shape[0] + 1)
    R, AZ = np.meshgrid(rvec, azvec)
    Z = np.concatenate((img, img[0:1, :]))
    elev_rad = elev * np.pi / 180
    X = R * np.sin(AZ) * np.cos(elev_rad)
    Y = R * np.cos(AZ) * np.cos(elev_rad)
    im = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm, shading='nearest')

    for r_km in range(50, 350, 50):
        r_m = r_km * 1000
        if r_m <= rvec[-1]:
            circ = plt.Circle((0, 0), r_m, linestyle='-', linewidth=0.4,
                              edgecolor=[.6, .6, .6], facecolor='none')
            ax.add_patch(circ)
    for az_deg in range(0, 360, 45):
        r_max = rvec[-1]
        x_end = r_max * np.sin(az_deg * np.pi / 180)
        y_end = r_max * np.cos(az_deg * np.pi / 180)
        ax.plot([0, x_end], [0, y_end], '-', color=[.6, .6, .6],
                alpha=0.5, linewidth=0.4)

    ax.set_aspect('equal')
    return im


def _km_formatter(x, pos):
    return f'{x / 1000:.0f}'


def plot_comparison(raw_vel, ours_vel, paper_vel, gt_vel, nyquist,
                    case_label, output_dir, ours_metrics, paper_metrics):
    """
    四欄極座標對比圖:
    RAW (Aliased) | Ours | Paper (UNet-VDA) | DA Ground Truth
    """
    vmax = max(abs(nyquist) * 2.5, 70)
    cmap = cm.get_cmap('seismic').copy()
    cmap.set_bad([.9, .9, .9])
    norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax)

    fig, axs = plt.subplots(1, 4, figsize=(22, 5.5))

    panels = [
        (raw_vel, 'RAW (Aliased)'),
        (ours_vel, f'Ours (RMSE={ours_metrics["rmse"]:.2f})'),
        (paper_vel, f'UNet-VDA (RMSE={paper_metrics["rmse"]:.2f})'),
        (gt_vel, 'DA Ground Truth'),
    ]

    for ax, (data, title) in zip(axs, panels):
        im = create_polar_plot(ax, data, cmap, norm)
        ax.set_title(title, fontsize=11)
        fmt = ticker.FuncFormatter(_km_formatter)
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        ax.set_xlabel('km')

    axs[0].set_ylabel('km')
    divider = make_axes_locatable(axs[-1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical', label='m/s')

    fig.suptitle(f'CWA NWP — {case_label}', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    safe_label = case_label.replace('/', '_').replace(' ', '_')
    out_path = os.path.join(output_dir, f"nwp_{safe_label}_comparison.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"   [VIZ] Saved: {out_path}")
    return out_path


def plot_error_polar(raw_vel, ours_vel, paper_vel, gt_vel, nyquist,
                     case_label, output_dir):
    """
    誤差極座標圖: Ours error | Paper error | |Paper err|-|Ours err|
    """
    ours_err = ours_vel - gt_vel
    paper_err = paper_vel - gt_vel
    err_lim = 2 * nyquist

    cmap_err = cm.get_cmap('RdBu_r').copy()
    cmap_err.set_bad([.9, .9, .9])
    norm_err = mpl.colors.Normalize(vmin=-err_lim, vmax=err_lim)

    diff = np.abs(paper_err) - np.abs(ours_err)
    cmap_diff = cm.get_cmap('RdYlGn').copy()
    cmap_diff.set_bad([.9, .9, .9])

    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))

    panels = [
        (ours_err, 'Ours Error (pred - GT)', cmap_err, norm_err),
        (paper_err, 'UNet-VDA Error (pred - GT)', cmap_err, norm_err),
        (diff, '|Paper err| - |Ours err|\n(Green=Ours better)', cmap_diff,
         mpl.colors.Normalize(vmin=-err_lim, vmax=err_lim)),
    ]

    for ax, (data, title, c, n) in zip(axs, panels):
        im = create_polar_plot(ax, data, c, n)
        ax.set_title(title, fontsize=10)
        fmt = ticker.FuncFormatter(_km_formatter)
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        ax.set_xlabel('km')
    axs[0].set_ylabel('km')

    fig.suptitle(f'Error Analysis — {case_label}', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    safe_label = case_label.replace('/', '_').replace(' ', '_')
    out_path = os.path.join(output_dir, f"nwp_{safe_label}_error.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"   [VIZ] Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# 7b. 地理可視化（直接複用 batch_test_nwp 的實作）
# ---------------------------------------------------------------------------

def _align_radar_to_velocity(radar_data, vel_matrix):
    """
    對齊 radar Radar 物件 ray-axis 與 velocity matrix 形狀（用於 geo viz）。

    用於 RCWF 720-ray + 360-ray velocity 情境：原始 .gz 讀進來是 720-ray 幾何，
    但 velocity 已 azimuth-aware 對齊到 GT 360（標準 0°起始, 1° spacing）。
    舊版 [::2] 只是 subsample raw azimuth (e.g. [69.71°, 70.71°, ...])，
    與 GT 標準排序 (0°, 1°, ...) 不一致，畫出來位置會錯。

    新版：重新生成 azimuth 為 GT 標準 (0° + j*1°)，跟 vel_matrix 物理位置一致。
    """
    import numpy as _np
    n_vel = vel_matrix.shape[0]
    n_radar = radar_data.azimuth['data'].shape[0]
    if n_radar == n_vel:
        return radar_data
    if n_radar != 2 * n_vel:
        print(f"   [GEO WARN] radar ray={n_radar} vs vel ray={n_vel}, 無法自動對齊")
        return radar_data

    # 重新生成 azimuth 為 GT 標準排序（0° 起始、360/n_vel 間隔）
    new_azm_sp = 360.0 / n_vel
    radar_data.azimuth['data']   = _np.arange(n_vel, dtype='float32') * new_azm_sp
    # elevation 保持原值（取前 n_vel 個即可）
    radar_data.elevation['data'] = radar_data.elevation['data'][:n_vel].copy()
    radar_data.time['data']      = radar_data.time['data'][:n_vel].copy()
    radar_data.sweep_end_ray_index['data'] = _np.array([n_vel - 1], dtype='int32')
    try:
        radar_data.nrays = n_vel
    except AttributeError:
        pass
    # 移除既有 720-ray fields data（避免 shape 衝突，反正會被新的 360 vel field 覆蓋）
    for k in list(radar_data.fields.keys()):
        f = radar_data.fields[k]
        if hasattr(f.get('data', None), 'shape') and f['data'].shape[0] == n_radar:
            f['data'] = f['data'][:n_vel].copy()
    return radar_data


def _render_geo_to_image(vel_matrix, gz_path, field_name, title, shape_path=None,
                         vmin=-80, vmax=80, use_cmap=None, use_norm=None, dpi=150):
    """獨立渲染一張 geo 圖到 PIL Image（不用 subplot，避免 basemap 空白問題）"""
    import io
    from PIL import Image as PILImage
    from pyart.graph import RadarMapDisplayBasemap
    from mixed_patch_test_model import read_cwb_radar_sweep, plot_taiwan_basemap, get_metadata
    from batch_test_nwp import CWA_VEL_CMAP, CWA_VEL_NORM

    if use_cmap is None:
        use_cmap = CWA_VEL_CMAP
    if use_norm is None:
        use_norm = CWA_VEL_NORM

    radar_data, radius_km, _ = read_cwb_radar_sweep(gz_path)
    # 處理 720-ray radar + 360-ray velocity 形狀錯位
    radar_data = _align_radar_to_velocity(radar_data, vel_matrix)
    lat = radar_data.latitude['data'][0]
    lon = radar_data.longitude['data'][0]

    vel_field = get_metadata('velocity')
    vel_field['data'] = vel_matrix
    radar_data.fields[field_name] = vel_field

    m = plot_taiwan_basemap(lon, lat, radius_km, shape_path)
    display = RadarMapDisplayBasemap(radar_data)
    display.plot_ppi_map(
        field_name, sweep=0, resolution='h', vmin=vmin, vmax=vmax,
        cmap=use_cmap, norm=use_norm,
        min_lon=lon - 1, max_lon=lon + 1,
        min_lat=lat - 1, max_lat=lat + 1,
        mask_outside=True, projection='aeqd', basemap=m
    )
    display.plot_range_rings([radius_km[-1]])
    plt.title(title, fontsize=11)

    buf = io.BytesIO()
    plt.savefig(buf, dpi=dpi, bbox_inches='tight', format='png')
    plt.close()
    buf.seek(0)
    return PILImage.open(buf).copy()


def _stitch_images_horizontal(images, suptitle=None, dpi=200):
    """將多張 PIL Image 水平拼接，可加總標題"""
    from PIL import Image as PILImage, ImageDraw, ImageFont

    # 統一高度
    max_h = max(img.height for img in images)
    resized = []
    for img in images:
        if img.height != max_h:
            ratio = max_h / img.height
            new_w = int(img.width * ratio)
            img = img.resize((new_w, max_h), PILImage.LANCZOS)
        resized.append(img)

    total_w = sum(img.width for img in resized)

    title_h = 40 if suptitle else 0
    canvas = PILImage.new('RGB', (total_w, max_h + title_h), (255, 255, 255))

    if suptitle:
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except (OSError, IOError):
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), suptitle, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(((total_w - tw) // 2, 8), suptitle, fill=(0, 0, 0), font=font)

    x_offset = 0
    for img in resized:
        canvas.paste(img, (x_offset, title_h))
        x_offset += img.width

    return canvas


def plot_geo_comparison(raw_vel, ours_vel, paper_vel, gt_vel,
                        triplet, case_label, output_dir,
                        ours_metrics=None, paper_metrics=None, shape_path=None):
    """
    氣象署風格 geo comparison 合圖：1×4（每 panel 獨立渲染再拼接）
    RAW (Aliased) | Ours | UNet-VDA | DA Ground Truth
    輸出: nwp_{case}_geo_comparison.png
    """
    if not _ensure_geo_imports():
        print("   [GEO] 不可用，跳過地理圖")
        return

    raw_path = triplet['raw']
    raw_gz = str(raw_path) if str(raw_path).endswith('.gz') else find_corresponding_gz_file(raw_path)
    if raw_gz is None:
        print("   [GEO] 找不到對應 .gz 檔案，跳過")
        return

    ours_rmse = f" (RMSE={ours_metrics['rmse']:.2f})" if ours_metrics else ""
    paper_rmse = f" (RMSE={paper_metrics['rmse']:.2f})" if paper_metrics else ""

    panels = [
        (raw_vel, 'raw_vel', 'RAW (Aliased)'),
        (ours_vel, 'ours_vel', f'Ours{ours_rmse}'),
        (paper_vel, 'paper_vel', f'UNet-VDA{paper_rmse}'),
        (gt_vel, 'gt_vel', 'DA Ground Truth'),
    ]

    try:
        images = []
        for vel, field_name, title in panels:
            img = _render_geo_to_image(vel, raw_gz, field_name, title,
                                       shape_path=shape_path)
            images.append(img)

        canvas = _stitch_images_horizontal(
            images, suptitle=f'CWA NWP (Geo) — {case_label}')

        safe_label = case_label.replace('/', '_').replace(' ', '_')
        out_path = os.path.join(output_dir, f"nwp_{safe_label}_geo_comparison.png")
        canvas.save(out_path, dpi=(200, 200))
        print(f"   [GEO] Saved: {out_path}")
    except Exception as e:
        print(f"   [GEO] comparison 失敗: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')


def plot_geo_error(raw_vel, ours_vel, paper_vel, gt_vel, nyquist,
                   triplet, case_label, output_dir, shape_path=None):
    """
    氣象署風格 geo error 合圖：1×3（每 panel 獨立渲染再拼接）
    Ours Error | UNet-VDA Error | |Paper err|-|Ours err|
    輸出: nwp_{case}_geo_error.png
    """
    if not _ensure_geo_imports():
        return

    raw_path = triplet['raw']
    raw_gz = str(raw_path) if str(raw_path).endswith('.gz') else find_corresponding_gz_file(raw_path)
    if raw_gz is None:
        return

    ours_err = ours_vel - gt_vel
    paper_err = paper_vel - gt_vel
    diff = np.abs(paper_err) - np.abs(ours_err)
    err_lim = 2 * nyquist

    cmap_err = cm.get_cmap('RdBu_r').copy()
    cmap_err.set_bad([.9, .9, .9])
    norm_err = mpl.colors.Normalize(vmin=-err_lim, vmax=err_lim)

    cmap_diff = cm.get_cmap('RdYlGn').copy()
    cmap_diff.set_bad([.9, .9, .9])
    norm_diff = mpl.colors.Normalize(vmin=-err_lim, vmax=err_lim)

    panels = [
        (ours_err, 'ours_err', 'Ours Error (pred - GT)', cmap_err, norm_err),
        (paper_err, 'paper_err', 'UNet-VDA Error (pred - GT)', cmap_err, norm_err),
        (diff, 'err_diff', '|Paper err| - |Ours err|\n(Green=Ours better)', cmap_diff, norm_diff),
    ]

    try:
        images = []
        for vel, field_name, title, c, n in panels:
            img = _render_geo_to_image(vel, raw_gz, field_name, title,
                                       shape_path=shape_path,
                                       vmin=-err_lim, vmax=err_lim,
                                       use_cmap=c, use_norm=n)
            images.append(img)

        canvas = _stitch_images_horizontal(
            images, suptitle=f'Error Analysis (Geo) — {case_label}')

        safe_label = case_label.replace('/', '_').replace(' ', '_')
        out_path = os.path.join(output_dir, f"nwp_{safe_label}_geo_error.png")
        canvas.save(out_path, dpi=(200, 200))
        print(f"   [GEO] Saved: {out_path}")
    except Exception as e:
        print(f"   [GEO] error 失敗: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')


# ---------------------------------------------------------------------------
# 8. 逐案測試
# ---------------------------------------------------------------------------

def test_single_case(model, paper_model, triplet, output_dir, enable_viz,
                     conf_threshold=0.0,
                     enable_vcorrected_pp=False, vcorrected_threshold=1.0,
                     enable_geo_viz=False, shape_path=None,
                     downsample_720=False, save_fields=False,
                     save_confidence=False):
    """測試單一 NWP 案例 (Ours + Paper)"""
    station = triplet['station']
    date = triplet.get('date', '')
    time_str = triplet.get('time', '')
    nwp_dir = triplet.get('nwp_dir', '')
    # 從 filename 抓仰角（e.g. RCKT.20230828.2352.bvel_raw.01 → '01'）
    # 必須加進 case_label，否則同站同時同檔不同仰角的圖會互相覆蓋
    _fn = triplet.get('filename', '')
    _elev = _fn.rsplit('.', 1)[-1] if '.' in _fn and _fn.rsplit('.', 1)[-1].isdigit() else ''
    case_label = f"{station}.{date}.{time_str}.elev{_elev}" if _elev else f"{station}.{date}.{time_str}"

    print(f"\n{'='*70}")
    print(f"[CASE] {case_label} (dir={nwp_dir})")
    print(f"   RAW: {triplet['raw']}")
    print(f"   DA:  {triplet['da']}")
    print(f"{'='*70}")

    # 1. 讀取（自動偵測 .gz 或 .parquet）
    print("\n[1/5] 讀取資料...")
    raw_vel, nyquist, raw_meta = load_velocity_data(triplet['raw'])
    da_vel, _, da_meta = load_velocity_data(triplet['da'])

    # 檔頭無 Nyquist（≤0，如 2007 bvel）時，改用 case 帶入之 nyquist override。
    # 2019+ 檔頭 Nyquist 有效 → 此分支不觸發 → locked-200 行為完全不變。
    if (nyquist is None or nyquist <= 0) and triplet.get('nyquist'):
        nyquist = float(triplet['nyquist'])
        print(f"   [nyquist override] 檔頭無 Nyquist，改用 case 值 {nyquist:.2f}")

    print(f"   RAW: shape={raw_vel.shape}, Nyquist={nyquist:.2f}, "
          f"azm_start={raw_meta.get('azm_start', 0):.2f}")
    print(f"   DA:  shape={da_vel.shape}, "
          f"azm_start={da_meta.get('azm_start', 0):.2f}")

    # ── RCWF 720-ray + GT 360-ray 處理（azimuth-aware 對齊）──
    # 舊邏輯 raw[::2] 假設 raw/GT 同 azimuth 起點，但 RCWF colza GT 已 normalized
    # 從 0° 開始、raw 從掃描起點開始（e.g. 69.71°），錯位達 70° → label/metric 全錯。
    # 新邏輯：raw 保留 720 給 inference，用 azimuth nearest-neighbor 對齊產生 raw_eval (360)。
    is_720_to_360 = (raw_vel.shape[0] == 720 and da_vel.shape[0] == 360)

    if is_720_to_360:
        print(f"   [720+360] azimuth-aware 對齊...")
        # range gate 不對齊則裁切，naz 保留差異
        if raw_vel.shape[1] != da_vel.shape[1]:
            min_rng = min(raw_vel.shape[1], da_vel.shape[1])
            raw_vel = raw_vel[:, :min_rng]
            da_vel = da_vel[:, :min_rng]
        # 用 azimuth 對齊把 raw 720 → raw_eval 360（每個 GT ray 對應 raw 中最近的 ray）
        raw_eval = align_raw_to_gt_by_azimuth(
            raw_vel,
            raw_azm_start=raw_meta.get('azm_start', 0.0),
            raw_azm_sp   =raw_meta.get('azm_sp', 0.5),
            gt_azm_start =da_meta.get('azm_start', 0.0),
            gt_azm_sp    =da_meta.get('azm_sp', 1.0),
            gt_naz       =da_vel.shape[0],
        )
        # GT 物理約束基於對齊後的 raw_eval
        gt_vel = apply_physics_constraint(da_vel, raw_eval, nyquist)
        # raw_vel 保持 720 給 inference
    else:
        # 一般情況：處理形狀不對齊
        if raw_vel.shape != da_vel.shape:
            print(f"[WARNING] 形狀不一致: raw={raw_vel.shape} vs da={da_vel.shape}")
            min_az = min(raw_vel.shape[0], da_vel.shape[0])
            min_rng = min(raw_vel.shape[1], da_vel.shape[1])
            raw_vel = raw_vel[:min_az, :min_rng]
            da_vel = da_vel[:min_az, :min_rng]
        gt_vel = apply_physics_constraint(da_vel, raw_vel, nyquist)
        raw_eval = raw_vel  # 一般情況 raw_eval 就是 raw_vel

    valid_gt = ~np.isnan(raw_eval) & ~np.isnan(gt_vel)
    print(f"   有效像素: {np.sum(valid_gt):,}/{gt_vel.size:,}")

    if np.sum(valid_gt) < 100:
        print("[SKIP] 有效像素太少")
        return None

    # ── Aliased ratio 計算 (供下游 subset 分析使用) ──
    # 折錯定義：|raw - gt| > 0.5 * Nyquist 視為被折錯像素
    diff_abs = np.abs(raw_eval[valid_gt] - gt_vel[valid_gt])
    aliased_pixels = int(np.sum(diff_abs > nyquist * 0.5))
    valid_count = int(np.sum(valid_gt))
    aliased_ratio = float(aliased_pixels / valid_count) if valid_count else 0.0
    print(f"   Aliased pixels: {aliased_pixels:,}/{valid_count:,} ({aliased_ratio:.2%})")

    # ── 720-ray interleaved split 推理判斷 ──
    # 720+360 case 強制走 interleaved split；其他情況依 downsample_720 旗標 + 閾值判斷
    naz_orig = raw_vel.shape[0]
    is_720 = is_720_to_360 or (downsample_720 and naz_orig > _720_THRESHOLD)
    if is_720:
        print(f"   [720-ray] 偵測到 {naz_orig}-ray，啟用 interleaved split 推理")
        raw_even, raw_odd = split_720_to_halves(raw_vel)
        print(f"   [720-ray] split → even {raw_even.shape}, odd {raw_odd.shape}")

    # 2. 我們的模型推理
    print("\n[2/5] Ours 模型推理...")
    conf_accums = []
    if is_720:
        # 兩組各自推理
        ours_even, cacc_even = run_our_model_on_cwa(model, raw_even, nyquist,
                                         conf_threshold=conf_threshold,
                                         return_confidence=save_confidence)
        ours_odd, cacc_odd  = run_our_model_on_cwa(model, raw_odd, nyquist,
                                         conf_threshold=conf_threshold,
                                         return_confidence=save_confidence)
        conf_accums = [cacc_even, cacc_odd]
        # 各自 physics constraint → 合併回 720
        ours_even_c = apply_physics_constraint(ours_even, raw_even, nyquist)
        ours_odd_c  = apply_physics_constraint(ours_odd, raw_odd, nyquist)
        ours_constrained = merge_halves_to_720(ours_even_c, ours_odd_c, raw_vel, nyquist)
        print(f"   [720-ray] Ours 合併完成: {ours_constrained.shape}")
    else:
        ours_raw, cacc = run_our_model_on_cwa(model, raw_vel, nyquist,
                                        conf_threshold=conf_threshold,
                                        return_confidence=save_confidence)
        conf_accums = [cacc]
        ours_constrained = apply_physics_constraint(ours_raw, raw_vel, nyquist)

    # 2b. V_corrected 物理後處理
    vcorr_stats = None
    if enable_vcorrected_pp:
        print("\n[2b/5] V_corrected 物理後處理...")
        ours_constrained, vcorr_stats = apply_vcorrected_postprocessing(
            ours_constrained, raw_vel, nyquist,
            threshold=vcorrected_threshold)
        print(f"   結果: FP回退 {vcorr_stats['fp_reverted']:,} 像素, "
              f"FN恢復 {vcorr_stats['fn_recovered']:,} 像素, "
              f"迭代 {vcorr_stats['iterations']} 次")

    # 3. 論文模型推理
    print("\n[3/5] Paper (UNet-VDA) 模型推理...")
    if is_720:
        paper_even = run_paper_model_on_cwa(paper_model, raw_even, nyquist)
        paper_odd  = run_paper_model_on_cwa(paper_model, raw_odd, nyquist)
        paper_even_c = apply_physics_constraint(paper_even, raw_even, nyquist)
        paper_odd_c  = apply_physics_constraint(paper_odd, raw_odd, nyquist)
        paper_constrained = merge_halves_to_720(paper_even_c, paper_odd_c, raw_vel, nyquist)
        print(f"   [720-ray] Paper 合併完成: {paper_constrained.shape}")
    else:
        paper_raw = run_paper_model_on_cwa(paper_model, raw_vel, nyquist)
        paper_constrained = apply_physics_constraint(paper_raw, raw_vel, nyquist)

    # ── 720→360 對齊（is_720_to_360 case 專用，azimuth-aware）──
    # 推理已於 720-ray 全部 ray 完成；下游 metric/viz 須對齊 360 GT 的 azimuth 位置。
    # 不能用 ours_constrained[::2]（舊版會丟掉奇數 ray、且位置不對應 GT），
    # 改用 azimuth nearest-neighbor 對齊（與 raw_eval 取同樣的 raw index）。
    if is_720_to_360:
        ours_constrained = align_raw_to_gt_by_azimuth(
            ours_constrained,
            raw_azm_start=raw_meta.get('azm_start', 0.0),
            raw_azm_sp   =raw_meta.get('azm_sp', 0.5),
            gt_azm_start =da_meta.get('azm_start', 0.0),
            gt_azm_sp    =da_meta.get('azm_sp', 1.0),
            gt_naz       =da_vel.shape[0],
        )
        paper_constrained = align_raw_to_gt_by_azimuth(
            paper_constrained,
            raw_azm_start=raw_meta.get('azm_start', 0.0),
            raw_azm_sp   =raw_meta.get('azm_sp', 0.5),
            gt_azm_start =da_meta.get('azm_start', 0.0),
            gt_azm_sp    =da_meta.get('azm_sp', 1.0),
            gt_naz       =da_vel.shape[0],
        )
        print(f"   [720→360 azimuth-aware] ours/paper 對齊 GT azimuth，shape={ours_constrained.shape}")

    # 4. 指標
    print("\n[4/5] 計算指標...")
    raw_metrics = compute_raw_metrics(raw_eval, gt_vel)
    ours_metrics = compute_metrics(ours_constrained, gt_vel, raw=raw_eval,
                                   nyq_val=nyquist, name="Ours",
                                   use_physics_method=True)
    paper_metrics = compute_metrics(paper_constrained, gt_vel, raw=raw_eval,
                                    nyq_val=nyquist, name="Paper",
                                    use_physics_method=True)

    # valid_pixels 一致性驗證
    mask_ours = ~np.isnan(ours_constrained) & ~np.isnan(gt_vel)
    mask_paper = ~np.isnan(paper_constrained) & ~np.isnan(gt_vel)
    mask_raw = ~np.isnan(raw_eval) & ~np.isnan(gt_vel)
    print(f"   Valid pixels: RAW={np.sum(mask_raw)}, Ours={np.sum(mask_ours)}, Paper={np.sum(mask_paper)}")

    # 打印結果
    print(f"\n[RESULT] {case_label}")
    print(f"   {'Method':<16} {'RMSE':>8} {'MAE':>8} {'Corr':>8} {'SR(Recall)':>10} {'FPR':>7} {'Prec':>7}")
    print(f"   {'RAW':<16} {raw_metrics['rmse']:8.3f} {raw_metrics['mae']:8.3f} "
          f"{raw_metrics['corr']:8.3f} {'N/A':>10} {'N/A':>7} {'N/A':>7}")

    sr_ours   = ours_metrics.get('correction_success_rate', None)
    fpr_ours  = ours_metrics.get('false_positive_rate', None)
    prec_ours = ours_metrics.get('precision', None)
    sr_paper   = paper_metrics.get('correction_success_rate', None)
    fpr_paper  = paper_metrics.get('false_positive_rate', None)
    prec_paper = paper_metrics.get('precision', None)
    print(f"   {'Ours':<16} {ours_metrics['rmse']:8.3f} {ours_metrics['mae']:8.3f} "
          f"{ours_metrics['corr']:8.3f} "
          f"{f'{sr_ours:.1%}' if sr_ours is not None else 'N/A':>10} "
          f"{f'{fpr_ours:.1%}' if fpr_ours is not None else 'N/A':>7} "
          f"{f'{prec_ours:.1%}' if prec_ours is not None else 'N/A':>7}")
    print(f"   {'UNet-VDA':<16} {paper_metrics['rmse']:8.3f} {paper_metrics['mae']:8.3f} "
          f"{paper_metrics['corr']:8.3f} "
          f"{f'{sr_paper:.1%}' if sr_paper is not None else 'N/A':>10} "
          f"{f'{fpr_paper:.1%}' if fpr_paper is not None else 'N/A':>7} "
          f"{f'{prec_paper:.1%}' if prec_paper is not None else 'N/A':>7}")

    # ── (選用)存下對齊後速度場,供事後計算物理一致性指標 / 下游 proxy(免重訓再跑模型)──
    if save_fields:
        try:
            _flbl = case_label.replace('/', '_').replace(' ', '_').replace('.', '_')
            _fdir = os.path.join(output_dir, '_fields')
            os.makedirs(_fdir, exist_ok=True)
            np.savez_compressed(
                os.path.join(_fdir, f"{_flbl}.npz"),
                raw=raw_eval.astype(np.float32),
                ours=ours_constrained.astype(np.float32),
                paper=paper_constrained.astype(np.float32),
                gt=gt_vel.astype(np.float32),
                nyquist=np.float32(nyquist),
                station=str(station), date=str(date), time=str(time_str),
                elevation=str(_elev), case_label=str(case_label),
                aliased_ratio=np.float32(locals().get('aliased_ratio', np.nan)),
            )
            print(f"   [save_fields] 已存速度場 → {_flbl}.npz")
        except Exception as _e:
            print(f"   [save_fields WARN] 存場失敗: {_e}")

    # 5. 視覺化（所有速度場使用 360 對齊版本：raw_eval/ours_constrained/paper_constrained/gt_vel）
    # is_720_to_360 case：ours/paper 已於上方 [::2]，與 raw_eval/gt_vel shape 一致
    if enable_viz:
        print("\n[5/5] 生成視覺化...")
        plot_comparison(raw_eval, ours_constrained, paper_constrained, gt_vel,
                       nyquist, case_label, output_dir, ours_metrics, paper_metrics)
        plot_error_polar(raw_eval, ours_constrained, paper_constrained, gt_vel,
                        nyquist, case_label, output_dir)
        if enable_geo_viz and _ensure_geo_imports():
            # 單張 geo（每個速度場獨立一張圖）
            raw_path = triplet['raw']
            if str(raw_path).endswith('.gz'):
                raw_gz = str(raw_path)   # real obs：本身就是 .gz
            else:
                raw_gz = find_corresponding_gz_file(raw_path)   # NWP：parquet → 反查 .gz
            if raw_gz is not None:
                # ⚠️ 把 case_label 中的 '.' 換成 '_'，避免 Path(filename).stem 把最後一段
                # 當 extension 砍掉，導致同站同日不同時段的圖檔互相覆蓋
                safe_label = case_label.replace('/', '_').replace(' ', '_').replace('.', '_')

                # 找對應的反射率 .gz（若存在，會多畫一張回波 geo）
                refl_gz = _find_reflectivity_gz(raw_gz)
                refl_vel = None
                if refl_gz:
                    try:
                        refl_vel, _, _ = read_gz_radar(refl_gz)
                        print(f"   [GEO] 反射率載入: {os.path.basename(refl_gz)} shape={refl_vel.shape}")
                    except Exception as e:
                        print(f"   [GEO WARN] 反射率讀取失敗 ({refl_gz}): {e}")
                        refl_vel = None

                # 速度場 4 張 + 反射率（若有）
                vel_panels = [
                    (raw_eval, 'raw', 'RAW (Aliased)'),
                    (ours_constrained, 'ours', 'Ours Dealiased'),
                    (paper_constrained, 'paper', 'UNet-VDA Dealiased'),
                    (gt_vel, 'gt', 'DA Ground Truth'),
                ]
                geo_results = []
                for vel, fname, ttl in vel_panels:
                    try:
                        out = save_single_geo_visualization(
                            vel, raw_gz, fname, ttl,
                            f"nwp_{safe_label}", output_dir,
                            shape_path=shape_path)
                        geo_results.append((fname, out is not None))
                    except Exception as e:
                        print(f"   [GEO WARN] {fname} 失敗: {e}")
                        geo_results.append((fname, False))

                # 反射率（dBZ 範圍跟色階都不同）
                if refl_vel is not None:
                    try:
                        out = save_single_geo_visualization(
                            refl_vel, raw_gz, 'refl', 'Reflectivity (dBZ)',
                            f"nwp_{safe_label}", output_dir,
                            shape_path=shape_path,
                            vmin=-10, vmax=70, cmap_name='reflectivity')
                        geo_results.append(('refl', out is not None))
                    except Exception as e:
                        print(f"   [GEO WARN] reflectivity 失敗: {e}")
                        geo_results.append(('refl', False))

                # 摘要
                ok = [n for n, s in geo_results if s]
                fail = [n for n, s in geo_results if not s]
                print(f"   [GEO] 單張完成: {ok}; 失敗: {fail if fail else '無'}")
            # 合圖 comparison（1×4）
            plot_geo_comparison(raw_eval, ours_constrained, paper_constrained, gt_vel,
                               triplet, case_label, output_dir,
                               ours_metrics=ours_metrics, paper_metrics=paper_metrics,
                               shape_path=shape_path)
            # 合圖 error（1×3）
            plot_geo_error(raw_eval, ours_constrained, paper_constrained, gt_vel,
                          nyquist, triplet, case_label, output_dir,
                          shape_path=shape_path)
    else:
        print("\n[5/5] 跳過視覺化")

    # 彙整
    case_result = {
        "case_label": case_label,
        "station": station,
        "date": date,
        "time": time_str,
        "elevation": _elev,  # ← 顯式欄位，方便 per-elev 統計
        "nwp_dir": nwp_dir,
        "filename": triplet['filename'],
        "nyquist": float(nyquist),
        "shape": list(raw_vel.shape),       # 原始 raw shape (720 for RCWF, otherwise 360)
        "eval_shape": list(gt_vel.shape),    # 評估解析度 (always 360 for 720-ray case)
        "is_720_to_360": bool(is_720_to_360),
        "aliased_ratio": aliased_ratio,      # 該 sweep 折錯像素比例 (供 subset 分析)
        "aliased_pixels": aliased_pixels,
        "valid_pixels": valid_count,
        "has_aliasing": bool(aliased_ratio > 0.01),  # > 1% 視為含折錯 sweep
        "valid_pixels_raw": int(np.sum(mask_raw)),
        "valid_pixels_ours": int(np.sum(mask_ours)),
        "valid_pixels_paper": int(np.sum(mask_paper)),
        "is_required_case": (triplet.get('nwp_dir', ''), triplet['filename']) in
                           {(c['nwp_dir'], c['file'] + '.parquet') for c in REQUIRED_CASES},
        "raw_metrics": raw_metrics,
        "ours_metrics": {k: v for k, v in ours_metrics.items() if k != 'dealiasing_metrics'},
        "paper_metrics": {k: v for k, v in paper_metrics.items() if k != 'dealiasing_metrics'},
    }

    if vcorr_stats is not None:
        case_result["vcorr_postprocessing"] = vcorr_stats

    if save_confidence:
        case_confidence = finalize_confidence(conf_accums)
        if case_confidence is not None:
            case_result["confidence"] = case_confidence

    return case_result


# ---------------------------------------------------------------------------
# 9. 結果彙整
# ---------------------------------------------------------------------------

def summarize_results(all_results, output_dir):
    """彙整所有案例 → summary JSON"""
    if not all_results:
        print("[WARNING] 無結果可彙整")
        return None

    n = len(all_results)

    def _to_float(obj):
        """將數值或數值字串轉成 float，失敗回傳 None。"""
        if obj is None:
            return None
        try:
            v = float(obj)
            return None if np.isnan(v) else v
        except (TypeError, ValueError):
            return None

    def avg_metric(key_path):
        vals = []
        for r in all_results:
            obj = r
            for k in key_path:
                obj = obj.get(k, {}) if isinstance(obj, dict) else None
                if obj is None:
                    break
            v = _to_float(obj)
            if v is not None:
                vals.append(v)
        return float(np.mean(vals)) if vals else None

    def std_metric(key_path):
        vals = []
        for r in all_results:
            obj = r
            for k in key_path:
                obj = obj.get(k, {}) if isinstance(obj, dict) else None
                if obj is None:
                    break
            v = _to_float(obj)
            if v is not None:
                vals.append(v)
        return float(np.std(vals)) if vals else None

    overall = {
        "raw_rmse": avg_metric(['raw_metrics', 'rmse']),
        "raw_mae": avg_metric(['raw_metrics', 'mae']),
        "raw_corr": avg_metric(['raw_metrics', 'corr']),
        "ours_rmse": avg_metric(['ours_metrics', 'rmse']),
        "ours_mae": avg_metric(['ours_metrics', 'mae']),
        "ours_corr": avg_metric(['ours_metrics', 'corr']),
        "ours_success_rate": avg_metric(['ours_metrics', 'correction_success_rate']),
        "ours_false_positive_rate": avg_metric(['ours_metrics', 'false_positive_rate']),
        "ours_precision": avg_metric(['ours_metrics', 'precision']),
        "paper_rmse": avg_metric(['paper_metrics', 'rmse']),
        "paper_mae": avg_metric(['paper_metrics', 'mae']),
        "paper_corr": avg_metric(['paper_metrics', 'corr']),
        "paper_success_rate": avg_metric(['paper_metrics', 'correction_success_rate']),
        "paper_false_positive_rate": avg_metric(['paper_metrics', 'false_positive_rate']),
        "paper_precision": avg_metric(['paper_metrics', 'precision']),
        "ours_rmse_std": std_metric(['ours_metrics', 'rmse']),
        "paper_rmse_std": std_metric(['paper_metrics', 'rmse']),
    }

    # Per-station 統計
    station_results = defaultdict(list)
    for r in all_results:
        station_results[r['station']].append(r)

    def _safe_mean(vals):
        v = [_to_float(x) for x in vals]
        v = [x for x in v if x is not None]
        return float(np.mean(v)) if v else None

    per_station = {}
    for st, cases in sorted(station_results.items()):
        st_n = len(cases)
        st_overall = {
            "n_cases": st_n,
            "ours_rmse":  _safe_mean([c['ours_metrics']['rmse']  for c in cases]),
            "paper_rmse": _safe_mean([c['paper_metrics']['rmse'] for c in cases]),
            "raw_rmse":   _safe_mean([c['raw_metrics']['rmse']   for c in cases]),
        }
        ours_sr  = [_to_float(c['ours_metrics'].get('correction_success_rate'))  for c in cases]
        paper_sr = [_to_float(c['paper_metrics'].get('correction_success_rate')) for c in cases]
        v = _safe_mean(ours_sr)
        if v is not None: st_overall['ours_success_rate'] = v
        v = _safe_mean(paper_sr)
        if v is not None: st_overall['paper_success_rate'] = v
        per_station[st] = st_overall

    # Per-typhoon-case 統計（依 nwp_dir 分組）
    typhoon_results = defaultdict(list)
    for r in all_results:
        typhoon_results[r.get('nwp_dir', 'unknown')].append(r)

    per_typhoon = {}
    for tc, cases in sorted(typhoon_results.items()):
        tc_entry = {
            "n_cases": len(cases),
            "stations": sorted(set(c['station'] for c in cases)),
            "ours_rmse":   _safe_mean([c['ours_metrics']['rmse']  for c in cases]),
            "paper_rmse":  _safe_mean([c['paper_metrics']['rmse'] for c in cases]),
            "raw_rmse":    _safe_mean([c['raw_metrics']['rmse']   for c in cases]),
            "ours_sr":     _safe_mean([c['ours_metrics'].get('correction_success_rate')  for c in cases]),
            "paper_sr":    _safe_mean([c['paper_metrics'].get('correction_success_rate') for c in cases]),
            "ours_fpr":    _safe_mean([c['ours_metrics'].get('false_positive_rate')  for c in cases]),
            "paper_fpr":   _safe_mean([c['paper_metrics'].get('false_positive_rate') for c in cases]),
        }
        per_typhoon[tc] = tc_entry

    # ── Aliased subset 摘要 (解決 SR 被無折錯 sweep 稀釋的問題) ──
    aliased_cases = [r for r in all_results if r.get('has_aliasing', False)]
    clean_cases   = [r for r in all_results if not r.get('has_aliasing', False)]

    def _subset_overall(cases):
        if not cases:
            return None
        def _sm(key_path):
            vals = []
            for r in cases:
                obj = r
                for k in key_path:
                    obj = obj.get(k, {}) if isinstance(obj, dict) else None
                    if obj is None:
                        break
                v = _to_float(obj)
                if v is not None:
                    vals.append(v)
            return float(np.mean(vals)) if vals else None
        return {
            "n_cases":                   len(cases),
            "raw_rmse":                  _sm(['raw_metrics', 'rmse']),
            "raw_mae":                   _sm(['raw_metrics', 'mae']),
            "ours_rmse":                 _sm(['ours_metrics', 'rmse']),
            "ours_mae":                  _sm(['ours_metrics', 'mae']),
            "ours_corr":                 _sm(['ours_metrics', 'corr']),
            "ours_success_rate":         _sm(['ours_metrics', 'correction_success_rate']),
            "ours_false_positive_rate":  _sm(['ours_metrics', 'false_positive_rate']),
            "ours_precision":            _sm(['ours_metrics', 'precision']),
            "paper_rmse":                _sm(['paper_metrics', 'rmse']),
            "paper_mae":                 _sm(['paper_metrics', 'mae']),
            "paper_corr":                _sm(['paper_metrics', 'corr']),
            "paper_success_rate":        _sm(['paper_metrics', 'correction_success_rate']),
            "paper_false_positive_rate": _sm(['paper_metrics', 'false_positive_rate']),
            "paper_precision":           _sm(['paper_metrics', 'precision']),
        }

    subset_summary = {
        "aliased_threshold": 0.01,
        "aliased_subset":    _subset_overall(aliased_cases),
        "clean_subset":      _subset_overall(clean_cases),
        "aliased_fraction":  float(len(aliased_cases) / n) if n else 0.0,
    }

    summary = {
        "n_cases": n,
        "overall": overall,
        "subset_analysis": subset_summary,
        "per_station": per_station,
        "per_typhoon": per_typhoon,
        "per_case": all_results,
    }

    # 記錄執行指令（方便日後追溯）
    summary["run_info"] = {
        "command": " ".join(sys.argv),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": sys.version.split()[0],
    }

    summary_path = os.path.join(output_dir, "nwp_comparison_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[SAVE] Summary: {summary_path}")

    # 打印
    o = overall
    print(f"\n{'='*80}")
    print(f"[SUMMARY] CWA NWP 跨模型比較結果 ({n} cases)")
    print(f"{'='*80}")
    ours_sr   = o.get('ours_success_rate')
    ours_fpr  = o.get('ours_false_positive_rate')
    ours_prec = o.get('ours_precision')
    paper_sr   = o.get('paper_success_rate')
    paper_fpr  = o.get('paper_false_positive_rate')
    paper_prec = o.get('paper_precision')
    ours_sr_str    = f"{ours_sr:.1%}"    if ours_sr   is not None else "N/A"
    ours_fpr_str   = f"{ours_fpr:.1%}"  if ours_fpr  is not None else "N/A"
    ours_prec_str  = f"{ours_prec:.1%}" if ours_prec  is not None else "N/A"
    paper_sr_str   = f"{paper_sr:.1%}"   if paper_sr  is not None else "N/A"
    paper_fpr_str  = f"{paper_fpr:.1%}" if paper_fpr  is not None else "N/A"
    paper_prec_str = f"{paper_prec:.1%}" if paper_prec is not None else "N/A"
    print(f"   {'Method':<16} {'RMSE':>10} {'MAE':>8} {'Corr':>8} {'SR(Recall)':>10} {'FPR':>7} {'Prec':>7}")
    def _fv(v, fmt='.3f'): return format(v, fmt) if v is not None else 'N/A'
    print(f"   {'RAW':<16} {_fv(o['raw_rmse']):>10} {_fv(o['raw_mae']):>8} "
          f"{_fv(o['raw_corr']):>8} {'N/A':>10} {'N/A':>7} {'N/A':>7}")
    ours_rmse_str = _fv(o['ours_rmse'])
    if o.get('ours_rmse_std'):
        ours_rmse_str += f"({o['ours_rmse_std']:.2f})"
    paper_rmse_str = _fv(o['paper_rmse'])
    if o.get('paper_rmse_std'):
        paper_rmse_str += f"({o['paper_rmse_std']:.2f})"
    print(f"   {'Ours':<16} {ours_rmse_str:>10} {_fv(o['ours_mae']):>8} "
          f"{_fv(o['ours_corr']):>8} {ours_sr_str:>10} {ours_fpr_str:>7} {ours_prec_str:>7}")
    print(f"   {'UNet-VDA':<16} {paper_rmse_str:>10} {_fv(o['paper_mae']):>8} "
          f"{_fv(o['paper_corr']):>8} {paper_sr_str:>10} {paper_fpr_str:>7} {paper_prec_str:>7}")

    print(f"\n   Per-station:")
    print(f"   {'Station':<10} {'N':>4} {'Ours RMSE':>10} {'Paper RMSE':>12} {'RAW RMSE':>10}")
    for st, st_o in sorted(per_station.items()):
        print(f"   {st:<10} {st_o['n_cases']:4d} {_fv(st_o['ours_rmse']):>10} "
              f"{_fv(st_o['paper_rmse']):>12} {_fv(st_o['raw_rmse']):>10}")

    print(f"\n   Per-typhoon-case:")
    print(f"   {'Case':<22} {'N':>4} {'Ours RMSE':>10} {'Paper RMSE':>12} {'RAW RMSE':>10} {'Ours SR':>8} {'Paper SR':>9} {'Ours FPR':>9} {'Paper FPR':>10}")
    for tc, tc_o in sorted(per_typhoon.items()):
        def _fmt(v): return f"{v:.3f}" if v is not None else "  N/A"
        def _fmtp(v): return f"{v:.1%}" if v is not None else "   N/A"
        print(f"   {tc:<22} {tc_o['n_cases']:4d} {_fmt(tc_o['ours_rmse']):>10} "
              f"{_fmt(tc_o['paper_rmse']):>12} {_fmt(tc_o['raw_rmse']):>10} "
              f"{_fmtp(tc_o['ours_sr']):>8} {_fmtp(tc_o['paper_sr']):>9} "
              f"{_fmtp(tc_o['ours_fpr']):>9} {_fmtp(tc_o['paper_fpr']):>10}")

    # ── Aliased subset 對照表 (解決 SR 被無折錯 sweep 稀釋問題) ──
    print(f"\n   Subset 分析 (aliased_ratio > 1% 視為含折錯 sweep):")
    print(f"   {'Subset':<18} {'N':>4} {'Frac':>6} {'Ours RMSE':>10} {'Ours SR':>9} {'Ours FPR':>9} {'Paper RMSE':>11} {'Paper SR':>10} {'Paper FPR':>10}")
    def _print_subset(label, s):
        if s is None or s['n_cases'] == 0:
            print(f"   {label:<18} {0:>4} {'N/A':>6} {'N/A':>10} {'N/A':>9} {'N/A':>9} {'N/A':>11} {'N/A':>10} {'N/A':>10}")
            return
        frac = s['n_cases'] / n if n else 0
        _fmt  = lambda v: f"{v:.3f}" if v is not None else "  N/A"
        _fmtp = lambda v: f"{v:.1%}" if v is not None else "   N/A"
        print(f"   {label:<18} {s['n_cases']:>4} {frac:>5.1%} "
              f"{_fmt(s['ours_rmse']):>10} "
              f"{_fmtp(s['ours_success_rate']):>9} {_fmtp(s['ours_false_positive_rate']):>9} "
              f"{_fmt(s['paper_rmse']):>11} "
              f"{_fmtp(s['paper_success_rate']):>10} {_fmtp(s['paper_false_positive_rate']):>10}")
    _print_subset("Aliased (含折錯)", subset_summary['aliased_subset'])
    _print_subset("Clean (無折錯)",   subset_summary['clean_subset'])
    _print_subset("All",                overall_subset_view := {
        'n_cases': n,
        'ours_rmse': overall.get('ours_rmse'),
        'ours_success_rate': overall.get('ours_success_rate'),
        'ours_false_positive_rate': overall.get('ours_false_positive_rate'),
        'paper_rmse': overall.get('paper_rmse'),
        'paper_success_rate': overall.get('paper_success_rate'),
        'paper_false_positive_rate': overall.get('paper_false_positive_rate'),
    })
    print(f"{'='*80}")

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run_inference_only(args):
    """純推論 + 可視化模式（無 GT、無 JSON）。

    --inference_input 指向單一 raw .gz 或含 raw 檔之資料夾；自 gz 檔頭讀 Nyquist，
    以與正典評估器相同之推論路徑（含 720→360 interleaved split + 物理約束）反折，
    輸出反折後速度場（--save_fields）與 geo 圖（--enable_geo_viz）。不需 GT、不算指標。
    """
    import glob
    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 80)
    print("純推論模式 (Inference-only, 無 GT / 無 JSON)")
    print(f"   輸入:     {args.inference_input}")
    print(f"   模型:     {args.model_path}")
    print(f"   輸出目錄: {args.output_dir}")
    print("=" * 80)

    inp = args.inference_input
    SKIP = ('vdaqc', 'sfda', 'bvel_da')  # GT 檔標記，純推論不當輸入
    if os.path.isfile(inp):
        raw_files = [inp]
    else:
        raw_files = [p for p in sorted(glob.glob(os.path.join(inp, '**', '*.gz'), recursive=True))
                     if not any(t in os.path.basename(p) for t in SKIP)]
    print(f"[INFER] 待處理 raw 檔: {len(raw_files)}")
    if not raw_files:
        print("[ERROR] 找不到任何 raw .gz 輸入"); return

    model = build_mixed_patch_model_for_inference(use_physics_model=args.use_physics_model)
    # Dummy forward 初始化（與正典評估器一致：子類別模型須先 build 才能 load_weights）
    _fr, _fn, _ = load_velocity_data(raw_files[0])
    _pad, _, _ = auto_zero_pad(_fr, layers=4, fill_value=0)
    _ = model({'vel': _pad[None, None, :, :, None].astype(np.float32),
               'nyq': np.array([[_fn if (_fn and _fn > 0) else 1.0]], dtype=np.float32)},
              training=False)
    model.load_weights(args.model_path)
    print(f"[OK] 模型載入完成: {args.model_path}")

    geo_ok = _ensure_geo_imports() if args.enable_geo_viz else False
    if args.enable_geo_viz and not geo_ok:
        print("[GEO WARN] pyart/basemap 不可用 → 跳過 geo 可視化（推論本身不受影響）")

    fld_dir = os.path.join(args.output_dir, '_fields')
    if args.save_fields:
        os.makedirs(fld_dir, exist_ok=True)

    results = []
    for i, raw_gz in enumerate(raw_files, 1):
        try:
            raw_vel, nyquist, meta = load_velocity_data(raw_gz)
            if nyquist is None or nyquist <= 0:
                if getattr(args, 'infer_nyquist', 0) and args.infer_nyquist > 0:
                    nyquist = float(args.infer_nyquist)
                    print(f"   [nyquist override] 檔頭無 Nyquist → 用 --infer_nyquist {nyquist:.2f}")
                else:
                    print(f"   [{i}/{len(raw_files)}] SKIP 檔頭無 Nyquist 且未給 --infer_nyquist: {raw_gz}")
                    continue

            naz = raw_vel.shape[0]
            is_720 = args.downsample_720 and naz > _720_THRESHOLD
            if is_720:
                raw_even, raw_odd = split_720_to_halves(raw_vel)
                oe, _ = run_our_model_on_cwa(model, raw_even, nyquist)
                oo, _ = run_our_model_on_cwa(model, raw_odd, nyquist)
                oe = apply_physics_constraint(oe, raw_even, nyquist)
                oo = apply_physics_constraint(oo, raw_odd, nyquist)
                ours = merge_halves_to_720(oe, oo, raw_vel, nyquist)
            else:
                o, _ = run_our_model_on_cwa(model, raw_vel, nyquist)
                ours = apply_physics_constraint(o, raw_vel, nyquist)

            base = os.path.splitext(os.path.basename(raw_gz))[0]
            safe = base.replace('.', '_')
            valid = ~np.isnan(raw_vel) & ~np.isnan(ours)
            corrected = int(np.sum(valid & (np.abs(ours - raw_vel) > 0.5 * nyquist)))

            if args.save_fields:
                np.savez_compressed(os.path.join(fld_dir, safe + '.npz'),
                                    raw=raw_vel.astype('float32'),
                                    ours=ours.astype('float32'),
                                    nyquist=float(nyquist))
            if geo_ok:
                for vel, fname, ttl in ((raw_vel, 'raw', 'RAW (Aliased)'),
                                        (ours, 'ours', 'Ours Dealiased')):
                    try:
                        save_single_geo_visualization(vel, raw_gz, fname, ttl,
                            f"infer_{safe}", args.output_dir, shape_path=args.shape_path)
                    except Exception as e:
                        print(f"   [GEO WARN] {fname}: {e}")

            results.append({'file': raw_gz, 'nyquist': float(nyquist),
                            'shape': list(raw_vel.shape), 'corrected_pixels': corrected})
            print(f"   [{i}/{len(raw_files)}] OK {base}  nyq={nyquist:.2f}  修正 {corrected:,} 像素")
        except Exception as e:
            print(f"   [{i}/{len(raw_files)}] FAIL {raw_gz}: {e}")

    summary = os.path.join(args.output_dir, 'inference_summary.json')
    with open(summary, 'w', encoding='utf-8') as f:
        json.dump({'n_cases': len(results), 'cases': results}, f, ensure_ascii=False, indent=2)
    print(f"\n[DONE] 完成 {len(results)}/{len(raw_files)} 案 → {args.output_dir}")
    print(f"       摘要: {summary}")


def main():
    parser = argparse.ArgumentParser(
        description='CWA NWP 跨模型比較: Ours vs UNet-VDA (Paper)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='我們的模型權重 (.h5)')
    parser.add_argument('--paper_model_path', type=str, default=DEFAULT_PAPER_MODEL,
                        help='論文模型 SavedModel 目錄')
    parser.add_argument('--nwp_root', type=str, default=None,
                        help='NWP 資料根目錄（realobs_root 模式時可省略）')
    parser.add_argument('--output_dir', type=str, default='Transfer_result/nwp_comparison',
                        help='輸出目錄')
    parser.add_argument('--h5_path', type=str, default=H5_PATH,
                        help='H5 訓練集路徑 (用於 test split 驗證)')
    parser.add_argument('--enable_viz', action='store_true',
                        help='啟用視覺化')
    parser.add_argument('--use_physics_model', action='store_true',
                        help='我們的模型使用物理約束版本')
    parser.add_argument('--max_cases', type=int, default=100,
                        help='最大測試案例數')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='隨機種子 (用於案例抽樣)')
    parser.add_argument('--clean_ratio', type=float, default=0.0,
                        help='Clean/low-alias case 佔比 (0~1)。'
                             '例如 0.2 = 20%% budget 給 clean case，用於公平 FPR 評估。'
                             '預設 0.0 = 只選 aliased case（向後相容）')
    parser.add_argument('--force_cases_file', type=str, default=None,
                        help='JSON 檔案指定測試案例 (跳過隨機選取，用於 apple-to-apple 比較)')
    parser.add_argument('--conf_threshold', type=float, default=0.0,
                        help='Confidence gate 閾值 (0~1)。'
                             '> 0 時：alias confidence < threshold 的像素保留 raw velocity，'
                             '不套用修正。可降低 FPR。建議測試值: 0.5, 0.7, 0.8, 0.9。'
                             '0.0 = 關閉 (預設)')
    parser.add_argument('--enable_vcorrected_pp', action='store_true',
                        help='啟用 V_corrected 物理後處理。'
                             '基於修正後速度空間連續性，回退 FP 修正並恢復漏修 FN。')
    parser.add_argument('--vcorrected_threshold', type=float, default=1.0,
                        help='V_corrected 後處理跳躍閾值（Nyquist 倍數）。'
                             '預設 1.0，較低值（如 0.8）更積極過濾。')
    parser.add_argument('--enable_geo_viz', action='store_true',
                        help='啟用氣象署風格地理可視化（需要 pyart + .gz 雷達檔案）')
    parser.add_argument('--shape_path', type=str, default=None,
                        help='台灣地圖 shapefile 路徑（例如 mapdata201805310314/COUNTY_MOI_1070516）')
    parser.add_argument('--downsample_720', action='store_true',
                        help='720-ray 站（如 RCWF）推理前以 interleaved split 拆為 2×360-ray，'
                             '解決 domain gap 造成的 FPR 過高問題。'
                             '推理後合併回原始解析度。')
    parser.add_argument('--exclude_stations', type=str, default=None,
                        help='排除指定站（逗號分隔），例如 --exclude_stations RCCG,RCGI')
    parser.add_argument('--save_fields', action='store_true', default=False,
                        help='把每個 case 對齊後的速度場(raw/ours/paper/gt + nyquist)存成 '
                             '{output_dir}/<case_subdir>/_fields/<label>.npz，供事後以 '
                             'compute_physics_consistency.py 計算物理一致性指標 / 下游 proxy。'
                             '僅做推論、不影響既有純量指標,預設關閉。')
    parser.add_argument('--save_confidence', action='store_true', default=False,
                        help='逐 case 輸出 softmax 分類信心統計（口委#8 confidence score），'
                             '寫入 summary 的 per_case.confidence（case_uncertainty=平均熵、'
                             'mean_max_prob、frac_low_conf 等）。僅做推論、不影響既有指標,預設關閉。')
    # ── 真實觀測模式（typhoonnew/ .gz 資料）────────────────────────────────────
    parser.add_argument('--realobs_root', type=str, default=None,
                        help='真實觀測資料根目錄（typhoonnew/）。'
                             '指定此參數時跳過 NWP/H5 相關步驟，'
                             '改從 typhoonnew/ 掃描 .gz 配對進行測試。')
    parser.add_argument('--realobs_elevation', type=str, default='all',
                        help='真實觀測仰角（01/02/... 或 all=全部，預設 all）')
    parser.add_argument('--realobs_sample_n', type=int, default=100,
                        help='真實觀測每站每案最多取樣掃描數（0=全部）')
    parser.add_argument('--realobs_force_cases_file', type=str, default=None,
                        help='Locked Real Obs test set JSON（由 build_locked_test_sets.py 產生）。'
                             '指定時跳過 realobs_root 掃描，直接讀 JSON 內 (raw, gt) pairs。'
                             '保證跨模型 100%% 一致的測試集。')
    parser.add_argument('--inference_input', type=str, default=None,
                        help='純推論模式（無 GT / 無 JSON）：指向單一 raw .gz 或含 raw 檔之資料夾。'
                             '設定時忽略 realobs/nwp 評估流程，僅做反折推論 + geo 可視化；'
                             'Nyquist 自 gz 檔頭讀取，不需 GT、不計算 SR/FPR。')
    parser.add_argument('--infer_nyquist', type=float, default=0.0,
                        help='純推論模式下，當 gz 檔頭無 Nyquist 時之後備值（預設 0=不套用，跳過該檔）。')
    args = parser.parse_args()

    # ── 基本檢查 ──────────────────────────────────────────────────────────────
    if not os.path.exists(args.model_path):
        print(f"[ERROR] 模型不存在: {args.model_path}")
        return

    # ── 純推論模式（無 GT / 無 JSON）：僅需 raw 檔 + 我們的模型，不碰 paper 模型與評估流程 ──
    if getattr(args, 'inference_input', None):
        _run_inference_only(args)
        return

    if not os.path.exists(args.paper_model_path):
        print(f"[ERROR] 論文模型不存在: {args.paper_model_path}")
        return

    realobs_mode = (args.realobs_root is not None) or (args.realobs_force_cases_file is not None)

    if not realobs_mode:
        if not os.path.isdir(args.nwp_root):
            print(f"[ERROR] NWP 目錄不存在: {args.nwp_root}")
            return
        if not os.path.exists(args.h5_path):
            print(f"[ERROR] H5 檔案不存在: {args.h5_path}")
            return

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    if realobs_mode:
        print("真實觀測 跨模型比較: Ours vs UNet-VDA (Paper)")
        print(f"   資料來源:   {args.realobs_root}")
    else:
        print("CWA NWP 跨模型比較: Ours vs UNet-VDA (Paper)")
        print(f"   NWP 資料:   {args.nwp_root}")
        print(f"   H5 檔案:    {args.h5_path}")
    print("=" * 80)
    print(f"   我們的模型: {args.model_path}")
    print(f"   論文模型:   {args.paper_model_path}")
    print(f"   輸出目錄:   {args.output_dir}")
    if args.enable_vcorrected_pp:
        print(f"   V_corr後處理: 啟用 (threshold={args.vcorrected_threshold})")
    if args.downsample_720:
        print(f"   720→360降採樣: 啟用 (naz>{_720_THRESHOLD} 時自動觸發)")

    # ── 資料選取 ──────────────────────────────────────────────────────────────
    exclude_set = set(s.strip() for s in (args.exclude_stations or '').split(',')
                      if s.strip())

    if realobs_mode:
        # ── 真實觀測模式 ──────────────────────────────────────────────────
        # 兩條路徑：
        #   1. --realobs_force_cases_file: 從 locked JSON 直接讀（推薦，跨模型 100% 一致）
        #   2. --realobs_root: 掃描資料夾 + sample_n（舊版，每跑可能採樣不同）
        print(f"\n{'='*60}")
        print("[PHASE 1] 真實觀測資料掃描")
        print(f"{'='*60}")
        if args.realobs_force_cases_file:
            test_cases = build_realobs_cases_from_force_file(args.realobs_force_cases_file)
            # 排除指定站（locked JSON 仍可套 station filter）
            if exclude_set:
                before = len(test_cases)
                test_cases = [tc for tc in test_cases if tc['station'] not in exclude_set]
                print(f"[FILTER] 排除站 {exclude_set}: {before} → {len(test_cases)} cases")
        else:
            sample_n = args.realobs_sample_n if args.realobs_sample_n > 0 else None
            test_cases = build_realobs_cases(
                args.realobs_root,
                elevation=args.realobs_elevation,
                sample_n=sample_n,
                exclude_stations=exclude_set)
        if not test_cases:
            print("[ERROR] 無可用真實觀測案例")
            return
    else:
        # ── NWP 模式（原有流程）───────────────────────────────────────────────
        print(f"\n{'='*60}")
        print("[PHASE 1] 資料選取與洩漏防護")
        print(f"{'='*60}")
        test_sources = extract_test_sources(args.h5_path)

        if args.force_cases_file:
            print(f"\n[MODE] 強制指定案例模式: {args.force_cases_file}")
            test_cases = build_forced_cases(
                args.nwp_root, args.force_cases_file, test_sources)
        else:
            test_cases = select_test_cases(
                args.nwp_root, test_sources,
                max_cases=args.max_cases,
                random_seed=args.random_seed,
                clean_ratio=args.clean_ratio)

        if not test_cases:
            print("[ERROR] 無可用測試案例")
            return

        if exclude_set:
            before = len(test_cases)
            test_cases = [tc for tc in test_cases if tc['station'] not in exclude_set]
            print(f"[FILTER] 排除站 {exclude_set}: {before} → {len(test_cases)} cases")
            if not test_cases:
                print("[ERROR] 排除後無可用測試案例")
                return

        # 驗證洩漏
        leak_count = 0
        for tc in test_cases:
            if tc['filename'] not in test_sources:
                print(f"[LEAK WARNING] {tc['filename']} 不在 test split 中!")
                leak_count += 1
        if leak_count > 0:
            print(f"[ERROR] 發現 {leak_count} 個洩漏案例!")
            return
        else:
            print(f"[OK] 所有 {len(test_cases)} 個案例均在 H5 test split 中 (無洩漏)")

    # 驗證: 5 個指定案例
    required_keys = {(c['nwp_dir'], c['file'] + '.parquet') for c in REQUIRED_CASES}
    found_required = set()
    for tc in test_cases:
        key = (tc.get('nwp_dir', ''), tc['filename'])
        if key in required_keys:
            found_required.add(key)
    print(f"[OK] 指定案例: {len(found_required)}/{len(REQUIRED_CASES)} 包含")
    if len(found_required) < len(REQUIRED_CASES):
        missing = required_keys - found_required
        print(f"[WARNING] 缺少: {missing}")

    # 保存選取的測試案例 JSON（可用 --force_cases_file 重現）
    selected_cases_for_save = []
    for tc in test_cases:
        # 格式與 --force_cases_file 相容
        fname = tc['filename']
        if fname.endswith('.parquet'):
            fname = fname[:-len('.parquet')]
        selected_cases_for_save.append({
            'nwp_dir': tc.get('nwp_dir', ''),
            'file': fname,
            'station': tc.get('station', ''),
            'date': tc.get('date', ''),
            'aliased_ratio': tc.get('aliased_ratio', -1),
        })
    cases_json_path = os.path.join(args.output_dir, "selected_test_cases.json")
    with open(cases_json_path, 'w') as f:
        json.dump(selected_cases_for_save, f, indent=2, default=str)
    print(f"[SAVE] 測試案例清單: {cases_json_path} ({len(selected_cases_for_save)} cases)")

    # 3. 建立模型
    print(f"\n{'='*60}")
    print("[PHASE 2] 模型載入")
    print(f"{'='*60}")

    # 3a. 我們的模型
    print("\n[MODEL] 建立我們的模型...")
    model = build_mixed_patch_model_for_inference(use_physics_model=args.use_physics_model)

    # Dummy forward 初始化
    first_raw, first_nyq, _ = load_velocity_data(test_cases[0]['raw'])
    padded_dummy, _, _ = auto_zero_pad(first_raw, layers=4, fill_value=0)
    dummy_vel = padded_dummy[None, None, :, :, None].astype(np.float32)
    dummy_nyq = np.array([[first_nyq]], dtype=np.float32)
    _ = model({'vel': dummy_vel, 'nyq': dummy_nyq}, training=False)
    model.load_weights(args.model_path)
    print(f"[OK] 我們的模型載入完成: {args.model_path}")

    # 3b. 論文模型
    print("\n[MODEL] 建立論文模型...")
    paper_model = build_paper_model(args.paper_model_path)

    # 4. 逐案測試
    print(f"\n{'='*60}")
    print(f"[PHASE 3] 逐案測試 ({len(test_cases)} cases)")
    print(f"{'='*60}")

    all_results = []
    for i, triplet in enumerate(test_cases):
        print(f"\n--- Case {i+1}/{len(test_cases)} ---")
        try:
            # 依個案名稱（nwp_dir）建立子資料夾
            case_subdir = triplet.get('nwp_dir', '')
            if case_subdir:
                case_output_dir = os.path.join(args.output_dir, case_subdir)
                os.makedirs(case_output_dir, exist_ok=True)
            else:
                case_output_dir = args.output_dir
            result = test_single_case(
                model, paper_model, triplet,
                case_output_dir, args.enable_viz,
                conf_threshold=args.conf_threshold,
                enable_vcorrected_pp=args.enable_vcorrected_pp,
                vcorrected_threshold=args.vcorrected_threshold,
                enable_geo_viz=args.enable_geo_viz,
                shape_path=args.shape_path,
                downsample_720=args.downsample_720,
                save_fields=args.save_fields,
                save_confidence=args.save_confidence)
            if result is not None:
                all_results.append(result)
        except Exception as e:
            print(f"[ERROR] {triplet['filename']} 失敗: {e}")
            import traceback
            traceback.print_exc()

    # 5. 彙整
    print(f"\n{'='*60}")
    print("[PHASE 4] 結果彙整")
    print(f"{'='*60}")
    summary = summarize_results(all_results, args.output_dir)

    # 最終驗證
    print(f"\n[VERIFY] 最終驗證:")
    if all_results:
        # RAW RMSE >> 0
        raw_rmses = [r['raw_metrics']['rmse'] for r in all_results
                     if not np.isnan(r['raw_metrics']['rmse'])]
        print(f"   RAW RMSE 平均: {np.mean(raw_rmses):.3f} (應 >> 0)")
        # 模型 RMSE < RAW
        ours_rmses = [r['ours_metrics']['rmse'] for r in all_results
                      if not np.isnan(r['ours_metrics']['rmse'])]
        paper_rmses = [r['paper_metrics']['rmse'] for r in all_results
                       if not np.isnan(r['paper_metrics']['rmse'])]
        print(f"   Ours RMSE 平均: {np.mean(ours_rmses):.3f} (應 < RAW)")
        print(f"   Paper RMSE 平均: {np.mean(paper_rmses):.3f} (應 < RAW)")

        # 覆蓋站數
        stations = set(r['station'] for r in all_results)
        print(f"   覆蓋站數: {len(stations)} ({', '.join(sorted(stations))})")

        # 指定案例
        req_found = sum(1 for r in all_results if r['is_required_case'])
        print(f"   指定案例: {req_found}/{len(REQUIRED_CASES)}")

    print(f"\n[DONE] 完成! 結果在 {args.output_dir}")


if __name__ == "__main__":
    main()
