#!/usr/bin/env python3
"""
Real Obs 資料轉 H5 Patch 工具

將 typhoonnew/ 的真實雷達觀測（binary gz）轉為與 transfer_learning_complete.py 相容的 H5 Patch 格式。

用途：NWP pretrain (M4-v1) → 真實資料 transfer learning

使用方式：
    python realobs_to_patches.py \
        --realobs_root /path/to/typhoonnew \
        --output_h5 data/realobs_train_patches.h5 \
        --train_cases 2021_Chanthu 2023_DOKSURI 2023_HAIKUI 2007_SQUALLLINE \
        --val_ratio 0.15 \
        --patch_size 128 \
        --num_patches_per_sweep 8 \
        --augment_rotations 3 \
        --augment_matrix 1

H5 格式（big array, 與 transfer_learning_complete.py 相容���：
    {split}/vel          (N, 128, 128, 1) float32  -- aliased radar obs
    {split}/gt_vel       (N, 128, 128, 1) float32  -- dealiased GT (vdaqc/sfda)
    {split}/alias_label  (N, 128, 128)    int32    -- fold label
    {split}/nyq          (N, 1)           float32  -- Nyquist velocity
    {split}/alias_ratio  (N,)             float32  -- ratio of aliased pixels
    {split}/patch_type   (N,)             string   -- 'aliased' or 'clean'
"""

import os
import sys
import numpy as np
import h5py
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from dealiasing_pre_patch_v2 import (
    auto_zero_pad,
    compute_alias_label_swapped,
    augment_sample,
)


# ================================================================
# Data Loading
# ================================================================

def read_gz_radar(gz_path: str):
    """
    讀取 CWA 雷達 .gz 二進位檔（支援 bvel_raw / bvel / bvel_sfda / bvel_vdaqc）。
    Returns: vel (H,W) float32 NaN=invalid, nyquist float, meta dict
    """
    import gzip, struct
    with gzip.open(gz_path, 'rb') as gf:
        f = gf.read()
    header_end = 160
    header = struct.unpack('<16s36i', f[0:header_end])
    info = np.array(header[1:-1])
    h_scale = info[0]
    info_flt = info / h_scale
    nray = int(info_flt[14])
    ngate = int(info_flt[15])
    nyq = float(info_flt[10])
    try:
        name = header[0][:4].decode('utf-8').strip('\x00').strip()
    except Exception:
        name = 'UNKN'
    yyyy = int(info_flt[4]); mm = int(info_flt[5]); dd = int(info_flt[6])
    hh = int(info_flt[7]); mn = int(info_flt[8])
    n = nray * ngate
    raw_int = np.array(struct.unpack('<' + str(n) + 'i', f[header_end:]))
    # CWA fill values: -990 (decoded: -99.0) and -9990 (decoded: -999.0)
    valid = (raw_int != -990) & (raw_int != -9990)
    vel = (raw_int / info_flt[20]).reshape(nray, ngate).astype(np.float32)
    vel[~valid.reshape(nray, ngate)] = np.nan
    # azm_start (header info_flt[16]) 為雷達實際掃描起始方位
    # azm_sp = 360.0 / nray (720-ray 是 0.5°, 360-ray 是 1°)
    # 用於 720 raw vs 360 GT 對齊（避免單純 raw[::2] 造成 azimuth 錯位）
    azm_start = float(info_flt[16])
    azm_sp = 360.0 / nray if nray > 0 else 1.0
    meta = {
        'station': name,
        'date': f"{yyyy:04d}{mm:02d}{dd:02d}",
        'time': f"{hh:02d}{mn:02d}",
        'nray': nray, 'ngate': ngate,
        'azm_start': azm_start,
        'azm_sp':    azm_sp,
    }
    return vel, nyq, meta


# ================================================================
# Azimuth-aware Alignment (修正 720-ray raw vs 360-ray GT 對齊錯位 bug)
# ================================================================

def align_raw_to_gt_by_azimuth(raw_vel, raw_azm_start, raw_azm_sp,
                                gt_azm_start, gt_azm_sp, gt_naz):
    """
    依方位角將 raw 重新採樣到 GT 的 ray 順序（每個 GT ray 對應 raw 中環形距離最近的 ray）。

    解決 RCWF 720-ray + GT 360-ray 時，舊版 raw[::2] 直接砍奇數 ray 會造成
    azimuth 錯位（例：raw 從 69.71° 開始、GT 從 0° 開始，[::2] 後角度差仍達 70°）。

    Args:
        raw_vel: (N_raw, N_gates) raw 速度場
        raw_azm_start: raw 起始方位 (degrees)
        raw_azm_sp:    raw 方位間隔 (degrees)
        gt_azm_start:  GT 起始方位 (degrees)
        gt_azm_sp:     GT 方位間隔 (degrees)
        gt_naz:        GT ray 數量

    Returns:
        raw_aligned: (gt_naz, N_gates) 與 GT 物理位置對齊的 raw 速度場
    """
    raw_naz = raw_vel.shape[0]
    raw_az = (raw_azm_start + np.arange(raw_naz) * raw_azm_sp) % 360.0
    gt_az  = (gt_azm_start  + np.arange(gt_naz)  * gt_azm_sp ) % 360.0
    diff = np.abs(gt_az[:, None] - raw_az[None, :])
    circ_diff = np.minimum(diff, 360.0 - diff)
    nearest_idx = np.argmin(circ_diff, axis=1)
    return raw_vel[nearest_idx, :]


# ================================================================
# Case Discovery
# ================================================================

def find_realobs_pairs(realobs_root, case_names, elevation='all'):
    """
    從指定的個案目錄找出所有 (raw, gt) 配對。

    支援：
      - 2023 颱風: polar/binary/STATION/  bvel_raw + bvel_sfda
      - 2007/2019 颮線: polar/binary/STATION/  bvel + bvel_sfda
      - 2021_Chanthu: 平面目錄  bvel_raw + bvel_vdaqc

    Returns: list of dict {raw, gt, station, case_name, date, time, elev}
    """
    root = Path(realobs_root)
    all_pairs = []

    for case_name in case_names:
        case_dir = root / case_name
        if not case_dir.exists():
            print(f"  [WARN] 個案目錄不存在: {case_dir}")
            continue

        pairs_in_case = []

        # 判斷目錄結構
        binary_root = case_dir / 'polar' / 'binary'

        # GT 檔名候選（依優先序）：vdaqc → sfda → da
        GT_SUFFIX_CANDIDATES = ['bvel_vdaqc', 'bvel_sfda', 'bvel_da']

        def find_gt_for_raw(raw_path_str, raw_token):
            """Try each GT suffix; return first existing path or None."""
            for suffix in GT_SUFFIX_CANDIDATES:
                gt = Path(raw_path_str.replace(raw_token, suffix))
                if gt.exists():
                    return gt
            return None

        if binary_root.exists():
            # 深層格式：polar/binary/STATION/
            for st_dir in sorted(d for d in binary_root.iterdir() if d.is_dir()):
                station = st_dir.name
                for raw_gz in sorted(st_dir.glob('*.gz')):
                    nm = raw_gz.name
                    # 跳過 GT 檔（含任一 GT suffix）
                    if any(suf in nm for suf in ['vdaqc', 'sfda', 'bvel_da']):
                        continue

                    # elevation filter
                    parts = nm.split('.')
                    elev = parts[-2] if len(parts) >= 2 else '01'
                    if elevation != 'all' and elev != elevation:
                        continue

                    # 找對應 GT（依 raw 命名格式）
                    if 'bvel_raw' in nm:
                        gt_gz = find_gt_for_raw(str(raw_gz), 'bvel_raw')
                    elif '.bvel.' in nm:
                        # 早期格式：RCCG.20070402.0804.bvel.01.gz
                        gt_gz = None
                        for suffix in GT_SUFFIX_CANDIDATES:
                            candidate = Path(str(raw_gz).replace('.bvel.', f'.{suffix}.'))
                            if candidate.exists():
                                gt_gz = candidate
                                break
                    else:
                        continue

                    if gt_gz is not None:
                        date = parts[1] if len(parts) > 1 else ''
                        time_str = parts[2] if len(parts) > 2 else ''
                        pairs_in_case.append({
                            'raw': str(raw_gz),
                            'gt': str(gt_gz),
                            'station': station,
                            'case_name': case_name,
                            'date': date,
                            'time': time_str,
                            'elev': elev,
                        })
        else:
            # 平面格式：case_name/STATION.YYYYMMDD.HHMM.bvel_raw.NN.gz
            station_files = defaultdict(list)
            for gz_file in sorted(case_dir.glob('*.gz')):
                nm = gz_file.name
                # 跳過 GT 檔
                if any(suf in nm for suf in ['vdaqc', 'sfda', 'bvel_da']):
                    continue
                if 'bvel_raw' not in nm:
                    continue

                parts = nm.split('.')
                station = parts[0] if parts else 'UNKN'
                elev = parts[-2] if len(parts) >= 2 else '01'
                if elevation != 'all' and elev != elevation:
                    continue

                gt_gz = find_gt_for_raw(str(gz_file), 'bvel_raw')

                if gt_gz is not None:
                    date = parts[1] if len(parts) > 1 else ''
                    time_str = parts[2] if len(parts) > 2 else ''
                    pairs_in_case.append({
                        'raw': str(gz_file),
                        'gt': str(gt_gz),
                        'station': station,
                        'case_name': case_name,
                        'date': date,
                        'time': time_str,
                        'elev': elev,
                    })

        print(f"  {case_name}: {len(pairs_in_case)} raw/gt pairs")
        all_pairs.extend(pairs_in_case)

    return all_pairs


# ================================================================
# Patch Extraction
# ================================================================

def extract_patches_from_pair(pair, patch_size=128, num_patches=5,
                              min_alias_pixels=10, clean_ratio_threshold=0.7,
                              augment_rotations=0, augment_matrix=0,
                              include_clean=True):
    """
    從一組 (raw, gt) 配對中提取 patches。

    Returns: list of patch dicts
    """
    try:
        raw_vel, nyq, meta = read_gz_radar(pair['raw'])
        gt_vel, _, gt_meta = read_gz_radar(pair['gt'])
    except Exception as e:
        print(f"  [ERR] 讀取失敗 {pair['raw']}: {e}")
        return []

    # 處理 RCWF 720-ray raw + 360-ray GT 對齊
    # 舊版 raw[::2] 直接砍奇數 ray 會 azimuth 錯位（raw 從 e.g. 69.71° 開始、
    # GT 從 0° 開始，[::2] 後 raw[i] 跟 gt[i] 仍對應不同物理方位，label 全錯）。
    # 新版用 azimuth-aware nearest-neighbor 對齊。
    if raw_vel.shape[0] != gt_vel.shape[0]:
        raw_vel = align_raw_to_gt_by_azimuth(
            raw_vel,
            raw_azm_start=meta['azm_start'], raw_azm_sp=meta['azm_sp'],
            gt_azm_start=gt_meta['azm_start'], gt_azm_sp=gt_meta['azm_sp'],
            gt_naz=gt_vel.shape[0],
        )
    # 若 ngate 不一致，裁切到較小值
    if raw_vel.shape == gt_vel.shape:
        pass  # 已對齊
    elif raw_vel.shape[0] == gt_vel.shape[0] and raw_vel.shape[1] != gt_vel.shape[1]:
        n = min(raw_vel.shape[1], gt_vel.shape[1])
        raw_vel = raw_vel[:, :n]
        gt_vel = gt_vel[:, :n]

    if raw_vel.shape != gt_vel.shape:
        print(f"  [ERR] shape 不一致: raw={raw_vel.shape}, gt={gt_vel.shape}")
        return []

    # 計算 alias label
    label_2d = compute_alias_label_swapped(raw_vel, gt_vel, nyq)

    # 準備資料版本（原始 + 方位角旋轉）
    data_versions = [('original', raw_vel, gt_vel, label_2d)]

    if augment_rotations > 0:
        n_az = raw_vel.shape[0]
        rotation_angles = [45, 90, 135, 180, 225, 270, 315]
        selected = np.random.choice(rotation_angles,
                                    min(augment_rotations, len(rotation_angles)),
                                    replace=False)
        for angle in selected:
            shift = round(n_az * angle / 360)
            rot_raw = np.roll(raw_vel, shift, axis=0)
            rot_gt = np.roll(gt_vel, shift, axis=0)
            rot_label = np.roll(label_2d, shift, axis=0)
            data_versions.append((f'rot_{angle}', rot_raw, rot_gt, rot_label))

    all_patches = []

    for version_name, v_raw, v_gt, v_label in data_versions:
        # Zero-pad to 16x multiple
        raw_pad, _, _ = auto_zero_pad(v_raw, layers=4, fill_value=np.nan)
        gt_pad, _, _ = auto_zero_pad(v_gt, layers=4, fill_value=np.nan)
        label_pad, _, _ = auto_zero_pad(v_label, layers=4, fill_value=0)
        H_pad, W_pad = raw_pad.shape

        if H_pad < patch_size or W_pad < patch_size:
            continue

        # 提取 aliased patches
        extracted = 0
        attempts = 0
        max_attempts = num_patches * 15

        while extracted < num_patches and attempts < max_attempts:
            attempts += 1
            top = np.random.randint(0, H_pad - patch_size + 1)
            left = np.random.randint(0, W_pad - patch_size + 1)

            p_raw = raw_pad[top:top+patch_size, left:left+patch_size]
            p_gt = gt_pad[top:top+patch_size, left:left+patch_size]
            p_label = label_pad[top:top+patch_size, left:left+patch_size]

            alias_mask = np.isin(p_label, [1, 2, 4, 5])
            alias_count = np.sum(alias_mask)
            total_valid = np.sum(p_label > 0)

            if total_valid == 0:
                continue

            if alias_count >= min_alias_pixels:
                alias_ratio = alias_count / total_valid
                patch_dict = _make_patch_dict(p_raw, p_gt, p_label, nyq,
                                             'aliased', alias_ratio, pair, version_name)
                all_patches.append(patch_dict)

                # Matrix augmentation
                for aug_i in range(augment_matrix):
                    aug_raw, aug_gt, aug_label = augment_sample(
                        p_raw.copy(), p_gt.copy(), p_label.copy())
                    aug_valid = np.sum(aug_label > 0)
                    aug_alias = np.sum(np.isin(aug_label, [1, 2, 4, 5]))
                    aug_ratio = aug_alias / aug_valid if aug_valid > 0 else 0.0
                    aug_dict = _make_patch_dict(aug_raw, aug_gt, aug_label, nyq,
                                               'aliased', aug_ratio, pair,
                                               f'{version_name}_aug{aug_i}')
                    all_patches.append(aug_dict)

                extracted += 1

        # 提取 clean patches
        if include_clean:
            clean_extracted = 0
            clean_target = max(1, num_patches // 3)
            attempts = 0

            while clean_extracted < clean_target and attempts < max_attempts:
                attempts += 1
                top = np.random.randint(0, H_pad - patch_size + 1)
                left = np.random.randint(0, W_pad - patch_size + 1)

                p_raw = raw_pad[top:top+patch_size, left:left+patch_size]
                p_gt = gt_pad[top:top+patch_size, left:left+patch_size]
                p_label = label_pad[top:top+patch_size, left:left+patch_size]

                total_valid = np.sum(p_label > 0)
                if total_valid == 0:
                    continue

                clean_count = np.sum(p_label == 3)
                clean_ratio = clean_count / total_valid

                if clean_ratio >= clean_ratio_threshold:
                    patch_dict = _make_patch_dict(p_raw, p_gt, p_label, nyq,
                                                 'clean', 0.0, pair, version_name)
                    all_patches.append(patch_dict)
                    clean_extracted += 1

    return all_patches


def _make_patch_dict(p_raw, p_gt, p_label, nyq, patch_type, alias_ratio, pair, version):
    """建立 patch dictionary"""
    # target_vel: aliased→修正為 gt, clean→保持原樣 (identity)
    if patch_type == 'aliased':
        target = p_gt[..., np.newaxis].astype(np.float32)
    else:
        target = p_raw[..., np.newaxis].astype(np.float32)

    return {
        'vel': p_raw[..., np.newaxis].astype(np.float32),
        'gt_vel': p_gt[..., np.newaxis].astype(np.float32),
        'target_vel': target,
        'alias_label': p_label.astype(np.int32),
        'nyq': np.array([nyq], dtype=np.float32),
        'patch_type': patch_type,
        'alias_ratio': float(alias_ratio),
        'station': pair['station'],
        'case_name': pair['case_name'],
        'source_file': f"{pair['station']}.{pair['date']}.{pair['time']}.{pair['elev']}",
        'augment_type': version,
    }


# ================================================================
# H5 Writing
# ================================================================

def _create_h5_split_group(h5, split_name, patch_size, chunk_n=4):
    """建立可動態擴展的 H5 group（maxshape 第一維為 None）。"""
    grp = h5.create_group(split_name)
    spatial_kw = dict(chunks=(chunk_n, patch_size, patch_size, 1),
                      compression='lzf', shuffle=True, maxshape=(None, patch_size, patch_size, 1))
    grp.create_dataset('vel', shape=(0, patch_size, patch_size, 1),
                       dtype=np.float32, **spatial_kw)
    grp.create_dataset('gt_vel', shape=(0, patch_size, patch_size, 1),
                       dtype=np.float32, **spatial_kw)
    grp.create_dataset('target_vel', shape=(0, patch_size, patch_size, 1),
                       dtype=np.float32, **spatial_kw)
    label_kw = dict(chunks=(chunk_n, patch_size, patch_size),
                    compression='lzf', shuffle=True, maxshape=(None, patch_size, patch_size))
    grp.create_dataset('alias_label', shape=(0, patch_size, patch_size),
                       dtype=np.int8, **label_kw)
    grp.create_dataset('nyq', shape=(0, 1), dtype=np.float32,
                       maxshape=(None, 1))
    grp.create_dataset('alias_ratio', shape=(0,), dtype=np.float32,
                       maxshape=(None,))
    dt = h5py.string_dtype()
    for name in ['patch_type', 'station', 'case_name', 'source_file', 'augment_type']:
        grp.create_dataset(name, shape=(0,), dtype=dt, maxshape=(None,))
    grp.attrs['format'] = 'big_array'
    return grp


def _append_patches_to_h5(grp, patches):
    """將一批 patches 追加到已建立的 H5 group 中。"""
    if not patches:
        return
    n_old = grp['vel'].shape[0]
    n_new = n_old + len(patches)
    # Resize all datasets
    ps = patches[0]['vel'].shape[0]
    grp['vel'].resize(n_new, axis=0)
    grp['gt_vel'].resize(n_new, axis=0)
    grp['target_vel'].resize(n_new, axis=0)
    grp['alias_label'].resize(n_new, axis=0)
    grp['nyq'].resize(n_new, axis=0)
    grp['alias_ratio'].resize(n_new, axis=0)
    for name in ['patch_type', 'station', 'case_name', 'source_file', 'augment_type']:
        grp[name].resize(n_new, axis=0)

    # Write
    grp['vel'][n_old:n_new] = np.stack([p['vel'] for p in patches])
    grp['gt_vel'][n_old:n_new] = np.stack([p['gt_vel'] for p in patches])
    grp['target_vel'][n_old:n_new] = np.stack([p['target_vel'] for p in patches])
    grp['alias_label'][n_old:n_new] = np.stack([p['alias_label'] for p in patches]).astype(np.int8)
    grp['nyq'][n_old:n_new] = np.stack([p['nyq'] for p in patches])
    grp['alias_ratio'][n_old:n_new] = np.array([p['alias_ratio'] for p in patches])
    grp['patch_type'][n_old:n_new] = [p['patch_type'] for p in patches]
    grp['station'][n_old:n_new] = [p['station'] for p in patches]
    grp['case_name'][n_old:n_new] = [p['case_name'] for p in patches]
    grp['source_file'][n_old:n_new] = [p['source_file'] for p in patches]
    grp['augment_type'][n_old:n_new] = [p['augment_type'] for p in patches]


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(description='Real Obs to H5 Patches')
    parser.add_argument('--realobs_root', type=str, required=True,
                        help='typhoonnew/ 根目錄')
    parser.add_argument('--output_h5', type=str, required=True,
                        help='輸出 H5 路徑')
    parser.add_argument('--train_cases', nargs='+', required=True,
                        help='訓練用個案名稱 (e.g. 2021_Chanthu 2023_DOKSURI)')
    parser.add_argument('--test_cases', nargs='+', default=None,
                        help='測試用個案名稱（可選，會另存 test split）')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='從 train_cases 中切出的 validation 比���')
    parser.add_argument('--elevation', type=str, default='all',
                        help='指定仰角 (e.g. 01, 02) 或 all')
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--num_patches_per_sweep', type=int, default=8,
                        help='每張掃描切幾個 aliased patches')
    parser.add_argument('--min_alias_pixels', type=int, default=10,
                        help='aliased patch 最少需要的 alias 像素數')
    parser.add_argument('--augment_rotations', type=int, default=3,
                        help='方位角旋轉增強次數 (0=不增強)')
    parser.add_argument('--augment_matrix', type=int, default=1,
                        help='每個 patch 的矩陣增強次數 (flip/rotate)')
    parser.add_argument('--minority_multiplier', type=int, default=5,
                        help='少數類別（颮線）的增強倍數（相對於颱風）')
    parser.add_argument('--include_clean', action='store_true', default=True,
                        help='包含 clean patches')
    parser.add_argument('--no_clean', action='store_true',
                        help='不包含 clean patches')
    parser.add_argument('--clean_ratio_threshold', type=float, default=0.7,
                        help='clean patch 最少需要的 clean pixel 比例')
    parser.add_argument('--seed', type=int, default=46)
    args = parser.parse_args()

    np.random.seed(args.seed)
    include_clean = not args.no_clean

    print("=" * 60)
    print("Real Obs → H5 Patch 轉換工具")
    print("=" * 60)
    print(f"  資料來源: {args.realobs_root}")
    print(f"  輸出: {args.output_h5}")
    print(f"  訓練個案: {args.train_cases}")
    if args.test_cases:
        print(f"  測試個案: {args.test_cases}")
    print(f"  Patch size: {args.patch_size}")
    print(f"  每掃描 patches: {args.num_patches_per_sweep}")
    print(f"  方位角���轉: {args.augment_rotations}")
    print(f"  矩陣��強: {args.augment_matrix}")
    print(f"  包含 clean patches: {include_clean}")
    print()

    # 找出所有 raw/gt 配對
    print("尋�� raw/gt 配對...")
    all_cases = args.train_cases + (args.test_cases or [])
    all_pairs = find_realobs_pairs(args.realobs_root, all_cases,
                                   elevation=args.elevation)

    if not all_pairs:
        print("[ERROR] 沒有找到任何 raw/gt 配對！")
        return

    # 分割 train / val / test
    train_pairs = [p for p in all_pairs if p['case_name'] in args.train_cases]
    test_pairs = [p for p in all_pairs if args.test_cases and p['case_name'] in args.test_cases]

    # 從 train_pairs 中切出 validation（按 source file 分組避免洩漏）
    source_groups = defaultdict(list)
    for p in train_pairs:
        key = (p['station'], p['date'], p['time'])
        source_groups[key].append(p)

    source_keys = list(source_groups.keys())
    np.random.shuffle(source_keys)
    n_val_sources = max(1, int(len(source_keys) * args.val_ratio))
    val_source_keys = set(map(tuple, source_keys[:n_val_sources]))
    train_source_keys = set(map(tuple, source_keys[n_val_sources:]))

    final_train_pairs = [p for p in train_pairs
                         if (p['station'], p['date'], p['time']) in train_source_keys]
    final_val_pairs = [p for p in train_pairs
                       if (p['station'], p['date'], p['time']) in val_source_keys]

    print(f"\n分割結果:")
    print(f"  Train: {len(final_train_pairs)} sweeps ({len(train_source_keys)} sources)")
    print(f"  Val: {len(final_val_pairs)} sweeps ({len(val_source_keys)} sources)")
    if test_pairs:
        print(f"  Test: {len(test_pairs)} sweeps")

    # 提取 patches — 邊提取邊寫入 H5，不在記憶體累積
    split_pairs = [
        ('train', final_train_pairs),
        ('val', final_val_pairs),
    ]
    if test_pairs:
        split_pairs.append(('test', test_pairs))

    import os
    os.makedirs(os.path.dirname(args.output_h5) or '.', exist_ok=True)

    with h5py.File(args.output_h5, 'w') as h5:
        for split_name, pairs in split_pairs:
            print(f"\n提取 {split_name} patches ({len(pairs)} sweeps)...")
            grp = _create_h5_split_group(h5, split_name, args.patch_size)
            n_aliased = 0
            n_clean = 0

            for pair in tqdm(pairs, desc=f"  {split_name}"):
                # 颮線個案自動增加增強倍數
                is_minority = 'SQUALLLINE' in pair['case_name'].upper()
                mult = args.minority_multiplier if is_minority else 1

                patches = extract_patches_from_pair(
                    pair,
                    patch_size=args.patch_size,
                    num_patches=args.num_patches_per_sweep * mult,
                    min_alias_pixels=args.min_alias_pixels,
                    clean_ratio_threshold=args.clean_ratio_threshold,
                    augment_rotations=min(7, args.augment_rotations * mult),
                    augment_matrix=args.augment_matrix * mult,
                    include_clean=include_clean,
                )
                if patches:
                    _append_patches_to_h5(grp, patches)
                    for p in patches:
                        if p['alias_ratio'] > 0.01:
                            n_aliased += 1
                        else:
                            n_clean += 1

            total_split = grp['vel'].shape[0]
            grp.attrs['num_patches'] = total_split
            print(f"  {split_name}: {total_split} patches (aliased={n_aliased}, clean={n_clean})")

    # Summary
    print("\n" + "=" * 60)
    print("完成！")
    with h5py.File(args.output_h5, 'r') as h5:
        total = 0
        for split_name in h5.keys():
            n = h5[split_name]['vel'].shape[0]
            total += n
            print(f"  {split_name}: {n} patches")
        print(f"  總 patches: {total}")
    print(f"  H5 saved: {args.output_h5}")
    print("=" * 60)


if __name__ == "__main__":
    main()
