#!/usr/bin/env python3
"""
Append new cases (e.g. additional squall lines) to existing realobs_train_patches.h5

Usage:
    python realobs_append_patches.py \
        --realobs_root /path/to/typhoonnew \
        --target_h5 data/realobs_train_patches.h5 \
        --append_cases 20190419_SQUALLLINE 20230418_SQUALLLINE \
                       20240330_SQUALLLINE 20240331_SQUALLLINE \
        --val_ratio 0.15 \
        --num_patches_per_sweep 8 \
        --augment_rotations 3 \
        --augment_matrix 1 \
        --minority_multiplier 5

Safety:
- Backs up H5 metadata before appending
- Verifies all datasets are resizable before starting
- Uses small batch writes (1000 patches each) for safer resume
"""

import os
import sys
import json
import argparse
import numpy as np
import h5py
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from realobs_to_patches import (
    find_realobs_pairs,
    extract_patches_from_pair,
)


def verify_h5_extendable(h5_path):
    """確認 H5 內所有 dataset 都可以擴展。"""
    with h5py.File(h5_path, 'r') as h5:
        for split in ['train', 'val']:
            if split not in h5:
                print(f"[WARN] {split} split 不存在於 H5")
                continue
            for key in h5[split].keys():
                ds = h5[split][key]
                if ds.maxshape[0] is not None:
                    raise RuntimeError(
                        f"{split}/{key} 不可擴展 maxshape={ds.maxshape}, "
                        f"無法 append。需要重建 H5。"
                    )
    return True


def backup_h5_metadata(h5_path, backup_path):
    """備份 H5 的 split sizes 跟 case_name 分布，方便回滾驗證。"""
    metadata = {}
    with h5py.File(h5_path, 'r') as h5:
        for split in h5.keys():
            if split == 'metadata':
                continue
            grp = h5[split]
            if 'case_name' not in grp:
                continue
            sizes = {}
            case_names = grp['case_name'][:].astype(str)
            from collections import Counter
            sizes['total'] = len(case_names)
            sizes['by_case'] = dict(Counter(case_names))
            metadata[split] = sizes

    with open(backup_path, 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  Metadata 備份至: {backup_path}")
    return metadata


def append_patches_to_h5(h5_path, split_name, patches):
    """把 patches list append 到 H5 指定 split。
    用 resize(n, axis=0) 形式跟原腳本 _append_patches_to_h5 完全對齊。
    """
    if not patches:
        return 0

    with h5py.File(h5_path, 'a') as h5:
        grp = h5[split_name]

        n_old = grp['vel'].shape[0]
        n_new = n_old + len(patches)

        # Resize 所有 dataset（沿用原腳本的 axis=0 風格，
        # 不依賴 patch_size hardcode）
        grp['vel'].resize(n_new, axis=0)
        grp['gt_vel'].resize(n_new, axis=0)
        grp['target_vel'].resize(n_new, axis=0)
        grp['alias_label'].resize(n_new, axis=0)
        grp['nyq'].resize(n_new, axis=0)
        grp['alias_ratio'].resize(n_new, axis=0)
        for name in ['patch_type', 'station', 'case_name',
                     'source_file', 'augment_type']:
            grp[name].resize(n_new, axis=0)

        # 寫入新資料（完全對齊 realobs_to_patches.py._append_patches_to_h5 邏輯）
        grp['vel'][n_old:n_new] = np.stack([p['vel'] for p in patches])
        grp['gt_vel'][n_old:n_new] = np.stack([p['gt_vel'] for p in patches])
        grp['target_vel'][n_old:n_new] = np.stack([p['target_vel'] for p in patches])
        grp['alias_label'][n_old:n_new] = np.stack(
            [p['alias_label'] for p in patches]).astype(np.int8)
        grp['nyq'][n_old:n_new] = np.stack([p['nyq'] for p in patches])
        grp['alias_ratio'][n_old:n_new] = np.array([p['alias_ratio'] for p in patches])
        grp['patch_type'][n_old:n_new] = [p['patch_type'] for p in patches]
        grp['station'][n_old:n_new] = [p['station'] for p in patches]
        grp['case_name'][n_old:n_new] = [p['case_name'] for p in patches]
        grp['source_file'][n_old:n_new] = [p['source_file'] for p in patches]
        grp['augment_type'][n_old:n_new] = [p['augment_type'] for p in patches]

        # 更新 attrs num_patches
        grp.attrs['num_patches'] = np.int32(n_new)

        return len(patches)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--realobs_root', required=True)
    parser.add_argument('--target_h5', required=True,
                        help='要 append 的目標 H5（會原地修改）')
    parser.add_argument('--append_cases', nargs='+', required=True,
                        help='要新增到 train+val 的個案名稱')
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--elevation', type=str, default='all')
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--num_patches_per_sweep', type=int, default=8)
    parser.add_argument('--min_alias_pixels', type=int, default=10)
    parser.add_argument('--augment_rotations', type=int, default=3)
    parser.add_argument('--augment_matrix', type=int, default=1)
    parser.add_argument('--minority_multiplier', type=int, default=5,
                        help='颮線（少數類別）的增強倍數')
    parser.add_argument('--clean_ratio_threshold', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=46)
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='每批 append 的 patch 數，越小越安全')
    parser.add_argument('--dry_run', action='store_true',
                        help='只列出會做的事，不實際 append')
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("=" * 60)
    print("Real Obs Append Patches 工具")
    print("=" * 60)
    print(f"  目標 H5: {args.target_h5}")
    print(f"  新增個案: {args.append_cases}")
    print(f"  Dry run: {args.dry_run}")
    print()

    # 安全檢查
    print("Step 1: 驗證 H5 可擴展...")
    verify_h5_extendable(args.target_h5)
    print("  ✓ 所有 dataset 都是 maxshape=(None,...)，可 resize")

    # 備份
    print("\nStep 2: 備份 metadata...")
    backup_path = args.target_h5.replace('.h5', '_metadata_before_append.json')
    metadata_before = backup_h5_metadata(args.target_h5, backup_path)
    for split, info in metadata_before.items():
        print(f"  {split}: {info['total']} patches, "
              f"cases={list(info['by_case'].keys())}")

    # 找新個案 raw/gt pairs
    print(f"\nStep 3: 尋找 raw/gt pairs (新增個案)...")
    all_pairs = find_realobs_pairs(args.realobs_root, args.append_cases,
                                   elevation=args.elevation)

    if not all_pairs:
        print("[ERROR] 沒有找到任何新增個案的 raw/gt 配對！")
        return

    # 按 source file 切 train/val
    print(f"\nStep 4: 切分 train/val (val_ratio={args.val_ratio})...")
    source_groups = defaultdict(list)
    for p in all_pairs:
        key = (p['station'], p['date'], p['time'])
        source_groups[key].append(p)

    source_keys = list(source_groups.keys())
    np.random.shuffle(source_keys)
    n_val_sources = max(1, int(len(source_keys) * args.val_ratio))
    val_source_keys = set(map(tuple, source_keys[:n_val_sources]))

    train_pairs = [p for p in all_pairs
                   if (p['station'], p['date'], p['time']) not in val_source_keys]
    val_pairs = [p for p in all_pairs
                 if (p['station'], p['date'], p['time']) in val_source_keys]

    print(f"  Train: {len(train_pairs)} sweeps "
          f"({len(source_keys) - n_val_sources} sources)")
    print(f"  Val: {len(val_pairs)} sweeps ({n_val_sources} sources)")

    if args.dry_run:
        print("\n[DRY RUN] 不實際 append，結束。")
        return

    # 提取 + append（分批）
    print(f"\nStep 5: 提取 patches + append 到 H5...")
    for split_name, pairs in [('train', train_pairs), ('val', val_pairs)]:
        if not pairs:
            continue
        print(f"\n  處理 {split_name} ({len(pairs)} sweeps)...")

        batch = []
        n_appended_total = 0
        for pair in tqdm(pairs, desc=f"  {split_name}"):
            # 颮線增強：minority_multiplier（與原腳本邏輯一致）
            # 原腳本 realobs_to_patches.py:530-544
            #   把 mult 乘進 num_patches / augment_rotations / augment_matrix
            #   而不是重複呼叫 extract_patches_from_pair
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
                include_clean=True,
            )
            batch.extend(patches)

            # 批次寫入
            while len(batch) >= args.batch_size:
                chunk = batch[:args.batch_size]
                batch = batch[args.batch_size:]
                n = append_patches_to_h5(args.target_h5, split_name, chunk)
                n_appended_total += n

        # 寫剩餘
        if batch:
            n = append_patches_to_h5(args.target_h5, split_name, batch)
            n_appended_total += n

        print(f"  {split_name} 新增 {n_appended_total} patches")

    # 最終驗證
    print(f"\nStep 6: 最終驗證...")
    after_path = args.target_h5.replace('.h5', '_metadata_after_append.json')
    metadata_after = backup_h5_metadata(args.target_h5, after_path)
    for split, info in metadata_after.items():
        before_total = metadata_before.get(split, {}).get('total', 0)
        delta = info['total'] - before_total
        print(f"  {split}: {info['total']} patches (+{delta})")
        print(f"    cases: {list(info['by_case'].keys())}")

    print(f"\n✓ Append 完成！備份檔：{backup_path}, {after_path}")


if __name__ == '__main__':
    main()
