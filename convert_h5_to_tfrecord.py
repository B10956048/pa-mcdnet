#!/usr/bin/env python3
"""
H5 → TFRecord 轉檔工具

把 mixed_patches H5 (v4.0 大 array 格式) 轉成 TFRecord shards
讓 tf.data 走 C++ pipeline，繞過 Python GIL 限制

使用範例：
    python convert_h5_to_tfrecord.py \
        --h5_path /dev/shm/realobs_train_patches.h5 \
        --output_dir /dev/shm/realobs_tfrecord \
        --splits train,val \
        --num_shards 32 \
        --compression GZIP

輸出檔案結構：
    output_dir/
        train-00000-of-00032.tfrecord.gz
        train-00001-of-00032.tfrecord.gz
        ...
        train_metadata.json
        val-00000-of-00032.tfrecord.gz
        ...
        val_metadata.json

對應 loader: mixed_patch_train_physics_constraints.load_mixed_patches_from_tfrecord
"""

import os
os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')
import argparse
import json
import time
from pathlib import Path

import numpy as np
import h5py
import tensorflow as tf
from tqdm import tqdm


# ============================================================================
# tf.train.Example helpers
# ============================================================================
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[float(value)]))


def serialize_patch(vel, nyq, alias_label, gt_vel, target_vel, patch_type):
    """
    把單一 patch 序列化成 tf.train.Example bytes。

    所有 array 用 raw bytes 存（C++ 解析最快），shape 從 metadata 還原。
    """
    feature = {
        'vel': _bytes_feature(np.ascontiguousarray(vel, dtype=np.float32).tobytes()),
        'nyq': _float_feature(float(nyq.item() if hasattr(nyq, 'item') else
                                    (nyq[0] if hasattr(nyq, '__len__') else nyq))),
        'alias_label': _bytes_feature(np.ascontiguousarray(alias_label, dtype=np.int32).tobytes()),
        'gt_vel': _bytes_feature(np.ascontiguousarray(gt_vel, dtype=np.float32).tobytes()),
        'target_vel': _bytes_feature(np.ascontiguousarray(target_vel, dtype=np.float32).tobytes()),
        'patch_type': _bytes_feature(
            patch_type.encode('utf-8') if isinstance(patch_type, str)
            else (patch_type if isinstance(patch_type, bytes) else str(patch_type).encode('utf-8'))
        ),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


# ============================================================================
# 主轉檔邏輯
# ============================================================================
def load_patch_types_vectorized(split_group, N):
    """向量化讀取所有 patch_type（與 loader 邏輯一致）"""
    if 'patch_type' in split_group:
        pt_raw = split_group['patch_type'][:]
        if pt_raw.dtype.kind == 'S':
            return np.char.decode(pt_raw, 'utf-8')
        elif pt_raw.dtype.kind == 'O':
            return np.array([
                (pt.decode('utf-8') if isinstance(pt, bytes) else str(pt))
                for pt in pt_raw
            ])
        else:
            return pt_raw.astype(str)
    return np.full(N, 'aliased', dtype=object)


def convert_split(h5_path, output_dir, split, num_shards, compression, chunk_size,
                   h5_cache_mb=512, seed=46, mode='sequential'):
    """
    轉換單一 split (train/val/test)。

    Mode = 'sequential' (預設，**快 50-100x**):
        1. 順序讀 H5（H5 內部 chunk 連續存取，無解壓爆量）
        2. Round-robin 寫入 NUM_SHARDS 個 writer
        3. 相鄰 patches 進不同 shard → 訓練時 interleave + shuffle buffer 提供隨機性
        4. 無需事先 shuffle indices（隨機性留給 training pipeline）

    Mode = 'shuffle' (舊版，慢但保留可選):
        1. 全局 shuffle indices
        2. 隨機 fancy index 讀 H5（每 patch 觸發內部 chunk 解壓 → 慢）
        3. 適用情況：H5 沒內部壓縮、且想要寫入時就完成所有 shuffling
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print(f"\n{'='*70}")
    print(f"轉換 split={split}（mode={mode}）")
    print(f"  H5 source : {h5_path}")
    print(f"  Output dir: {output_dir}")
    print(f"  Shards    : {num_shards}")
    print(f"  Compress  : {compression or 'NONE'}")
    print(f"{'='*70}")

    h5_cache_bytes = h5_cache_mb * 1024 * 1024

    with h5py.File(h5_path, 'r', rdcc_nbytes=h5_cache_bytes, rdcc_nslots=100003) as h5f:
        if split not in h5f:
            raise ValueError(f"H5 中找不到 split '{split}'，可用: {list(h5f.keys())}")

        sg = h5f[split]
        if 'vel' not in sg:
            raise ValueError(f"split '{split}' 沒有 'vel' dataset")

        N = sg['vel'].shape[0]
        has_target_vel = 'target_vel' in sg
        vel_shape = tuple(sg['vel'].shape[1:])
        label_shape = tuple(sg['alias_label'].shape[1:])
        gt_shape = tuple(sg['gt_vel'].shape[1:])
        nyq_shape = tuple(sg['nyq'].shape[1:]) if sg['nyq'].ndim > 1 else (1,)

        # 印 H5 內部 chunking（診斷用）
        vel_chunks = sg['vel'].chunks
        vel_compression = sg['vel'].compression
        print(f"\n📊 共 {N} patches")
        print(f"   vel shape        : {vel_shape}")
        print(f"   vel H5 chunks    : {vel_chunks}  （內部 chunk size）")
        print(f"   vel H5 compress  : {vel_compression}  （內部壓縮類型）")
        print(f"   alias_label shape: {label_shape}")
        print(f"   gt_vel shape     : {gt_shape}")
        print(f"   nyq shape        : {nyq_shape}")
        print(f"   has target_vel   : {has_target_vel}")

        # 向量化讀 patch_type
        print(f"\n🔍 讀取 patch_type（向量化）...")
        t = time.time()
        all_patch_types = load_patch_types_vectorized(sg, N)
        print(f"   完成，用 {time.time()-t:.1f}s")

        # 統計 patch_type 分布
        unique, counts = np.unique(all_patch_types, return_counts=True)
        patch_type_counts = {str(u): int(c) for u, c in zip(unique, counts)}
        print(f"   分布: {patch_type_counts}")

        # 切 shards
        shard_size = (N + num_shards - 1) // num_shards
        print(f"\n📦 每 shard 約 {shard_size} patches")

        tf_options = tf.io.TFRecordOptions(compression_type=compression)
        shard_paths = []
        ext = '.tfrecord.gz' if compression == 'GZIP' else '.tfrecord'

        if mode == 'sequential':
            # ════════════════════════════════════════════════════════════════
            # SEQUENTIAL MODE（推薦，快 50-100x）
            # 同時開 NUM_SHARDS 個 writer，順序讀 H5，round-robin 分派
            # ════════════════════════════════════════════════════════════════
            writers = []
            for shard_id in range(num_shards):
                shard_path = output_dir / f'{split}-{shard_id:05d}-of-{num_shards:05d}{ext}'
                shard_paths.append(shard_path.name)
                writers.append(tf.io.TFRecordWriter(str(shard_path), options=tf_options))

            try:
                with tqdm(total=N, desc=f'寫入 {split}', unit='patch') as pbar:
                    # 順序大 chunk 讀（H5 chunk 連續，無 random fancy index 爆量）
                    for chunk_start in range(0, N, chunk_size):
                        chunk_end = min(chunk_start + chunk_size, N)
                        # H5 slice (連續) — 最快讀法
                        chunk_vel = sg['vel'][chunk_start:chunk_end]
                        chunk_nyq = sg['nyq'][chunk_start:chunk_end]
                        chunk_label = sg['alias_label'][chunk_start:chunk_end]
                        chunk_gt = sg['gt_vel'][chunk_start:chunk_end]
                        chunk_pt = all_patch_types[chunk_start:chunk_end]
                        if has_target_vel:
                            chunk_target = sg['target_vel'][chunk_start:chunk_end]

                        for j in range(chunk_end - chunk_start):
                            idx = chunk_start + j
                            shard_id = idx % num_shards  # round-robin
                            patch_type = str(chunk_pt[j])
                            if has_target_vel:
                                target_vel = chunk_target[j]
                            else:
                                target_vel = (chunk_vel[j].copy()
                                              if patch_type == 'clean'
                                              else chunk_gt[j].copy())
                            serialized = serialize_patch(
                                chunk_vel[j], chunk_nyq[j],
                                chunk_label[j], chunk_gt[j],
                                target_vel, patch_type
                            )
                            writers[shard_id].write(serialized)
                        pbar.update(chunk_end - chunk_start)
            finally:
                for w in writers:
                    w.close()

        elif mode == 'shuffle':
            # ════════════════════════════════════════════════════════════════
            # SHUFFLE MODE（舊版，慢，保留可選）
            # 全局 shuffle indices → 隨機 fancy index 讀 H5
            # ════════════════════════════════════════════════════════════════
            rng = np.random.RandomState(seed)
            indices = np.arange(N)
            rng.shuffle(indices)
            print(f"\n🔀 全局 shuffle 完成 (seed={seed})")

            with tqdm(total=N, desc=f'寫入 {split}', unit='patch') as pbar:
                for shard_id in range(num_shards):
                    shard_idx = indices[shard_id * shard_size : (shard_id + 1) * shard_size]
                    if len(shard_idx) == 0:
                        continue
                    shard_path = output_dir / f'{split}-{shard_id:05d}-of-{num_shards:05d}{ext}'
                    shard_paths.append(shard_path.name)
                    with tf.io.TFRecordWriter(str(shard_path), options=tf_options) as writer:
                        for chunk_start in range(0, len(shard_idx), chunk_size):
                            chunk_raw = shard_idx[chunk_start : chunk_start + chunk_size]
                            sorted_idx = np.sort(chunk_raw)
                            chunk_vel = sg['vel'][sorted_idx]
                            chunk_nyq = sg['nyq'][sorted_idx]
                            chunk_label = sg['alias_label'][sorted_idx]
                            chunk_gt = sg['gt_vel'][sorted_idx]
                            chunk_pt = all_patch_types[sorted_idx]
                            if has_target_vel:
                                chunk_target = sg['target_vel'][sorted_idx]
                            for j in range(len(sorted_idx)):
                                patch_type = str(chunk_pt[j])
                                if has_target_vel:
                                    target_vel = chunk_target[j]
                                else:
                                    target_vel = (chunk_vel[j].copy()
                                                  if patch_type == 'clean'
                                                  else chunk_gt[j].copy())
                                serialized = serialize_patch(
                                    chunk_vel[j], chunk_nyq[j],
                                    chunk_label[j], chunk_gt[j],
                                    target_vel, patch_type
                                )
                                writer.write(serialized)
                            pbar.update(len(sorted_idx))
        else:
            raise ValueError(f"未知 mode: {mode}（可用：'sequential' 或 'shuffle'）")

        elapsed = time.time() - t_start
        print(f"\n✅ 全部 shard 寫入完成，總耗時 {elapsed/60:.1f} 分鐘")

    # 寫 metadata.json
    meta = {
        'split': split,
        'num_patches': int(N),
        'num_shards': int(len([s for s in shard_paths])),
        'compression': compression or '',
        'has_target_vel': bool(has_target_vel),
        'h5_source': str(h5_path),
        'vel_shape': list(vel_shape),
        'alias_label_shape': list(label_shape),
        'gt_vel_shape': list(gt_shape),
        'nyq_shape': list(nyq_shape),
        'patch_type_counts': patch_type_counts,
        'shard_files': shard_paths,
        'seed': int(seed),
        'conversion_time_seconds': float(elapsed),
    }
    meta_path = output_dir / f'{split}_metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"📝 Metadata 已寫入: {meta_path}")

    # 檢查輸出大小
    total_size = sum((output_dir / s).stat().st_size for s in shard_paths)
    print(f"💾 總輸出大小: {total_size / 1024**3:.2f} GB")
    print(f"   平均每 patch: {total_size / N / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(
        description='H5 patches → TFRecord shards 轉檔工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--h5_path', type=str, required=True,
                        help='來源 H5 檔（v4.0 大 array 格式）')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='輸出目錄（會自動建立）')
    parser.add_argument('--splits', type=str, default='train,val',
                        help='要轉的 split，逗號分隔（預設: train,val）')
    parser.add_argument('--num_shards', type=int, default=32,
                        help='每個 split 的 shard 數量（預設: 32）')
    parser.add_argument('--compression', type=str, default='GZIP',
                        choices=['GZIP', 'NONE', ''],
                        help='壓縮類型（預設: GZIP，跟 H5 差不多大；NONE 更快但大 3.5x）')
    parser.add_argument('--chunk_size', type=int, default=512,
                        help='H5 chunked read 大小（預設: 512 patches）')
    parser.add_argument('--h5_cache_mb', type=int, default=512,
                        help='H5 read cache 大小 MB（預設: 512）')
    parser.add_argument('--seed', type=int, default=46,
                        help='Shuffle seed（預設: 46，跟訓練一致；僅 mode=shuffle 用）')
    parser.add_argument('--mode', type=str, default='sequential',
                        choices=['sequential', 'shuffle'],
                        help='讀法：sequential（快 50-100x，推薦）/ shuffle（舊版慢）')
    args = parser.parse_args()

    compression = '' if args.compression == 'NONE' else args.compression
    splits = [s.strip() for s in args.splits.split(',') if s.strip()]

    print(f"配置:")
    print(f"  H5         : {args.h5_path}")
    print(f"  Output     : {args.output_dir}")
    print(f"  Splits     : {splits}")
    print(f"  Shards     : {args.num_shards} per split")
    print(f"  Compression: {compression or 'NONE'}")
    print(f"  Chunk size : {args.chunk_size}")
    print(f"  H5 cache   : {args.h5_cache_mb} MB")
    print(f"  Seed       : {args.seed}")

    for split in splits:
        convert_split(
            h5_path=args.h5_path,
            output_dir=args.output_dir,
            split=split,
            num_shards=args.num_shards,
            compression=compression,
            chunk_size=args.chunk_size,
            h5_cache_mb=args.h5_cache_mb,
            seed=args.seed,
            mode=args.mode,
        )

    print(f"\n{'='*70}")
    print(f"全部完成！輸出目錄: {args.output_dir}")
    print(f"{'='*70}")
    print(f"\n下一步：訓練時 --nwp_h5 改成指向這個目錄")
    print(f"  python transfer_learning_complete.py --nwp_h5 {args.output_dir} ...")


if __name__ == '__main__':
    main()
