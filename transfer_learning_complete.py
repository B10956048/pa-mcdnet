#!/usr/bin/env python3
"""
完整版遷移學習腳本：真實雷達 → NWP（RCCG/RCWF）

策略：
1. 載入預訓練模型（真實雷達訓練的完整模型）
2. 凍結 80% 的層（保護颱風 pattern 能力）
3. 微調決策層（學習 NWP 特殊 pattern）
4. 雙驗證集監控（每 epoch 測試颱風 + NWP）
5. Early Stopping（颱風性能下降超過 2% 則停止）
6. 保留所有 5 個分支（分類、回歸、物理、軟分類、置信度）

使用範例：
    # 步驟 1: 先轉換 NWP 為 H5 Patches
    python nwp_to_patches.py \
        --nwp_root /path/to/nwp_data_csv \
        --output_h5 data/nwp_patches_rccg_rcwf.h5 \
        --target_stations RCCG RCWF \
        --max_dirs 10

    # 步驟 2: 遷移學習
    python transfer_learning_complete.py \
        --pretrained_model "results/加入空間平滑_調整weight_spatial_improved_plan_b.1011_20251011_121622/spatial_improved_plan_b.1011_best.h5" \
        --nwp_h5 data/nwp_patches_rccg_rcwf.h5 \
        --typhoon_h5 data/mixed_patches_reorganized_aliased70p.h5 \
        --output_dir results/transfer_learning_rccg_rcwf \
        --freeze_ratio 0.8 \
        --learning_rate 5e-5 \
        --epochs 50 \
        --patience 5 \
        --baseline_typhoon_success 84.0
"""

# 重要：必須在所有其他導入之前執行修復
import fix_typing  # 這行必須放在最前面

import os
# Windows WDDM 無法申請大連續 CUDA 塊（BFC grow chunk 到 2GB 就失敗）
# 必須用 cuda_malloc_async：使用 CUDA 非同步記憶體池，不需要大連續塊
# 但 cuda_malloc_async 長時間訓練後池子碎片化 → resume 時 cuDNN algorithm profiling OOM
# 解法：同時停用 cuDNN autotune，避免 profiling 大量臨時分配造成碎片
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'  # 停用 cuDNN 演算法搜尋，避免 profiling 觸發碎片 OOM
import sys
import shutil  # 🆕 用於檔案操作
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import json
import pickle
import time
import argparse
import gc
from tqdm import tqdm
import h5py

# 導入模型和訓練相關
from unet_model.dealias_mulit_v2_physics import VelocityDealiaser, make_velocity_mask
from unet_model.feature_extraction_mulit_v2 import (
    create_downsampler,
    create_upsampler_cls,
    create_upsampler_reg
)

# 導入數據載入函數
# load_mixed_patches 是統一入口，自動偵測 H5 或 TFRecord 目錄
from mixed_patch_train_physics_constraints import (
    load_mixed_patches_from_h5,
    load_mixed_patches,
    mixed_patch_focal_loss,
)

# 設置隨機種子
SEED = 46
np.random.seed(SEED)
tf.random.set_seed(SEED)


def manual_transfer_weights(source_h5_path, target_h5_path, model):
    """
    手動轉移權重以修復 BatchNormalization 順序問題

    Args:
        source_h5_path: 源權重文件路徑（可能有順序問題）
        target_h5_path: 目標權重文件路徑（修復後）
        model: 已初始化的模型實例

    Returns:
        bool: 是否成功修復
    """
    print(f"\n🔧 執行手動權重轉移...")
    print(f"   源文件: {source_h5_path}")
    print(f"   目標文件: {target_h5_path}")

    def find_weight_in_h5(h5_file, weight_name):
        """在 H5 文件中遞迴查找權重"""
        clean_name = weight_name.replace(':0', '')

        def search_recursive(group, path=""):
            for key in group.keys():
                item = group[key]
                current_path = f"{path}/{key}" if path else key

                if isinstance(item, h5py.Dataset):
                    if current_path.endswith(clean_name) or current_path.endswith(weight_name):
                        return item[()]

                    parts = clean_name.split('/')
                    path_parts = current_path.split('/')
                    if len(path_parts) >= len(parts):
                        if path_parts[-len(parts):] == parts:
                            return item[()]

                elif isinstance(item, h5py.Group):
                    result = search_recursive(item, current_path)
                    if result is not None:
                        return result

            return None

        return search_recursive(h5_file)

    matched = 0
    failed = 0

    try:
        with h5py.File(source_h5_path, 'r') as h5f:
            for weight in model.weights:
                weight_name = weight.name
                weight_data = find_weight_in_h5(h5f, weight_name)

                if weight_data is not None:
                    if weight_data.shape == tuple(weight.shape):
                        weight.assign(weight_data)
                        matched += 1
                    else:
                        print(f"   ⚠️  形狀不匹配: {weight_name}")
                        failed += 1
                else:
                    print(f"   ❌ 找不到: {weight_name}")
                    failed += 1

        if failed > 0:
            print(f"   ❌ 轉移失敗: {failed} 個權重無法設置")
            return False

        # 保存修復後的權重
        model.save_weights(target_h5_path, save_format='h5')
        print(f"   ✅ 權重轉移成功: {matched}/{len(model.weights)}")
        return True

    except Exception as e:
        print(f"   ❌ 轉移過程出錯: {e}")
        return False


def verify_and_fix_weights(model_path, model_class_factory):
    """
    驗證權重文件是否可以正常載入，如果不行則自動修復

    Args:
        model_path: 權重文件路徑
        model_class_factory: 創建模型實例的工廠函數

    Returns:
        str: 可用的權重文件路徑（可能是原始或修復後的）
    """
    print(f"\n🔍 驗證權重文件: {model_path}")

    # 創建測試模型
    test_model = model_class_factory()

    # 初始化模型
    dummy_input = {
        'vel': tf.zeros((1, 1, 128, 128, 1), dtype=tf.float32),
        'nyq': tf.ones((1, 1), dtype=tf.float32)
    }
    _ = test_model(dummy_input, training=False)

    # 嘗試載入權重
    try:
        test_model.load_weights(model_path)
        print(f"   ✅ 權重文件正常，可以直接使用")
        return model_path
    except Exception as e:
        print(f"   ⚠️  載入失敗: {e}")
        print(f"   🔧 嘗試自動修復...")

        # 生成修復後的文件名
        fixed_path = model_path.replace('.h5', '_fixed.h5')

        # 執行手動轉移
        success = manual_transfer_weights(model_path, fixed_path, test_model)

        if success:
            # 再次驗證修復後的權重
            test_model2 = model_class_factory()
            _ = test_model2(dummy_input, training=False)
            try:
                test_model2.load_weights(fixed_path)
                print(f"   ✅ 修復成功！使用: {fixed_path}")
                return fixed_path
            except Exception as e2:
                print(f"   ❌ 修復後仍無法載入: {e2}")
                raise RuntimeError(f"無法修復權重文件: {model_path}")
        else:
            raise RuntimeError(f"權重轉移失敗: {model_path}")


def freeze_layers(model, freeze_ratio=0.8):
    """
    凍結模型的部分層

    Parameters:
    -----------
    model : tf.keras.Model
        要凍結的模型
    freeze_ratio : float
        凍結比例（0.8 = 凍結前 80% 的層）

    Returns:
    --------
    model : 凍結後的模型
    """
    total_layers = len(model.layers)
    freeze_until = int(total_layers * freeze_ratio)

    print(f"\n❄️  凍結策略:")
    print(f"   總層數: {total_layers}")
    print(f"   凍結前 {freeze_until} 層 ({freeze_ratio*100:.0f}%)")
    print(f"   訓練後 {total_layers - freeze_until} 層 ({(1-freeze_ratio)*100:.0f}%)")

    for i, layer in enumerate(model.layers):
        if i < freeze_until:
            layer.trainable = False
        else:
            layer.trainable = True

    return model


def print_trainable_summary(model):
    """打印模型的可訓練參數統計"""
    trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_count = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    total_count = trainable_count + non_trainable_count

    print("\n" + "="*60)
    print("模型參數統計")
    print("="*60)
    print(f"可訓練參數:     {trainable_count:,} ({trainable_count/total_count*100:.1f}%)")
    print(f"不可訓練參數:   {non_trainable_count:,} ({non_trainable_count/total_count*100:.1f}%)")
    print(f"總參數:         {total_count:,}")
    print("="*60 + "\n")


def compute_spatial_smoothness_loss(classification_logits, smoothness_type='fold_jump',
                                    raw_vel=None, nyq_vel=None):
    """
    計算空間平滑損失

    Args:
        classification_logits: (B, H, W, 6) 分類 logits
        smoothness_type: 平滑類型 ('fold_jump', 'total_variation', 'prob_diff', 'v_corrected')
        raw_vel: (B, H, W, 1) 原始速度，smoothness_type='v_corrected' 時必須提供
        nyq_vel: (B, H, W, 1) Nyquist 速度，smoothness_type='v_corrected' 時必須提供

    Returns:
        smoothness_loss: 標量
    """
    with tf.name_scope('spatial_smoothness'):
        if smoothness_type == 'v_corrected':
            # V_corrected smoothness：懲罰修正後速度的不連續
            # V_corr = V_raw + fold × 2Nyq
            # 正確修正：V_raw 跳但 fold 補償 → V_corr 平滑 → 不罰
            # FP 亂修：V_raw 平滑但 fold 亂跳 → V_corr 跳 2Nyq → 重罰
            probs = tf.nn.softmax(classification_logits, axis=-1)  # (B, H, W, 6)

            # fold_values: [class0=no-alias(0fold), class1=-2fold, class2=-1fold,
            #               class3=0fold, class4=+1fold, class5=+2fold]
            fold_values = tf.constant([0.0, -2.0, -1.0, 0.0, 1.0, 2.0], dtype=tf.float32)
            fold_values = tf.reshape(fold_values, [1, 1, 1, 6])

            # 期望 fold（可微分）
            expected_fold = tf.reduce_sum(probs * fold_values, axis=-1, keepdims=True)  # (B,H,W,1)

            # 處理 raw_vel 中的 NaN（無效像素填 0，不參與 smoothness）
            raw_vel_safe = tf.where(tf.math.is_nan(raw_vel), tf.zeros_like(raw_vel), raw_vel)
            valid_mask = tf.cast(~tf.math.is_nan(raw_vel[:, :, :, 0]), tf.float32)  # (B,H,W)

            # V_corrected = V_raw + expected_fold × 2 × Nyq
            v_corrected = raw_vel_safe + expected_fold * 2.0 * nyq_vel  # (B,H,W,1)
            v_corr_2d = v_corrected[:, :, :, 0]  # (B,H,W)

            # 相鄰像素的修正後速度差異
            diff_x = v_corr_2d[:, :, 1:] - v_corr_2d[:, :, :-1]
            diff_y = v_corr_2d[:, 1:, :] - v_corr_2d[:, :-1, :]

            # 只計算兩端都是有效像素的 pair
            valid_pair_x = valid_mask[:, :, 1:] * valid_mask[:, :, :-1]  # (B,H,W-1)
            valid_pair_y = valid_mask[:, 1:, :] * valid_mask[:, :-1, :]  # (B,H-1,W)

            # 用 Nyquist 值歸一化，使 loss 不受雷達 Nyquist 差異影響
            nyq_per_sample = tf.reduce_mean(nyq_vel, axis=[1, 2, 3], keepdims=True)
            nyq_3d = nyq_per_sample[:, :, :, 0]  # (B,1,1) → broadcast 到 (B,H,W-1)
            diff_x_norm = diff_x / (nyq_3d + 1e-6)
            diff_y_norm = diff_y / (nyq_3d + 1e-6)

            # Huber-like: 小梯度不罰（自然風切），大跳躍（>1 Nyquist）重罰
            threshold = 1.0  # 歸一化後，1.0 ≈ 1個 Nyquist 速度
            large_jump_x = tf.maximum(tf.abs(diff_x_norm) - threshold, 0.0)
            large_jump_y = tf.maximum(tf.abs(diff_y_norm) - threshold, 0.0)

            # 只對超過 threshold 的大跳躍 pair 計算 loss（避免被平滑 pair 稀釋）
            masked_x = tf.square(large_jump_x) * valid_pair_x
            masked_y = tf.square(large_jump_y) * valid_pair_y
            # 分母：只計大跳躍 pair 數量，不是全部 valid pair
            large_jump_mask_x = tf.cast(tf.abs(diff_x_norm) > threshold, tf.float32) * valid_pair_x
            large_jump_mask_y = tf.cast(tf.abs(diff_y_norm) > threshold, tf.float32) * valid_pair_y
            n_large_x = tf.reduce_sum(large_jump_mask_x) + 1e-6
            n_large_y = tf.reduce_sum(large_jump_mask_y) + 1e-6
            smoothness_loss = tf.reduce_sum(masked_x) / n_large_x + tf.reduce_sum(masked_y) / n_large_y

        elif smoothness_type == 'fold_jump':
            # 原版：直接懲罰fold跳躍
            probs = tf.nn.softmax(classification_logits, axis=-1)  # (B, H, W, 6)
            fold_probs = probs[:, :, :, 1:6]  # 取class 1-5: [-2, -1, 0, +1, +2] fold
            fold_probs = fold_probs / tf.reduce_sum(fold_probs, axis=-1, keepdims=True)  # 重新歸一化

            fold_values = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=tf.float32)
            fold_values = tf.reshape(fold_values, [1, 1, 1, 5])

            expected_fold = tf.reduce_sum(fold_probs * fold_values, axis=-1)  # (B, H, W)

            fold_diff_x = expected_fold[:, :, 1:] - expected_fold[:, :, :-1]
            fold_diff_y = expected_fold[:, 1:, :] - expected_fold[:, :-1, :]

            threshold = 0.5
            large_jump_x = tf.maximum(tf.abs(fold_diff_x) - threshold, 0.0)
            large_jump_y = tf.maximum(tf.abs(fold_diff_y) - threshold, 0.0)

            smoothness_loss = tf.reduce_mean(tf.square(large_jump_x)) + tf.reduce_mean(tf.square(large_jump_y))

        elif smoothness_type == 'total_variation':
            fold_values = tf.constant([0.0, -2.0, -1.0, 0.0, 1.0, 2.0], dtype=tf.float32)
            probs = tf.nn.softmax(classification_logits, axis=-1)
            expected_fold = tf.reduce_sum(probs * fold_values, axis=-1)

            tv_h = tf.abs(expected_fold[:, 1:, :] - expected_fold[:, :-1, :])
            tv_w = tf.abs(expected_fold[:, :, 1:] - expected_fold[:, :, :-1])

            smoothness_loss = tf.reduce_mean(tv_h) + tf.reduce_mean(tv_w)

        else:  # 'prob_diff' - 原始方法
            probs = tf.nn.softmax(classification_logits, axis=-1)
            prob_dx = probs[:, :, 1:, :] - probs[:, :, :-1, :]
            prob_dy = probs[:, 1:, :, :] - probs[:, :-1, :, :]
            smoothness_loss = tf.reduce_mean(tf.square(prob_dx)) + tf.reduce_mean(tf.square(prob_dy))

        return smoothness_loss


def _evaluate_validation_distributed(model, ds_val, val_steps,
                                       lambda_cls, lambda_reg, lambda_physics, lambda_soft, lambda_confidence,
                                       mask_strategy, enable_spatial_smoothness, lambda_spatial, smoothness_type,
                                       aliased_weight, clean_weight, lambda_fpr, lambda_fnr,
                                       focal_gamma, class_weight, strategy):
    """
    Multi-GPU 版本：用 strategy.run 把 val 切到所有 GPU 同步計算。
    跟 single GPU 版本相同 metric，但快 ~4x（4 卡）+ 有 tqdm 進度條。
    """
    from tqdm import tqdm as _tqdm

    # 確保 dataset 已 distribute（防禦寫法：用 hasattr 偵測 DistributedDataset 特性）
    # 不同 TF 版本 DistributedDataset 的 import path 不同，用「是否有 _strategy」判別
    _already_distributed = hasattr(ds_val, '_strategy') or 'Distributed' in type(ds_val).__name__
    if not _already_distributed:
        ds_val = strategy.experimental_distribute_dataset(ds_val)

    @tf.function
    def val_step(x_dict, y_dict):
        """
        單 batch loss 計算 — multi-GPU 安全版。
        關鍵：用 mask 乘法 + reduce_sum 取代 tf.boolean_mask（避免 @tf.function + 多 GPU
        下 boolean_mask 產生 0-size kernel launch 觸發 CUDA assertion）。
        """
        alias_label = y_dict['alias_label']
        target_vel = y_dict['target_vel']
        patch_types = y_dict['patch_type']

        out = model(x_dict, training=False)
        classification_logits = out['alias_mask']
        velocity_residual = out['velocity_residual']
        physics_dealiased_vel = out['physics_dealiased_vel']
        soft_classification_vel = out['soft_classification_vel']
        confidence_guided_vel = out['confidence_guided_vel']

        if mask_strategy == 'v1':
            monitor_mask = (~tf.math.is_nan(target_vel) &
                            tf.not_equal(alias_label[..., tf.newaxis], 0))
        else:
            monitor_mask = ~tf.math.is_nan(target_vel)

        cls_loss = mixed_patch_focal_loss(alias_label, classification_logits, patch_types,
                                          gamma=focal_gamma, class_weight=class_weight)

        raw_vel = x_dict['vel'][:, -1]
        nyq_single = x_dict['nyq'][:, -1]
        shp_raw = tf.shape(raw_vel)
        nyq_vel = tf.tile(nyq_single[:, None, None, None],
                          [1, shp_raw[1], shp_raw[2], shp_raw[3]])

        cat = tf.argmax(classification_logits, axis=-1, output_type=tf.int32)
        cat = tf.expand_dims(cat, axis=-1)
        corr = tf.zeros_like(raw_vel)
        corr = tf.where(tf.equal(cat, 1), -4.0*nyq_vel, corr)
        corr = tf.where(tf.equal(cat, 2), -2.0*nyq_vel, corr)
        corr = tf.where(tf.equal(cat, 4),  2.0*nyq_vel, corr)
        corr = tf.where(tf.equal(cat, 5),  4.0*nyq_vel, corr)

        base_prediction = raw_vel + corr
        target_residual = target_vel - base_prediction

        # Mask 乘法版本（避免 tf.boolean_mask 0-size 風險）
        # 把 NaN 換 0，mask 用乘法廣播
        mask_f = tf.cast(monitor_mask, tf.float32)
        n_valid = tf.reduce_sum(mask_f) + 1e-9
        target_vel_safe = tf.where(monitor_mask, target_vel, tf.zeros_like(target_vel))
        target_residual_safe = tf.where(monitor_mask, target_residual, tf.zeros_like(target_residual))
        velocity_residual_safe = tf.where(monitor_mask, velocity_residual, tf.zeros_like(velocity_residual))
        physics_safe = tf.where(monitor_mask, physics_dealiased_vel, tf.zeros_like(physics_dealiased_vel))
        soft_safe = tf.where(monitor_mask, soft_classification_vel, tf.zeros_like(soft_classification_vel))
        confidence_safe = tf.where(monitor_mask, confidence_guided_vel, tf.zeros_like(confidence_guided_vel))

        reg_loss = tf.reduce_sum(tf.square(target_residual_safe - velocity_residual_safe) * mask_f) / n_valid
        physics_loss = tf.reduce_sum(tf.square(target_vel_safe - physics_safe) * mask_f) / n_valid
        soft_cls_loss = tf.reduce_sum(tf.square(target_vel_safe - soft_safe) * mask_f) / n_valid
        confidence_loss = tf.reduce_sum(tf.square(target_vel_safe - confidence_safe) * mask_f) / n_valid

        batch_size_f = tf.cast(tf.shape(target_vel)[0], tf.float32)
        aliased_count = tf.reduce_sum(tf.cast(tf.equal(patch_types, 'aliased'), tf.float32))
        clean_count = tf.reduce_sum(tf.cast(tf.equal(patch_types, 'clean'), tf.float32))
        total_weight = aliased_count * aliased_weight + clean_count * clean_weight
        normalized_weight = tf.where(
            tf.greater(total_weight, 0.0),
            total_weight / batch_size_f,
            tf.constant(1.0, dtype=tf.float32)
        )

        batch_loss = normalized_weight * (
            lambda_cls * cls_loss + lambda_reg * reg_loss +
            lambda_physics * physics_loss + lambda_soft * soft_cls_loss +
            lambda_confidence * confidence_loss
        )

        if enable_spatial_smoothness:
            spatial_loss = compute_spatial_smoothness_loss(
                classification_logits, smoothness_type,
                raw_vel=raw_vel, nyq_vel=nyq_vel)
            batch_loss += lambda_spatial * spatial_loss

        return batch_loss

    @tf.function
    def distributed_val_step(x, y):
        per_replica = strategy.run(val_step, args=(x, y))
        return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica, axis=None)

    total_loss = 0.0
    count = 0
    val_iter = iter(ds_val)
    progress = _tqdm(range(val_steps), desc='驗證中（multi-GPU）', leave=False)
    for _ in progress:
        try:
            x_dict, y_dict = next(val_iter)
        except StopIteration:
            break
        loss = distributed_val_step(x_dict, y_dict)
        loss_val = float(loss)
        total_loss += loss_val
        count += 1
        progress.set_postfix({'loss': f'{loss_val:.4f}'})

    return total_loss / max(count, 1)


def evaluate_validation_set(model, ds_val, val_steps, lambda_cls, lambda_reg,
                            lambda_physics, lambda_soft, lambda_confidence,
                            mask_strategy, enable_spatial_smoothness=False,
                            lambda_spatial=0.0, smoothness_type='fold_jump',
                            aliased_weight=0.8, clean_weight=0.2, lambda_fpr=0.0, lambda_fnr=0.0,
                            focal_gamma=2.0, class_weight=None,
                            strategy=None, n_gpus=1):
    """
    評估驗證集性能

    Returns:
    --------
    avg_loss : float
        平均損失
    """
    total_loss = 0.0
    count = 0

    # ============================================================
    # Multi-GPU 快速路徑：distribute val dataset + strategy.run
    # ============================================================
    if n_gpus > 1 and strategy is not None:
        return _evaluate_validation_distributed(
            model, ds_val, val_steps,
            lambda_cls, lambda_reg, lambda_physics, lambda_soft, lambda_confidence,
            mask_strategy, enable_spatial_smoothness, lambda_spatial, smoothness_type,
            aliased_weight, clean_weight, lambda_fpr, lambda_fnr,
            focal_gamma, class_weight, strategy
        )

    # ============================================================
    # 單 GPU 路徑（原邏輯，加 tqdm）
    # ============================================================
    # INFER_MICRO=8：與訓練 MICRO=8 一致，保守 GPU 記憶體（避免 [8,128,128,256]=134MB OOM）
    INFER_MICRO = 8

    from tqdm import tqdm as _tqdm
    val_progress = _tqdm(ds_val.take(val_steps), total=val_steps, desc='驗證中', leave=False)
    for x_dict, y_dict in val_progress:
        alias_label = y_dict['alias_label']
        target_vel = y_dict['target_vel']
        patch_types = y_dict['patch_type']

        # 前向傳播：分 micro-batch=8 避免大 activation tensor OOM
        # int(tf.shape(...)[0]) 取實際 runtime batch size（避免 dynamic shape None 問題）
        batch_sz = int(tf.shape(alias_label)[0])
        n_full = batch_sz // INFER_MICRO
        micro_outs = []
        for i in range(n_full):
            s = i * INFER_MICRO
            x_m = {k: v[s:s + INFER_MICRO] for k, v in x_dict.items()}
            micro_outs.append(model(x_m, training=False))
            del x_m
        # 尾部不足一個 micro-batch 的樣本也要處理
        tail = batch_sz - n_full * INFER_MICRO
        if tail > 0:
            x_m = {k: v[n_full * INFER_MICRO:] for k, v in x_dict.items()}
            micro_outs.append(model(x_m, training=False))
            del x_m
        if not micro_outs:
            del x_dict, y_dict
            continue
        out = {k: tf.concat([mo[k] for mo in micro_outs], axis=0)
               for k in micro_outs[0] if not isinstance(micro_outs[0][k], dict)}
        del micro_outs

        classification_logits = out['alias_mask']
        velocity_residual = out['velocity_residual']
        physics_dealiased_vel = out['physics_dealiased_vel']
        soft_classification_vel = out['soft_classification_vel']
        confidence_guided_vel = out['confidence_guided_vel']

        # 計算 mask
        if mask_strategy == 'v1':
            monitor_mask = (
                ~tf.math.is_nan(target_vel) &
                tf.not_equal(alias_label[..., tf.newaxis], 0)
            )
        else:  # v2
            monitor_mask = ~tf.math.is_nan(target_vel)

        # 1. 分類損失 - 使用 Focal Loss（與主訓練一致）
        cls_loss = mixed_patch_focal_loss(alias_label, classification_logits, patch_types,
                                          gamma=focal_gamma, class_weight=class_weight)

        # 2. 回歸殘差損失
        target_vel_valid = tf.boolean_mask(target_vel, monitor_mask)
        velocity_residual_valid = tf.boolean_mask(velocity_residual, monitor_mask)

        raw_vel = x_dict['vel'][:, -1]
        nyq_single = x_dict['nyq'][:, -1]
        shp_raw = tf.shape(raw_vel)
        nyq_vel = tf.tile(nyq_single[:, None, None, None],
                         [1, shp_raw[1], shp_raw[2], shp_raw[3]])

        cat = tf.argmax(classification_logits, axis=-1, output_type=tf.int32)
        cat = tf.expand_dims(cat, axis=-1)
        corr = tf.zeros_like(raw_vel)
        corr = tf.where(tf.equal(cat, 1), -4.0*nyq_vel, corr)  # -2 fold
        corr = tf.where(tf.equal(cat, 2), -2.0*nyq_vel, corr)  # -1 fold
        corr = tf.where(tf.equal(cat, 4),  2.0*nyq_vel, corr)  # +1 fold
        corr = tf.where(tf.equal(cat, 5),  4.0*nyq_vel, corr)  # +2 fold

        base_prediction = raw_vel + corr
        target_residual = target_vel - base_prediction
        target_residual_valid = tf.boolean_mask(target_residual, monitor_mask)

        if tf.size(target_residual_valid) > 0:
            reg_loss = tf.reduce_mean(tf.square(target_residual_valid - velocity_residual_valid))
        else:
            reg_loss = tf.constant(0.0, dtype=tf.float32)

        # 3. 物理約束回歸損失
        physics_vel_valid = tf.boolean_mask(physics_dealiased_vel, monitor_mask)

        if tf.size(target_vel_valid) > 0:
            physics_loss = tf.reduce_mean(tf.square(target_vel_valid - physics_vel_valid))
        else:
            physics_loss = tf.constant(0.0, dtype=tf.float32)

        # 4. 軟分類分支損失
        soft_cls_vel_valid = tf.boolean_mask(soft_classification_vel, monitor_mask)

        if tf.size(target_vel_valid) > 0:
            soft_cls_loss = tf.reduce_mean(tf.square(target_vel_valid - soft_cls_vel_valid))
        else:
            soft_cls_loss = tf.constant(0.0, dtype=tf.float32)

        # 5. 置信度導向混合分支損失
        confidence_vel_valid = tf.boolean_mask(confidence_guided_vel, monitor_mask)

        if tf.size(target_vel_valid) > 0:
            confidence_loss = tf.reduce_mean(tf.square(target_vel_valid - confidence_vel_valid))
        else:
            confidence_loss = tf.constant(0.0, dtype=tf.float32)

        # 計算 patch 權重歸一化（與訓練一致）
        batch_size = tf.cast(tf.shape(target_vel)[0], tf.float32)
        aliased_count = tf.reduce_sum(tf.cast(tf.equal(patch_types, 'aliased'), tf.float32))
        clean_count = tf.reduce_sum(tf.cast(tf.equal(patch_types, 'clean'), tf.float32))

        # 使用傳入的權重參數
        total_weight = aliased_count * aliased_weight + clean_count * clean_weight
        normalized_weight = tf.cond(
            tf.greater(total_weight, 0.0),
            lambda: total_weight / batch_size,
            lambda: tf.constant(1.0, dtype=tf.float32)
        )

        # FPR Auxiliary Loss: 懲罰 clean pixels (label=3) 預測為 aliased 的機率
        fpr_loss = tf.constant(0.0, dtype=tf.float32)
        if lambda_fpr > 0:
            probs = tf.nn.softmax(classification_logits, axis=-1)
            alias_prob = probs[:,:,:,1] + probs[:,:,:,2] + probs[:,:,:,4] + probs[:,:,:,5]
            clean_mask = tf.equal(alias_label, 3)
            clean_alias_probs = tf.boolean_mask(alias_prob, clean_mask)
            fpr_loss = tf.cond(
                tf.greater(tf.size(clean_alias_probs), 0),
                lambda: tf.reduce_mean(clean_alias_probs),
                lambda: tf.constant(0.0, dtype=tf.float32)
            )

        # FNR Auxiliary Loss: 懲罰 aliased pixels (label in 1,2,4,5) 預測為 no-alias (class 3) 的機率
        fnr_loss = tf.constant(0.0, dtype=tf.float32)
        if lambda_fnr > 0:
            probs = tf.nn.softmax(classification_logits, axis=-1)
            no_alias_prob = probs[:,:,:,3]
            aliased_mask = (tf.equal(alias_label, 1) | tf.equal(alias_label, 2) |
                            tf.equal(alias_label, 4) | tf.equal(alias_label, 5))
            aliased_no_alias_probs = tf.boolean_mask(no_alias_prob, aliased_mask)
            fnr_loss = tf.cond(
                tf.greater(tf.size(aliased_no_alias_probs), 0),
                lambda: tf.reduce_mean(aliased_no_alias_probs),
                lambda: tf.constant(0.0, dtype=tf.float32)
            )

        # 總損失（加上 normalized_weight）
        batch_loss = normalized_weight * (lambda_cls * cls_loss + lambda_reg * reg_loss +
                     lambda_physics * physics_loss + lambda_soft * soft_cls_loss +
                     lambda_confidence * confidence_loss)
        batch_loss += lambda_fpr * fpr_loss + lambda_fnr * fnr_loss

        # 空間平滑損失（與訓練步驟一致）
        spatial_smoothness_loss = 0.0
        if enable_spatial_smoothness:
            spatial_loss = compute_spatial_smoothness_loss(
                classification_logits, smoothness_type,
                raw_vel=raw_vel, nyq_vel=nyq_vel)
            spatial_smoothness_loss = lambda_spatial * spatial_loss.numpy()
            batch_loss += spatial_smoothness_loss

        total_loss += float(batch_loss)
        count += 1
        # 明確釋放本步驗證的所有大 tensor，避免 GPU 記憶體累積
        del x_dict, y_dict, out, classification_logits, velocity_residual
        del physics_dealiased_vel, soft_classification_vel, confidence_guided_vel
        del alias_label, target_vel, patch_types, monitor_mask

    avg_loss = total_loss / max(count, 1)
    return avg_loss


def save_checkpoint(result_dir, model, optimizer, epoch, best_combined_score,
                    best_epoch, baseline_typhoon_loss, current_lr,
                    lr_reduction_count, patience_counter, early_stop_counter,
                    history):
    """
    儲存訓練 checkpoint（每個 epoch 結束後呼叫）

    儲存內容：
    - checkpoint_meta.json：訓練狀態
    - latest_model.h5：最新模型權重
    - optimizer_state.pkl：optimizer 內部狀態（Adam m/v）
    """
    ckpt_dir = os.path.join(result_dir, "checkpoint")
    os.makedirs(ckpt_dir, exist_ok=True)

    # 1. 儲存模型權重
    model_path = os.path.join(ckpt_dir, "latest_model.h5")
    model.save_weights(model_path, save_format='h5')

    # 2. 儲存 optimizer 狀態
    opt_path = os.path.join(ckpt_dir, "optimizer_state.pkl")
    opt_weights = optimizer.get_weights()
    with open(opt_path, 'wb') as f:
        pickle.dump(opt_weights, f)

    # 3. 儲存 metadata
    meta = {
        'epoch': epoch,
        'best_combined_score': float(best_combined_score),
        'best_epoch': best_epoch,
        'baseline_typhoon_loss': float(baseline_typhoon_loss) if baseline_typhoon_loss is not None else None,
        'current_lr': float(current_lr),
        'lr_reduction_count': lr_reduction_count,
        'patience_counter': patience_counter,
        'early_stop_counter': early_stop_counter,
        'history': {k: [float(v) if isinstance(v, (np.floating, float)) else v
                        for v in vals]
                    for k, vals in history.items()}
    }
    meta_path = os.path.join(ckpt_dir, "checkpoint_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"   💾 Checkpoint 已儲存 (epoch {epoch + 1})")


def load_checkpoint(resume_dir):
    """
    從 checkpoint 目錄載入訓練狀態

    Args:
        resume_dir: 先前的 result 目錄路徑

    Returns:
        dict: 包含所有訓練狀態的 metadata，以及 optimizer weights
    """
    ckpt_dir = os.path.join(resume_dir, "checkpoint")

    # 1. 載入 metadata
    meta_path = os.path.join(ckpt_dir, "checkpoint_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"找不到 checkpoint metadata: {meta_path}")

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    # 2. 載入 optimizer 狀態
    opt_path = os.path.join(ckpt_dir, "optimizer_state.pkl")
    if not os.path.exists(opt_path):
        raise FileNotFoundError(f"找不到 optimizer 狀態: {opt_path}")

    with open(opt_path, 'rb') as f:
        meta['optimizer_weights'] = pickle.load(f)

    # 3. 確認模型權重存在
    model_path = os.path.join(ckpt_dir, "latest_model.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型權重: {model_path}")
    meta['model_path'] = model_path

    print(f"   ✅ Checkpoint 載入成功 (epoch {meta['epoch'] + 1})")
    return meta


def transfer_learning_complete(
    pretrained_model_path,
    nwp_h5_path,
    typhoon_h5_path,
    output_dir="results/transfer_learning",
    freeze_ratio=0.8,
    learning_rate=5e-5,
    epochs=50,
    batch_size=4,
    patience=5,
    baseline_typhoon_success=84.0,
    # 學習率調度參數（與原始訓練一致）
    lr_scheduling=True,
    lr_factor=0.5,
    lr_patience=5,
    lr_min=1e-6,
    # 損失權重（必須與預訓練模型完全一致！）
    lambda_cls=2.0,      # ⚠️ 原始訓練使用 2.0
    lambda_reg=0.1,
    lambda_physics=0.1,
    lambda_soft=0.5,
    lambda_confidence=0.1,
    # 空間平滑約束參數（與原始訓練一致）
    enable_spatial_smoothness=True,
    lambda_spatial=50.0,  # ⚠️ 原始訓練使用 50.0
    smoothness_type='v_corrected',
    mask_strategy='v1',
    filter_patch_type='all',
    # Patch 類型權重（控制 aliased/clean patches 的損失權重）
    aliased_weight=0.8,
    clean_weight=0.2,
    # Focal Loss gamma 參數 (0.0=CrossEntropy, 2.0=原始Focal)
    focal_gamma=0.0,
    # Per-class weight for classification loss [cat0..cat5]
    # cat0=invalid(masked), cat1=fold-2, cat2=fold-1, cat3=fold0, cat4=fold+1, cat5=fold+2
    class_weight=None,
    # FPR / FNR Auxiliary Loss 權重
    lambda_fpr=0.0,
    lambda_fnr=0.0,
    # 正則化參數
    weight_decay=0.0,
    # 斷點續訓參數
    resume_dir=None,
    # 加速訓練：覆寫 steps_per_epoch
    steps_per_epoch=None
):
    """
    完整版遷移學習主函數

    Parameters:
    -----------
    pretrained_model_path : str
        預訓練模型路徑
    nwp_h5_path : str
        NWP H5 patches 路徑
    typhoon_h5_path : str
        颱風驗證 H5 patches 路徑
    freeze_ratio : float
        凍結比例（0.8 = 凍結 80% 層）
    baseline_typhoon_success : float
        基線颱風成功率（用於 Early Stopping）
    """

    start_time = time.time()

    # result_dir：resume 時重用原目錄，否則建新目錄
    if resume_dir is not None:
        result_dir = resume_dir
        print("🔄 斷點續訓模式")
        print(f"   續訓目錄: {result_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = f"{output_dir}_{timestamp}"
        os.makedirs(result_dir, exist_ok=True)

    print("🚀 完整版遷移學習：真實雷達 → NWP")
    print(f"   預訓練模型: {pretrained_model_path or '(從零訓練)'}")
    print(f"   NWP H5: {nwp_h5_path}")
    print(f"   颱風驗證 H5: {typhoon_h5_path or '(無，純 NWP 訓練)'}")
    print(f"   凍結比例: {freeze_ratio*100:.0f}% {'(無預訓練模型，不凍結)' if pretrained_model_path is None else ''}")
    print(f"   學習率: {learning_rate}")
    print(f"   基線颱風成功率: {baseline_typhoon_success:.1f}%")
    print(f"   Patch 權重: aliased={aliased_weight}, clean={clean_weight}")
    print(f"   結果目錄: {result_dir}")

    # ============================================================
    # 多 GPU 自動偵測（MirroredStrategy）
    # ============================================================
    # 邏輯：
    #   - 1 GPU (e.g. 本地 Windows 3090) → 用 default strategy（等同單 GPU）
    #   - 2+ GPU (server 多卡) → MirroredStrategy，所有卡同步訓練 1 個 model
    # 對使用者：完全自動，--batch_size 視為 global batch（自動切到各 GPU）
    _gpus = tf.config.list_physical_devices('GPU')
    _n_gpus = len(_gpus)
    if _n_gpus > 1:
        strategy = tf.distribute.MirroredStrategy()
        _per_replica_batch = batch_size // _n_gpus
        print(f"\n🚀 Multi-GPU 模式: 偵測到 {_n_gpus} 張 GPU")
        print(f"   啟用 tf.distribute.MirroredStrategy")
        print(f"   Global batch_size = {batch_size}, per-GPU batch = {_per_replica_batch}")
        if batch_size % _n_gpus != 0:
            print(f"   ⚠️  batch_size ({batch_size}) 不能被 GPU 數 ({_n_gpus}) 整除，可能 drop 部分樣本")
    else:
        strategy = tf.distribute.get_strategy()  # default = 單 GPU/CPU
        print(f"\n🖥️  Single GPU/CPU 模式: {_n_gpus} 張 GPU 可見")

    # ============================================================
    # 步驟 1: 建立模型架構
    # ============================================================
    print("\n" + "="*80)
    print("步驟 1/6: 建立模型架構")
    print("="*80)

    # 模型 + optimizer 都需要在 strategy.scope() 內建立，所有 variable 才會在
    # 各 replica 同步。Single GPU mode 下 strategy 是 default，scope 無 overhead。
    with strategy.scope():
        # 創建特徵提取器和上採樣器（與訓練時一致）
        extractor = create_downsampler(input_channels=1, start_neurons=32)
        upsampler_cls = create_upsampler_cls(n_inputs=1, start_neurons=32, classes=6)
        upsampler_reg = create_upsampler_reg(n_inputs=1, start_neurons=32)

        # 建立模型
        model = VelocityDealiaser(extractor, upsampler_cls, upsampler_reg)

    print("✅ 模型架構已建立")

    # ============================================================
    # 步驟 2: 載入預訓練權重
    # ============================================================
    print("\n" + "="*80)
    print("步驟 2/6: 載入預訓練權重")
    print("="*80)

    # 需要先進行一次前向傳播來初始化變數
    # 模型期待 5 維輸入: (batch, n_times, n_az, n_rad, 1)
    # init forward / load weights / freeze 都要在 strategy.scope() 內以同步 variable
    with strategy.scope():
        dummy_input = {
            'vel': tf.zeros((1, 1, 128, 128, 1), dtype=tf.float32),
            'nyq': tf.ones((1, 1), dtype=tf.float32)
        }
        _ = model(dummy_input, training=False)

        if resume_dir is not None:
            # Resume 模式：從 checkpoint 載入權重（稍後在 optimizer 建立後完成）
            print(f"   （Resume 模式：權重將從 checkpoint 載入）")
        elif pretrained_model_path is not None:
            # 正常模式：載入預訓練權重
            model.load_weights(pretrained_model_path)
            print(f"✅ 已載入預訓練權重: {pretrained_model_path}")
        else:
            print("✅ 從零初始化（無預訓練模型，所有層隨機初始化）")

        # ============================================================
        # 步驟 3: 應用凍結策略
        # ============================================================
        print("\n" + "="*80)
        print("步驟 3/6: 應用凍結策略")
        print("="*80)

        if pretrained_model_path is not None and freeze_ratio > 0.0:
            model = freeze_layers(model, freeze_ratio=freeze_ratio)
        else:
            print("   （不凍結任何層，全部層可訓練）")
        print_trainable_summary(model)

    # ============================================================
    # 步驟 4: 載入訓練和驗證數據
    # ============================================================
    print("\n" + "="*80)
    print("步驟 4/6: 載入訓練和驗證數據")
    print("="*80)

    # NWP 數據
    print("\n📂 載入 NWP 數據...")

    if typhoon_h5_path is not None:
        # 混合訓練：75% NWP + 25% 颱風
        nwp_batch_size = int(batch_size * 0.75)
        typhoon_batch_size = batch_size - nwp_batch_size
        print(f"   混合訓練策略: {nwp_batch_size} NWP + {typhoon_batch_size} 颱風 = {batch_size} 總 batch size")
    else:
        # 純 NWP 訓練
        nwp_batch_size = batch_size
        typhoon_batch_size = 0
        print(f"   純 NWP 訓練策略: {nwp_batch_size} NWP = {batch_size} 總 batch size")

    ds_nwp_train, nwp_train_patches = load_mixed_patches(
        nwp_h5_path, split='train', batch_size=nwp_batch_size, shuffle=True, repeat=True,
        filter_patch_type=filter_patch_type
    )

    ds_nwp_val, nwp_val_patches = load_mixed_patches(
        nwp_h5_path, split='val', batch_size=batch_size, shuffle=False, repeat=False,
        filter_patch_type=filter_patch_type
    )

    print(f"   NWP 訓練: {nwp_train_patches} patches")
    print(f"   NWP 驗證: {nwp_val_patches} patches")

    nwp_train_steps = max(1, nwp_train_patches // nwp_batch_size)
    nwp_val_steps = max(1, nwp_val_patches // batch_size)

    # 覆寫 steps_per_epoch（加速訓練用）
    if steps_per_epoch is not None and steps_per_epoch > 0:
        original_steps = nwp_train_steps
        nwp_train_steps = steps_per_epoch
        print(f"   ⚠️ 覆寫每 epoch 步數: {original_steps} → {nwp_train_steps}")
        print(f"   (每 epoch 看 ~{nwp_train_steps * nwp_batch_size} 個 patches，"
              f"非整個 train dataset)")

    if typhoon_h5_path is not None:
        # 颱風數據（訓練 + 驗證）
        print("\n📂 載入颱風數據...")
        ds_typhoon_train, typhoon_train_patches = load_mixed_patches(
            typhoon_h5_path, split='train', batch_size=typhoon_batch_size, shuffle=True, repeat=True,
            filter_patch_type=filter_patch_type
        )

        ds_typhoon_val, typhoon_val_patches = load_mixed_patches(
            typhoon_h5_path, split='val', batch_size=batch_size, shuffle=False, repeat=False,
            filter_patch_type=filter_patch_type
        )

        print(f"   颱風訓練: {typhoon_train_patches} patches")
        print(f"   颱風驗證: {typhoon_val_patches} patches")

        # 創建混合訓練數據集（zip NWP + 颱風）
        print(f"\n🔀 創建混合訓練數據集...")

        def merge_batches(nwp_batch, typhoon_batch):
            """合併 NWP 和颱風 batch"""
            nwp_x, nwp_y = nwp_batch
            typhoon_x, typhoon_y = typhoon_batch
            merged_x = {
                'vel': tf.concat([nwp_x['vel'], typhoon_x['vel']], axis=0),
                'nyq': tf.concat([nwp_x['nyq'], typhoon_x['nyq']], axis=0)
            }
            merged_y = {
                'alias_label': tf.concat([nwp_y['alias_label'], typhoon_y['alias_label']], axis=0),
                'gt_vel': tf.concat([nwp_y['gt_vel'], typhoon_y['gt_vel']], axis=0),
                'target_vel': tf.concat([nwp_y['target_vel'], typhoon_y['target_vel']], axis=0),
                'patch_type': tf.concat([nwp_y['patch_type'], typhoon_y['patch_type']], axis=0)
            }
            return merged_x, merged_y

        ds_mixed_train = tf.data.Dataset.zip((ds_nwp_train, ds_typhoon_train))
        ds_mixed_train = ds_mixed_train.map(merge_batches,
                                            num_parallel_calls=tf.data.AUTOTUNE
                                           ).prefetch(tf.data.AUTOTUNE)
        typhoon_train_steps = max(1, typhoon_train_patches // typhoon_batch_size)
        typhoon_val_steps = max(1, typhoon_val_patches // batch_size)
        mixed_train_steps = nwp_train_steps  # 基於 NWP，颱風循環使用
        print(f"   ✅ 混合數據集創建完成（iterator 每 epoch 重建以防止 CPU RAM 累積）")
    else:
        # 純 NWP：直接使用 NWP dataset
        ds_mixed_train = ds_nwp_train
        typhoon_train_steps = 0
        typhoon_val_steps = 0
        mixed_train_steps = nwp_train_steps
        print(f"\n📂 純 NWP 模式，不載入颱風數據")

    print(f"\n📊 訓練步數設定:")
    print(f"   NWP 訓練步數: {nwp_train_steps}")
    if typhoon_h5_path is not None:
        print(f"   颱風訓練步數: {typhoon_train_steps}")
        print(f"   混合訓練步數: {mixed_train_steps} (基於 NWP，颱風循環使用)")
    else:
        print(f"   混合訓練步數: {mixed_train_steps} (純 NWP)")

    # ============================================================
    # 步驟 5: 設置優化器和訓練循環
    # ============================================================
    print("\n" + "="*80)
    print("步驟 5/6: 開始微調訓練")
    print("="*80)

    # 優化器（使用可變學習率，支援 weight decay via L2 regularization）
    # Optimizer 必須在 strategy.scope() 內建立才能正確 all-reduce 梯度
    with strategy.scope():
        lr_variable = tf.Variable(learning_rate, dtype=tf.float32, trainable=False)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_variable)
    if weight_decay > 0:
        print(f"   使用 Adam optimizer + L2 weight decay ({weight_decay})")
    current_lr = learning_rate

    # Multi-GPU 時 distribute train dataset（val 不 distribute，evaluate_validation_set
    # 仍走 single device 簡化邏輯；val 時間遠 < train，影響小）
    if _n_gpus > 1:
        print(f"   📡 Distribute train dataset 到 {_n_gpus} 張 GPU...")
        ds_mixed_train = strategy.experimental_distribute_dataset(ds_mixed_train)

    # Early Stopping 變數
    best_combined_score = -np.inf
    best_epoch = 0
    patience_counter = 0
    baseline_typhoon_loss = None  # 在第一個 epoch 設定

    # 學習率調度變數
    lr_reduction_count = 0
    early_stop_counter = 0  # 用於學習率調度的計數器

    # 訓練歷史
    history = {
        'epoch': [],
        'nwp_train_loss': [],
        'nwp_val_loss': [],
        'typhoon_val_loss': [],
        'combined_score': [],
        'learning_rate': [],
        'spatial_loss': []  # 🆕 記錄空間平滑損失
    }

    # 預設 start_epoch
    start_epoch = 0

    # ============================================================
    # Resume：從 checkpoint 還原訓練狀態
    # ============================================================
    if resume_dir is not None:
        print("\n🔄 從 checkpoint 還原訓練狀態...")
        ckpt_meta = load_checkpoint(resume_dir)

        # Multi-GPU: 所有 weight 操作必須在 strategy.scope() 內
        with strategy.scope():
            # 載入模型權重
            model.load_weights(ckpt_meta['model_path'])
            print(f"   ✅ 模型權重已從 checkpoint 載入")

            # 做一次 dummy gradient step 讓 Adam 為所有 trainable variable 建立內部 slot
            # 必須使用所有分支的輸出，否則部分變數不會產生梯度
            # MirroredStrategy 下 gradient 必須在 replica context 內，故包進 strategy.run；
            # 單卡 default strategy 時 strategy.run(fn) 等同直接呼叫 fn()，向後相容。
            def _dummy_step():
                with tf.GradientTape() as tape:
                    dummy_out = model(dummy_input, training=True)
                    dummy_loss = (
                        tf.reduce_mean(dummy_out['alias_mask']) +
                        tf.reduce_mean(dummy_out['velocity_residual']) +
                        tf.reduce_mean(dummy_out['physics_dealiased_vel']) +
                        tf.reduce_mean(dummy_out['soft_classification_vel']) +
                        tf.reduce_mean(dummy_out['confidence_guided_vel'])
                    )
                grads = tape.gradient(dummy_loss, model.trainable_variables)
                # 將 None 梯度替換為零，確保所有變數都建立 optimizer slot
                grads = [g if g is not None else tf.zeros_like(v)
                         for g, v in zip(grads, model.trainable_variables)]
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            strategy.run(_dummy_step)

            # 重新載入模型權重（因為 dummy step 會改變權重）
            model.load_weights(ckpt_meta['model_path'])

            # 載入 optimizer 狀態
            optimizer.set_weights(ckpt_meta['optimizer_weights'])
            print(f"   ✅ Optimizer 狀態已還原")

        # 還原訓練狀態變數
        start_epoch = ckpt_meta['epoch'] + 1
        best_combined_score = ckpt_meta['best_combined_score']
        best_epoch = ckpt_meta['best_epoch']
        baseline_typhoon_loss = ckpt_meta['baseline_typhoon_loss']
        current_lr = ckpt_meta['current_lr']
        lr_variable.assign(current_lr)
        lr_reduction_count = ckpt_meta['lr_reduction_count']
        patience_counter = ckpt_meta['patience_counter']
        early_stop_counter = ckpt_meta['early_stop_counter']
        history = ckpt_meta['history']

        print(f"   ✅ 訓練狀態已還原:")
        print(f"      從 epoch {start_epoch + 1} 繼續（已完成 {start_epoch} epochs）")
        print(f"      最佳 epoch: {best_epoch}, 最佳評分: {best_combined_score:.4f}")
        print(f"      學習率: {current_lr:.2e}")
        print(f"      Patience counter: {patience_counter}/{patience}")

    print(f"\n訓練配置:")
    print(f"   Epochs: {epochs}")
    if typhoon_h5_path is not None:
        print(f"   Batch size: {batch_size} ({nwp_batch_size} NWP + {typhoon_batch_size} 颱風)")
    else:
        print(f"   Batch size: {batch_size} (純 NWP)")
    print(f"   混合訓練 steps/epoch: {mixed_train_steps}")
    print(f"   Early Stopping patience: {patience}")
    print(f"   Early Stopping 條件: 颱風損失增加 > 20% 或 {patience} epochs 無改善")
    if class_weight is not None:
        print(f"   Class weight: {class_weight}")
        print(f"     cat1(fold-2)={class_weight[1]:.2f}, cat2(fold-1)={class_weight[2]:.2f}, "
              f"cat3(fold0)={class_weight[3]:.2f}, cat4(fold+1)={class_weight[4]:.2f}, cat5(fold+2)={class_weight[5]:.2f}")
    else:
        print(f"   Class weight: None (均等權重)")

    # 訓練步驟：梯度累積完全在 @tf.function 內部（unrolled），不跨越 Python/TF 邊界
    # 原理：把 loss 計算提取成普通函數 _compute_loss，@tf.function train_step
    # 在同一個 compiled graph 裡跑 ACCUM 個 micro-batch，TF 自動管理所有 tensor 生命週期
    #
    # Multi-GPU 注意：MICRO 用 per-replica batch 算（不是 global），因為 strategy.run
    # 進到 train_step 時，x_dict/y_dict 已是 per-replica 切片大小。
    #   single GPU: per_replica = batch_size，MICRO = batch_size // ACCUM
    #   4 GPUs:     per_replica = batch_size // 4，MICRO = (batch_size // 4) // ACCUM

    ACCUM = 4                                    # 固定 4-way accumulation
    PER_REPLICA_BATCH = batch_size // _n_gpus if _n_gpus > 1 else batch_size
    MICRO = PER_REPLICA_BATCH // ACCUM           # Python int，trace 時成為常數
    print(f"   📦 Gradient accumulation: ACCUM={ACCUM}, per_replica_batch={PER_REPLICA_BATCH}, MICRO={MICRO}")

    def _compute_loss(x_dict, y_dict):
        """Loss 計算（不含 GradientTape，由呼叫方包）"""
        alias_label = y_dict['alias_label']
        target_vel = y_dict['target_vel']
        patch_types = y_dict['patch_type']

        out = model(x_dict, training=True)
        classification_logits = out['alias_mask']
        velocity_residual = out['velocity_residual']
        physics_dealiased_vel = out['physics_dealiased_vel']
        soft_classification_vel = out['soft_classification_vel']
        confidence_guided_vel = out['confidence_guided_vel']

        if mask_strategy == 'v1':
            monitor_mask = (
                ~tf.math.is_nan(target_vel) &
                tf.not_equal(alias_label[..., tf.newaxis], 0)
            )
        else:
            monitor_mask = ~tf.math.is_nan(target_vel)

        cls_loss = mixed_patch_focal_loss(alias_label, classification_logits, patch_types,
                                          gamma=focal_gamma, class_weight=class_weight)

        target_vel_valid = tf.boolean_mask(target_vel, monitor_mask)
        velocity_residual_valid = tf.boolean_mask(velocity_residual, monitor_mask)

        raw_vel = x_dict['vel'][:, -1]
        nyq_single = x_dict['nyq'][:, -1]
        shp_raw = tf.shape(raw_vel)
        nyq_vel = tf.tile(nyq_single[:, None, None, None],
                         [1, shp_raw[1], shp_raw[2], shp_raw[3]])

        cat = tf.argmax(classification_logits, axis=-1, output_type=tf.int32)
        cat = tf.expand_dims(cat, axis=-1)
        corr = tf.zeros_like(raw_vel)
        corr = tf.where(tf.equal(cat, 1), -4.0*nyq_vel, corr)
        corr = tf.where(tf.equal(cat, 2), -2.0*nyq_vel, corr)
        corr = tf.where(tf.equal(cat, 4),  2.0*nyq_vel, corr)
        corr = tf.where(tf.equal(cat, 5),  4.0*nyq_vel, corr)

        base_prediction = raw_vel + corr
        target_residual = target_vel - base_prediction
        target_residual_valid = tf.boolean_mask(target_residual, monitor_mask)

        reg_loss = tf.cond(
            tf.greater(tf.size(target_residual_valid), 0),
            lambda: tf.reduce_mean(tf.square(target_residual_valid - velocity_residual_valid)),
            lambda: tf.constant(0.0, dtype=tf.float32)
        )

        physics_vel_valid = tf.boolean_mask(physics_dealiased_vel, monitor_mask)
        physics_loss = tf.cond(
            tf.greater(tf.size(target_vel_valid), 0),
            lambda: tf.reduce_mean(tf.square(target_vel_valid - physics_vel_valid)),
            lambda: tf.constant(0.0, dtype=tf.float32)
        )

        soft_cls_vel_valid = tf.boolean_mask(soft_classification_vel, monitor_mask)
        soft_cls_loss = tf.cond(
            tf.greater(tf.size(target_vel_valid), 0),
            lambda: tf.reduce_mean(tf.square(target_vel_valid - soft_cls_vel_valid)),
            lambda: tf.constant(0.0, dtype=tf.float32)
        )

        confidence_vel_valid = tf.boolean_mask(confidence_guided_vel, monitor_mask)
        confidence_loss = tf.cond(
            tf.greater(tf.size(target_vel_valid), 0),
            lambda: tf.reduce_mean(tf.square(target_vel_valid - confidence_vel_valid)),
            lambda: tf.constant(0.0, dtype=tf.float32)
        )

        batch_size_float = tf.cast(tf.shape(target_vel)[0], tf.float32)
        aliased_count = tf.reduce_sum(tf.cast(tf.equal(patch_types, 'aliased'), tf.float32))
        clean_count = tf.reduce_sum(tf.cast(tf.equal(patch_types, 'clean'), tf.float32))
        total_weight = aliased_count * aliased_weight + clean_count * clean_weight
        normalized_weight = tf.cond(
            tf.greater(total_weight, 0.0),
            lambda: total_weight / batch_size_float,
            lambda: tf.constant(1.0, dtype=tf.float32)
        )

        fpr_loss = tf.constant(0.0, dtype=tf.float32)
        if lambda_fpr > 0:
            probs = tf.nn.softmax(classification_logits, axis=-1)
            alias_prob = probs[:,:,:,1] + probs[:,:,:,2] + probs[:,:,:,4] + probs[:,:,:,5]
            clean_mask = tf.equal(alias_label, 3)
            clean_alias_probs = tf.boolean_mask(alias_prob, clean_mask)
            fpr_loss = tf.cond(
                tf.greater(tf.size(clean_alias_probs), 0),
                lambda: tf.reduce_mean(clean_alias_probs),
                lambda: tf.constant(0.0, dtype=tf.float32)
            )

        # FNR Auxiliary Loss: 懲罰 aliased pixels (label in 1,2,4,5) 預測為 no-alias (class 3) 的機率
        fnr_loss = tf.constant(0.0, dtype=tf.float32)
        if lambda_fnr > 0:
            probs = tf.nn.softmax(classification_logits, axis=-1)
            no_alias_prob = probs[:,:,:,3]
            aliased_mask = (tf.equal(alias_label, 1) | tf.equal(alias_label, 2) |
                            tf.equal(alias_label, 4) | tf.equal(alias_label, 5))
            aliased_no_alias_probs = tf.boolean_mask(no_alias_prob, aliased_mask)
            fnr_loss = tf.cond(
                tf.greater(tf.size(aliased_no_alias_probs), 0),
                lambda: tf.reduce_mean(aliased_no_alias_probs),
                lambda: tf.constant(0.0, dtype=tf.float32)
            )

        total_loss = normalized_weight * (lambda_cls * cls_loss + lambda_reg * reg_loss +
                     lambda_physics * physics_loss + lambda_soft * soft_cls_loss +
                     lambda_confidence * confidence_loss)
        total_loss += lambda_fpr * fpr_loss + lambda_fnr * fnr_loss

        spatial_smoothness_loss = tf.constant(0.0, dtype=tf.float32)
        if enable_spatial_smoothness:
            spatial_smoothness_loss = compute_spatial_smoothness_loss(
                classification_logits, smoothness_type,
                raw_vel=raw_vel, nyq_vel=nyq_vel)
            total_loss += lambda_spatial * spatial_smoothness_loss

        if weight_decay > 0:
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables
                                if 'bias' not in v.name and 'batch_normalization' not in v.name])
            total_loss += weight_decay * l2_loss

        return total_loss, spatial_smoothness_loss

    def diagnose_loss_breakdown(x_dict, y_dict):
        """Eager mode 診斷：印出各分支 loss 數值與佔比（只在第 1 epoch 呼叫一次）"""
        alias_label = y_dict['alias_label']
        target_vel = y_dict['target_vel']
        patch_types = y_dict['patch_type']

        out = model(x_dict, training=False)
        classification_logits = out['alias_mask']
        velocity_residual = out['velocity_residual']
        physics_dealiased_vel = out['physics_dealiased_vel']
        soft_classification_vel = out['soft_classification_vel']
        confidence_guided_vel = out['confidence_guided_vel']

        if mask_strategy == 'v1':
            monitor_mask = (
                ~tf.math.is_nan(target_vel) &
                tf.not_equal(alias_label[..., tf.newaxis], 0)
            )
        else:
            monitor_mask = ~tf.math.is_nan(target_vel)

        cls_loss = mixed_patch_focal_loss(alias_label, classification_logits, patch_types,
                                          gamma=focal_gamma, class_weight=class_weight)
        target_vel_valid = tf.boolean_mask(target_vel, monitor_mask)
        velocity_residual_valid = tf.boolean_mask(velocity_residual, monitor_mask)

        raw_vel = x_dict['vel'][:, -1]
        nyq_single = x_dict['nyq'][:, -1]
        shp_raw = tf.shape(raw_vel)
        nyq_vel = tf.tile(nyq_single[:, None, None, None],
                         [1, shp_raw[1], shp_raw[2], shp_raw[3]])

        cat = tf.argmax(classification_logits, axis=-1, output_type=tf.int32)
        cat = tf.expand_dims(cat, axis=-1)
        corr = tf.zeros_like(raw_vel)
        corr = tf.where(tf.equal(cat, 1), -4.0*nyq_vel, corr)
        corr = tf.where(tf.equal(cat, 2), -2.0*nyq_vel, corr)
        corr = tf.where(tf.equal(cat, 4),  2.0*nyq_vel, corr)
        corr = tf.where(tf.equal(cat, 5),  4.0*nyq_vel, corr)
        base_prediction = raw_vel + corr
        target_residual = target_vel - base_prediction
        target_residual_valid = tf.boolean_mask(target_residual, monitor_mask)

        reg_loss = tf.cond(
            tf.greater(tf.size(target_residual_valid), 0),
            lambda: tf.reduce_mean(tf.square(target_residual_valid - velocity_residual_valid)),
            lambda: tf.constant(0.0, dtype=tf.float32))

        physics_vel_valid = tf.boolean_mask(physics_dealiased_vel, monitor_mask)
        physics_loss = tf.cond(
            tf.greater(tf.size(target_vel_valid), 0),
            lambda: tf.reduce_mean(tf.square(target_vel_valid - physics_vel_valid)),
            lambda: tf.constant(0.0, dtype=tf.float32))

        soft_cls_vel_valid = tf.boolean_mask(soft_classification_vel, monitor_mask)
        soft_cls_loss = tf.cond(
            tf.greater(tf.size(target_vel_valid), 0),
            lambda: tf.reduce_mean(tf.square(target_vel_valid - soft_cls_vel_valid)),
            lambda: tf.constant(0.0, dtype=tf.float32))

        confidence_vel_valid = tf.boolean_mask(confidence_guided_vel, monitor_mask)
        confidence_loss = tf.cond(
            tf.greater(tf.size(target_vel_valid), 0),
            lambda: tf.reduce_mean(tf.square(target_vel_valid - confidence_vel_valid)),
            lambda: tf.constant(0.0, dtype=tf.float32))

        spatial_loss_val = tf.constant(0.0, dtype=tf.float32)
        if enable_spatial_smoothness:
            spatial_loss_val = compute_spatial_smoothness_loss(
                classification_logits, smoothness_type,
                raw_vel=raw_vel, nyq_vel=nyq_vel)

        # 計算加權後的值
        w_cls = lambda_cls * float(cls_loss)
        w_reg = lambda_reg * float(reg_loss)
        w_physics = lambda_physics * float(physics_loss)
        w_soft = lambda_soft * float(soft_cls_loss)
        w_conf = lambda_confidence * float(confidence_loss)
        w_spatial = lambda_spatial * float(spatial_loss_val)

        total_weighted = w_cls + w_reg + w_physics + w_soft + w_conf + w_spatial
        if total_weighted == 0:
            total_weighted = 1.0  # 避免除零

        print(f"\n{'='*70}")
        print(f"  Loss Breakdown 診斷（1 batch, eager mode）")
        print(f"{'='*70}")
        print(f"  {'分支':<22} {'raw loss':>10} {'lambda':>8} {'加權值':>10} {'佔比':>8}")
        print(f"  {'-'*60}")
        print(f"  {'Classification':<22} {float(cls_loss):>10.4f} {lambda_cls:>8.2f} {w_cls:>10.4f} {w_cls/total_weighted*100:>7.1f}%")
        print(f"  {'Soft Classification':<22} {float(soft_cls_loss):>10.4f} {lambda_soft:>8.2f} {w_soft:>10.4f} {w_soft/total_weighted*100:>7.1f}%")
        print(f"  {'Regression':<22} {float(reg_loss):>10.4f} {lambda_reg:>8.2f} {w_reg:>10.4f} {w_reg/total_weighted*100:>7.1f}%")
        print(f"  {'Physics':<22} {float(physics_loss):>10.4f} {lambda_physics:>8.2f} {w_physics:>10.4f} {w_physics/total_weighted*100:>7.1f}%")
        print(f"  {'Confidence':<22} {float(confidence_loss):>10.4f} {lambda_confidence:>8.2f} {w_conf:>10.4f} {w_conf/total_weighted*100:>7.1f}%")
        print(f"  {'Spatial Smoothness':<22} {float(spatial_loss_val):>10.6f} {lambda_spatial:>8.1f} {w_spatial:>10.4f} {w_spatial/total_weighted*100:>7.1f}%")
        print(f"  {'-'*60}")
        print(f"  {'Total (加權)':<22} {'':>10} {'':>8} {total_weighted:>10.4f} {'100.0':>7}%")
        print(f"{'='*70}\n")

    @tf.function
    def train_step(x_dict, y_dict):
        """Gradient accumulation 完全在 @tf.function 內部（unrolled ACCUM 次）
        所有 tensor 在同一個 compiled graph 內建立與釋放，TF 自動管理記憶體
        batch_size=32 → MICRO=8（梯度 128MB）；batch_size=64 → MICRO=16（梯度 256MB）"""
        total_loss = tf.constant(0.0)
        total_sp   = tf.constant(0.0)

        # 每個 micro-batch 獨立 GradientTape，計算完梯度立即加到累積器
        # 第一個 micro-batch 初始化累積器，後續疊加
        with tf.GradientTape() as tape:
            loss0, sp0 = _compute_loss(
                {k: v[0*MICRO:1*MICRO] for k, v in x_dict.items()},
                {k: v[0*MICRO:1*MICRO] for k, v in y_dict.items()})
        grads = tape.gradient(loss0, model.trainable_variables)
        accum = [g * (1.0/ACCUM) if g is not None else None for g in grads]
        total_loss += loss0 * (1.0/ACCUM)
        total_sp   += sp0   * (1.0/ACCUM)

        with tf.GradientTape() as tape:
            loss1, sp1 = _compute_loss(
                {k: v[1*MICRO:2*MICRO] for k, v in x_dict.items()},
                {k: v[1*MICRO:2*MICRO] for k, v in y_dict.items()})
        grads = tape.gradient(loss1, model.trainable_variables)
        accum = [a + g * (1.0/ACCUM) if (a is not None and g is not None) else a
                 for a, g in zip(accum, grads)]
        total_loss += loss1 * (1.0/ACCUM)
        total_sp   += sp1   * (1.0/ACCUM)

        with tf.GradientTape() as tape:
            loss2, sp2 = _compute_loss(
                {k: v[2*MICRO:3*MICRO] for k, v in x_dict.items()},
                {k: v[2*MICRO:3*MICRO] for k, v in y_dict.items()})
        grads = tape.gradient(loss2, model.trainable_variables)
        accum = [a + g * (1.0/ACCUM) if (a is not None and g is not None) else a
                 for a, g in zip(accum, grads)]
        total_loss += loss2 * (1.0/ACCUM)
        total_sp   += sp2   * (1.0/ACCUM)

        with tf.GradientTape() as tape:
            loss3, sp3 = _compute_loss(
                {k: v[3*MICRO:4*MICRO] for k, v in x_dict.items()},
                {k: v[3*MICRO:4*MICRO] for k, v in y_dict.items()})
        grads = tape.gradient(loss3, model.trainable_variables)
        accum = [a + g * (1.0/ACCUM) if (a is not None and g is not None) else a
                 for a, g in zip(accum, grads)]
        total_loss += loss3 * (1.0/ACCUM)
        total_sp   += sp3   * (1.0/ACCUM)

        # 裁剪並更新權重
        clipped = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in accum]
        optimizer.apply_gradients(zip(clipped, model.trainable_variables))

        return total_loss, total_sp

    # Multi-GPU 包裝：strategy.run 將 train_step 分發到所有 replica
    # 各 replica 獨立做 gradient accumulation，apply_gradients 時自動 all-reduce
    @tf.function
    def distributed_train_step(x_dict, y_dict):
        per_replica_loss, per_replica_sp = strategy.run(train_step, args=(x_dict, y_dict))
        if _n_gpus > 1:
            loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=None)
            sp = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_sp, axis=None)
        else:
            loss = per_replica_loss
            sp = per_replica_sp
        return loss, sp

    # 訓練循環
    print("\n開始訓練...\n")

    for epoch in range(start_epoch, epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*80}")

        # 每個 epoch 重建 iterator（不重建 dataset），確保舊的 shuffle buffer / prefetch 被釋放
        # ds_mixed_train 只建立一次（在 loop 外），避免反覆建立 TF graph node 造成 GPU 記憶體碎片化
        mixed_iter = iter(ds_mixed_train)

        # 訓練一個 epoch（使用混合數據集）
        epoch_losses = []
        epoch_spatial_losses = []  # 🆕 記錄 spatial loss
        progress_bar = tqdm(range(mixed_train_steps), desc="訓練中")

        for step in progress_bar:
            x_dict, y_dict = next(mixed_iter)
            loss, spatial_loss = distributed_train_step(x_dict, y_dict)  # Multi-GPU 包裝
            loss_val = float(loss)
            spatial_val = float(spatial_loss)
            epoch_losses.append(loss_val)
            epoch_spatial_losses.append(spatial_val)
            progress_bar.set_postfix({'loss': f"{loss_val:.4f}",
                                     'spatial': f"{spatial_val:.4f}"})
            del x_dict, y_dict  # 立即釋放本步 batch，不等 Python GC
            if step % 100 == 0:
                gc.collect()

        # epoch 結束後立即刪除 iterator，釋放 shuffle buffer / prefetch buffer
        del mixed_iter
        gc.collect()

        avg_train_loss = np.mean(epoch_losses)
        avg_spatial_loss = np.mean(epoch_spatial_losses)  # 🆕 計算平均 spatial loss

        # 第 1 epoch 結束後：用一個 batch 診斷各分支 loss 佔比
        if epoch == start_epoch:
            try:
                diag_iter = iter(ds_mixed_train)
                diag_x, diag_y = next(diag_iter)
                # 只取一個 micro-batch 大小，避免 OOM
                diag_x_micro = {k: v[:MICRO] for k, v in diag_x.items()}
                diag_y_micro = {k: v[:MICRO] for k, v in diag_y.items()}
                diagnose_loss_breakdown(diag_x_micro, diag_y_micro)
                del diag_iter, diag_x, diag_y, diag_x_micro, diag_y_micro
                gc.collect()
            except Exception as e:
                print(f"  [診斷跳過] {e}")

        # 驗證：NWP
        print("\n驗證 NWP...")
        nwp_val_loss = evaluate_validation_set(
            model, ds_nwp_val, nwp_val_steps,
            lambda_cls, lambda_reg, lambda_physics, lambda_soft, lambda_confidence,
            mask_strategy, enable_spatial_smoothness, lambda_spatial, smoothness_type,
            aliased_weight, clean_weight, lambda_fpr, lambda_fnr,
            focal_gamma=focal_gamma, class_weight=class_weight,
            strategy=strategy, n_gpus=_n_gpus
        )

        # 驗證：颱風（僅在有颱風資料時）
        if typhoon_h5_path is not None:
            print("驗證颱風...")
            typhoon_val_loss = evaluate_validation_set(
                model, ds_typhoon_val, typhoon_val_steps,
                lambda_cls, lambda_reg, lambda_physics, lambda_soft, lambda_confidence,
                mask_strategy, enable_spatial_smoothness, lambda_spatial, smoothness_type,
                aliased_weight, clean_weight, lambda_fpr, lambda_fnr,
                focal_gamma=focal_gamma, class_weight=class_weight,
                strategy=strategy, n_gpus=_n_gpus
            )
            # 綜合評分：NWP 50% + 颱風 50%
            combined_score = -0.5 * typhoon_val_loss - 0.5 * nwp_val_loss
        else:
            typhoon_val_loss = None
            # 純 NWP 模式：直接用 NWP val loss
            combined_score = -nwp_val_loss

        # 記錄歷史
        history['epoch'].append(epoch+1)
        history['nwp_train_loss'].append(avg_train_loss)
        history['nwp_val_loss'].append(nwp_val_loss)
        history['typhoon_val_loss'].append(typhoon_val_loss if typhoon_val_loss is not None else 0.0)
        history['combined_score'].append(combined_score)
        history['learning_rate'].append(current_lr)
        history['spatial_loss'].append(avg_spatial_loss)  # 🆕 記錄 spatial loss

        # 顯示結果
        print(f"\n結果:")
        print(f"   混合訓練損失:    {avg_train_loss:.4f}")
        print(f"   空間平滑損失:    {avg_spatial_loss:.4f}")  # 🆕 顯示 spatial loss
        print(f"   NWP 驗證損失:    {nwp_val_loss:.4f}")
        if typhoon_val_loss is not None:
            print(f"   颱風驗證損失:    {typhoon_val_loss:.4f}")
        print(f"   綜合評分:        {combined_score:.4f}")
        print(f"   當前學習率:      {current_lr:.2e}")

        # Catastrophic Forgetting 監控（僅在有颱風資料時）
        if typhoon_h5_path is not None and typhoon_val_loss is not None:
            if baseline_typhoon_loss is None:
                baseline_typhoon_loss = typhoon_val_loss
                print(f"   📊 Baseline 颱風損失: {baseline_typhoon_loss:.4f}")
            else:
                loss_increase_ratio = (typhoon_val_loss - baseline_typhoon_loss) / baseline_typhoon_loss
                if loss_increase_ratio > 0.10:
                    print(f"   ⚠️  颱風損失增加 {loss_increase_ratio*100:.1f}% (Baseline: {baseline_typhoon_loss:.4f})")
                if loss_increase_ratio > 0.20:
                    print(f"\n🚨 颱風損失顯著增加 ({loss_increase_ratio*100:.1f}%)！")
                    print(f"   這可能表示 Catastrophic Forgetting，建議檢查訓練策略")
                    # 不強制停止，讓 combined_score early stopping 決定

        # 保存最佳模型
        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_epoch = epoch + 1
            patience_counter = 0
            early_stop_counter = 0  # 重置計數器

            # 保存最佳模型（與原始訓練一致）
            best_model_path = os.path.join(result_dir, "best_model.h5")
            model.save_weights(best_model_path, save_format='h5')
            print(f"   ✅ 新的最佳模型！（epoch {best_epoch}）")
            print(f"   💾 權重已保存: {best_model_path}")

            # 🆕 立即驗證權重是否可以正常載入
            print(f"   🔍 驗證權重文件...")
            test_model = VelocityDealiaser(extractor, upsampler_cls, upsampler_reg)
            dummy_test_input = {
                'vel': tf.zeros((1, 1, 128, 128, 1), dtype=tf.float32),
                'nyq': tf.ones((1, 1), dtype=tf.float32)
            }
            _ = test_model(dummy_test_input, training=False)

            try:
                test_model.load_weights(best_model_path)
                print(f"   ✅ 權重驗證通過")
            except Exception as e:
                print(f"   ⚠️  權重載入失敗，執行自動修復: {e}")
                fixed_path = best_model_path.replace('.h5', '_fixed.h5')
                success = manual_transfer_weights(best_model_path, fixed_path, test_model)
                if success:
                    # 用修復後的權重覆蓋原文件
                    shutil.move(fixed_path, best_model_path)
                    print(f"   ✅ 權重已自動修復並覆蓋原文件")
                else:
                    print(f"   ❌ 自動修復失敗，請手動檢查")
        else:
            patience_counter += 1
            early_stop_counter += 1
            print(f"   無改善 ({patience_counter}/{patience})")

            # 學習率調度
            if lr_scheduling and early_stop_counter > 0 and early_stop_counter % lr_patience == 0:
                old_lr = current_lr
                current_lr = max(old_lr * lr_factor, lr_min)
                lr_variable.assign(current_lr)
                lr_reduction_count += 1
                print(f"   📉 學習率調整: {old_lr:.2e} -> {current_lr:.2e} (第 {lr_reduction_count} 次)")

        # 儲存 checkpoint（每個 epoch 結束後，Early Stopping 檢查之前）
        save_checkpoint(
            result_dir, model, optimizer, epoch,
            best_combined_score, best_epoch, baseline_typhoon_loss,
            current_lr, lr_reduction_count, patience_counter,
            early_stop_counter, history
        )

        # Early Stopping（在 checkpoint 儲存之後）
        if patience_counter >= patience:
            print(f"\n✅ {patience} epochs 無改善，Early Stopping")
            break

    # ============================================================
    # 步驟 6: 保存結果
    # ============================================================
    print("\n" + "="*80)
    print("步驟 6/6: 保存訓練結果")
    print("="*80)

    # 保存歷史
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(result_dir, "training_history.csv"), index=False)

    # 繪製訓練曲線
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # NWP 訓練損失
    axes[0, 0].plot(history['epoch'], history['nwp_train_loss'])
    axes[0, 0].set_title('NWP Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)

    # NWP 驗證損失
    axes[0, 1].plot(history['epoch'], history['nwp_val_loss'])
    axes[0, 1].set_title('NWP Validation Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True)

    # 颱風驗證損失
    if typhoon_h5_path is not None:
        axes[1, 0].plot(history['epoch'], history['typhoon_val_loss'])
        axes[1, 0].set_title('Typhoon Validation Loss')
    else:
        axes[1, 0].text(0.5, 0.5, 'No typhoon data\n(NWP-only mode)',
                        ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Typhoon Validation Loss (N/A)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True)

    # 綜合評分
    axes[1, 1].plot(history['epoch'], history['combined_score'])
    score_title = 'Combined Score (50% Typhoon + 50% NWP)' if typhoon_h5_path is not None else 'NWP Val Score'
    axes[1, 1].set_title(score_title)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'training_history.png'), dpi=150)
    plt.close()

    # 保存配置
    config = {
        'pretrained_model': pretrained_model_path,
        'nwp_h5': nwp_h5_path,
        'typhoon_h5': typhoon_h5_path,
        'freeze_ratio': freeze_ratio,
        'learning_rate': learning_rate,
        'epochs_trained': len(history['epoch']),
        'best_epoch': best_epoch,
        'best_combined_score': float(best_combined_score),
        'final_nwp_val_loss': float(history['nwp_val_loss'][-1]),
        'final_typhoon_val_loss': float(history['typhoon_val_loss'][-1]) if typhoon_h5_path is not None else None,
        'final_spatial_loss': float(history['spatial_loss'][-1]),  # 🆕 記錄最終 spatial loss
        'baseline_typhoon_success': baseline_typhoon_success,
        'loss_weights': {
            'lambda_cls': lambda_cls,
            'lambda_reg': lambda_reg,
            'lambda_physics': lambda_physics,
            'lambda_soft': lambda_soft,
            'lambda_confidence': lambda_confidence,
            'lambda_spatial': lambda_spatial  # 🆕 記錄 spatial 權重
        },
        'spatial_smoothness': {  # 🆕 完整記錄 spatial smoothness 配置
            'enabled': enable_spatial_smoothness,
            'lambda_spatial': lambda_spatial,
            'smoothness_type': smoothness_type
        },
        'patch_weights': {  # Patch 類型權重配置
            'aliased_weight': aliased_weight,
            'clean_weight': clean_weight
        },
        'focal_gamma': focal_gamma,
        'lambda_fpr': lambda_fpr,
        'lambda_fnr': lambda_fnr
    }

    with open(os.path.join(result_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    end_time = time.time()
    training_time = (end_time - start_time) / 60

    print(f"\n✅ 訓練完成！")
    print(f"   最佳 epoch: {best_epoch}")
    print(f"   最佳綜合評分: {best_combined_score:.4f}")
    print(f"   訓練時間: {training_time:.2f} 分鐘")
    print(f"   結果保存於: {result_dir}")

    return result_dir


def main():
    parser = argparse.ArgumentParser(description='完整版遷移學習：真實雷達 → NWP')

    parser.add_argument('--pretrained_model', type=str, default=None,
                       help='預訓練模型路徑（省略則從零訓練）')
    parser.add_argument('--nwp_h5', type=str, required=True,
                       help='NWP 訓練資料路徑：.h5 檔 或 TFRecord 目錄（自動偵測）')
    parser.add_argument('--typhoon_h5', type=str, default=None,
                       help='颱風驗證資料路徑：.h5 檔 或 TFRecord 目錄（省略則純 NWP 訓練）')
    parser.add_argument('--output_dir', type=str, default='results/transfer_learning',
                       help='輸出目錄')
    parser.add_argument('--freeze_ratio', type=float, default=0.8,
                       help='凍結比例（0.8 = 凍結 80% 層）')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='學習率')
    parser.add_argument('--epochs', type=int, default=50,
                       help='最大訓練輪數')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--chunk_size', type=int, default=None,
                       help='H5 一次讀多少 patches (預設 256, 純原始順序模式設 1)。'
                            '影響流暢度 + throughput trade-off')
    parser.add_argument('--shuffle_buffer', type=int, default=None,
                       help='TF shuffle buffer 大小 (預設 max(2048, batch×8, CHUNK×4))。'
                            '必須 ≥ chunk×4 才不會卡頓')
    parser.add_argument('--steps_per_epoch', type=int, default=None,
                       help='覆寫每 epoch 的訓練步數（預設用整個 train dataset）。'
                            '用來加速 epoch，搭配增加 epochs 數使用')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping 耐心值')
    parser.add_argument('--baseline_typhoon_success', type=float, default=84.0,
                       help='基線颱風成功率（用於 Early Stopping）')
    # 學習率調度參數
    parser.add_argument('--lr_scheduling', type=bool, default=True,
                       help='是否啟用學習率調度')
    parser.add_argument('--lr_factor', type=float, default=0.5,
                       help='學習率縮減因子')
    parser.add_argument('--lr_patience', type=int, default=5,
                       help='學習率調度耐心值')
    parser.add_argument('--lr_min', type=float, default=1e-6,
                       help='最小學習率')
    # 損失權重參數
    parser.add_argument('--lambda_cls', type=float, default=2.0,
                       help='分類損失權重 (預訓練模型使用 2.0)')
    parser.add_argument('--lambda_reg', type=float, default=0.1,
                       help='回歸損失權重')
    parser.add_argument('--lambda_physics', type=float, default=0.1,
                       help='物理約束損失權重')
    parser.add_argument('--lambda_soft', type=float, default=0.5,
                       help='軟分類損失權重')
    parser.add_argument('--lambda_confidence', type=float, default=0.1,
                       help='置信度損失權重')
    # 空間平滑約束參數
    parser.add_argument('--disable_spatial_smoothness', dest='enable_spatial_smoothness',
                       action='store_false',
                       help='禁用空間平滑約束（默認啟用）')
    parser.set_defaults(enable_spatial_smoothness=True)
    parser.add_argument('--lambda_spatial', type=float, default=50.0,
                       help='空間平滑損失權重 (預訓練模型使用 50.0)')
    parser.add_argument('--smoothness_type', type=str, default='v_corrected',
                       choices=['fold_jump', 'total_variation', 'prob_diff', 'v_corrected'],
                       help='空間平滑類型 (v_corrected=懲罰修正後速度不連續, fold_jump=原版)')
    parser.add_argument('--mask_strategy', type=str, default='v1',
                       choices=['v1', 'v2'],
                       help='Mask 策略')
    parser.add_argument('--filter_patch_type', type=str, default='all',
                       choices=['all', 'aliased', 'clean'],
                       help='過濾 patch 類型')
    # Patch 類型權重參數
    parser.add_argument('--aliased_weight', type=float, default=0.8,
                       help='Aliased patches 的損失權重 (預設 0.8)')
    parser.add_argument('--clean_weight', type=float, default=0.2,
                       help='Clean patches 的損失權重 (預設 0.2)')
    parser.add_argument('--focal_gamma', type=float, default=0.0,
                       help='Focal Loss gamma 參數 (0=CE, 2.0=原始Focal)')
    parser.add_argument('--class_weight', type=float, nargs=6, default=None,
                       help='Per-class weight [cat0 cat1 cat2 cat3 cat4 cat5] '
                            '(cat0=invalid, 1=fold-2, 2=fold-1, 3=fold0, 4=fold+1, 5=fold+2)')
    parser.add_argument('--lambda_fpr', type=float, default=0.0,
                       help='FPR Auxiliary Loss 權重：懲罰 clean pixels 被預測為 aliased (0=關閉, 建議 0.5~2.0)')
    parser.add_argument('--lambda_fnr', type=float, default=0.0,
                       help='FNR Auxiliary Loss 權重：懲罰 aliased pixels 被預測為 no-alias (0=關閉, 建議 0.5~2.0)')
    # 正則化參數
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='Weight decay 正則化強度 (0=不使用, 建議 1e-4)')
    # 斷點續訓參數
    parser.add_argument('--resume', type=str, default=None,
                       help='從先前的 result 目錄接續訓練（例如 results/transfer_xxx_20260125_093649）')

    args = parser.parse_args()

    # CLI args 覆寫 H5 loader 的 env var（在 import load_mixed_patches 之前不行，
    # 但 loader 內部讀 os.environ 是在 function call 時，所以這裡設仍有效）
    if args.chunk_size is not None:
        os.environ['CHUNK_SIZE'] = str(args.chunk_size)
        print(f"   ⚙️  CHUNK_SIZE 設為 {args.chunk_size} (CLI override)")
    if args.shuffle_buffer is not None:
        os.environ['SHUFFLE_BUFFER'] = str(args.shuffle_buffer)
        print(f"   ⚙️  SHUFFLE_BUFFER 設為 {args.shuffle_buffer} (CLI override)")

    transfer_learning_complete(
        pretrained_model_path=args.pretrained_model,
        nwp_h5_path=args.nwp_h5,
        typhoon_h5_path=args.typhoon_h5,
        output_dir=args.output_dir,
        freeze_ratio=args.freeze_ratio,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        baseline_typhoon_success=args.baseline_typhoon_success,
        lr_scheduling=args.lr_scheduling,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        lr_min=args.lr_min,
        lambda_cls=args.lambda_cls,
        lambda_reg=args.lambda_reg,
        lambda_physics=args.lambda_physics,
        lambda_soft=args.lambda_soft,
        lambda_confidence=args.lambda_confidence,
        enable_spatial_smoothness=args.enable_spatial_smoothness,
        lambda_spatial=args.lambda_spatial,
        smoothness_type=args.smoothness_type,
        mask_strategy=args.mask_strategy,
        filter_patch_type=args.filter_patch_type,
        aliased_weight=args.aliased_weight,
        clean_weight=args.clean_weight,
        focal_gamma=args.focal_gamma,
        class_weight=args.class_weight,
        lambda_fpr=args.lambda_fpr,
        lambda_fnr=args.lambda_fnr,
        weight_decay=args.weight_decay,
        resume_dir=args.resume,
        steps_per_epoch=args.steps_per_epoch
    )


if __name__ == "__main__":
    main()
