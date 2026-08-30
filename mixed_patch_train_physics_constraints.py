#!/usr/bin/env python3
"""
混合Patch訓練腳本 - 物理約束版本
基於 mixed_patch_train_v2.py 修改，添加物理約束功能

新增功能:
- 局部方差最小化約束 (local_variance) - 防止像素間劇烈跳躍
- 拉普拉斯算子約束 (laplacian) - 最小曲率原理 
- 相對梯度懲罰約束 (relative_gradient) - 自適應梯度約束
- 統計分位數約束 (statistical) - 基於梯度分布的異常檢測
- 無約束選項 (none) - 原始損失函數

使用範例:
  # 測試局部方差約束
  python mixed_patch_train_physics_constraints.py \
      --h5_file mixed_patches.h5 \
      --experiment_name "test_local_variance" \
      --physics_constraint local_variance \
      --constraint_weight 0.1

  # 測試拉普拉斯約束
  python mixed_patch_train_physics_constraints.py \
      --h5_file mixed_patches.h5 \
      --experiment_name "test_laplacian" \
      --physics_constraint laplacian \
      --constraint_weight 0.05

  # 無約束對照組
  python mixed_patch_train_physics_constraints.py \
      --h5_file mixed_patches.h5 \
      --experiment_name "test_no_constraint" \
      --physics_constraint none
"""

import os
# 🔓 必須在 import h5py 之前設定：關閉 HDF5 file locking
# 否則多 H5 file handle 對同檔仍會序列化，方案 B 並行讀無效
# read-only + tmpfs (/dev/shm) 場景完全安全（沒寫操作不會 race）
os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')
import h5py
import numpy as np
import tensorflow as tf
import time
from tqdm import tqdm
import sys
import pandas as pd
import matplotlib.pyplot as plt
import json
import argparse
from tensorflow.keras.callbacks import ReduceLROnPlateau
#import tensorflow_probability as tfp

# 導入改進的物理約束
from improved_physics_constraints import (
    direct_gradient_constraint_with_mask,
    hierarchical_smoothness_constraint_with_mask,
    adaptive_threshold_constraint_with_mask,
    local_variance_constraint_simple
)

# 設置路徑 - 修改導入路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
# 使用v2版本以獲得更好的通道適配和靈活性
from unet_model.dealias_mulit_v2 import VelocityDealiaser
from unet_model.feature_extraction_mulit_v2 import create_downsampler, create_upsampler_cls, create_upsampler_reg

# 設置隨機種子
SEED = 46
np.random.seed(SEED)
tf.random.set_seed(SEED)

###############################################################################
# 物理約束函數實現
###############################################################################

def local_variance_constraint(y_pred, window_size=3):
    """
    局部方差最小化約束 (只在有效區域計算)
    在3x3窗口內最小化方差，自動懲罰不連續跳躍
    
    Parameters:
    -----------
    y_pred : tf.Tensor
        預測速度場 (B,H,W,1) - 可能包含NaN表示無效區域
    window_size : int
        局部窗口大小
        
    Returns:
    --------
    constraint_loss : tf.Tensor
        約束損失值
    """
    # 移除最後一個維度進行卷積計算
    y_pred_2d = tf.squeeze(y_pred, axis=-1)  # (B,H,W)
    
    # 創建有效區域mask（非NaN的區域）
    valid_mask = ~tf.math.is_nan(y_pred_2d)
    
    # 只有當有效區域足夠大時才計算約束
    valid_count = tf.reduce_sum(tf.cast(valid_mask, tf.float32))
    total_count = tf.cast(tf.reduce_prod(tf.shape(y_pred_2d)), tf.float32)
    valid_ratio = valid_count / total_count
    
    # 如果有效區域不足50%，跳過此約束
    if tf.less(valid_ratio, 0.5):
        return tf.constant(0.0, dtype=tf.float32)
    
    # ✅ 修正：保持原始值，避免0填充造成人工邊界
    # 不填充0，直接使用原始預測值
    
    # 創建平均池化kernel
    kernel = tf.ones([window_size, window_size, 1, 1]) / (window_size * window_size)
    
    # 擴展維度用於卷積 - 使用原始預測值
    y_pred_4d = tf.expand_dims(y_pred_2d, axis=-1)  # (B,H,W,1)
    valid_mask_4d = tf.expand_dims(tf.cast(valid_mask, tf.float32), axis=-1)
    
    # 計算局部平均（只考慮有效區域）
    local_sum = tf.nn.conv2d(y_pred_4d, kernel, strides=1, padding='SAME')
    local_count = tf.nn.conv2d(valid_mask_4d, kernel, strides=1, padding='SAME')
    local_mean = local_sum / (local_count + 1e-8)  # 避免除零
    
    # 計算局部方差（只在有效區域）
    squared_diff = tf.square(y_pred_4d - local_mean) * valid_mask_4d
    local_variance = tf.nn.conv2d(squared_diff, kernel, strides=1, padding='SAME')
    local_variance = local_variance / (local_count + 1e-8)
    
    # 只對有效區域的方差進行平均
    masked_variance = local_variance * valid_mask_4d
    total_valid_variance = tf.reduce_sum(masked_variance)
    total_valid_pixels = tf.reduce_sum(valid_mask_4d)
    
    result = total_valid_variance / (total_valid_pixels + 1e-8)
    return result

def local_variance_constraint_with_mask(y_pred, mask_valid, window_size=3):
    """
    使用與回歸損失相同的mask計算局部方差約束
    
    Parameters:
    -----------
    y_pred : tf.Tensor
        預測速度場 (B,H,W,1)
    mask_valid : tf.Tensor  
        有效區域mask (B,H,W,1) - 與回歸損失相同
    window_size : int
        局部窗口大小
    """
    # 移除最後一個維度
    y_pred_2d = tf.squeeze(y_pred, axis=-1)  # (B,H,W)
    mask_2d = tf.squeeze(mask_valid, axis=-1)  # (B,H,W)
    
    # 檢查是否有足夠的有效像素 - 批次大小感知版本
    valid_count = tf.reduce_sum(tf.cast(mask_2d, tf.float32))
    batch_size = tf.shape(mask_2d)[0]
    min_valid_per_patch = tf.cast(window_size * window_size, tf.float32)  # 至少需要window_size x window_size的區域
    min_total_valid = min_valid_per_patch * tf.cast(batch_size, tf.float32)
    
    if tf.less(valid_count, min_total_valid):
        return tf.constant(0.0, dtype=tf.float32)
    
    # ❌ 修正關鍵錯誤：不要用0填充無效區域，這會產生人工邊界梯度
    # ✅ 正確做法：保持原始預測值，用mask來過濾計算結果
    
    # 創建卷積kernel - 不歸一化，計算絕對數量
    kernel = tf.ones([window_size, window_size, 1, 1])  # 移除歸一化

    # 🛡️ 關鍵修復：在計算前清理NaN，避免污染整個計算流程
    y_pred_2d_clean = tf.where(tf.math.is_finite(y_pred_2d), y_pred_2d, tf.zeros_like(y_pred_2d))

    #tf.print("[LocalVar] CLEANED - y_pred_2d range:", tf.reduce_min(y_pred_2d_clean), "to", tf.reduce_max(y_pred_2d_clean))

    # 擴展維度 - 使用清理後的預測值
    y_pred_4d = tf.expand_dims(y_pred_2d_clean, axis=-1)  # 使用清理後的值
    mask_4d = tf.expand_dims(tf.cast(mask_2d, tf.float32), axis=-1)

    # 只對有效區域計算局部統計 - 使用mask加權
    local_sum = tf.nn.conv2d(y_pred_4d * mask_4d, kernel, strides=1, padding='SAME')
    local_count = tf.nn.conv2d(mask_4d, kernel, strides=1, padding='SAME')

    # 🔍 添加調試信息
    #tf.print("[LocalVar] local_count max:", tf.reduce_max(local_count), "min:", tf.reduce_min(local_count))

    # 只在有足夠有效像素的窗口計算平均值
    min_window_pixels = 2.0  # 降低閾值：窗口內至少需要2個有效像素
    valid_window_mask = tf.greater(local_count, min_window_pixels)

    local_mean = tf.where(
        valid_window_mask,
        local_sum / (local_count + 1e-8),
        tf.zeros_like(local_sum)
    )

    # 計算方差 (只對有效窗口和有效像素)
    y_diff = y_pred_4d - local_mean

    # 🔍 檢查y_diff中的異常值
    #tf.print("[LocalVar] y_diff range:", tf.reduce_min(y_diff), "to", tf.reduce_max(y_diff))
    #tf.print("[LocalVar] local_mean range:", tf.reduce_min(local_mean), "to", tf.reduce_max(local_mean))

    # 只對有效像素和有效窗口計算方差差分
    # 檢查維度並修正
    #tf.print("[LocalVar] mask_4d shape:", tf.shape(mask_4d))
    #tf.print("[LocalVar] valid_window_mask shape:", tf.shape(valid_window_mask))

    # 確保維度匹配 - valid_window_mask 是 (B,H,W,1)，需要廣播到 mask_4d 的維度
    mask_4d_bool = tf.cast(mask_4d, tf.bool)  # (B,H,W,1)
    valid_window_mask_3d = tf.squeeze(valid_window_mask, axis=-1)  # (B,H,W)
    valid_window_mask_4d = tf.expand_dims(valid_window_mask_3d, axis=-1)  # (B,H,W,1)

    combined_mask = mask_4d_bool & valid_window_mask_4d

    safe_y_diff = tf.where(
        combined_mask,
        y_diff,
        tf.zeros_like(y_diff)
    )

    squared_diff = tf.square(safe_y_diff)
    #tf.print("[LocalVar] squared_diff range:", tf.reduce_min(squared_diff), "to", tf.reduce_max(squared_diff))

    local_variance_sum = tf.nn.conv2d(squared_diff, kernel, strides=1, padding='SAME')

    local_variance = tf.where(
        valid_window_mask,
        local_variance_sum / (local_count + 1e-8),
        tf.zeros_like(local_variance_sum)
    )

    #tf.print("[LocalVar] local_variance range:", tf.reduce_min(local_variance), "to", tf.reduce_max(local_variance))

    # 只對有效窗口的方差進行平均
    valid_variance_pixels = tf.reduce_sum(tf.cast(valid_window_mask, tf.float32))

    # 檢查是否有足夠的有效窗口
    #tf.print("[LocalVar] valid_variance_pixels:", valid_variance_pixels)
    if tf.less(valid_variance_pixels, 1.0):
        tf.print("[LocalVar] SKIP: valid_variance_pixels < 1")
        return tf.constant(0.0, dtype=tf.float32)

    # 只統計有效窗口的方差
    total_valid_variance = tf.reduce_sum(local_variance * tf.cast(valid_window_mask, tf.float32))
    raw_result = total_valid_variance / (valid_variance_pixels + 1e-8)

    # 直接使用原始方差損失，不進行自適應調整
    result = raw_result

    # 🔍 檢查並修正NaN/Inf
    result = tf.where(tf.math.is_finite(result), result, tf.constant(0.0, dtype=tf.float32))

    return result

def laplacian_constraint_with_mask(y_pred, mask_valid):
    """
    帶mask的拉普拉斯約束 - 真正的二階導數實現
    計算拉普拉斯算子: ∇²f = ∂²f/∂x² + ∂²f/∂y²
    """
    y_pred_2d = tf.squeeze(y_pred, axis=-1)  # (B,H,W)
    mask_2d = tf.squeeze(mask_valid, axis=-1)  # (B,H,W)
    
    # ✅ 修正：不填充0，保持原始值避免人工邊界
    
    # 檢查尺寸是否足夠計算二階導數
    h, w = tf.shape(y_pred_2d)[1], tf.shape(y_pred_2d)[2]
    if tf.less(h, 3) or tf.less(w, 3):
        return tf.constant(0.0, dtype=tf.float32)
        
    # 檢查是否有足夠的有效像素
    valid_count = tf.reduce_sum(tf.cast(mask_2d, tf.float32))
    batch_size = tf.shape(mask_2d)[0]
    min_valid_per_patch = 9.0  # 至少需要3x3區域
    min_total_valid = min_valid_per_patch * tf.cast(batch_size, tf.float32)
    
    if tf.less(valid_count, min_total_valid):
        return tf.constant(0.0, dtype=tf.float32)
    
    # 計算x方向二階偏導數 ∂²f/∂x² - 使用原始值
    d2_dx2 = y_pred_2d[:, :, 2:] - 2*y_pred_2d[:, :, 1:-1] + y_pred_2d[:, :, :-2]

    # 計算y方向二階偏導數 ∂²f/∂y² - 使用原始值
    d2_dy2 = y_pred_2d[:, 2:, :] - 2*y_pred_2d[:, 1:-1, :] + y_pred_2d[:, :-2, :]
    
    # 確保維度匹配
    min_h = tf.minimum(tf.shape(d2_dx2)[1], tf.shape(d2_dy2)[1])
    min_w = tf.minimum(tf.shape(d2_dx2)[2], tf.shape(d2_dy2)[2])
    
    d2_dx2_cropped = d2_dx2[:, :min_h, :min_w]
    d2_dy2_cropped = d2_dy2[:, :min_h, :min_w]
    
    # 拉普拉斯算子
    laplacian = d2_dx2_cropped + d2_dy2_cropped
    
    # 只對有效區域的拉普拉斯值進行統計
    # 創建對應的有效區域mask
    mask_cropped = mask_2d[:, 1:min_h+1, 1:min_w+1]  # 對應裁剪後的區域
    
    # ✅ 修正：只統計有效區域的拉普拉斯值，避免0填充
    valid_laplacian_values = tf.boolean_mask(laplacian, mask_cropped)
    valid_pixels = tf.size(valid_laplacian_values)

    # 如果有效像素太少，返回0
    if tf.less(valid_pixels, 1):
        return tf.constant(0.0, dtype=tf.float32)

    result = tf.reduce_mean(tf.square(valid_laplacian_values))
    return result

def relative_gradient_constraint_with_mask(y_pred, mask_valid):
    """
    帶mask的相對梯度約束 - 真正的相對梯度實現
    懲罰相對於平均梯度過大的變化
    """
    y_pred_2d = tf.squeeze(y_pred, axis=-1)  # (B,H,W)
    mask_2d = tf.squeeze(mask_valid, axis=-1)  # (B,H,W)
    
    # 檢查是否有足夠的有效像素
    valid_count = tf.reduce_sum(tf.cast(mask_2d, tf.float32))
    batch_size = tf.shape(mask_2d)[0]
    min_valid_per_patch = 4.0  # 至少需要2x2區域計算梯度
    min_total_valid = min_valid_per_patch * tf.cast(batch_size, tf.float32)
    
    if tf.less(valid_count, min_total_valid):
        return tf.constant(0.0, dtype=tf.float32)
    
    # ✅ 修正：直接使用原始值計算梯度，避免0填充造成人工邊界

    # 計算水平和垂直梯度 - 使用原始值
    grad_x = tf.abs(y_pred_2d[:, :, 1:] - y_pred_2d[:, :, :-1])
    grad_y = tf.abs(y_pred_2d[:, 1:, :] - y_pred_2d[:, :-1, :])
    
    # 創建對應的有效區域mask
    mask_x = mask_2d[:, :, 1:] & mask_2d[:, :, :-1]  # 兩個相鄰像素都有效
    mask_y = mask_2d[:, 1:, :] & mask_2d[:, :-1, :]
    
    # 只對有效梯度計算統計
    valid_grad_x = tf.where(mask_x, grad_x, tf.zeros_like(grad_x))
    valid_grad_y = tf.where(mask_y, grad_y, tf.zeros_like(grad_y))
    
    # 計算有效梯度的平均值
    valid_count_x = tf.reduce_sum(tf.cast(mask_x, tf.float32))
    valid_count_y = tf.reduce_sum(tf.cast(mask_y, tf.float32))
    
    mean_grad_x = tf.reduce_sum(valid_grad_x) / (valid_count_x + 1e-8)
    mean_grad_y = tf.reduce_sum(valid_grad_y) / (valid_count_y + 1e-8)
    
    # 計算相對梯度懲罰 (gradient/mean_gradient)² - 只對有效梯度
    # 需要提取真正的有效梯度值（非零遮罩版本）
    valid_grad_x_values = tf.boolean_mask(grad_x, mask_x)
    valid_grad_y_values = tf.boolean_mask(grad_y, mask_y)

    result_x = tf.cond(tf.greater(valid_count_x, 0),
                      lambda: tf.reduce_mean(tf.square(valid_grad_x_values / (mean_grad_x + 1e-8))),
                      lambda: tf.constant(0.0, dtype=tf.float32))
    result_y = tf.cond(tf.greater(valid_count_y, 0),
                      lambda: tf.reduce_mean(tf.square(valid_grad_y_values / (mean_grad_y + 1e-8))),
                      lambda: tf.constant(0.0, dtype=tf.float32))

    # 檢查是否有足夠的有效梯度
    if tf.less(valid_count_x, 1) and tf.less(valid_count_y, 1):
        return tf.constant(0.0, dtype=tf.float32)
    
    return result_x + result_y

def statistical_constraint_with_mask(y_pred, mask_valid):
    """
    帶mask的統計約束 - 基於梯度分布的異常檢測
    只懲罰超過95%分位數的異常梯度
    """
    y_pred_2d = tf.squeeze(y_pred, axis=-1)  # (B,H,W)
    mask_2d = tf.squeeze(mask_valid, axis=-1)  # (B,H,W)
    
    # ✅ 修正：直接使用原始值計算梯度，避免0填充造成人工邊界

    # 計算梯度 - 使用原始值
    grad_x = tf.abs(y_pred_2d[:, :, 1:] - y_pred_2d[:, :, :-1])
    grad_y = tf.abs(y_pred_2d[:, 1:, :] - y_pred_2d[:, :-1, :])
    
    # 創建對應的有效區域mask
    mask_x = mask_2d[:, :, 1:] & mask_2d[:, :, :-1]
    mask_y = mask_2d[:, 1:, :] & mask_2d[:, :-1, :]
    
    # 收集所有有效的梯度值
    valid_grad_x = tf.boolean_mask(grad_x, mask_x)
    valid_grad_y = tf.boolean_mask(grad_y, mask_y)
    
    # 檢查是否有足夠的梯度值
    total_valid_grads = tf.size(valid_grad_x) + tf.size(valid_grad_y)
    if tf.less(total_valid_grads, 20):  # 至少需要20個有效梯度
        return tf.constant(0.0, dtype=tf.float32)
    
    # 合併所有有效梯度
    all_valid_gradients = tf.concat([valid_grad_x, valid_grad_y], 0)
    
    # 使用tf.nn.top_k計算95%分位數
    total_size = tf.size(all_valid_gradients)
    k = tf.cast(tf.cast(total_size, tf.float32) * 0.05, tf.int32)  # 前5%
    k = tf.maximum(k, 1)
    k = tf.minimum(k, total_size)
    
    top_k_values, _ = tf.nn.top_k(all_valid_gradients, k=k)
    percentile_95 = tf.reduce_min(top_k_values)
    
    # 只懲罰超過95%分位數的梯度
    excessive_grad_x = tf.maximum(0.0, valid_grad_x - percentile_95)
    excessive_grad_y = tf.maximum(0.0, valid_grad_y - percentile_95)
    
    # 計算平均超額梯度
    result = tf.reduce_mean(excessive_grad_x) + tf.reduce_mean(excessive_grad_y)
    return result

def laplacian_constraint(y_pred):
    """
    拉普拉斯算子約束 (安全版本)
    基於最小曲率原理，懲罰二階導數
    
    Parameters:
    -----------
    y_pred : tf.Tensor
        預測速度場 (B,H,W,1)
        
    Returns:
    --------
    constraint_loss : tf.Tensor
        約束損失值
    """
    # 檢查輸入是否包含NaN
    if tf.reduce_any(tf.math.is_nan(y_pred)):
        return tf.constant(0.0, dtype=tf.float32)
    
    # 移除最後一個維度
    y_pred_2d = tf.squeeze(y_pred, axis=-1)  # (B,H,W)
    
    # 檢查尺寸是否足夠計算二階導數
    h, w = tf.shape(y_pred_2d)[1], tf.shape(y_pred_2d)[2]
    if h < 3 or w < 3:
        return tf.constant(0.0, dtype=tf.float32)
    
    # 計算x方向二階偏導數 ∂²f/∂x²
    d2_dx2 = y_pred_2d[:, :, 2:] - 2*y_pred_2d[:, :, 1:-1] + y_pred_2d[:, :, :-2]
    
    # 計算y方向二階偏導數 ∂²f/∂y²
    d2_dy2 = y_pred_2d[:, 2:, :] - 2*y_pred_2d[:, 1:-1, :] + y_pred_2d[:, :-2, :]
    
    # 拉普拉斯算子 ∇²f = ∂²f/∂x² + ∂²f/∂y²
    # 確保維度匹配
    min_h = min(tf.shape(d2_dx2)[1], tf.shape(d2_dy2)[1])
    min_w = min(tf.shape(d2_dx2)[2], tf.shape(d2_dy2)[2])
    
    d2_dx2_cropped = d2_dx2[:, :min_h, :min_w]
    d2_dy2_cropped = d2_dy2[:, :min_h, :min_w]
    
    laplacian = d2_dx2_cropped + d2_dy2_cropped
    
    # 最小化拉普拉斯算子的平方，安全處理NaN
    result = tf.reduce_mean(tf.square(laplacian))
    return tf.where(tf.math.is_nan(result), tf.constant(0.0, dtype=tf.float32), result)

def relative_gradient_constraint(y_pred):
    """
    相對梯度懲罰約束 (安全版本)
    基於梯度相對於平均梯度的比例進行懲罰，自適應約束
    
    Parameters:
    -----------
    y_pred : tf.Tensor
        預測速度場 (B,H,W,1)
        
    Returns:
    --------
    constraint_loss : tf.Tensor
        約束損失值
    """
    # 檢查輸入是否包含NaN
    if tf.reduce_any(tf.math.is_nan(y_pred)):
        return tf.constant(0.0, dtype=tf.float32)
    
    # 移除最後一個維度
    y_pred_2d = tf.squeeze(y_pred, axis=-1)  # (B,H,W)
    
    # 檢查尺寸是否足夠計算梯度
    h, w = tf.shape(y_pred_2d)[1], tf.shape(y_pred_2d)[2]
    if h < 2 or w < 2:
        return tf.constant(0.0, dtype=tf.float32)
    
    # 計算水平和垂直梯度
    grad_x = tf.abs(y_pred_2d[:, :, 1:] - y_pred_2d[:, :, :-1])
    grad_y = tf.abs(y_pred_2d[:, 1:, :] - y_pred_2d[:, :-1, :])
    
    # 計算平均梯度，增強數值穩定性
    mean_grad_x = tf.reduce_mean(grad_x) + 1e-8  # 避免除零
    mean_grad_y = tf.reduce_mean(grad_y) + 1e-8
    
    # 相對梯度懲罰：(gradient/mean_gradient)²
    relative_penalty_x = tf.reduce_mean(tf.square(grad_x / mean_grad_x))
    relative_penalty_y = tf.reduce_mean(tf.square(grad_y / mean_grad_y))
    
    # 安全處理結果
    result = relative_penalty_x + relative_penalty_y
    return tf.where(tf.math.is_nan(result), tf.constant(0.0, dtype=tf.float32), result)

def statistical_constraint(y_pred):
    """
    統計分位數約束 (修復版 - 不依賴tfp，安全版本)
    基於梯度分布統計，只懲罰超過95%分位數的異常梯度
    
    Parameters:
    -----------
    y_pred : tf.Tensor
        預測速度場 (B,H,W,1)
        
    Returns:
    --------
    constraint_loss : tf.Tensor
        約束損失值
    """
    # 檢查輸入是否包含NaN
    if tf.reduce_any(tf.math.is_nan(y_pred)):
        return tf.constant(0.0, dtype=tf.float32)
    
    # 移除最後一個維度
    y_pred_2d = tf.squeeze(y_pred, axis=-1)  # (B,H,W)
    
    # 檢查尺寸是否足夠計算梯度
    h, w = tf.shape(y_pred_2d)[1], tf.shape(y_pred_2d)[2]
    if h < 2 or w < 2:
        return tf.constant(0.0, dtype=tf.float32)
    
    # 計算所有梯度
    grad_x = tf.abs(y_pred_2d[:, :, 1:] - y_pred_2d[:, :, :-1])
    grad_y = tf.abs(y_pred_2d[:, 1:, :] - y_pred_2d[:, :-1, :])
    
    # 將所有梯度合併為一維
    all_gradients = tf.concat([tf.reshape(grad_x, [-1]), tf.reshape(grad_y, [-1])], 0)
    
    # 檢查梯度數量是否足夠
    total_size = tf.size(all_gradients)
    if total_size < 10:  # 如果梯度數量太少，跳過約束
        return tf.constant(0.0, dtype=tf.float32)
    
    # 使用tf.nn.top_k替代tfp.stats.percentile計算95%分位數
    k = tf.cast(tf.cast(total_size, tf.float32) * 0.05, tf.int32)  # 前5%最大值
    k = tf.maximum(k, 1)  # 確保k至少為1
    k = tf.minimum(k, total_size)  # 確保k不超過總數
    
    top_k_values, _ = tf.nn.top_k(all_gradients, k=k)
    percentile_95 = tf.reduce_min(top_k_values)  # 95%分位數 = 前5%的最小值
    
    # 只懲罰超過95%分位數的梯度
    excessive_grad_x = tf.maximum(0.0, grad_x - percentile_95)
    excessive_grad_y = tf.maximum(0.0, grad_y - percentile_95)
    
    # 安全處理結果
    result = tf.reduce_mean(excessive_grad_x) + tf.reduce_mean(excessive_grad_y)
    return tf.where(tf.math.is_nan(result), tf.constant(0.0, dtype=tf.float32), result)

def apply_physics_constraint(y_pred, constraint_type='none', constraint_weight=0.1):
    """
    應用指定的物理約束 (舊版本，向後兼容)
    """
    if constraint_type == 'none':
        return tf.constant(0.0, dtype=tf.float32)
    elif constraint_type == 'local_variance':
        return constraint_weight * local_variance_constraint(y_pred)
    elif constraint_type == 'laplacian':
        return constraint_weight * laplacian_constraint(y_pred)
    elif constraint_type == 'relative_gradient':
        return constraint_weight * relative_gradient_constraint(y_pred)
    elif constraint_type == 'statistical':
        return constraint_weight * statistical_constraint(y_pred)
    else:
        raise ValueError(f"未知的物理約束類型: {constraint_type}")

def apply_physics_constraint_with_mask(y_pred, mask_valid, constraint_type='none', constraint_weight=0.1, debug=False):
    """
    使用與回歸損失相同的mask應用物理約束
    
    Parameters:
    -----------
    y_pred : tf.Tensor
        預測速度場 (B,H,W,1)
    mask_valid : tf.Tensor
        有效區域mask (B,H,W,1) - 與回歸損失使用相同的mask
    constraint_type : str
        約束類型: 'none', 'local_variance', 'laplacian', 'relative_gradient', 'statistical'
    constraint_weight : float
        約束權重
    debug : bool
        是否輸出調試信息
        
    Returns:
    --------
    constraint_loss : tf.Tensor
        約束損失值
    """
    if constraint_type == 'none':
        return tf.constant(0.0, dtype=tf.float32)
    
    # 檢查有效區域是否足夠
    valid_count = tf.reduce_sum(tf.cast(mask_valid, tf.float32))
    total_count = tf.cast(tf.reduce_prod(tf.shape(mask_valid)), tf.float32)
    valid_ratio = valid_count / total_count
    
    # 🔍 獲取batch信息進行診斷
    batch_size = tf.shape(mask_valid)[0]
    pixels_per_patch = total_count / tf.cast(batch_size, tf.float32)
    valid_per_patch = valid_count / tf.cast(batch_size, tf.float32)
    
    # 調試信息
    #if debug:
    #    tf.print(f"[Physics] Batch size: {batch_size}")
    #    tf.print(f"[Physics] Total pixels: {total_count} ({pixels_per_patch:.0f} per patch)")
    #    tf.print(f"[Physics] Valid pixels: {valid_count} ({valid_per_patch:.0f} per patch)")
    #    tf.print(f"[Physics] Valid ratio: {valid_ratio:.3f}")
    
    # 🎯 修正閾值邏輯：基於每個patch的平均有效像素
    min_valid_per_patch = 5.0  # 降低閾值：每個patch至少5個有效像素
    min_total_valid = min_valid_per_patch * tf.cast(batch_size, tf.float32)

    # 🔍 總是輸出調試信息來診斷問題
    #tf.print("[Physics]", constraint_type, ": valid=", valid_count, ", need=", min_total_valid, ", ratio=", valid_ratio)

    if tf.less(valid_count, min_total_valid):
        tf.print("[Physics] SKIP", constraint_type, ": valid=", valid_count, "< need=", min_total_valid)
        return tf.constant(0.0, dtype=tf.float32)
    
    # 使用mask中的有效區域計算約束
    if constraint_type == 'local_variance':
        result = constraint_weight * local_variance_constraint_with_mask(y_pred, mask_valid)
    elif constraint_type == 'laplacian':
        result = constraint_weight * laplacian_constraint_with_mask(y_pred, mask_valid)
    elif constraint_type == 'relative_gradient':
        result = constraint_weight * relative_gradient_constraint_with_mask(y_pred, mask_valid)
    elif constraint_type == 'statistical':
        result = constraint_weight * statistical_constraint_with_mask(y_pred, mask_valid)
    # 🆕 新增的改進約束
    elif constraint_type == 'direct_gradient':
        result = constraint_weight * direct_gradient_constraint_with_mask(y_pred, mask_valid,20.0)
    elif constraint_type == 'hierarchical':
        result = constraint_weight * hierarchical_smoothness_constraint_with_mask(y_pred, mask_valid)
    elif constraint_type == 'adaptive_threshold':
        result = constraint_weight * adaptive_threshold_constraint_with_mask(y_pred, mask_valid)
    elif constraint_type == 'simple_local_variance':
        result = constraint_weight * local_variance_constraint_simple(y_pred, mask_valid)
    else:
        raise ValueError(f"未知的物理約束類型: {constraint_type}")
    
    # 🔍 總是輸出約束值來診斷問題
    #tf.print("[Physics]", constraint_type, "result:", result)

    # 🛡️ 最終安全檢查：確保不返回NaN或Inf
    result = tf.where(tf.math.is_finite(result), result, tf.constant(0.0, dtype=tf.float32))
    #tf.print("[Physics]", constraint_type, "final result:", result)

    return result

###############################################################################
# 從 TFRecord 讀 patch - C++ pipeline，繞過 Python GIL
###############################################################################
def load_mixed_patches_from_tfrecord(tfrecord_dir, split='train', batch_size=4,
                                       shuffle=True, repeat=True, filter_patch_type='all'):
    """
    從 TFRecord shards 目錄讀取 patches。

    與 load_mixed_patches_from_h5 介面完全相容（同樣 signature 和 return）。

    Parameters
    ----------
    tfrecord_dir : str
        包含 {split}-XXXXX-of-XXXXX.tfrecord(.gz) 檔和 {split}_metadata.json 的目錄。
        用 convert_h5_to_tfrecord.py 產生。
    其餘參數同 load_mixed_patches_from_h5。

    Returns
    -------
    ds : tf.data.Dataset
        已 batched、加 time dim、prefetched 的 dataset。
    num_patches : int
        該 split 的 patch 總數（套用 filter_patch_type 後）。
    """
    import json as _json
    from pathlib import Path as _Path

    _t_meta = time.time()
    tfrecord_dir = _Path(tfrecord_dir)

    # ── 讀 metadata ──
    meta_path = tfrecord_dir / f'{split}_metadata.json'
    if not meta_path.exists():
        raise FileNotFoundError(
            f"找不到 metadata: {meta_path}\n"
            f"請確認 tfrecord_dir 正確、有跑過 convert_h5_to_tfrecord.py"
        )
    with open(meta_path) as f:
        meta = _json.load(f)

    total_patches = int(meta['num_patches'])
    vel_shape = tuple(meta['vel_shape'])            # e.g. (128, 128, 1)
    label_shape = tuple(meta['alias_label_shape'])  # e.g. (128, 128)
    gt_shape = tuple(meta['gt_vel_shape'])          # e.g. (128, 128, 1)
    compression = meta.get('compression', '') or ''
    patch_type_counts = meta.get('patch_type_counts', {})

    # 套用 patch_type filter 後的實際 count
    if filter_patch_type == 'all':
        num_patches = total_patches
    else:
        num_patches = int(patch_type_counts.get(filter_patch_type, 0))
        if num_patches == 0:
            print(f"⚠️  filter_patch_type='{filter_patch_type}' 找不到任何 patch（可用: {list(patch_type_counts.keys())}）")

    print(f"📦 TFRecord 格式，{split} 共 {total_patches} patches，"
          f"過濾後 {num_patches} (metadata loaded in {time.time()-_t_meta:.2f}s)")
    print(f"   patch_type 分布: {patch_type_counts}")

    # ── 找 shard 檔案 ──
    if compression == 'GZIP':
        shard_pattern = str(tfrecord_dir / f'{split}-*.tfrecord.gz')
    else:
        shard_pattern = str(tfrecord_dir / f'{split}-*.tfrecord')
    shard_files = sorted(tf.io.gfile.glob(shard_pattern))
    if len(shard_files) == 0:
        # 不分 gz/non-gz 都試試
        shard_files = sorted(tf.io.gfile.glob(str(tfrecord_dir / f'{split}-*.tfrecord*')))
    if len(shard_files) == 0:
        raise FileNotFoundError(f"沒找到 TFRecord 檔: {shard_pattern}")
    print(f"   📁 {len(shard_files)} shard 檔，compression={compression or 'NONE'}")

    # ── parse function ──
    feature_description = {
        'vel':         tf.io.FixedLenFeature([], tf.string),
        'nyq':         tf.io.FixedLenFeature([], tf.float32),
        'alias_label': tf.io.FixedLenFeature([], tf.string),
        'gt_vel':      tf.io.FixedLenFeature([], tf.string),
        'target_vel':  tf.io.FixedLenFeature([], tf.string),
        'patch_type':  tf.io.FixedLenFeature([], tf.string),
    }

    def _parse(serialized):
        ex = tf.io.parse_single_example(serialized, feature_description)
        vel = tf.reshape(tf.io.decode_raw(ex['vel'], tf.float32), vel_shape)
        nyq = tf.expand_dims(ex['nyq'], axis=0)  # () -> (1,)
        alias_label = tf.reshape(tf.io.decode_raw(ex['alias_label'], tf.int32), label_shape)
        gt_vel = tf.reshape(tf.io.decode_raw(ex['gt_vel'], tf.float32), gt_shape)
        target_vel = tf.reshape(tf.io.decode_raw(ex['target_vel'], tf.float32), gt_shape)
        return (
            {'vel': vel, 'nyq': nyq},
            {'alias_label': alias_label, 'gt_vel': gt_vel,
             'target_vel': target_vel, 'patch_type': ex['patch_type']}
        )

    # ── 建 dataset：interleave 多 shard 並行讀（C++ 真平行）──
    files_ds = tf.data.Dataset.from_tensor_slices(shard_files)
    if shuffle:
        files_ds = files_ds.shuffle(len(shard_files))

    # cycle_length: 同時 open 幾個 shard 並交錯讀
    # AUTOTUNE 會根據 CPU 自動調整（通常 = min(N_shards, n_CPU)）
    ds = files_ds.interleave(
        lambda fp: tf.data.TFRecordDataset(
            fp,
            compression_type=compression,
            buffer_size=8 * 1024 * 1024,  # 8 MB read buffer per file
        ),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not shuffle,  # shuffle 模式下不要求順序，最大化吞吐
    )

    # Parse（C++ 並行）
    ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)

    # Filter（在 parse 後做，因為要讀 patch_type）
    if filter_patch_type != 'all':
        _filter_pt = tf.constant(filter_patch_type)
        ds = ds.filter(lambda x, y: tf.equal(y['patch_type'], _filter_pt))

    # Shuffle（TFRecord 沒 numpy in-generator shuffle，這裡要大一點才有足夠隨機性）
    if shuffle:
        _env_sb = os.environ.get('SHUFFLE_BUFFER')
        if _env_sb:
            shuffle_buffer = min(int(_env_sb), max(num_patches, 1))
            _sb_src = f'env SHUFFLE_BUFFER={_env_sb}'
        else:
            # 預設公式：max(4096, batch×16) — 跟 batch 大小 scale，大 batch 也夠 lookahead
            #  - batch=128:  max(4096, 2048)  = 4096
            #  - batch=384:  max(4096, 6144)  = 6144
            #  - batch=724:  max(4096, 11584) = 11584
            shuffle_buffer = min(max(4096, batch_size * 16), max(num_patches, 1))
            _sb_src = f'default = max(4096, batch×16)'
        ds = ds.shuffle(shuffle_buffer)
        print(f"   📦 Shuffle buffer: {shuffle_buffer} ({_sb_src}, "
              f"batch={batch_size}, ratio={shuffle_buffer/batch_size:.1f}x)")

    # Batch
    ds = ds.batch(batch_size, drop_remainder=True)

    # 加 time dimension（與 H5 path 一致）
    def _add_time_dim(x_dict, y_dict):
        vel = tf.expand_dims(x_dict['vel'], axis=1)  # (B,H,W,1) -> (B,1,H,W,1)
        return {'vel': vel, 'nyq': x_dict['nyq']}, y_dict

    ds = ds.map(_add_time_dim, num_parallel_calls=tf.data.AUTOTUNE)

    if repeat:
        ds = ds.repeat()

    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds, num_patches


###############################################################################
# 統一入口：自動偵測 H5 / TFRecord
###############################################################################
def load_mixed_patches(data_path, split='train', batch_size=4, shuffle=True,
                        repeat=True, filter_patch_type='all'):
    """
    自動偵測格式並轉派到對應 loader。

    - data_path 是 .h5 檔 → load_mixed_patches_from_h5
    - data_path 是目錄（含 {split}_metadata.json）→ load_mixed_patches_from_tfrecord

    Returns
    -------
    (ds, num_patches) — 兩種 backend 介面一致。
    """
    from pathlib import Path as _Path
    p = _Path(data_path)

    if p.is_file():
        # H5 或其他單檔格式
        if p.suffix.lower() not in ('.h5', '.hdf5'):
            print(f"⚠️  {p.name} 副檔名非 .h5，仍嘗試以 H5 載入")
        return load_mixed_patches_from_h5(
            data_path, split=split, batch_size=batch_size,
            shuffle=shuffle, repeat=repeat, filter_patch_type=filter_patch_type
        )
    elif p.is_dir():
        # TFRecord shard 目錄
        meta_check = p / f'{split}_metadata.json'
        if not meta_check.exists():
            raise FileNotFoundError(
                f"{data_path} 是目錄但找不到 {split}_metadata.json\n"
                f"請先跑 convert_h5_to_tfrecord.py 把 H5 轉成 TFRecord"
            )
        return load_mixed_patches_from_tfrecord(
            data_path, split=split, batch_size=batch_size,
            shuffle=shuffle, repeat=repeat, filter_patch_type=filter_patch_type
        )
    else:
        raise FileNotFoundError(f"路徑不存在: {data_path}")


###############################################################################
# 從混合 H5 文件讀取 patch - 支持patch_type (保持與原版相同)
###############################################################################
def load_mixed_patches_from_h5(h5_path, split='train', batch_size=4, shuffle=True, repeat=True, filter_patch_type='all'):
    """
    從混合patch H5文件讀取數據，支持patch_type信息和過濾。
    自動偵測 H5 格式：大 array（v4.0）或 per-group（舊版），向後兼容。
    """
    # 用 with 短暫打開 H5 只為了讀 metadata（不要整個 function 都包住）
    import time as _time
    _t_meta = _time.time()
    with h5py.File(h5_path, 'r') as h5f:
        if split not in h5f:
            raise ValueError(f"H5文件中找不到 '{split}' 數據集")

        split_group = h5f[split]

        # ── 偵測格式 ──
        is_big_array = ('vel' in split_group and isinstance(split_group['vel'], h5py.Dataset)
                        and split_group['vel'].ndim >= 3)

        if is_big_array:
            N = split_group['vel'].shape[0]
            print(f"✅ 偵測到大 array 格式 (v4.0)，{split} 共 {N} patches")

            # 🚀 向量化讀 patch_type：避免 Python list comp 對 3.7M strings (慢 10-50x)
            if 'patch_type' in split_group:
                pt_raw = split_group['patch_type'][:]
                # numpy 向量化 decode：對 bytes/object dtype 自動處理
                if pt_raw.dtype.kind == 'S':
                    all_patch_types = np.char.decode(pt_raw, 'utf-8')
                elif pt_raw.dtype.kind == 'O':
                    # h5py 預設用 object dtype 存 vlen string
                    # 仍需 loop 但用更快的方式（無 isinstance 判斷）
                    all_patch_types = np.array([
                        (pt.decode('utf-8') if isinstance(pt, bytes) else str(pt))
                        for pt in pt_raw
                    ])
                else:
                    all_patch_types = pt_raw.astype(str)
            else:
                all_patch_types = np.full(N, 'aliased', dtype=object)

            # 🚀 向量化過濾（numpy mask 而非 Python list comp）
            if filter_patch_type == 'all':
                filtered_indices = np.arange(N)
            else:
                filtered_indices = np.where(all_patch_types == filter_patch_type)[0]

            num_patches = len(filtered_indices)
            has_target_vel = 'target_vel' in split_group
            print(f"📊 {split}集 (過濾: {filter_patch_type}) 包含 {num_patches} 個patches "
                  f"(metadata loaded in {_time.time() - _t_meta:.1f}s)")

            # ════════════════════════════════════════════════════════════════════
            # 🚀 方案 B：並行多 H5 readers（parallel sharded generators）
            # ════════════════════════════════════════════════════════════════════
            # 機制：
            #  1. 把 indices 切成 NUM_SHARDS 份
            #  2. 每個 shard 一個獨立 generator + 獨立 H5 file handle
            #  3. tf.data.Dataset.sample_from_datasets 平行收集
            #  4. 各 shard 的 prefetch(2) 讓 TF 在背景同時做多個 H5 read
            #  5. H5py read 釋放 GIL → 多執行緒 disk I/O 真的並行
            #
            # 跨平台相容性：
            #  - h5py 多 file handle 對同檔案：所有 OS 支援（read-only）
            #  - tf.data.sample_from_datasets：所有 OS 支援
            #  - multiprocessing.cpu_count()：Windows/Linux/Mac 皆支援
            #
            # 對模型精度：
            #  - 訓練資料覆蓋率：每個 epoch 每個 patch 仍被看過一次（無重複/遺漏）
            #  - 隨機性：master shuffle + 每 shard 內 shuffle + downstream shuffle buffer
            #    三層 → 比方案 A 略有不同的 batch 組成，但對 Adam optimizer 影響微小
            #  - 預期精度差：±0.1-0.5%（在訓練雜訊範圍內）
            # ════════════════════════════════════════════════════════════════════
            # CHUNK_SIZE 控制 H5 一次讀多少 patches。
            # ⚠️ 重要平衡：CHUNK_SIZE 必須 << shuffle_buffer，否則 buffer 耗盡時
            # 訓練會「等」H5 讀下一個 chunk，造成每 N step 卡頓。
            # 預設 256：buffer 2048 可容 8 chunks，prefetch 完全覆蓋讀取延遲。
            # 可用 env var CHUNK_SIZE 覆寫。
            CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', '256'))

            # 自動偵測 CPU 數
            # 預設：CPU × 75%（留 25% 給 TF 自己跑 AUTOTUNE map、scheduling、prefetch）
            # 可用環境變數 NUM_DATA_SHARDS 覆寫（例如想激進設 100%、或保守設 50%）
            import multiprocessing as _mp
            _NUM_CPUS = _mp.cpu_count()
            _env_override = os.environ.get('NUM_DATA_SHARDS')
            if _env_override:
                NUM_SHARDS = max(1, int(_env_override))
                _src = f'env var NUM_DATA_SHARDS={_env_override}'
            else:
                # ⚠️ 預設 1：實測 sample_from_datasets + 多 H5 handles 反而慢
                # （主要因 HDF5 file lock 沒真的關掉，env var 設定太晚）
                # 想開並行：用 env var NUM_DATA_SHARDS=4 等
                NUM_SHARDS = 1
                _src = 'default 1 (single generator)'
            # 每 shard 的 H5 cache（總 ~1.5 GB 分散給各 shard，min 64 MB）
            H5_CACHE_BYTES = max(64 * 1024 * 1024, (1024 ** 3 + 512 * 1024 * 1024) // NUM_SHARDS)
            print(f"   🔀 並行 readers: {NUM_SHARDS} shards (CPU={_NUM_CPUS}, {_src}, "
                  f"cache={H5_CACHE_BYTES // (1024*1024)} MB/shard)")

            # 一次性 master shuffle（用 numpy in-place shuffle，比 Python list shuffle 快 10x）
            _master_indices = np.asarray(filtered_indices, dtype=np.int64)
            if shuffle:
                np.random.shuffle(_master_indices)  # in-place
            _shard_size = (len(_master_indices) + NUM_SHARDS - 1) // NUM_SHARDS
            _shard_indices_list = [
                _master_indices[i * _shard_size : (i + 1) * _shard_size]
                for i in range(NUM_SHARDS)
            ]
            # 處理 remainder：移除空 shard（最後 shard 切完可能為空）
            _shard_indices_list = [s for s in _shard_indices_list if len(s) > 0]

            # ════════════════════════════════════════════════════════════════════
            # 🚀 PRE-BATCHED generator：直接 yield 已 batched 的 tensor
            # 跳過 TF foreground 的 per-patch overhead (10-15 ms × batch = 5-8 秒)
            # 改成 per-batch overhead (~10-50 ms total)
            # 期待 step time: 10-20s → 1-3s
            # ════════════════════════════════════════════════════════════════════
            def _make_shard_generator(shard_id, shard_indices):
                """為單一 shard 建立 PRE-BATCHED generator factory."""
                def gen():
                    with h5py.File(h5_path, 'r',
                                    rdcc_nbytes=H5_CACHE_BYTES,
                                    rdcc_nslots=100003) as hf:
                        sd = hf[split]
                        # 用 numpy in-place shuffle（比 list 快 10x）
                        local_idx = shard_indices.copy() if hasattr(shard_indices, 'copy') else np.array(shard_indices)
                        if shuffle:
                            np.random.shuffle(local_idx)

                        # 直接以 batch_size 切片，每次 yield 一個完整 batch
                        for batch_start in range(0, len(local_idx), batch_size):
                            batch_idx = local_idx[batch_start:batch_start + batch_size]
                            # H5 fancy indexing 必須 sorted unique → np.sort 比 sorted() 快 5x
                            sorted_idx = np.sort(batch_idx)
                            # 不夠一個 batch 就跳過（避免不固定 shape）
                            if len(sorted_idx) < batch_size:
                                continue

                            # 一次批次讀整個 batch (5 個 H5 calls 取代 5×batch_size)
                            batch_vel   = sd['vel'][sorted_idx]      # (B,128,128,1)
                            batch_nyq   = sd['nyq'][sorted_idx]      # (B,1)
                            batch_label = sd['alias_label'][sorted_idx]  # (B,128,128)
                            batch_gt    = sd['gt_vel'][sorted_idx]   # (B,128,128,1)
                            # 向量化取 patch_types（避免 N 次 Python loop access）
                            batch_pt = all_patch_types[sorted_idx]
                            if has_target_vel:
                                batch_target = sd['target_vel'][sorted_idx]
                            else:
                                # 從 patch_type 決定 target（aliased 用 gt，clean 用 raw）
                                batch_target = np.where(
                                    (batch_pt == 'clean')[:, None, None, None],
                                    batch_vel, batch_gt
                                )

                            yield (
                                {'vel': batch_vel.astype(np.float32),
                                 'nyq': batch_nyq.astype(np.float32)},
                                {'alias_label': batch_label.astype(np.int32),
                                 'gt_vel': batch_gt.astype(np.float32),
                                 'target_vel': batch_target.astype(np.float32),
                                 'patch_type': batch_pt}
                            )
                return gen

            # 預設關掉 pre-batched（實測會變慢，可能因為大 numpy → TF tensor 轉換）
            # 想開啟設環境變數 PREBATCH=1
            _PREBATCHED = os.environ.get('PREBATCH', '0') == '1'

            if not _PREBATCHED:
                # 退回 per-patch chunked generator（簡單版，最快實測 ~9.6s/step）
                def _make_shard_generator(shard_id, shard_indices):
                    def gen():
                        with h5py.File(h5_path, 'r',
                                        rdcc_nbytes=H5_CACHE_BYTES,
                                        rdcc_nslots=100003) as hf:
                            sd = hf[split]
                            # numpy in-place shuffle (vs list shuffle 快 10x)
                            local_idx = shard_indices.copy() if hasattr(shard_indices, 'copy') else np.array(shard_indices)
                            if shuffle:
                                np.random.shuffle(local_idx)
                            for chunk_start in range(0, len(local_idx), CHUNK_SIZE):
                                chunk_raw = local_idx[chunk_start:chunk_start + CHUNK_SIZE]
                                # np.sort 比 sorted() 快 5-10x
                                sorted_idx = np.sort(chunk_raw)
                                chunk_vel   = sd['vel'][sorted_idx]
                                chunk_nyq   = sd['nyq'][sorted_idx]
                                chunk_label = sd['alias_label'][sorted_idx]
                                chunk_gt    = sd['gt_vel'][sorted_idx]
                                if has_target_vel:
                                    chunk_target = sd['target_vel'][sorted_idx]
                                # 向量化取 patch_types (避免 N 次 Python loop access)
                                chunk_pt = all_patch_types[sorted_idx]
                                for j in range(len(sorted_idx)):
                                    patch_type = chunk_pt[j]
                                    if has_target_vel:
                                        target_vel = chunk_target[j]
                                    else:
                                        target_vel = (chunk_vel[j].copy() if patch_type == 'clean'
                                                       else chunk_gt[j].copy())
                                    yield (
                                        {'vel': chunk_vel[j], 'nyq': chunk_nyq[j]},
                                        {'alias_label': chunk_label[j], 'gt_vel': chunk_gt[j],
                                         'target_vel': target_vel, 'patch_type': patch_type}
                                    )
                    return gen

            mixed_patch_generator = _make_shard_generator(0, _master_indices)

        else:
            # ── 舊版 per-group 格式 ──
            sample_patch_id = list(split_group.keys())[0]
            sample_patch = split_group[sample_patch_id]

            supports_mixed_patches = False
            if 'patch_type' in sample_patch.attrs:
                supports_mixed_patches = True
                print(f"✅ 檢測到混合patch格式，支持patch_type")
            elif 'patch_type' in sample_patch:
                supports_mixed_patches = True
                print(f"✅ 檢測到混合patch格式，支持patch_type")
            else:
                print(f"⚠️  未檢測到patch_type，將所有patch視為aliased類型")

            def _read_patch_type(pg):
                if supports_mixed_patches:
                    if 'patch_type' in pg.attrs:
                        pt = pg.attrs['patch_type']
                    elif 'patch_type' in pg:
                        pt = pg['patch_type'][()]
                    else:
                        return 'aliased'
                    return pt.decode() if isinstance(pt, bytes) else pt
                return 'aliased'

            # 計算過濾後數量
            filtered_patch_count = 0
            for pid in split_group:
                pt = _read_patch_type(split_group[pid])
                if filter_patch_type == 'all' or pt == filter_patch_type:
                    filtered_patch_count += 1

            num_patches = filtered_patch_count
            print(f"📊 {split}集 (過濾: {filter_patch_type}) 包含 {num_patches} 個patches")

            def mixed_patch_generator():
                with h5py.File(h5_path, 'r') as hf:
                    patch_ids = list(hf[split].keys())
                    if shuffle:
                        np.random.shuffle(patch_ids)

                    for patch_id in patch_ids:
                        patch_group = hf[split][patch_id]
                        patch_type = _read_patch_type(patch_group)

                        if filter_patch_type != 'all' and patch_type != filter_patch_type:
                            continue

                        vel = patch_group['vel'][:]
                        nyq = patch_group['nyq'][:]
                        alias_label = patch_group['alias_label'][:]
                        gt_vel = patch_group['gt_vel'][:]

                        if supports_mixed_patches and 'target_vel' in patch_group:
                            target_vel = patch_group['target_vel'][:]
                        else:
                            target_vel = vel.copy() if patch_type == 'clean' else gt_vel.copy()

                        yield (
                            {'vel': vel, 'nyq': nyq},
                            {'alias_label': alias_label, 'gt_vel': gt_vel,
                             'target_vel': target_vel, 'patch_type': patch_type}
                        )

    # 建立 tf.data.Dataset
    _is_prebatched = locals().get('_PREBATCHED', False)
    if _is_prebatched:
        # Pre-batched: 每個 element 已經是 (B, H, W, ...) shape
        output_signature = (
            {
                'vel': tf.TensorSpec(shape=(None, None, None, 1), dtype=tf.float32),
                'nyq': tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            },
            {
                'alias_label': tf.TensorSpec(shape=(None, None, None), dtype=tf.int32),
                'gt_vel': tf.TensorSpec(shape=(None, None, None, 1), dtype=tf.float32),
                'target_vel': tf.TensorSpec(shape=(None, None, None, 1), dtype=tf.float32),
                'patch_type': tf.TensorSpec(shape=(None,), dtype=tf.string),
            }
        )
    else:
        # Per-patch (舊版 fallback)
        output_signature = (
            {
                'vel': tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
                'nyq': tf.TensorSpec(shape=(1,), dtype=tf.float32),
            },
            {
                'alias_label': tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                'gt_vel': tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
                'target_vel': tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
                'patch_type': tf.TensorSpec(shape=(), dtype=tf.string),
            }
        )

    # 建立 dataset：v4.0 big array 格式用方案 B 並行 shards；舊版 fallback 單 generator
    _use_parallel_shards = (is_big_array
                            and '_shard_indices_list' in locals()
                            and len(_shard_indices_list) > 1)
    if _use_parallel_shards:
        # 方案 B：每個 shard 各自一個 dataset + prefetch + sample_from_datasets
        # Pre-batched 模式: 每 element = 1 個 batch，prefetch 小（2-4）即可
        # Per-patch 模式: prefetch = batch_size × 4 // NUM_SHARDS
        if _is_prebatched:
            _per_shard_prefetch = max(2, 32 // len(_shard_indices_list))
        else:
            _per_shard_prefetch = max(128, (batch_size * 4) // len(_shard_indices_list))
        print(f"   📤 Per-shard prefetch: {_per_shard_prefetch} "
              f"(total buffered: {_per_shard_prefetch * len(_shard_indices_list)} patches, "
              f"~{(_per_shard_prefetch * len(_shard_indices_list)) // batch_size} batches)")

        _shard_datasets = []
        for _i, _shard_idx in enumerate(_shard_indices_list):
            _ds_shard = tf.data.Dataset.from_generator(
                _make_shard_generator(_i, _shard_idx),
                output_signature=output_signature
            ).prefetch(_per_shard_prefetch)
            _shard_datasets.append(_ds_shard)

        # 均勻取樣：每個 shard 同等權重
        ds = tf.data.Dataset.sample_from_datasets(
            _shard_datasets,
            weights=[1.0 / len(_shard_datasets)] * len(_shard_datasets),
            stop_on_empty_dataset=False
        )
    else:
        # Fallback：舊版 per-group 格式或單 shard
        ds = tf.data.Dataset.from_generator(
            mixed_patch_generator,
            output_signature=output_signature
        )

    if _is_prebatched:
        # Pre-batched 模式：每個 element 已是 batch，**不需要 shuffle 也不需要 .batch()**
        # shuffle 已在 generator 內做（np.random.shuffle 全部 indices）
        # 每個 batch 是隨機 sample，跨 batch 也是隨機順序
        print(f"   ⚡ Pre-batched generator 模式: 每 yield 直接是 {batch_size}-patch batch（跳過 TF shuffle/batch）")
    else:
        if shuffle:
            # 🎯 shuffle_buffer 大小考量：
            #   **必須 ≥ CHUNK_SIZE × 4**，否則每讀新 chunk 就 buffer 耗盡 → 訓練卡頓
            #   公式: max(2048, batch_size × 8, CHUNK_SIZE × 4)
            #   ─ batch×8 確保 8 batches 隨機性
            #   ─ CHUNK_SIZE×4 確保 prefetch 完全覆蓋 H5 chunk read 延遲
            #   ─ 2048 是 floor（極小 batch 場景仍要夠 random）
            #   env var SHUFFLE_BUFFER 可自訂
            _env_sb = os.environ.get('SHUFFLE_BUFFER')
            _chunk_size_for_calc = int(os.environ.get('CHUNK_SIZE', '256'))
            if _env_sb:
                shuffle_buffer = min(int(_env_sb), num_patches)
                _sb_src = f'env SHUFFLE_BUFFER={_env_sb}'
            else:
                shuffle_buffer = min(
                    max(2048, batch_size * 8, _chunk_size_for_calc * 4),
                    num_patches
                )
                _sb_src = f'default = max(2048, batch×8, CHUNK×4)'
            ds = ds.shuffle(shuffle_buffer)
            print(f"   📦 Shuffle buffer: {shuffle_buffer} ({_sb_src}, "
                  f"batch_size={batch_size}, CHUNK_SIZE={_chunk_size_for_calc}, "
                  f"ratio={shuffle_buffer/batch_size:.1f}x batch, "
                  f"{shuffle_buffer/_chunk_size_for_calc:.1f}x chunk)")
        ds = ds.batch(batch_size)

    def add_time_dimension(x_dict, y_dict):
        vel = x_dict['vel']                  # (B,H,W,1)
        vel = tf.expand_dims(vel, axis=1)    # => (B,1,H,W,1)
        return {'vel': vel, 'nyq': x_dict['nyq']}, y_dict

    ds = ds.map(add_time_dimension, num_parallel_calls=tf.data.AUTOTUNE)

    if repeat:
        ds = ds.repeat()

    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds, num_patches

###############################################################################
# 混合Patch損失函數 - 加入物理約束
###############################################################################
def mixed_patch_focal_loss(labels, logits, patch_types, gamma=2.0, alpha=1.0, class_weight=None):
    """
    混合patch的focal loss - 根據patch_type調整

    Parameters:
    -----------
    class_weight : list/tuple of 6 floats, optional
        Per-class weight for [cat0, cat1, ..., cat5].
        cat0 (invalid) 被 mask 掉所以值無意義。
        若為 None 則所有 class 權重相同。
    """
    # 基礎focal loss (忽略label=0)
    mask_valid = tf.not_equal(labels, 0)
    label_clamped = tf.where(tf.equal(labels, 0), tf.fill(tf.shape(labels), 3), labels)

    y_true = tf.one_hot(label_clamped, depth=6)  # (B,H,W,6)
    y_pred = tf.nn.softmax(logits, axis=-1)
    y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0)
    ce = -y_true * tf.math.log(y_pred)
    # 當 gamma < 1 時，(1-p)^(gamma-1) 在 p→1 處梯度趨向 inf
    # 需要 clip 避免 0^(負數) = inf → inf×0 = NaN
    one_minus_p = tf.maximum(1.0 - y_pred, 1e-6)
    focal_factor = tf.pow(one_minus_p, gamma)
    loss_map = alpha * focal_factor * ce  # (B,H,W,6)

    # Per-class weighting
    if class_weight is not None:
        cw = tf.constant(class_weight, dtype=tf.float32)  # (6,)
        loss_map = loss_map * cw[tf.newaxis, tf.newaxis, tf.newaxis, :]  # broadcast to (B,H,W,6)

    loss_per_pixel = tf.reduce_sum(loss_map, axis=-1)  # => (B,H,W)

    valid_pixels = tf.boolean_mask(loss_per_pixel, mask_valid)
    
    def no_valid(): 
        return tf.constant(0.0, dtype=tf.float32)
    def compute_mean(): 
        return tf.reduce_mean(valid_pixels)
        
    base_loss = tf.cond(tf.size(valid_pixels)>0, compute_mean, no_valid)
    
    return base_loss

def mixed_patch_regression_loss_with_physics(y_true, y_pred, target_vel, patch_types, alias_labels, 
                                           mask_strategy='v1', nan_penalty_weight=10.0,
                                           physics_constraint='none', constraint_weight=0.1):
    """
    混合patch的回歸損失 - 加入物理約束
    
    Parameters:
    -----------
    y_true : tf.Tensor
        原始GT速度 (B,H,W,1) - 用於參考
    y_pred : tf.Tensor
        預測速度 (B,H,W,1) 
    target_vel : tf.Tensor
        實際訓練目標 (B,H,W,1) - 根據patch_type決定
    patch_types : tf.Tensor
        patch類型 (B,) - 'aliased' or 'clean'
    alias_labels : tf.Tensor
        標籤用於mask (B,H,W)
    mask_strategy : str
        'v1' - 原版策略，嚴格要求RAW和GT都有效
        'v2' - 改進策略，支持RAW缺失但GT有效的數據重建學習
    nan_penalty_weight : float
        NaN預測的懲罰權重 (僅用於v2策略)
    physics_constraint : str
        物理約束類型: 'none', 'local_variance', 'laplacian', 'relative_gradient', 'statistical'
    constraint_weight : float
        物理約束權重
    """
    # 🔍 檢查y_pred中的NaN情況
    nan_count = tf.reduce_sum(tf.cast(tf.math.is_nan(y_pred), tf.float32))
    #tf.print("[Mask] y_pred NaN count:", nan_count)

    # 🎯 使用與回歸損失相同的mask策略
    if mask_strategy == 'v1':
        # 原版mask策略：嚴格要求RAW和GT都有效
        mask_valid = (
            ~tf.math.is_nan(target_vel) &                                    # target必須有效
            ~tf.math.is_nan(y_pred) &                                       # 預測必須有效
            tf.not_equal(alias_labels[..., tf.newaxis], 0)                  # 排除標籤=0區域
        )
    elif mask_strategy == 'v2':
        # 改進mask策略：允許RAW為NaN但GT有值的位置參與訓練
        mask_valid = (
            ~tf.math.is_nan(target_vel) &                                   # target必須有效
            tf.not_equal(alias_labels[..., tf.newaxis], 0)                  # 排除標籤=0區域
        )
    else:
        raise ValueError(f"未知的mask策略: {mask_strategy}")
    
    target_valid = tf.boolean_mask(target_vel, mask_valid)
    pred_valid = tf.boolean_mask(y_pred, mask_valid)
    
    def no_valid(): 
        return tf.constant(0.0, dtype=tf.float32)
    
    def compute_loss():
        if mask_strategy == 'v2':
            # 對NaN預測給予懲罰，鼓勵模型輸出有效值
            nan_penalty = tf.where(
                tf.math.is_nan(pred_valid),
                tf.ones_like(pred_valid) * nan_penalty_weight,  # NaN懲罰
                tf.zeros_like(pred_valid)
            )
            
            # 將NaN預測替換為0進行MSE計算
            pred_clean = tf.where(
                tf.math.is_nan(pred_valid),
                tf.zeros_like(pred_valid),
                pred_valid
            )
            
            base_loss = tf.reduce_mean(tf.square(target_valid - pred_clean))
            penalty_loss = tf.reduce_mean(nan_penalty)
            regression_loss = base_loss + penalty_loss
        else:
            # v1策略：簡單MSE
            regression_loss = tf.reduce_mean(tf.square(target_valid - pred_valid))
        
        # 🎯 物理約束使用相同的mask - 傳遞mask給約束函數
        physics_loss = apply_physics_constraint_with_mask(
            y_pred, mask_valid, physics_constraint, constraint_weight
        )
        
        return regression_loss + physics_loss
        
    return tf.cond(tf.size(target_valid)>0, compute_loss, no_valid)

###############################################################################
# 混合Patch訓練函數 - 加入物理約束參數
###############################################################################
def train_mixed_patches_with_physics(h5_patch_path,
                        extractor, upsampler_cls, upsampler_reg,
                        experiment_name="mixed_patch_physics_exp",
                        epochs=30,
                        batch_size=4,
                        lambda_cls=1.0,
                        lambda_reg=1.0,
                        aliased_weight=0.8,
                        clean_weight=0.2,
                        mask_strategy='v1',
                        nan_penalty_weight=10.0,
                        # 新增：patch類型過濾參數
                        filter_patch_type='all',
                        # 新增：物理約束參數
                        physics_constraint='none',
                        constraint_weight=0.1,
                        # 早停參數
                        early_stopping=True,
                        patience=10,
                        min_delta=1e-4,
                        # 學習率調整參數  
                        lr_scheduling=True,
                        learning_rate=1e-4,
                        lr_factor=0.5,
                        lr_patience=5,
                        lr_min=1e-6):
    """
    混合patch訓練函數 - 加入物理約束
    
    Parameters:
    -----------
    physics_constraint : str
        物理約束類型: 'none', 'local_variance', 'laplacian', 'relative_gradient', 'statistical'
    constraint_weight : float
        物理約束權重
    """
    print(f"[INFO] 混合patch訓練實驗 (物理約束版): {experiment_name}")
    print(f"      Patch file: {h5_patch_path}")
    print(f"      🎯 Patch類型過濾: {filter_patch_type}")
    print(f"      🧪 物理約束: {physics_constraint} (權重: {constraint_weight})")
    print(f"      Aliased權重: {aliased_weight}, Clean權重: {clean_weight}")
    print(f"      Mask策略: {mask_strategy} {'(原版-嚴格)' if mask_strategy=='v1' else '(改進版-數據重建)'}")
    if mask_strategy == 'v2':
        print(f"      NaN懲罰權重: {nan_penalty_weight}")
    
    # 創建結果目錄
    result_dir = os.path.join("results", experiment_name)
    os.makedirs(result_dir, exist_ok=True)
    model_dir = os.path.join(result_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # 構建模型
    model = VelocityDealiaser(extractor, upsampler_cls, upsampler_reg)
    # 創建優化器和學習率調度器
    if lr_scheduling:
        # 使用可變學習率變量，便於後續手動調整
        initial_learning_rate = learning_rate
        lr_variable = tf.Variable(initial_learning_rate, trainable=False, name='learning_rate')
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_variable)
        print(f"🎛️ 啟用學習率調度，初始學習率: {initial_learning_rate}")
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        print(f"🔒 固定學習率: {learning_rate}")
    
    # 早停設置
    if early_stopping:
        early_stop_counter = 0
        no_improve_epochs = []
        print(f"⏰ 啟用早停，耐心值: {patience}, 最小改進: {min_delta}")
    else:
        early_stop_counter = 0  # 即使不使用早停也初始化，避免錯誤
        print("🚫 關閉早停功能")
    
    # 加載訓練和驗證數據
    ds_train, train_total_patches = load_mixed_patches_from_h5(
        h5_patch_path, split='train', batch_size=batch_size, shuffle=True, repeat=True, filter_patch_type=filter_patch_type
    )
    ds_val, val_total_patches = load_mixed_patches_from_h5(
        h5_patch_path, split='val', batch_size=batch_size, shuffle=False, repeat=False, filter_patch_type=filter_patch_type
    )
    
    print(f"Train patches = {train_total_patches}, Val patches = {val_total_patches}")
    steps_per_epoch = max(1, train_total_patches // batch_size)
    val_steps = max(1, val_total_patches // batch_size)
    
    # TensorBoard
    train_log_dir = os.path.join(result_dir, "logs", "train")
    val_log_dir = os.path.join(result_dir, "logs", "val")
    os.makedirs(train_log_dir, exist_ok=True)
    os.makedirs(val_log_dir, exist_ok=True)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)
    
    global_step = 0
    best_val_loss = np.inf
    best_epoch = 0
    current_lr = learning_rate
    lr_reduction_count = 0
    
    # 訓練步驟 - 支持混合patch和物理約束
    @tf.function
    def mixed_train_step(x_dict, y_dict, model, opt, lambda_cls, lambda_reg, mask_strat, nan_penalty, phys_constraint, const_weight):
        alias_label = y_dict['alias_label']  # (B,H,W)
        gt_vel = y_dict['gt_vel']           # (B,H,W,1) - 原始GT參考
        target_vel = y_dict['target_vel']   # (B,H,W,1) - 實際訓練目標
        patch_types = y_dict['patch_type']  # (B,) - patch類型

        with tf.GradientTape() as tape:
            out = model(x_dict, training=True)
            logits = out['alias_mask']      # (B,H,W,6)
            vel_pred = out['dealiased_vel'] # (B,H,W,1)

            # 使用帶物理約束的混合patch損失函數
            cls_loss = mixed_patch_focal_loss(alias_label, logits, patch_types)
            reg_loss = mixed_patch_regression_loss_with_physics(
                gt_vel, vel_pred, target_vel, patch_types, alias_label, 
                mask_strat, nan_penalty, phys_constraint, const_weight
            )
            total_loss = lambda_cls * cls_loss + lambda_reg * reg_loss

        grads = tape.gradient(total_loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        
        # 計算物理約束損失用於監控 - 使用與訓練相同的mask版本
        # 需要重新構建mask來保持一致性
        if mask_strat == 'v1':
            monitor_mask = (
                ~tf.math.is_nan(target_vel) &
                ~tf.math.is_nan(vel_pred) &
                tf.not_equal(alias_label[..., tf.newaxis], 0)
            )
        else:  # v2
            monitor_mask = (
                ~tf.math.is_nan(target_vel) &
                tf.not_equal(alias_label[..., tf.newaxis], 0)
            )
        physics_loss = apply_physics_constraint_with_mask(
            vel_pred, monitor_mask, phys_constraint, const_weight, debug=False
        )
        
        return total_loss, cls_loss, reg_loss, physics_loss

    @tf.function
    def mixed_val_step(x_dict, y_dict, model, lambda_cls, lambda_reg, mask_strat, nan_penalty, phys_constraint, const_weight):
        alias_label = y_dict['alias_label']
        gt_vel = y_dict['gt_vel']
        target_vel = y_dict['target_vel']
        patch_types = y_dict['patch_type']
        
        out = model(x_dict, training=False)
        logits = out['alias_mask']
        vel_pred = out['dealiased_vel']

        cls_loss = mixed_patch_focal_loss(alias_label, logits, patch_types)
        reg_loss = mixed_patch_regression_loss_with_physics(
            gt_vel, vel_pred, target_vel, patch_types, alias_label, 
            mask_strat, nan_penalty, phys_constraint, const_weight
        )
        total_loss = lambda_cls * cls_loss + lambda_reg * reg_loss
        
        # 計算物理約束損失用於監控 - 使用與訓練相同的mask版本
        # 需要重新構建mask來保持一致性
        if mask_strat == 'v1':
            monitor_mask = (
                ~tf.math.is_nan(target_vel) &
                ~tf.math.is_nan(vel_pred) &
                tf.not_equal(alias_label[..., tf.newaxis], 0)
            )
        else:  # v2
            monitor_mask = (
                ~tf.math.is_nan(target_vel) &
                tf.not_equal(alias_label[..., tf.newaxis], 0)
            )
        physics_loss = apply_physics_constraint_with_mask(
            vel_pred, monitor_mask, phys_constraint, const_weight, debug=False
        )
        
        return total_loss, cls_loss, reg_loss, physics_loss, vel_pred, logits
    
    # 訓練歷史記錄 - 加入物理約束記錄
    history = {
        'train_loss': [],
        'train_cls_loss': [],
        'train_reg_loss': [],
        'train_physics_loss': [],
        'val_loss': [],
        'val_cls_loss': [],
        'val_reg_loss': [],
        'val_physics_loss': [],
        'epoch_time': [],
        'aliased_patches_count': [],
        'clean_patches_count': [],
        'learning_rate': [],
        'early_stop_counter': [],
        'best_val_loss_so_far': []
    }
    
    # 記錄訓練開始時間
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")
        avg_loss, avg_cls, avg_reg, avg_physics = 0.0, 0.0, 0.0, 0.0
        
        # 統計patch類型分布
        aliased_count = 0
        clean_count = 0

        # 訓練循環
        for step, (x_batch, y_batch) in enumerate(tqdm(ds_train.take(steps_per_epoch), total=steps_per_epoch)):
            t_loss, c_loss, r_loss, p_loss = mixed_train_step(
                x_batch, y_batch, model, optimizer, lambda_cls, lambda_reg, 
                mask_strategy, nan_penalty_weight, physics_constraint, constraint_weight
            )
            avg_loss += t_loss.numpy()
            avg_cls += c_loss.numpy()
            avg_reg += r_loss.numpy()
            avg_physics += p_loss.numpy()
            
            # 統計patch類型
            for pt in y_batch['patch_type']:
                pt_str = pt.numpy().decode() if hasattr(pt.numpy(), 'decode') else str(pt.numpy())
                if pt_str == 'aliased':
                    aliased_count += 1
                else:
                    clean_count += 1

            # 記錄到TensorBoard
            with train_summary_writer.as_default():
                tf.summary.scalar("train_step_total_loss", t_loss, step=global_step)
                tf.summary.scalar("train_step_cls_loss", c_loss, step=global_step)
                tf.summary.scalar("train_step_reg_loss", r_loss, step=global_step)
                tf.summary.scalar("train_step_physics_loss", p_loss, step=global_step)
            global_step += 1

            if step % 20 == 0:
                print(f"  step [{step}/{steps_per_epoch}] total={t_loss.numpy():.4f}"
                      f" cls={c_loss.numpy():.4f} reg={r_loss.numpy():.4f} phys={p_loss.numpy():.4f}")

        avg_loss /= steps_per_epoch
        avg_cls /= steps_per_epoch
        avg_reg /= steps_per_epoch
        avg_physics /= steps_per_epoch
        
        # 驗證循環
        val_loss, val_cls, val_reg, val_physics = 0.0, 0.0, 0.0, 0.0
        val_count = 0
        for x_val, y_val in ds_val.take(val_steps):
            t_loss, c_loss, r_loss, p_loss, _, _ = mixed_val_step(
                x_val, y_val, model, lambda_cls, lambda_reg, 
                mask_strategy, nan_penalty_weight, physics_constraint, constraint_weight
            )
            val_loss += t_loss.numpy()
            val_cls += c_loss.numpy()
            val_reg += r_loss.numpy()
            val_physics += p_loss.numpy()
            val_count += 1
            
        if val_count > 0:
            val_loss /= val_count
            val_cls /= val_count
            val_reg /= val_count
            val_physics /= val_count
        
        # 計算這個epoch的耗時
        epoch_time = time.time() - epoch_start_time
        
        print(f"[Epoch {epoch+1}] train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}")
        print(f"             cls={val_cls:.4f}, reg={val_reg:.4f}, physics={val_physics:.4f}, time={epoch_time:.1f}s")
        print(f"             Patch分布: {aliased_count} aliased + {clean_count} clean")
        
        with train_summary_writer.as_default():
            tf.summary.scalar("train_loss_epoch", avg_loss, step=epoch)
            tf.summary.scalar("train_cls_epoch", avg_cls, step=epoch)
            tf.summary.scalar("train_reg_epoch", avg_reg, step=epoch)
            tf.summary.scalar("train_physics_epoch", avg_physics, step=epoch)
            tf.summary.scalar("aliased_patches_count", aliased_count, step=epoch)
            tf.summary.scalar("clean_patches_count", clean_count, step=epoch)
        
        with val_summary_writer.as_default():
            tf.summary.scalar("val_loss_epoch", val_loss, step=epoch)
            tf.summary.scalar("val_cls_epoch", val_cls, step=epoch)
            tf.summary.scalar("val_reg_epoch", val_reg, step=epoch)
            tf.summary.scalar("val_physics_epoch", val_physics, step=epoch)
            tf.summary.scalar("epoch_time", epoch_time, step=epoch)
        
        # 記錄歷史數據
        history['train_loss'].append(float(avg_loss))
        history['train_cls_loss'].append(float(avg_cls))
        history['train_reg_loss'].append(float(avg_reg))
        history['train_physics_loss'].append(float(avg_physics))
        history['val_loss'].append(float(val_loss))
        history['val_cls_loss'].append(float(val_cls))
        history['val_reg_loss'].append(float(val_reg))
        history['val_physics_loss'].append(float(val_physics))
        history['epoch_time'].append(float(epoch_time))
        history['aliased_patches_count'].append(aliased_count)
        history['clean_patches_count'].append(clean_count)
        # 記錄新的訓練指標
        history['learning_rate'].append(float(current_lr))
        history['early_stop_counter'].append(early_stop_counter if early_stopping else 0)
        history['best_val_loss_so_far'].append(float(best_val_loss))
        
        # 早停檢查
        if early_stopping and early_stop_counter >= patience:
            print(f"\n🛑 Early Stopping! 連續 {patience} 個epoch無改進")
            print(f"最佳驗證損失: {best_val_loss:.4f} (Epoch {best_epoch})")
            print(f"停止訓練於 Epoch {epoch + 1}")
            break
        
        # 獲取當前學習率
        if lr_scheduling:
            # 獲取學習率變量的值
            current_lr = float(optimizer.learning_rate.numpy()) if hasattr(optimizer.learning_rate, 'numpy') else learning_rate
        else:
            current_lr = float(optimizer.learning_rate.numpy()) if hasattr(optimizer.learning_rate, 'numpy') else learning_rate
        
        # 保存最佳模型和早停邏輯
        improved = False
        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_model_path = os.path.join(model_dir, f"{experiment_name}_best.h5")
            model.save_weights(best_model_path)
            print(f"📈 保存最佳模型 (Epoch {best_epoch}), val_loss={val_loss:.4f} (改進 {(history['val_loss'][epoch-1] if epoch > 0 else val_loss) - val_loss:.4f})")
            improved = True
            if early_stopping:
                early_stop_counter = 0
        else:
            if early_stopping:
                early_stop_counter += 1
                no_improve_epochs.append(epoch + 1)
                print(f"⏰ 驗證損失無改進 ({early_stop_counter}/{patience})，當前最佳: {best_val_loss:.4f} (Epoch {best_epoch})")
        
        # 自適應學習率調整 (基於驗證損失)
        if lr_scheduling and epoch > 0 and not improved:
            # 檢查是否需要降低學習率
            if len(history['val_loss']) >= lr_patience:
                recent_losses = history['val_loss'][-lr_patience:]
                if all(loss >= (best_val_loss - min_delta) for loss in recent_losses):
                    # 降低學習率
                    new_lr = max(current_lr * lr_factor, lr_min)
                    if new_lr < current_lr:
                        # 手動設置學習率變量
                        if lr_scheduling:
                            optimizer.learning_rate.assign(new_lr)
                        lr_reduction_count += 1
                        print(f"🎛️ 學習率調整: {current_lr:.2e} → {new_lr:.2e} (第{lr_reduction_count}次調整)")
                        current_lr = new_lr
    
    # 計算總訓練時間
    total_time = time.time() - start_time
    actual_epochs = len(history['train_loss'])
    
    print(f"\n=== 混合patch訓練完成! (物理約束版) ===")
    print(f"實際訓練: {actual_epochs}/{epochs} epochs")
    print(f"總耗時: {total_time/60:.1f}分鐘 (平均 {total_time/actual_epochs:.1f}s/epoch)")
    print(f"最佳驗證損失: {best_val_loss:.4f} (Epoch {best_epoch})")
    print(f"🧪 物理約束: {physics_constraint} (權重: {constraint_weight})")
    if lr_scheduling:
        print(f"學習率調整次數: {lr_reduction_count}")
        print(f"最終學習率: {current_lr:.2e}")
    if early_stopping and early_stop_counter >= patience:
        print(f"早停原因: 連續 {patience} 個epoch無改進")
    elif early_stopping:
        print(f"訓練完成，無早停觸發 (最後 {early_stop_counter} 個epoch無改進)")
    
    # 保存最終模型
    final_model_path = os.path.join(model_dir, f"{experiment_name}_final.h5")
    model.save_weights(final_model_path)
    print(f"最終模型已保存到 {final_model_path}")
    
    # 保存訓練歷史
    history_path = os.path.join(result_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    # 繪製增強版訓練曲線，包含物理約束監控
    plt.figure(figsize=(20, 16))
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    # 總損失曲線
    plt.subplot(3, 4, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train', linewidth=2)
    plt.plot(epochs_range, history['val_loss'], label='Validation', linewidth=2)
    plt.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best (Epoch {best_epoch})')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 組件損失
    plt.subplot(3, 4, 2)
    plt.plot(epochs_range, history['train_cls_loss'], label='Train Cls', linewidth=1.5)
    plt.plot(epochs_range, history['train_reg_loss'], label='Train Reg', linewidth=1.5)
    plt.plot(epochs_range, history['val_cls_loss'], label='Val Cls', linewidth=1.5)
    plt.plot(epochs_range, history['val_reg_loss'], label='Val Reg', linewidth=1.5)
    plt.title('Component Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 🆕 物理約束損失
    plt.subplot(3, 4, 3)
    plt.plot(epochs_range, history['train_physics_loss'], label='Train Physics', linewidth=2, color='purple')
    plt.plot(epochs_range, history['val_physics_loss'], label='Val Physics', linewidth=2, color='orange')
    plt.title(f'Physics Constraint Loss ({physics_constraint})')
    plt.xlabel('Epoch')
    plt.ylabel('Physics Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Patch分布統計
    plt.subplot(3, 4, 4)
    plt.plot(epochs_range, history['aliased_patches_count'], label='Aliased Patches', linewidth=2)
    plt.plot(epochs_range, history['clean_patches_count'], label='Clean Patches', linewidth=2)
    plt.title('Patch Type Distribution per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 訓練時間
    plt.subplot(3, 4, 5)
    plt.plot(epochs_range, history['epoch_time'], linewidth=2)
    plt.title('Training Time per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.grid(True, alpha=0.3)
    
    # 學習率變化
    plt.subplot(3, 4, 6)
    plt.plot(epochs_range, history['learning_rate'], linewidth=2, color='green')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # 早停計數器
    if early_stopping:
        plt.subplot(3, 4, 7)
        plt.plot(epochs_range, history['early_stop_counter'], linewidth=2, color='orange')
        plt.axhline(y=patience, color='red', linestyle='--', label=f'Patience ({patience})')
        plt.title('Early Stopping Counter')
        plt.xlabel('Epoch')
        plt.ylabel('No Improvement Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 最佳驗證損失追蹤
    plt.subplot(3, 4, 8)
    plt.plot(epochs_range, history['best_val_loss_so_far'], linewidth=2, color='purple')
    plt.title('Best Validation Loss Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Best Val Loss So Far')
    plt.grid(True, alpha=0.3)
    
    # 驗證損失改進情況
    plt.subplot(3, 4, 9)
    if len(history['val_loss']) > 1:
        val_loss_diff = [0] + [history['val_loss'][i] - history['val_loss'][i-1] 
                              for i in range(1, len(history['val_loss']))]
        colors = ['red' if diff > 0 else 'green' for diff in val_loss_diff]
        plt.bar(epochs_range, val_loss_diff, color=colors, alpha=0.7)
        plt.title('Validation Loss Change per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Val Loss Δ')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
    
    # 損失比例圖
    plt.subplot(3, 4, 10)
    if len(history['val_cls_loss']) > 0 and len(history['val_reg_loss']) > 0:
        total_val_loss = [cls + reg for cls, reg in zip(history['val_cls_loss'], history['val_reg_loss'])]
        cls_ratio = [cls/total if total > 0 else 0.5 for cls, total in zip(history['val_cls_loss'], total_val_loss)]
        reg_ratio = [reg/total if total > 0 else 0.5 for reg, total in zip(history['val_reg_loss'], total_val_loss)]
        phys_ratio = [phys/total if total > 0 else 0 for phys, total in zip(history['val_physics_loss'], total_val_loss)]
        
        plt.plot(epochs_range, cls_ratio, label='Cls Loss Ratio', linewidth=2)
        plt.plot(epochs_range, reg_ratio, label='Reg Loss Ratio', linewidth=2)
        plt.plot(epochs_range, phys_ratio, label='Physics Loss Ratio', linewidth=2)
        plt.title('Loss Component Ratios')
        plt.xlabel('Epoch')
        plt.ylabel('Ratio')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 🆕 物理約束效果分析
    plt.subplot(3, 4, 11)
    if physics_constraint != 'none':
        # 計算物理約束與總損失的關聯
        physics_normalized = [(p / max(history['val_physics_loss'])) for p in history['val_physics_loss']]
        total_normalized = [(t / max(history['val_loss'])) for t in history['val_loss']]
        
        plt.plot(epochs_range, physics_normalized, label='Physics Loss (normalized)', linewidth=2, color='red')
        plt.plot(epochs_range, total_normalized, label='Total Loss (normalized)', linewidth=2, color='blue')
        plt.title('Constraint vs Total Loss (Normalized)')
        plt.xlabel('Epoch')
        plt.ylabel('Normalized Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No Physics Constraint', ha='center', va='center', transform=plt.gca().transAxes)
    
    # 約束效果統計
    plt.subplot(3, 4, 12)
    if physics_constraint != 'none':
        physics_reduction = [(history['val_physics_loss'][0] - p) / history['val_physics_loss'][0] * 100 
                           for p in history['val_physics_loss']]
        plt.plot(epochs_range, physics_reduction, linewidth=2, color='green')
        plt.title('Physics Constraint Improvement (%)')
        plt.xlabel('Epoch')
        plt.ylabel('Improvement (%)')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No Physics Constraint', ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "physics_training_curves.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 創建訓練總結報告 - 加入物理約束信息
    summary_report = {
        'training_completed': True,
        'total_epochs_planned': epochs,
        'actual_epochs_trained': actual_epochs,
        'early_stopped': early_stopping and early_stop_counter >= patience,
        'best_validation_loss': float(best_val_loss),
        'best_epoch': int(best_epoch),
        'final_learning_rate': float(current_lr),
        'lr_reduction_count': int(lr_reduction_count),
        'training_time_minutes': float(total_time/60),
        'average_epoch_time_seconds': float(total_time/actual_epochs),
        'early_stopping_enabled': early_stopping,
        'lr_scheduling_enabled': lr_scheduling,
        # 新增物理約束信息
        'physics_constraint': physics_constraint,
        'constraint_weight': float(constraint_weight),
        'final_physics_loss': float(history['val_physics_loss'][-1]) if history['val_physics_loss'] else 0.0,
        'physics_improvement_percent': float(
            (history['val_physics_loss'][0] - history['val_physics_loss'][-1]) / history['val_physics_loss'][0] * 100
        ) if history['val_physics_loss'] and history['val_physics_loss'][0] > 0 else 0.0
    }
    
    summary_path = os.path.join(result_dir, "training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary_report, f, indent=4)
    
    print(f"📊 物理約束訓練曲線已保存: {os.path.join(result_dir, 'physics_training_curves.png')}")
    print(f"📋 訓練總結已保存: {summary_path}")
    
    return model, history, best_model_path

###############################################################################
# 主函數
###############################################################################
def main():
    parser = argparse.ArgumentParser(description='混合patch訓練腳本 - 物理約束版本')
    parser.add_argument('--h5_file', type=str, required=True,
                        help='混合patch H5文件路徑')
    parser.add_argument('--experiment_name', type=str, default='mixed_patch_physics_exp',
                        help='實驗名稱')
    parser.add_argument('--epochs', type=int, default=30,
                        help='訓練輪數')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小')
    parser.add_argument('--lambda_cls', type=float, default=1.0,
                        help='分類損失權重')
    parser.add_argument('--lambda_reg', type=float, default=1.0,
                        help='回歸損失權重')
    parser.add_argument('--aliased_weight', type=float, default=0.8,
                        help='摺錯patch權重')
    parser.add_argument('--clean_weight', type=float, default=0.2,
                        help='乾淨patch權重')
    parser.add_argument('--mask_strategy', type=str, default='v1', 
                        choices=['v1', 'v2'],
                        help='Mask策略: v1=原版(嚴格), v2=改進版(支持數據重建)')
    parser.add_argument('--nan_penalty_weight', type=float, default=10.0,
                        help='V2策略中NaN預測的懲罰權重')
    
    # patch類型過濾參數
    parser.add_argument('--filter_patch_type', type=str, default='all',
                        choices=['all', 'aliased', 'clean'],
                        help='過濾patch類型: all=使用所有類型, aliased=只使用有折錯, clean=只使用乾淨')
    
    # 🆕 物理約束參數
    parser.add_argument('--physics_constraint', type=str, default='none',
                        choices=['none', 'local_variance', 'laplacian', 'relative_gradient', 'statistical',
                                'direct_gradient', 'hierarchical', 'adaptive_threshold', 'simple_local_variance'],
                        help='物理約束類型: none=無約束, local_variance=局部方差約束, laplacian=拉普拉斯約束, relative_gradient=相對梯度約束, statistical=統計分位數約束, direct_gradient=直接梯度約束, hierarchical=階層約束, adaptive_threshold=自適應閾值約束, simple_local_variance=簡化局部方差約束')
    parser.add_argument('--constraint_weight', type=float, default=0.1,
                        help='物理約束權重')
    
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='初始學習率')
    
    # 早停參數
    parser.add_argument('--early_stopping', action='store_true', default=True,
                        help='啟用早停功能')
    parser.add_argument('--no_early_stopping', dest='early_stopping', action='store_false',
                        help='關閉早停功能')
    parser.add_argument('--patience', type=int, default=10,
                        help='早停耐心值（多少個epoch無改進後停止）')
    parser.add_argument('--min_delta', type=float, default=1e-4,
                        help='早停最小改進量')
    
    # 學習率調整參數
    parser.add_argument('--lr_scheduling', action='store_true', default=True,
                        help='啟用學習率調整')
    parser.add_argument('--no_lr_scheduling', dest='lr_scheduling', action='store_false',
                        help='關閉學習率調整')
    parser.add_argument('--lr_factor', type=float, default=0.5,
                        help='學習率衰減因子')
    parser.add_argument('--lr_patience', type=int, default=5,
                        help='學習率調整耐心值')
    parser.add_argument('--lr_min', type=float, default=1e-6,
                        help='最小學習率')
    
    args = parser.parse_args()
    
    # 檢查H5文件是否存在
    if not os.path.exists(args.h5_file):
        print(f"錯誤: H5文件 {args.h5_file} 不存在!")
        return
    
    print("=== 混合Patch訓練模式 (物理約束版) ===")
    print(f"🧪 物理約束: {args.physics_constraint} (權重: {args.constraint_weight})")
    
    # 建立模型組件
    extractor = create_downsampler(input_channels=1, start_neurons=32)
    up_cls = create_upsampler_cls(n_inputs=1, start_neurons=32, classes=6)  
    up_reg = create_upsampler_reg(n_inputs=1, start_neurons=32)
    
    # 混合patch訓練 (物理約束版)
    model, history, best_model_path = train_mixed_patches_with_physics(
        h5_patch_path=args.h5_file,
        extractor=extractor,
        upsampler_cls=up_cls,
        upsampler_reg=up_reg,
        experiment_name=args.experiment_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lambda_cls=args.lambda_cls,
        lambda_reg=args.lambda_reg,
        aliased_weight=args.aliased_weight,
        clean_weight=args.clean_weight,
        mask_strategy=args.mask_strategy,
        nan_penalty_weight=args.nan_penalty_weight,
        filter_patch_type=args.filter_patch_type,
        # 🆕 物理約束參數
        physics_constraint=args.physics_constraint,
        constraint_weight=args.constraint_weight,
        # 早停參數
        early_stopping=args.early_stopping,
        patience=args.patience,
        min_delta=args.min_delta,
        # 學習率調整參數  
        lr_scheduling=args.lr_scheduling,
        learning_rate=args.learning_rate,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        lr_min=args.lr_min
    )
    
    print("混合patch訓練完成 (物理約束版)!")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"總執行時間: {(end_time - start_time)/60:.2f} 分鐘")