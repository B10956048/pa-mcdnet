#!/usr/bin/env python3
"""
改進的物理約束實現
專門針對相鄰像素大跳變問題設計的約束函數
"""

import tensorflow as tf

def smart_fill_invalid_pixels(y_pred_2d, mask_2d):
    """
    智能填充無效像素，減少人工邊界效應
    用局部有效像素的平均值填充，而不是0
    """
    # 先清理原始NaN
    y_clean = tf.where(tf.math.is_finite(y_pred_2d), y_pred_2d, tf.zeros_like(y_pred_2d))

    # 使用更簡單的方法：就近填充
    # 避免複雜的卷積操作，直接用均值填充

    # 計算有效像素的全局均值作為填充值
    valid_pixels = tf.boolean_mask(y_clean, mask_2d)

    global_mean = tf.cond(
        tf.size(valid_pixels) > 0,
        lambda: tf.reduce_mean(valid_pixels),
        lambda: tf.constant(0.0, dtype=y_clean.dtype)
    )

    # 簡單填充：有效區域保持原值，無效區域用全局均值
    local_mean_2d = tf.where(
        mask_2d,
        y_clean,      # 有效像素保持原值
        global_mean   # 無效像素用全局均值填充
    )

    return local_mean_2d

def direct_gradient_constraint_with_mask(y_pred, mask_valid, threshold=None):
    """
    直接梯度約束 - 專門約束相鄰像素的大跳變

    Parameters:
    -----------
    y_pred : tf.Tensor (B,H,W,1)
        預測速度場
    mask_valid : tf.Tensor (B,H,W,1)
        有效區域mask
    threshold : float, optional
        跳變閾值，超過此值的梯度會被懲罰。如果為None，則自動計算
    """
    y_pred_2d = tf.squeeze(y_pred, axis=-1)  # (B,H,W)
    mask_2d = tf.squeeze(mask_valid, axis=-1)  # (B,H,W)

    # 🔧 智能填充策略：避免NaN污染，同時減少人工邊界

    # 方法1: 先確定完全有效的像素對
    mask_x = mask_2d[:, :, 1:] & mask_2d[:, :, :-1]  # 兩個相鄰像素都有效
    mask_y = mask_2d[:, 1:, :] & mask_2d[:, :-1, :]

    # 方法2: 智能填充無效像素
    # 對於無效像素，用相鄰有效像素的平均值填充，而不是0
    y_pred_2d_filled = smart_fill_invalid_pixels(y_pred_2d, mask_2d)

    # 計算梯度（現在沒有NaN風險）
    grad_x = tf.abs(y_pred_2d_filled[:, :, 1:] - y_pred_2d_filled[:, :, :-1])
    grad_y = tf.abs(y_pred_2d_filled[:, 1:, :] - y_pred_2d_filled[:, :-1, :])

    # 自動計算閾值（如果未提供）
    if threshold is None:
        # 收集所有有效梯度
        valid_grad_x = tf.boolean_mask(grad_x, mask_x)
        valid_grad_y = tf.boolean_mask(grad_y, mask_y)
        all_gradients = tf.concat([valid_grad_x, valid_grad_y], 0)

        if tf.size(all_gradients) > 10:
            # 使用90%分位數作為自動閾值
            total_size = tf.size(all_gradients)
            k = tf.cast(tf.cast(total_size, tf.float32) * 0.1, tf.int32)  # 前10%
            k = tf.maximum(k, 1)
            k = tf.minimum(k, total_size)
            top_k_values, _ = tf.nn.top_k(all_gradients, k=k)
            threshold = tf.reduce_min(top_k_values)
        else:
            threshold = 20.0  # 默認值

    # 只對超過閾值的梯度進行懲罰
    penalty_x = tf.maximum(0.0, grad_x - threshold)
    penalty_y = tf.maximum(0.0, grad_y - threshold)

    # 只在有效區域計算懲罰
    valid_penalty_x = tf.where(mask_x, penalty_x, tf.zeros_like(penalty_x))
    valid_penalty_y = tf.where(mask_y, penalty_y, tf.zeros_like(penalty_y))

    # 計算平均懲罰
    total_valid_x = tf.reduce_sum(tf.cast(mask_x, tf.float32))
    total_valid_y = tf.reduce_sum(tf.cast(mask_y, tf.float32))

    if tf.less(total_valid_x + total_valid_y, 1.0):
        return tf.constant(0.0, dtype=tf.float32)

    penalty_sum = tf.reduce_sum(valid_penalty_x) + tf.reduce_sum(valid_penalty_y)
    valid_count = total_valid_x + total_valid_y

    result = penalty_sum / (valid_count + 1e-8)

    return tf.where(tf.math.is_finite(result), result, tf.constant(0.0, dtype=tf.float32))

def hierarchical_smoothness_constraint_with_mask(y_pred, mask_valid):
    """
    階層平滑約束 - 結合多個尺度的約束
    1階: 相鄰像素 (最重要) - 針對尖銳跳變
    2階: 3x3局部區域 - 針對局部變異
    3階: 5x5區域 - 針對大尺度不連續
    """
    # 1階: 直接梯度約束 (權重最高)
    direct_penalty = direct_gradient_constraint_with_mask(y_pred, mask_valid, threshold=15.0)

    # 2階: 局部方差約束 (中等權重)
    local_variance = local_variance_constraint_simple(y_pred, mask_valid, window_size=3)

    # 3階: 區域平滑約束 (權重最低)
    regional_smoothness = local_variance_constraint_simple(y_pred, mask_valid, window_size=5)

    # 階層權重組合
    result = 0.7 * direct_penalty + 0.2 * local_variance + 0.1 * regional_smoothness

    return result

def local_variance_constraint_simple(y_pred, mask_valid, window_size=3):
    """
    簡化版Local Variance約束 - 移除自適應權重，保持簡潔
    """
    y_pred_2d = tf.squeeze(y_pred, axis=-1)  # (B,H,W)
    mask_2d = tf.squeeze(mask_valid, axis=-1)  # (B,H,W)

    # 檢查有效像素數量
    valid_count = tf.reduce_sum(tf.cast(mask_2d, tf.float32))
    batch_size = tf.shape(mask_2d)[0]
    min_required = tf.cast(window_size * window_size * batch_size, tf.float32)

    if tf.less(valid_count, min_required):
        return tf.constant(0.0, dtype=tf.float32)

    # 清理NaN
    y_pred_2d_clean = tf.where(tf.math.is_finite(y_pred_2d), y_pred_2d, tf.zeros_like(y_pred_2d))

    # 使用extract_patches計算局部方差
    y_pred_4d = tf.expand_dims(y_pred_2d_clean, axis=-1)
    mask_4d = tf.expand_dims(tf.cast(mask_2d, tf.float32), axis=-1)

    # 提取patches
    patches = tf.image.extract_patches(
        y_pred_4d,
        sizes=[1, window_size, window_size, 1],
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding='SAME'
    )  # (B, H, W, window_size^2)

    mask_patches = tf.image.extract_patches(
        mask_4d,
        sizes=[1, window_size, window_size, 1],
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding='SAME'
    )  # (B, H, W, window_size^2)

    # 只對有效窗口計算方差
    valid_patches = patches * mask_patches
    valid_counts = tf.reduce_sum(mask_patches, axis=-1)  # (B, H, W)

    # 計算每個窗口的方差
    patch_mean = tf.reduce_sum(valid_patches, axis=-1) / (valid_counts + 1e-8)  # (B, H, W)
    patch_var = tf.reduce_sum(mask_patches * tf.square(patches - patch_mean[..., None]), axis=-1) / (valid_counts + 1e-8)

    # 只統計有足夠有效像素的窗口
    valid_window_mask = tf.greater(valid_counts, 2.0)
    valid_variances = tf.boolean_mask(patch_var, valid_window_mask)

    if tf.size(valid_variances) == 0:
        return tf.constant(0.0, dtype=tf.float32)

    result = tf.reduce_mean(valid_variances)
    return tf.where(tf.math.is_finite(result), result, tf.constant(0.0, dtype=tf.float32))

def adaptive_threshold_constraint_with_mask(y_pred, mask_valid):
    """
    自適應閾值約束 - 根據局部梯度分布動態調整閾值
    """
    y_pred_2d = tf.squeeze(y_pred, axis=-1)
    mask_2d = tf.squeeze(mask_valid, axis=-1)

    # 清理NaN
    y_pred_2d_clean = tf.where(tf.math.is_finite(y_pred_2d), y_pred_2d, tf.zeros_like(y_pred_2d))

    # 計算所有有效梯度
    grad_x = tf.abs(y_pred_2d_clean[:, :, 1:] - y_pred_2d_clean[:, :, :-1])
    grad_y = tf.abs(y_pred_2d_clean[:, 1:, :] - y_pred_2d_clean[:, :-1, :])

    mask_x = mask_2d[:, :, 1:] & mask_2d[:, :, :-1]
    mask_y = mask_2d[:, 1:, :] & mask_2d[:, :-1, :]

    # 收集所有有效梯度
    valid_grad_x = tf.boolean_mask(grad_x, mask_x)
    valid_grad_y = tf.boolean_mask(grad_y, mask_y)
    all_gradients = tf.concat([valid_grad_x, valid_grad_y], 0)

    if tf.size(all_gradients) < 10:
        return tf.constant(0.0, dtype=tf.float32)

    # 計算動態閾值 (95%分位數)
    total_size = tf.size(all_gradients)
    k = tf.cast(tf.cast(total_size, tf.float32) * 0.05, tf.int32)
    k = tf.maximum(k, 1)
    k = tf.minimum(k, total_size)

    top_k_values, _ = tf.nn.top_k(all_gradients, k=k)
    dynamic_threshold = tf.reduce_min(top_k_values)

    # 懲罰超過動態閾值的梯度
    penalty_x = tf.maximum(0.0, grad_x - dynamic_threshold)
    penalty_y = tf.maximum(0.0, grad_y - dynamic_threshold)

    valid_penalty_x = tf.where(mask_x, penalty_x, tf.zeros_like(penalty_x))
    valid_penalty_y = tf.where(mask_y, penalty_y, tf.zeros_like(penalty_y))

    total_penalty = tf.reduce_sum(valid_penalty_x) + tf.reduce_sum(valid_penalty_y)
    total_valid = tf.reduce_sum(tf.cast(mask_x, tf.float32)) + tf.reduce_sum(tf.cast(mask_y, tf.float32))

    result = total_penalty / (total_valid + 1e-8)
    return tf.where(tf.math.is_finite(result), result, tf.constant(0.0, dtype=tf.float32))

# 約束函數映射表
IMPROVED_CONSTRAINTS = {
    'direct_gradient': direct_gradient_constraint_with_mask,
    'hierarchical': hierarchical_smoothness_constraint_with_mask,
    'adaptive_threshold': adaptive_threshold_constraint_with_mask,
    'simple_local_variance': local_variance_constraint_simple
}