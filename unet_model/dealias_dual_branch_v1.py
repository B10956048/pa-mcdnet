#!/usr/bin/env python3
"""
雙分支速度去折錯模型 - 針對混合patch訓練優化

基於 dealias_mulit_v2.py 修改，新增雙分支架構：
- Aliased Branch: 專門學習修復折錯patches (RAW → GT)
- Clean Branch: 專門學習保持乾淨patches (RAW → RAW)
- Shared Encoder: 共享特徵提取器

使用方法:
    model = VelocityDealiaserDualBranch(extractor, upsampler_cls, upsampler_reg_aliased, upsampler_reg_clean)

輸入:
    {
        'vel': [B, n_times, n_az, n_rad, 1],
        'nyq': [B, n_times, 1],
        'patch_type': [B] - 'aliased' or 'clean'  # 新增
    }

輸出:
    {
        'alias_mask': [B, n_az, n_rad, 6],
        'dealiased_vel': [B, n_az, n_rad, 1]
    }
"""

import tensorflow as tf

def make_velocity_mask(vel, fill_val=None):
    """
    创建速度掩码，确保类型兼容性
    """
    # 获取输入张量的dtype
    input_dtype = vel.dtype

    # 如果未指定fill_val，使用与输入相同类型的NaN
    if fill_val is None:
        fill_val = tf.constant(float('nan'), dtype=input_dtype)
    else:
        # 确保fill_val与输入类型匹配
        fill_val = tf.cast(fill_val, input_dtype)

    # 创建掩码
    bad_mask = tf.math.is_nan(vel)
    # 使用类型匹配的fill_val
    vel = tf.where(bad_mask, fill_val, vel)

    # 创建有效掩码
    valid_mask = ~bad_mask

    return vel, valid_mask

class VelocityDealiaserDualBranch(tf.keras.Model):
    """
    雙分支速度去折錯模型

    架構:
    - Shared Encoder: 共享特徵提取器
    - Classification Head: 分類頭部（共享）
    - Aliased Branch: 專門處理需要修復的patches
    - Clean Branch: 專門處理需要保持的patches
    """

    def __init__(self, extractor, upsampler_cls, upsampler_reg_aliased, upsampler_reg_clean=None):
        super(VelocityDealiaserDualBranch, self).__init__()

        # 共享組件
        self.extractor = extractor          # 共享特徵提取器
        self.upsampler_cls = upsampler_cls  # 共享分類頭部

        # 分支組件
        self.upsampler_reg_aliased = upsampler_reg_aliased  # 修復分支
        self.upsampler_reg_clean = upsampler_reg_clean or upsampler_reg_aliased  # 保持分支（可選）

        # 通道適配器
        self.start_neurons = 32
        self.channel_adapters = [
            tf.keras.layers.Conv2D(self.start_neurons, 1, padding='same', name='adapter_0'),
            tf.keras.layers.Conv2D(self.start_neurons*2, 1, padding='same', name='adapter_1'),
            tf.keras.layers.Conv2D(self.start_neurons*4, 1, padding='same', name='adapter_2'),
            tf.keras.layers.Conv2D(self.start_neurons*8, 1, padding='same', name='adapter_3')
        ]

        # 分支選擇層 (可選)
        self.branch_attention = tf.keras.layers.Dense(128, activation='relu', name='branch_attention')

    def call(self, inputs, training=None):
        """
        前向傳播

        Parameters:
        -----------
        inputs : dict
            包含 'vel', 'nyq', 'patch_type' 的字典
        """
        vel_in = inputs['vel']    # (B, n_times, n_az, n_rad, 1)
        nyq = inputs['nyq']       # (B, n_times, 1)
        patch_types = inputs.get('patch_type', None)  # (B,) - 'aliased' or 'clean'

        # 如果沒有patch_type，預設為aliased（向後兼容）
        if patch_types is None:
            batch_size = tf.shape(vel_in)[0]
            patch_types = tf.fill([batch_size], 'aliased')

        # ===== 特徵提取部分 (共享) =====
        # 處理輸入和遮罩
        if 'valid_mask' in inputs:
            valid_mask = inputs['valid_mask']
            valid_mask = tf.expand_dims(valid_mask, axis=1)
            shp_inputs = tf.shape(vel_in)
            valid_mask = tf.broadcast_to(valid_mask, [shp_inputs[0], shp_inputs[1], shp_inputs[2], shp_inputs[3]])
        else:
            _, valid_mask = make_velocity_mask(vel_in)

        # Nyquist歸一化
        shp_inputs = tf.shape(vel_in)
        nyq_tiled = tf.tile(nyq[:, :, None, None, None],
                           [1, 1, shp_inputs[2], shp_inputs[3], shp_inputs[4]])
        vel_norm = vel_in / nyq_tiled
        vel_norm, bad_mask = make_velocity_mask(vel_norm, fill_val=-3.0)

        # Reshape for encoder
        x = tf.transpose(vel_norm, (0, 2, 3, 1, 4))  # (B, n_az, n_rad, n_times, 1)
        xshp = tf.shape(x)
        x = tf.reshape(x, (xshp[0], xshp[1], xshp[2], xshp[3]*xshp[4]))  # (B, n_az, n_rad, n_times)

        # 共享特徵提取
        f_list = self.extractor(x)  # [f0, f1, f2, f3]

        # 通道調整
        f_list_adjusted = []
        for i, f in enumerate(f_list):
            f_adjusted = self.channel_adapters[i](f)
            f_list_adjusted.append(f_adjusted)

        # ===== 分類頭部 (共享) =====
        classification_logits = self.upsampler_cls(f_list_adjusted)  # (B, n_az, n_rad, 6)
        classification_logits = tf.clip_by_value(classification_logits, -10.0, 10.0)

        # ===== 雙分支回歸部分 =====
        # Aliased分支：專門學習修復
        aliased_residual = self.upsampler_reg_aliased(f_list_adjusted)  # (B, n_az, n_rad, 1)

        # Clean分支：專門學習保持
        if self.upsampler_reg_clean is not None:
            clean_residual = self.upsampler_reg_clean(f_list_adjusted)  # (B, n_az, n_rad, 1)
        else:
            # 共享模式：clean分支使用相同upsampler但期望產生較小的residual
            clean_residual = self.upsampler_reg_aliased(f_list_adjusted) * 0.1  # 縮放因子

        # ===== 根據patch_type動態選擇分支 =====
        # 創建patch type mask
        aliased_mask = tf.equal(patch_types, 'aliased')  # (B,)

        # 擴展mask維度以匹配residual形狀
        aliased_mask_expanded = aliased_mask[:, None, None, None]  # (B, 1, 1, 1)

        # 動態選擇residual
        velocity_residual = tf.where(
            aliased_mask_expanded,
            aliased_residual,    # 使用修復分支
            clean_residual       # 使用保持分支
        )

        # ===== 最終去折錯計算 =====
        raw_last = vel_in[:, -1]      # (B, n_az, n_rad, 1)
        nyq_last = nyq_tiled[:, -1]   # (B, n_az, n_rad, 1)

        # 分類結果轉換為週期校正
        cat = tf.argmax(classification_logits, axis=-1, output_type=tf.int32)  # (B, n_az, n_rad)
        cat = tf.expand_dims(cat, axis=-1)  # (B, n_az, n_rad, 1)
        corr = tf.zeros_like(raw_last)

        # 週期校正計算（每個 fold = 2×Nyquist）
        corr = tf.where(tf.equal(cat, 1), -4.0*nyq_last, corr)  # -2 fold
        corr = tf.where(tf.equal(cat, 2), -2.0*nyq_last, corr)  # -1 fold
        corr = tf.where(tf.equal(cat, 4),  2.0*nyq_last, corr)  # +1 fold
        corr = tf.where(tf.equal(cat, 5),  4.0*nyq_last, corr)  # +2 fold

        # 最終預測
        vel_pred = raw_last + corr + velocity_residual

        # ===== 應用遮罩 =====
        if len(valid_mask.shape) > 3:
            valid_mask_last = valid_mask[:, -1]
        else:
            valid_mask_last = valid_mask

        valid_mask_for_pred = tf.reshape(valid_mask_last, tf.shape(vel_pred))
        valid_mask_for_pred = tf.cast(valid_mask_for_pred, tf.bool)

        # 創建NaN mask
        nan_tensor = tf.cast(tf.constant(float('nan')), vel_pred.dtype)
        vel_pred = tf.where(
            valid_mask_for_pred,
            vel_pred,
            tf.zeros_like(vel_pred) * nan_tensor
        )

        return {
            'alias_mask': classification_logits,
            'dealiased_vel': vel_pred,
            # 調試信息
            'aliased_residual': aliased_residual,
            'clean_residual': clean_residual,
            'velocity_residual': velocity_residual,
            'branch_selection': aliased_mask
        }

class VelocityDealiaserDualBranchV2(tf.keras.Model):
    """
    雙分支模型的改進版本 - 使用注意力機制

    特色：
    1. 基於patch_type的注意力權重
    2. 更靈活的分支融合
    3. 支持漸進式訓練
    """

    def __init__(self, extractor, upsampler_cls, upsampler_reg, use_attention=True):
        super(VelocityDealiaserDualBranchV2, self).__init__()

        self.extractor = extractor
        self.upsampler_cls = upsampler_cls
        self.upsampler_reg = upsampler_reg
        self.use_attention = use_attention

        # 通道適配器
        self.start_neurons = 32
        self.channel_adapters = [
            tf.keras.layers.Conv2D(self.start_neurons, 1, padding='same', name='adapter_0'),
            tf.keras.layers.Conv2D(self.start_neurons*2, 1, padding='same', name='adapter_1'),
            tf.keras.layers.Conv2D(self.start_neurons*4, 1, padding='same', name='adapter_2'),
            tf.keras.layers.Conv2D(self.start_neurons*8, 1, padding='same', name='adapter_3')
        ]

        # 注意力機制（可選）
        if self.use_attention:
            # Patch-type嵌入
            self.patch_type_embedding = tf.keras.layers.Embedding(2, 64, name='patch_type_embed')  # aliased=0, clean=1

            # 特徵注意力層
            self.feature_attention = tf.keras.layers.MultiHeadAttention(
                num_heads=4, key_dim=64, name='feature_attention'
            )

            # 分支權重預測器
            self.branch_weight_predictor = tf.keras.Sequential([
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(2, activation='softmax', name='branch_weights')
            ])

    def call(self, inputs, training=None):
        """前向傳播 - 注意力版本"""
        vel_in = inputs['vel']
        nyq = inputs['nyq']
        patch_types = inputs.get('patch_type', None)

        if patch_types is None:
            batch_size = tf.shape(vel_in)[0]
            patch_types = tf.fill([batch_size], 'aliased')

        # ===== 共享特徵提取 =====
        # (與原版相同的預處理)
        if 'valid_mask' in inputs:
            valid_mask = inputs['valid_mask']
            valid_mask = tf.expand_dims(valid_mask, axis=1)
            shp_inputs = tf.shape(vel_in)
            valid_mask = tf.broadcast_to(valid_mask, [shp_inputs[0], shp_inputs[1], shp_inputs[2], shp_inputs[3]])
        else:
            _, valid_mask = make_velocity_mask(vel_in)

        shp_inputs = tf.shape(vel_in)
        nyq_tiled = tf.tile(nyq[:, :, None, None, None],
                           [1, 1, shp_inputs[2], shp_inputs[3], shp_inputs[4]])
        vel_norm = vel_in / nyq_tiled
        vel_norm, bad_mask = make_velocity_mask(vel_norm, fill_val=-3.0)

        x = tf.transpose(vel_norm, (0, 2, 3, 1, 4))
        xshp = tf.shape(x)
        x = tf.reshape(x, (xshp[0], xshp[1], xshp[2], xshp[3]*xshp[4]))

        f_list = self.extractor(x)
        f_list_adjusted = []
        for i, f in enumerate(f_list):
            f_adjusted = self.channel_adapters[i](f)
            f_list_adjusted.append(f_adjusted)

        # ===== 注意力機制 =====
        if self.use_attention:
            # 將patch_type轉換為數值
            patch_type_ids = tf.where(
                tf.equal(patch_types, 'aliased'),
                tf.zeros_like(patch_types, dtype=tf.int32),
                tf.ones_like(patch_types, dtype=tf.int32)
            )

            # Patch type嵌入
            patch_embed = self.patch_type_embedding(patch_type_ids)  # (B, 64)

            # 對最深層特徵應用注意力
            deepest_feature = f_list_adjusted[-1]  # (B, H, W, C)

            # 將patch embedding擴展到空間維度
            spatial_shape = tf.shape(deepest_feature)
            patch_embed_spatial = tf.tile(
                patch_embed[:, None, None, :],
                [1, spatial_shape[1], spatial_shape[2], 1]
            )  # (B, H, W, 64)

            # Reshape為序列進行attention
            b, h, w, c = spatial_shape[0], spatial_shape[1], spatial_shape[2], spatial_shape[3]
            feature_flat = tf.reshape(deepest_feature, [b, h*w, c])
            embed_flat = tf.reshape(patch_embed_spatial, [b, h*w, 64])

            # 自注意力
            attended_features, _ = self.feature_attention(
                query=feature_flat,
                key=embed_flat,
                value=feature_flat,
                training=training
            )

            # Reshape回空間維度
            attended_features = tf.reshape(attended_features, [b, h, w, c])
            f_list_adjusted[-1] = attended_features

        # ===== 分類和回歸 =====
        classification_logits = self.upsampler_cls(f_list_adjusted)
        classification_logits = tf.clip_by_value(classification_logits, -10.0, 10.0)

        velocity_residual = self.upsampler_reg(f_list_adjusted)

        # ===== 最終計算 =====
        raw_last = vel_in[:, -1]
        nyq_last = nyq_tiled[:, -1]

        cat = tf.argmax(classification_logits, axis=-1, output_type=tf.int32)
        cat = tf.expand_dims(cat, axis=-1)
        corr = tf.zeros_like(raw_last)

        corr = tf.where(tf.equal(cat, 1), -4.0*nyq_last, corr)  # -2 fold
        corr = tf.where(tf.equal(cat, 2), -2.0*nyq_last, corr)  # -1 fold
        corr = tf.where(tf.equal(cat, 4),  2.0*nyq_last, corr)  # +1 fold
        corr = tf.where(tf.equal(cat, 5),  4.0*nyq_last, corr)  # +2 fold

        vel_pred = raw_last + corr + velocity_residual

        # 應用遮罩
        if len(valid_mask.shape) > 3:
            valid_mask_last = valid_mask[:, -1]
        else:
            valid_mask_last = valid_mask

        valid_mask_for_pred = tf.reshape(valid_mask_last, tf.shape(vel_pred))
        valid_mask_for_pred = tf.cast(valid_mask_for_pred, tf.bool)

        nan_tensor = tf.cast(tf.constant(float('nan')), vel_pred.dtype)
        vel_pred = tf.where(
            valid_mask_for_pred,
            vel_pred,
            tf.zeros_like(vel_pred) * nan_tensor
        )

        return {
            'alias_mask': classification_logits,
            'dealiased_vel': vel_pred
        }

# 工廠函數
def create_dual_branch_model(extractor, upsampler_cls, upsampler_reg_aliased, upsampler_reg_clean=None, version='v1'):
    """
    創建雙分支模型的工廠函數

    Parameters:
    -----------
    version : str
        'v1' - 基本雙分支架構
        'v2' - 注意力增強版本
    """
    if version == 'v1':
        return VelocityDealiaserDualBranch(
            extractor, upsampler_cls, upsampler_reg_aliased, upsampler_reg_clean
        )
    elif version == 'v2':
        return VelocityDealiaserDualBranchV2(
            extractor, upsampler_cls, upsampler_reg_aliased, use_attention=True
        )
    else:
        raise ValueError(f"不支持的版本: {version}")