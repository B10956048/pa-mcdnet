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

class PhysicsAwareRegressionHead(tf.keras.layers.Layer):
    """
    物理約束的回歸頭，預測fold number並強制物理規律
    """
    def __init__(self, **kwargs):
        super(PhysicsAwareRegressionHead, self).__init__(**kwargs)
        # 預測5個fold number的機率：n ∈ {-2,-1,0,+1,+2}
        self.fold_predictor = tf.keras.layers.Dense(5, activation=None, name='fold_logits')

    def call(self, features):
        """
        Args:
            features: 特徵張量 (B, H, W, C)
        Returns:
            fold_probs: fold number的軟機率分布 (B, H, W, 5)
        """
        # 預測fold number的logits
        fold_logits = self.fold_predictor(features)  # (B, H, W, 5)

        # 應用softmax得到機率分布（軟約束，可微分）
        fold_probs = tf.nn.softmax(fold_logits, axis=-1)  # (B, H, W, 5)

        return fold_probs

def apply_physics_constraint(fold_probs, raw_vel, nyquist_vel):
    """
    應用物理約束：V = Vraw + n×2×Vnyq

    Args:
        fold_probs: fold number機率分布 (B, H, W, 5)
        raw_vel: 原始觀測速度 (B, H, W, 1)
        nyquist_vel: Nyquist速度 (B, H, W, 1)

    Returns:
        constrained_vel: 物理約束後的速度 (B, H, W, 1)
    """
    # fold values: [-2, -1, 0, +1, +2]
    fold_values = tf.constant([-2., -1., 0., +1., +2.], dtype=fold_probs.dtype)
    fold_values = tf.reshape(fold_values, [1, 1, 1, 5])  # 廣播形狀

    # 計算期望的fold number（軟加權平均）
    expected_fold = tf.reduce_sum(fold_probs * fold_values, axis=-1, keepdims=True)  # (B, H, W, 1)

    # 應用物理約束：V = Vraw + n×2×Vnyq（每個 fold = 2×Nyquist）
    constrained_vel = raw_vel + expected_fold * 2 * nyquist_vel

    return constrained_vel

class VelocityDealiaser(tf.keras.Model):
    """
    包含物理約束的速度反折錯模型
    輸入:
      {
        'vel': [B, n_times, n_az, n_rad, 1],
        'nyq': [B, n_times, 1]
      }
    輸出:
      {
        'alias_mask':           [B, n_az, n_rad, 6],
        'velocity_residual':    [B, n_az, n_rad, 1],  # 原始regression殘差
        'dealiased_vel':        [B, n_az, n_rad, 1],  # 分類結果
        'fold_probs':           [B, n_az, n_rad, 5],  # fold number機率分布
        'physics_dealiased_vel':[B, n_az, n_rad, 1]   # 物理約束regression結果
      }
    """
    def __init__(self, extractor, upsampler_cls, upsampler_reg):
        super(VelocityDealiaser, self).__init__()
        self.extractor = extractor          # downsampler
        self.upsampler_cls = upsampler_cls  # 用於輸出 6 channels (分類)
        self.upsampler_reg = upsampler_reg  # 用於輸出 1 channel (回歸殘差)

        # 物理約束regression head
        self.physics_regression_head = PhysicsAwareRegressionHead()

        # 在初始化時創建通道適配器
        self.start_neurons = 32
        self.channel_adapters = [
            tf.keras.layers.Conv2D(self.start_neurons * (2**i), 1, padding='same', name=f'adapter_{i}')
            for i in range(4)  # [32, 64, 128, 256]
        ]

    def call(self, inputs, training=None):
        """
        Args:
            inputs: {'vel': (B,T,H,W,1), 'nyq': (B,T,1)}
        """
        vel_in = inputs['vel']
        nyq = inputs['nyq']

        # 處理速度場mask
        vel_in, valid_mask = make_velocity_mask(vel_in, fill_val=-3.0)

        shp_inputs = tf.shape(vel_in)
        # 重複 nyq 使其跟 vel_in 同形狀
        nyq_tiled = tf.tile(nyq[:, :, None, None, None],
                            [1, 1, shp_inputs[2], shp_inputs[3], shp_inputs[4]])
        # 歸一化 + mask
        vel_norm = vel_in / nyq_tiled
        vel_norm, bad_mask = make_velocity_mask(vel_norm, fill_val=-3.0)

        # reshape => (B, n_az, n_rad, n_times)
        x = tf.transpose(vel_norm, (0, 2, 3, 1, 4))  # (B, n_az, n_rad, n_times,1)
        xshp = tf.shape(x)
        x = tf.reshape(x, (xshp[0], xshp[1], xshp[2], xshp[3]*xshp[4]))  # => (B, n_az, n_rad, n_times)

        # 用 downsampler 提取多層特徵 f_list=[f0,f1,f2,f3]
        f_list = self.extractor(x)

        # 調整每個特徵的通道數
        f_list_adjusted = []
        for i, f in enumerate(f_list):
            f_adjusted = self.channel_adapters[i](f)
            f_list_adjusted.append(f_adjusted)

        # === 原有的分類和回歸分支 ===
        classification_logits = self.upsampler_cls(f_list_adjusted)   # => (B, n_az, n_rad, 6)
        velocity_residual = self.upsampler_reg(f_list_adjusted)       # => (B, n_az, n_rad, 1)

        # clip logits 避免數值過大
        classification_logits = tf.clip_by_value(classification_logits, -10.0, 10.0)

        # 根據分類結果計算大週期校正（原有邏輯保持不變）
        raw_last = vel_in[:, -1]   # (B,n_az,n_rad,1)
        nyq_last = nyq_tiled[:, -1]   # (B,n_az,n_rad,1)

        # === 原始硬分類分支 ===
        cat = tf.argmax(classification_logits, axis=-1, output_type=tf.int32)  # => (B,n_az,n_rad)
        cat = tf.expand_dims(cat, axis=-1)  # => (B,n_az,n_rad,1)
        corr_hard = tf.zeros_like(raw_last)

        # 注意: 要確保和 compute_alias_label 的定義一致
        # compute_alias_label_swapped: n = diff/(2*nyq), 每個 fold = 2×Nyquist
        corr_hard = tf.where(tf.equal(cat, 1), -4.0*nyq_last, corr_hard)  # -2 fold
        corr_hard = tf.where(tf.equal(cat, 2), -2.0*nyq_last, corr_hard)  # -1 fold
        corr_hard = tf.where(tf.equal(cat, 4),  2.0*nyq_last, corr_hard)  # +1 fold
        corr_hard = tf.where(tf.equal(cat, 5),  4.0*nyq_last, corr_hard)  # +2 fold

        # === 新增：軟分類分支 (物理約束版本) ===
        # 將6類分類轉換為5類fold機率分布
        fold_probs_from_cls = tf.nn.softmax(classification_logits, axis=-1)[:, :, :, 1:6]  # (B,H,W,5)
        # 重新歸一化，因為移除了no-alias類別
        fold_probs_from_cls = fold_probs_from_cls / tf.reduce_sum(fold_probs_from_cls, axis=-1, keepdims=True)

        # fold values: [-2, -1, 0, +1, +2]
        fold_values = tf.constant([-2., -1., 0., +1., +2.], dtype=fold_probs_from_cls.dtype)
        fold_values = tf.reshape(fold_values, [1, 1, 1, 5])  # 廣播形狀

        # 計算期望的fold number（軟加權平均）
        expected_fold_cls = tf.reduce_sum(fold_probs_from_cls * fold_values, axis=-1, keepdims=True)  # (B,H,W,1)

        # 應用軟分類物理約束（每個 fold = 2×Nyquist）
        corr_soft = expected_fold_cls * 2 * nyq_last
        dealiased_vel_soft = raw_last + corr_soft

        # 暫時保持原有硬分類作為主要輸出
        dealiased_vel = raw_last + corr_hard

        # === 新增：物理約束的回歸分支 ===
        # 使用最後一層特徵預測fold number
        final_features = f_list_adjusted[-1]  # 使用最深層特徵

        # 上採樣到原始解析度（簡單的雙線性插值）
        target_shape = tf.shape(raw_last)
        final_features_upsampled = tf.image.resize(
            final_features,
            [target_shape[1], target_shape[2]],
            method='bilinear'
        )

        # 預測fold number機率分布
        fold_probs = self.physics_regression_head(final_features_upsampled)  # (B, H, W, 5)

        # 🔍 調試形狀
        #tf.print("🔧 Physics branch debug:")
        #tf.print("  final_features_upsampled shape:", tf.shape(final_features_upsampled))
        #tf.print("  fold_probs shape:", tf.shape(fold_probs))
        #tf.print("  raw_last shape:", tf.shape(raw_last))
        #tf.print("  nyq_last shape:", tf.shape(nyq_last))

        # 應用物理約束
        physics_dealiased_vel = apply_physics_constraint(fold_probs, raw_last, nyq_last)
        #tf.print("  physics_dealiased_vel shape:", tf.shape(physics_dealiased_vel))

        # 🔧 重要：對所有輸出應用相同的遮罩處理
        #tf.print("🔍 Valid mask debug:")
        #tf.print("  valid_mask shape:", tf.shape(valid_mask))
        #tf.print("  valid_mask.shape.ndims:", len(valid_mask.shape))

        valid_mask_last = valid_mask[:, -1] if len(valid_mask.shape) > 3 else valid_mask
        #tf.print("  valid_mask_last shape (after slicing):", tf.shape(valid_mask_last))

        # 🔧 修正：如果valid_mask_last已經是4維(B,H,W,1)，不需要expand_dims
        if len(valid_mask_last.shape) == 3:  # (B,H,W) -> (B,H,W,1)
            valid_mask_last = tf.expand_dims(valid_mask_last, axis=-1)
        #tf.print("  valid_mask_last shape (final):", tf.shape(valid_mask_last))

        valid_mask_bool = tf.cast(valid_mask_last, tf.bool)

        # 在無效區域設置NaN而非0
        nan_tensor = tf.cast(tf.constant(float('nan')), physics_dealiased_vel.dtype)

        # 對物理約束輸出應用遮罩
        physics_dealiased_vel = tf.where(
            valid_mask_bool,
            physics_dealiased_vel,
            tf.zeros_like(physics_dealiased_vel) * nan_tensor
        )

        # 對軟分類輸出應用遮罩
        dealiased_vel_soft = tf.where(
            valid_mask_bool,
            dealiased_vel_soft,
            tf.zeros_like(dealiased_vel_soft) * nan_tensor
        )

        # 🔧 新增：對velocity_residual應用遮罩，確保無觀測區域為NaN
        velocity_residual = tf.where(
            valid_mask_bool,
            velocity_residual,
            tf.zeros_like(velocity_residual) * nan_tensor
        )

        # 🔧 新增：對dealiased_vel也應用遮罩，保持一致性
        dealiased_vel = tf.where(
            valid_mask_bool,
            dealiased_vel,
            tf.zeros_like(dealiased_vel) * nan_tensor
        )

        # === 🆕 新增：置信度導向的軟硬混合分支 ===
        # 🔧 修正：只考慮需要校正的類別 (排除no-alias類別)
        classification_probs = tf.nn.softmax(classification_logits, axis=-1)  # (B,H,W,6)

        # 提取需要校正的類別機率 (classes 1-5: [-2,-1,0,+1,+2])
        alias_probs = classification_probs[:, :, :, 1:6]  # (B,H,W,5) 排除class 0 (no-alias)

        # 重新歸一化alias類別機率
        alias_probs_sum = tf.reduce_sum(alias_probs, axis=-1, keepdims=True)
        alias_probs_normalized = tf.where(
            tf.greater(alias_probs_sum, 1e-8),  # 避免除零
            alias_probs / alias_probs_sum,
            alias_probs  # 如果sum為0，保持原值
        )

        # 計算alias類別中的最大置信度
        max_alias_confidence = tf.reduce_max(alias_probs_normalized, axis=-1, keepdims=True)  # (B,H,W,1)

        # 置信度閾值（可調參數）
        confidence_threshold = 0.8

        # 計算置信度權重：高置信度 → 更多硬分類，低置信度 → 更多軟分類
        # 使用sigmoid平滑過渡，避免硬切換
        confidence_margin = (max_alias_confidence - confidence_threshold) * 10  # 放大差距
        hard_weight = tf.sigmoid(confidence_margin)  # 高置信度時接近1，低置信度時接近0
        soft_weight = 1.0 - hard_weight

        # 混合硬分類和軟分類結果
        confidence_guided_correction = hard_weight * corr_hard + soft_weight * corr_soft
        confidence_guided_dealiased = raw_last + confidence_guided_correction

        # 應用相同的遮罩處理
        confidence_guided_dealiased = tf.where(
            valid_mask_bool,
            confidence_guided_dealiased,
            tf.zeros_like(confidence_guided_dealiased) * nan_tensor
        )

        return {
            'alias_mask': classification_logits,
            'velocity_residual': velocity_residual,  # 🔧 已應用mask處理
            'dealiased_vel': dealiased_vel,  # 🔧 已應用mask處理（原硬分類）
            'fold_probs': fold_probs,  # 物理約束分支的fold機率
            'physics_dealiased_vel': physics_dealiased_vel,  # 物理約束回歸輸出
            'soft_classification_vel': dealiased_vel_soft,  # 軟分類輸出
            'fold_probs_from_cls': fold_probs_from_cls,  # 從分類轉換的fold機率
            'confidence_guided_vel': confidence_guided_dealiased,  # 🆕 置信度導向混合輸出
            'alias_confidence': max_alias_confidence,  # 🔧 修正：alias類別置信度（排除no-alias）
            'classification_confidence': tf.reduce_max(classification_probs, axis=-1, keepdims=True),  # 🆕 完整分類置信度（含no-alias，用於分析）
            'hard_soft_weights': {'hard_weight': hard_weight, 'soft_weight': soft_weight}  # 🆕 混合權重（用於分析）
        }