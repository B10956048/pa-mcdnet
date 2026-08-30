import tensorflow as tf

def make_velocity_mask(vel, fill_val=0):
    
    bad_mask = tf.math.is_nan(vel) | tf.math.is_inf(vel)
    # 添加顯式類型轉換
    fill_val = tf.cast(fill_val, vel.dtype)
    vel = tf.where(bad_mask, fill_val, vel)
    return vel, bad_mask

class VelocityDealiaser(tf.keras.Model):
    """
    輸入:
      {
        'vel': [B, n_times, n_az, n_rad, 1],
        'nyq': [B, n_times, 1]
      }
    輸出:
      {
        'alias_mask':      [B, n_az, n_rad, 6],
        'velocity_residual':[B, n_az, n_rad, 1],
        'dealiased_vel':   [B, n_az, n_rad, 1]
      }
    """
    def __init__(self, extractor, upsampler_cls, upsampler_reg):
        super(VelocityDealiaser, self).__init__()
        self.extractor = extractor          # downsampler
        self.upsampler_cls = upsampler_cls  # 用於輸出 6 channels (分類)
        self.upsampler_reg = upsampler_reg  # 用於輸出 1 channel (回歸殘差)

    def call(self, inputs):
        nyq = inputs['nyq']     # (B, n_times, 1)
        vel_in = inputs['vel']  # (B, n_times, n_az, n_rad, 1)
        print("vel_in shape:", vel_in.shape)

        # 在VelocityDealiaser.call方法開頭增加檢查
        if 'valid_mask' in inputs:
          valid_mask = inputs['valid_mask']  # 使用提供的有效區域遮罩
          # 確保valid_mask的形狀與vel_in兼容
          # 由於valid_mask的形狀可能是(B, n_az, n_rad)，而vel_in是(B, n_times, n_az, n_rad, 1)
          # 我們需要添加缺失的維度
          valid_mask = tf.expand_dims(valid_mask, axis=1)  # 添加時間維度
          shp_inputs = tf.shape(vel_in)
          valid_mask = tf.broadcast_to(valid_mask, [shp_inputs[0], shp_inputs[1], shp_inputs[2], shp_inputs[3]])
        else:
          # 創建默認遮罩（僅排除NaN值）
          _, valid_mask = make_velocity_mask(vel_in)
          valid_mask = ~valid_mask
        print("valid_mask shape:", valid_mask.shape)

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
        f_list = self.extractor(x) # each shape: (B, naz/2^k, nrad/2^k, c)
        # 分別丟給兩個 upsampler (分類 / 回歸)
        classification_logits = self.upsampler_cls(f_list)   # => (B, n_az, n_rad, 6)
        velocity_residual = self.upsampler_reg(f_list)       # => (B, n_az, n_rad, 1)

        # clip logits 避免數值過大
        classification_logits = tf.clip_by_value(classification_logits, -10.0, 10.0)

        # 根據分類結果計算大週期校正
        raw_last = vel_in[:, -1]   # (B,n_az,n_rad,1)
        nyq_last = nyq_tiled[:, -1]   # (B,n_az,n_rad,1)

        cat = tf.argmax(classification_logits, axis=-1, output_type=tf.int32)  # => (B,n_az,n_rad)
        cat = tf.expand_dims(cat, axis=-1)  # => (B,n_az,n_rad,1)
        corr = tf.zeros_like(raw_last)

        # 注意: 要確保和 compute_alias_label 的定義一致
        # compute_alias_label_swapped: n = diff/(2*nyq), 每個 fold = 2×Nyquist
        # label=1 => n=-2 => correction = -4×Nyquist
        # label=2 => n=-1 => correction = -2×Nyquist
        # label=4 => n=+1 => correction = +2×Nyquist
        # label=5 => n=+2 => correction = +4×Nyquist
        corr = tf.where(tf.equal(cat, 1), -4.0*nyq_last, corr)  # -2 fold
        corr = tf.where(tf.equal(cat, 2), -2.0*nyq_last, corr)  # -1 fold
        corr = tf.where(tf.equal(cat, 4),  2.0*nyq_last, corr)  # +1 fold
        corr = tf.where(tf.equal(cat, 5),  4.0*nyq_last, corr)  # +2 fold
        # cat=0 或 3 => 0

        # 最終去折錯速度 = 原始 + 分類大週期校正 + 回歸殘差
        vel_pred = raw_last + corr + velocity_residual
        print("vel_pred shape:", vel_pred.shape)
        if len(valid_mask.shape) > 3:
          valid_mask_last = valid_mask[:, -1]  # 取最後時間步
        else:
          valid_mask_last = valid_mask
        # 關鍵修改：確保valid_mask的形狀與vel_pred兼容
        # vel_pred的形狀是(B,n_az,n_rad,1)
        # 將valid_mask處理成相同形狀
        # 確保形狀與vel_pred完全一致
        valid_mask_for_pred = tf.reshape(valid_mask_last, tf.shape(vel_pred))
        valid_mask_for_pred = tf.cast(valid_mask_for_pred, tf.bool)
        print("valid_mask_for_pred shape:", valid_mask_for_pred.shape)
            # 應用遮罩
        vel_pred = tf.where(
            valid_mask_for_pred,
            vel_pred, 
            tf.zeros_like(vel_pred) * tf.constant(float('nan'))
        )
        print("vel_pred shape:", vel_pred.shape)

        # 確保填充區域保持為NaN或某個特定值
        #vel_pred = tf.where(
        #tf.cast(tf.expand_dims(valid_mask, -1), tf.bool),  # 僅在此處轉換
        #vel_pred, 
        #tf.zeros_like(vel_pred) * tf.constant(float('nan'))
        #)        
        return {
            'alias_mask': classification_logits,     # 分類 logits
            'velocity_residual': velocity_residual,  # 回歸殘差
            'dealiased_vel': vel_pred,                # 最終校正速度
            'valid_mask': valid_mask  # 添加這一行
        }
