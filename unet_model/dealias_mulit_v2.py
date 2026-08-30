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
        # 在初始化時創建通道適配器 - 動態匹配upsampler期望
        # 這些通道數需要匹配upsampler的期望輸入
        # 對於start_neurons=32: upsampler期望 [32, 64, 128, 256] (i0,i1,i2,i3)
        self.start_neurons = 32  # 固定使用32，與訓練腳本保持一致
        self.channel_adapters = [
            tf.keras.layers.Conv2D(self.start_neurons, 1, padding='same'),      # Feature 0 -> 32
            tf.keras.layers.Conv2D(self.start_neurons*2, 1, padding='same'),    # Feature 1 -> 64  
            tf.keras.layers.Conv2D(self.start_neurons*4, 1, padding='same'),    # Feature 2 -> 128
            tf.keras.layers.Conv2D(self.start_neurons*8, 1, padding='same')     # Feature 3 -> 256
        ]
    def call(self, inputs):
        nyq = inputs['nyq']     # (B, n_times, 1)
        vel_in = inputs['vel']  # (B, n_times, n_az, n_rad, 1)
        #print("vel_in shape:", vel_in.shape)
        # 診斷輸入數據
        #tf.print("=== 診斷開始 ===")
        #tf.print("輸入 vel_in 非NaN像素:", tf.reduce_sum(tf.cast(~tf.math.is_nan(vel_in), tf.int32)))
         # 在VelocityDealiaser.call方法開頭增加檢查
            

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
          #valid_mask = ~valid_mask
        #tf.print("有效遮罩形狀:", tf.shape(valid_mask))
        #tf.print("有效遮罩中True的數量:", tf.reduce_sum(tf.cast(valid_mask, tf.int32)))

        # ... 其他代碼 ...

        
        #print("valid_mask shape:", valid_mask.shape)

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
        # 調試：打印特徵形狀
        #for i, f in enumerate(f_list):
        #    print(f"Feature {i} shape: {f.shape}")

        # 創建通道調整層（如果尚未創建）
        #if not hasattr(self, 'channel_adapter'):
        # 創建一個1x1卷積層來調整通道數
        #   self.channel_adapter = tf.keras.layers.Conv2D(128, 1, padding='same')
        # 根據上採樣器期望的通道數創建調整層
        #    self.channel_adapters = [
        #        tf.keras.layers.Conv2D(64, 1, padding='same'),   # Feature 0
        #        tf.keras.layers.Conv2D(128, 1, padding='same'),  # Feature 1
        #        tf.keras.layers.Conv2D(256, 1, padding='same'),  # Feature 2
        #        tf.keras.layers.Conv2D(512, 1, padding='same')   # Feature 3
        #    ]
        # 調整每個特徵的通道數
        f_list_adjusted = []
        for i, f in enumerate(f_list):
            f_adjusted = self.channel_adapters[i](f)
            f_list_adjusted.append(f_adjusted)
            #print(f"Adjusted feature {i} shape: {f_adjusted.shape}")
        #調試：檢查調整後的特徵形狀
        #print(f"Adjusted feature shape: {f_list_adjusted[-1].shape}")

        # 分別丟給兩個 upsampler (分類 / 回歸)
        #classification_logits = self.upsampler_cls(f_list)   # => (B, n_az, n_rad, 6)
        #velocity_residual = self.upsampler_reg(f_list)       # => (B, n_az, n_rad, 1)
        classification_logits = self.upsampler_cls(f_list_adjusted)   # => (B, n_az, n_rad, 6)
        velocity_residual = self.upsampler_reg(f_list_adjusted)       # => (B, n_az, n_rad, 1)

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

        # 在計算 vel_pred 之前檢查中間變量
        #tf.print("raw_last 非NaN:", tf.reduce_sum(tf.cast(~tf.math.is_nan(raw_last), tf.int32)))
        #tf.print("corr 非零:", tf.reduce_sum(tf.cast(tf.not_equal(corr, 0), tf.int32)))
        #tf.print("velocity_residual 非NaN:", tf.reduce_sum(tf.cast(~tf.math.is_nan(velocity_residual), tf.int32)))

        # 最終去折錯速度 = 原始 + 分類大週期校正 + 回歸殘差
        vel_pred = raw_last + corr + velocity_residual
        #tf.print("計算後 vel_pred 非NaN:", tf.reduce_sum(tf.cast(~tf.math.is_nan(vel_pred), tf.int32)))

    
        #print("vel_pred shape:", vel_pred.shape)
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
        #tf.print("遮罩應用前 valid_mask_for_pred True的數量:", tf.reduce_sum(tf.cast(valid_mask_for_pred, tf.int32)))

        #print("valid_mask_for_pred shape:", valid_mask_for_pred.shape)
        # 創建與 vel_pred 相同類型的 NaN
        nan_tensor = tf.cast(tf.constant(float('nan')), vel_pred.dtype)
            # 應用遮罩
        #vel_pred = tf.where(
        #    valid_mask_for_pred,
        #    vel_pred, 
        #    tf.zeros_like(vel_pred) * tf.constant(float('nan'))
        #)

        # 不要將所有值都設為NaN，只在真正無效的區域設置NaN
        nan_tensor = tf.cast(tf.constant(float('nan')), vel_pred.dtype)
        vel_pred = tf.where(
            valid_mask_for_pred,
            vel_pred, 
            tf.zeros_like(vel_pred) * nan_tensor
        )
        #tf.print("最終 vel_pred 非NaN:", tf.reduce_sum(tf.cast(~tf.math.is_nan(vel_pred), tf.int32)))
        #tf.print("=== 診斷結束 ===")
        #vel_pred = tf.where(
        #    valid_mask_for_pred,
        #    vel_pred, 
        #    tf.zeros_like(vel_pred) * nan_tensor
        #)


        # 確保填充區域保持為NaN或某個特定值
        #vel_pred = tf.where(
        #tf.cast(tf.expand_dims(valid_mask, -1), tf.bool),  # 僅在此處轉換
        #vel_pred, 
        #tf.zeros_like(vel_pred) * tf.constant(float('nan'))
        #)        
        return {
        'alias_mask': classification_logits,     # 分類 logits
        'velocity_residual': velocity_residual,  # 回歸殘差
        'dealiased_vel': vel_pred,               # 最終校正速度
        'valid_mask': valid_mask                 # 添加這一行
    }
