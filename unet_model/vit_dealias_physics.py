"""
vit_dealias_physics.py — Vision Transformer 版速度反折錯模型

與 dealias_mulit_v2_physics.py 的 VelocityDealiaser 介面完全相容：
  - 輸入: {'vel': (B, T, H, W, 1), 'nyq': (B, T, 1)}
  - 輸出: 與 VelocityDealiaser 相同的 11-key dict

設計要點：
  - PatchEmbed: Conv2D(stride=patch_size) + LayerNorm
  - LearnedPosEmbed: 可學習 2D 位置編碼，bilinear 插值支援不同解析度
  - Pre-LN TransformerBlock: MHSA + FFN (GELU)
  - DensePredDecoder: progressive ×2 UpSampling + Conv2D（分類/回歸各一套）
  - 物理校正邏輯（硬/軟分類、fold 機率）與原 CNN 版完全相同

預設組態 (ViT-S for 128×128 patches):
  patch_size=8 → 16×16=256 tokens
  embed_dim=192, depth=6, num_heads=6, mlp_ratio=4.0
  參數量 ~18M (CNN 版 ~7M，但有更大的全域感受野)

使用方式（替換 transfer_learning_complete.py 中的模型建立段落）:
    from unet_model.vit_dealias_physics import create_vit_dealiaser
    model = create_vit_dealiaser()          # 取代 extractor+upsampler+VelocityDealiaser
    model = create_vit_dealiaser('tiny')    # 更小的 ViT-Tiny 設定
"""

import math
import tensorflow as tf


# ══════════════════════════════════════════════════════════════════════════════
# 共用工具（與 dealias_mulit_v2_physics.py 相同）
# ══════════════════════════════════════════════════════════════════════════════

def make_velocity_mask(vel, fill_val=None):
    """建立速度遮罩：NaN → fill_val，並回傳 valid_mask。"""
    input_dtype = vel.dtype
    if fill_val is None:
        fill_val = tf.constant(float('nan'), dtype=input_dtype)
    else:
        fill_val = tf.cast(fill_val, input_dtype)
    bad_mask = tf.math.is_nan(vel)
    vel = tf.where(bad_mask, fill_val, vel)
    valid_mask = ~bad_mask
    return vel, valid_mask


def apply_physics_constraint(fold_probs, raw_vel, nyquist_vel):
    """物理校正: V = Vraw + E[n] × 2 × Vnyq，fold n ∈ {-2,-1,0,+1,+2}。"""
    fold_values = tf.constant([-2., -1., 0., +1., +2.], dtype=fold_probs.dtype)
    fold_values = tf.reshape(fold_values, [1, 1, 1, 5])
    expected_fold = tf.reduce_sum(fold_probs * fold_values, axis=-1, keepdims=True)
    return raw_vel + expected_fold * 2.0 * nyquist_vel


# ══════════════════════════════════════════════════════════════════════════════
# Patch Embedding
# ══════════════════════════════════════════════════════════════════════════════

class PatchEmbed(tf.keras.layers.Layer):
    """
    (B, H, W, C) → (B, N, embed_dim)，N = (H/ps)×(W/ps)

    以 Conv2D(kernel=patch_size, stride=patch_size) 實作線性投影，
    等價於將每個 patch 展平後做 Linear，但 Conv2D 實作更為高效。
    """

    def __init__(self, patch_size=8, embed_dim=192, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = tf.keras.layers.Conv2D(
            embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding='valid',
            use_bias=True,
            name='proj',
        )
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='norm')

    def call(self, x):
        """回傳 (tokens, H_p, W_p)。"""
        x = self.proj(x)                          # (B, H_p, W_p, D)
        B  = tf.shape(x)[0]
        H_p = tf.shape(x)[1]
        W_p = tf.shape(x)[2]
        x = tf.reshape(x, (B, H_p * W_p, self.embed_dim))  # (B, N, D)
        x = self.norm(x)
        return x, H_p, W_p


# ══════════════════════════════════════════════════════════════════════════════
# Positional Encoding
# ══════════════════════════════════════════════════════════════════════════════

class LearnedPosEmbed(tf.keras.layers.Layer):
    """
    可學習的 2D 位置編碼。

    訓練時 grid 固定（128/8 = 16×16），推理時若 grid 改變則做 bilinear 插值。
    初始化為 truncated normal（標準 ViT 做法）。
    """

    def __init__(self, num_patches, embed_dim=192, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches   # 訓練時期望的 N = H_p * W_p
        self.embed_dim = embed_dim
        # grid_size: 假設為正方形 grid，用於插值
        self.grid_size = int(math.isqrt(num_patches))

    def build(self, input_shape):
        self.pos_embed = self.add_weight(
            name='pos_embed',
            shape=(1, self.num_patches, self.embed_dim),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, tokens, H_p, W_p):
        """
        若 N == num_patches，直接加。
        否則將 pos_embed reshape → bilinear resize → 加回。

        注意：使用 tf.cond 而非 Python if，避免在 @tf.function 圖模式下
        以 Symbolic Tensor 作為 Python bool 而報錯。
        """
        N = H_p * W_p

        def add_directly():
            return tokens + self.pos_embed

        def interpolate():
            pos = tf.reshape(
                self.pos_embed,
                (1, self.grid_size, self.grid_size, self.embed_dim),
            )
            pos = tf.image.resize(pos, [H_p, W_p], method='bilinear')
            pos = tf.reshape(pos, (1, N, self.embed_dim))
            return tokens + pos

        return tf.cond(tf.equal(N, self.num_patches), add_directly, interpolate)


# ══════════════════════════════════════════════════════════════════════════════
# Transformer Block 元件
# ══════════════════════════════════════════════════════════════════════════════

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    """Scaled dot-product MHSA。QKV 以單一 Dense 合併計算。"""

    def __init__(self, embed_dim, num_heads, attn_drop=0.0, proj_drop=0.0, **kwargs):
        super().__init__(**kwargs)
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) 必須整除 num_heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv  = tf.keras.layers.Dense(embed_dim * 3, use_bias=True, name='qkv')
        self.proj = tf.keras.layers.Dense(embed_dim,      use_bias=True, name='proj')
        self.attn_drop = tf.keras.layers.Dropout(attn_drop)
        self.proj_drop = tf.keras.layers.Dropout(proj_drop)

    def call(self, x, training=None):
        B   = tf.shape(x)[0]
        N   = tf.shape(x)[1]
        C   = self.num_heads * self.head_dim

        qkv = self.qkv(x)                                         # (B, N, 3C)
        qkv = tf.reshape(qkv, (B, N, 3, self.num_heads, self.head_dim))
        qkv = tf.transpose(qkv, (2, 0, 3, 1, 4))                  # (3, B, h, N, d)
        q, k, v = qkv[0], qkv[1], qkv[2]                          # (B, h, N, d)

        attn = tf.matmul(q, k, transpose_b=True) * self.scale      # (B, h, N, N)
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)

        out = tf.matmul(attn, v)                                    # (B, h, N, d)
        out = tf.transpose(out, (0, 2, 1, 3))                       # (B, N, h, d)
        out = tf.reshape(out, (B, N, C))
        out = self.proj(out)
        out = self.proj_drop(out, training=training)
        return out


class MLP(tf.keras.layers.Layer):
    """Transformer FFN: Linear(GELU) → Dropout → Linear → Dropout。"""

    def __init__(self, embed_dim, mlp_ratio=4.0, drop=0.0, **kwargs):
        super().__init__(**kwargs)
        hidden = int(embed_dim * mlp_ratio)
        self.fc1  = tf.keras.layers.Dense(hidden,    activation='gelu', use_bias=True, name='fc1')
        self.fc2  = tf.keras.layers.Dense(embed_dim, use_bias=True, name='fc2')
        self.drop = tf.keras.layers.Dropout(drop)

    def call(self, x, training=None):
        x = self.fc1(x)
        x = self.drop(x, training=training)
        x = self.fc2(x)
        x = self.drop(x, training=training)
        return x


class TransformerBlock(tf.keras.layers.Layer):
    """
    Pre-LN Transformer Block（比 Post-LN 訓練更穩定）:
        x = x + MHSA(LN(x))
        x = x + MLP(LN(x))
    """

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0,
                 attn_drop=0.0, proj_drop=0.0, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='norm1')
        self.attn  = MultiHeadSelfAttention(
            embed_dim, num_heads, attn_drop, proj_drop, name='attn'
        )
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='norm2')
        self.mlp   = MLP(embed_dim, mlp_ratio, proj_drop, name='mlp')

    def call(self, x, training=None):
        x = x + self.attn(self.norm1(x), training=training)
        x = x + self.mlp(self.norm2(x), training=training)
        return x


# ══════════════════════════════════════════════════════════════════════════════
# Dense Prediction Decoder
# ══════════════════════════════════════════════════════════════════════════════

class UpBlock(tf.keras.layers.Layer):
    """UpSampling2D(×2) + Conv2D×2 + BN + ReLU。"""

    def __init__(self, out_ch, **kwargs):
        super().__init__(**kwargs)
        self.up   = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.conv1 = tf.keras.layers.Conv2D(out_ch, 3, padding='same', use_bias=False)
        self.bn1   = tf.keras.layers.BatchNormalization()
        self.act1  = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv2D(out_ch, 3, padding='same', use_bias=False)
        self.bn2   = tf.keras.layers.BatchNormalization()
        self.act2  = tf.keras.layers.ReLU()

    def call(self, x, training=None):
        x = self.up(x)
        x = self.act1(self.bn1(self.conv1(x), training=training))
        x = self.act2(self.bn2(self.conv2(x), training=training))
        return x


class DensePredDecoder(tf.keras.layers.Layer):
    """
    (B, N, embed_dim) → (B, H_full, W_full, out_ch)

    流程：
      1. token_proj (Dense) 可降維
      2. reshape → (B, H_p, W_p, D)
      3. num_ups 次 UpBlock（每次 ×2）
      4. 最終 Conv2D(out_ch, 1)

    以 patch_size=8 為例：
      tokens 16×16 → 32×32 → 64×64 → 128×128  (num_ups=3)
    """

    def __init__(self, embed_dim=192, out_ch=6, num_ups=3,
                 base_ch=64, **kwargs):
        super().__init__(**kwargs)
        self.num_ups   = num_ups
        self.embed_dim = embed_dim
        self.out_ch    = out_ch

        # first_ch = embed_dim：token_proj 做等維度線性投影即可。
        # 舊做法 max(base_ch * 2^(num_ups-1), embed_dim) 在 tiny 模型下
        # 會把 embed_dim=96 膨脹到 256/512，既浪費記憶體也違反「tiny 要輕量」的設計意圖。
        first_ch = embed_dim
        self.token_proj = tf.keras.layers.Dense(first_ch, use_bias=True, name='token_proj')

        # UpBlocks：通道數逐步減半（不低於 base_ch）
        # 先全部建立再整批賦值，確保 Keras ListWrapper 完整追蹤所有子層
        up_blocks = []
        in_ch = first_ch
        for i in range(num_ups):
            out = max(base_ch, in_ch // 2)
            up_blocks.append(UpBlock(out, name=f'up_{i}'))
            in_ch = out
        self.up_blocks = up_blocks

        self.head = tf.keras.layers.Conv2D(out_ch, 1, padding='same', name='head')

    def call(self, tokens, H_p, W_p, training=None):
        """
        tokens: (B, N, embed_dim)  N = H_p * W_p
        Returns: (B, H_p * 2^num_ups, W_p * 2^num_ups, out_ch)
        """
        B = tf.shape(tokens)[0]
        x = self.token_proj(tokens)                  # (B, N, first_ch)
        x = tf.reshape(x, (B, H_p, W_p, -1))        # (B, H_p, W_p, first_ch)
        for up in self.up_blocks:
            x = up(x, training=training)
        return self.head(x)                          # (B, H_full, W_full, out_ch)


# ══════════════════════════════════════════════════════════════════════════════
# Physics Regression Head（與 CNN 版相同）
# ══════════════════════════════════════════════════════════════════════════════

class PhysicsAwareRegressionHead(tf.keras.layers.Layer):
    """預測 fold number 機率分布 {-2,-1,0,+1,+2}。"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fold_predictor = tf.keras.layers.Dense(5, activation=None, name='fold_logits')

    def call(self, features):
        return tf.nn.softmax(self.fold_predictor(features), axis=-1)


# ══════════════════════════════════════════════════════════════════════════════
# 主模型
# ══════════════════════════════════════════════════════════════════════════════

class ViTVelocityDealiaser(tf.keras.Model):
    """
    Vision Transformer 版速度反折錯模型。

    輸入/輸出介面與 VelocityDealiaser (dealias_mulit_v2_physics.py) 完全相同，
    可直接替換訓練腳本中的模型建立段落。

    預設（ViT-S, patch_size=8, 128×128 input）：
      - 256 tokens per sample
      - Attention O(N²) = 256² = 65k，batch=32 時約 50MB/layer → VRAM 友善
      - 參數量 ~18M

    較大感受野（相比 CNN）的優勢：
      - 每個 token 從第一個 block 就能看到整個掃描圖
      - 對 alias 邊界的全局連貫性判斷更好
    """

    def __init__(self,
                 patch_size=8,
                 embed_dim=192,
                 depth=6,
                 num_heads=6,
                 mlp_ratio=4.0,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 input_size=128,
                 classes=6,
                 **kwargs):
        super().__init__(**kwargs)
        self.patch_size  = patch_size
        self.embed_dim   = embed_dim
        self.classes     = classes

        grid_size  = input_size // patch_size       # e.g., 128//8 = 16
        num_patches = grid_size * grid_size          # e.g., 256
        num_ups    = int(math.log2(patch_size))     # e.g., log2(8)=3

        # ── Encoder ──────────────────────────────────────────────────────────
        self.patch_embed = PatchEmbed(patch_size, embed_dim, name='patch_embed')
        self.pos_embed   = LearnedPosEmbed(num_patches, embed_dim, name='pos_embed')
        self.blocks      = [
            TransformerBlock(embed_dim, num_heads, mlp_ratio,
                             attn_drop, proj_drop, name=f'block_{i}')
            for i in range(depth)
        ]
        self.encoder_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name='encoder_norm'
        )

        # ── Decoders ─────────────────────────────────────────────────────────
        self.decoder_cls = DensePredDecoder(
            embed_dim=embed_dim, out_ch=classes,
            num_ups=num_ups, base_ch=64, name='decoder_cls',
        )
        self.decoder_reg = DensePredDecoder(
            embed_dim=embed_dim, out_ch=1,
            num_ups=num_ups, base_ch=64, name='decoder_reg',
        )

        # ── Physics Regression Head ───────────────────────────────────────────
        # 先用 Conv1×1 將 token 從 embed_dim 壓縮至 32 維，再 bilinear resize 到全解析度
        # 避免直接 resize embed_dim=192 的 tokens（會產生 B×128×128×192 ≈ 400MB 中間張量）
        self.physics_proj = tf.keras.layers.Dense(32, use_bias=False, name='physics_proj')
        self.physics_head = PhysicsAwareRegressionHead(name='physics_head')

    # ─────────────────────────────────────────────────────────────────────────

    def call(self, inputs, training=None):
        """
        inputs: {'vel': (B, T, H, W, 1), 'nyq': (B, T, 1)}
        返回與 VelocityDealiaser 完全相同格式的 dict。
        """
        vel_in = inputs['vel']   # (B, T, H, W, 1)
        nyq    = inputs['nyq']   # (B, T, 1)

        # ── 前處理（與 CNN 版相同） ───────────────────────────────────────────
        vel_in, valid_mask = make_velocity_mask(vel_in, fill_val=-3.0)
        shp = tf.shape(vel_in)
        nyq_tiled = tf.tile(
            nyq[:, :, None, None, None],
            [1, 1, shp[2], shp[3], shp[4]],
        )
        vel_norm, _ = make_velocity_mask(vel_in / nyq_tiled, fill_val=-3.0)

        # reshape: (B, T, H, W, 1) → (B, H, W, T)
        x = tf.transpose(vel_norm, (0, 2, 3, 1, 4))
        xshp = tf.shape(x)
        x = tf.reshape(x, (xshp[0], xshp[1], xshp[2], xshp[3] * xshp[4]))

        # ── ViT Encoder ──────────────────────────────────────────────────────
        tokens, H_p, W_p = self.patch_embed(x)          # (B, N, D)
        tokens = self.pos_embed(tokens, H_p, W_p)        # (B, N, D) + pos

        for block in self.blocks:
            tokens = block(tokens, training=training)    # (B, N, D)

        tokens = self.encoder_norm(tokens)               # (B, N, D)

        # ── Dense Prediction ─────────────────────────────────────────────────
        classification_logits = self.decoder_cls(        # (B, H, W, 6)
            tokens, H_p, W_p, training=training
        )
        velocity_residual = self.decoder_reg(            # (B, H, W, 1)
            tokens, H_p, W_p, training=training
        )
        classification_logits = tf.clip_by_value(classification_logits, -10.0, 10.0)

        # ── 後處理（與 CNN 版完全相同） ───────────────────────────────────────
        raw_last = vel_in[:, -1]      # (B, H, W, 1) — 最後一幀原始速度
        nyq_last = nyq_tiled[:, -1]   # (B, H, W, 1) — 對應 Nyquist 速度

        # --- 硬分類校正 ---
        cat = tf.argmax(classification_logits, axis=-1, output_type=tf.int32)
        cat = tf.expand_dims(cat, axis=-1)
        corr_hard = tf.zeros_like(raw_last)
        corr_hard = tf.where(tf.equal(cat, 1), -4.0 * nyq_last, corr_hard)  # -2 fold
        corr_hard = tf.where(tf.equal(cat, 2), -2.0 * nyq_last, corr_hard)  # -1 fold
        corr_hard = tf.where(tf.equal(cat, 4),  2.0 * nyq_last, corr_hard)  # +1 fold
        corr_hard = tf.where(tf.equal(cat, 5),  4.0 * nyq_last, corr_hard)  # +2 fold
        dealiased_vel = raw_last + corr_hard

        # --- 軟分類校正 ---
        fold_probs_from_cls = tf.nn.softmax(classification_logits, axis=-1)[:, :, :, 1:6]
        fold_probs_from_cls = fold_probs_from_cls / (
            tf.reduce_sum(fold_probs_from_cls, axis=-1, keepdims=True) + 1e-8
        )
        fold_values = tf.reshape(
            tf.constant([-2., -1., 0., +1., +2.], dtype=fold_probs_from_cls.dtype),
            [1, 1, 1, 5]
        )
        expected_fold_cls = tf.reduce_sum(
            fold_probs_from_cls * fold_values, axis=-1, keepdims=True
        )
        corr_soft = expected_fold_cls * 2.0 * nyq_last
        dealiased_vel_soft = raw_last + corr_soft

        # --- 物理回歸分支 ---
        # 重要：先壓縮到 32 維再 resize，避免 B×128×128×embed_dim（~400MB）的巨大中間張量
        B = tf.shape(tokens)[0]
        token_spatial = tf.reshape(tokens, (B, H_p, W_p, self.embed_dim))  # (B, 16, 16, D)
        token_small   = self.physics_proj(token_spatial)                    # (B, 16, 16, 32)
        token_full    = tf.image.resize(
            tf.cast(token_small, tf.float32),
            [tf.shape(raw_last)[1], tf.shape(raw_last)[2]],
            method='bilinear',
        )                                                                    # (B, H, W, 32)
        fold_probs = self.physics_head(token_full)                          # (B, H, W, 5)
        physics_dealiased_vel = apply_physics_constraint(fold_probs, raw_last, nyq_last)

        # --- 遮罩處理（無效區域 → NaN） ---
        valid_mask_last = valid_mask[:, -1] if len(valid_mask.shape) > 3 else valid_mask
        if len(valid_mask_last.shape) == 3:
            valid_mask_last = tf.expand_dims(valid_mask_last, axis=-1)
        valid_mask_bool = tf.cast(valid_mask_last, tf.bool)
        # valid_mask_bool shape: (B, H, W, 1)，可廣播至所有 (B, H, W, C) 輸出

        def mask_nan(t):
            # 用 t.dtype 建立 NaN，確保 dtype 匹配（防止混合精度場景下的 type error）
            nan_t = tf.zeros_like(t) * tf.cast(float('nan'), t.dtype)
            return tf.where(valid_mask_bool, t, nan_t)

        physics_dealiased_vel = mask_nan(physics_dealiased_vel)
        dealiased_vel_soft    = mask_nan(dealiased_vel_soft)
        velocity_residual     = mask_nan(velocity_residual)
        dealiased_vel         = mask_nan(dealiased_vel)

        # --- 置信度導向混合分支 ---
        classification_probs = tf.nn.softmax(classification_logits, axis=-1)
        alias_probs = classification_probs[:, :, :, 1:6]
        alias_probs_sum = tf.reduce_sum(alias_probs, axis=-1, keepdims=True)
        alias_probs_norm = tf.where(
            tf.greater(alias_probs_sum, 1e-8),
            alias_probs / alias_probs_sum,
            alias_probs,
        )
        max_alias_conf = tf.reduce_max(alias_probs_norm, axis=-1, keepdims=True)
        confidence_threshold = 0.8
        hard_weight = tf.sigmoid((max_alias_conf - confidence_threshold) * 10.0)
        soft_weight = 1.0 - hard_weight
        confidence_guided_dealiased = mask_nan(
            raw_last + hard_weight * corr_hard + soft_weight * corr_soft
        )

        return {
            'alias_mask':                classification_logits,
            'velocity_residual':         velocity_residual,
            'dealiased_vel':             dealiased_vel,
            'fold_probs':                fold_probs,
            'physics_dealiased_vel':     physics_dealiased_vel,
            'soft_classification_vel':   dealiased_vel_soft,
            'fold_probs_from_cls':       fold_probs_from_cls,
            'confidence_guided_vel':     confidence_guided_dealiased,
            'alias_confidence':          max_alias_conf,
            'classification_confidence': tf.reduce_max(
                classification_probs, axis=-1, keepdims=True
            ),
            'hard_soft_weights': {
                'hard_weight': hard_weight,
                'soft_weight': soft_weight,
            },
        }


# ══════════════════════════════════════════════════════════════════════════════
# 工廠函數
# ══════════════════════════════════════════════════════════════════════════════

# 預設組態
_PRESETS = {
    # VRAM 需求從小到大
    'tiny':   dict(patch_size=8,  embed_dim=96,  depth=4, num_heads=3, mlp_ratio=4.0),
    'small':  dict(patch_size=8,  embed_dim=192, depth=6, num_heads=6, mlp_ratio=4.0),   # 預設
    'base':   dict(patch_size=8,  embed_dim=384, depth=8, num_heads=8, mlp_ratio=4.0),
    # patch_size=16 可大幅降低 token 數（64 tokens），適合 VRAM 緊張時
    'tiny16': dict(patch_size=16, embed_dim=96,  depth=4, num_heads=3, mlp_ratio=4.0),
    'small16':dict(patch_size=16, embed_dim=192, depth=6, num_heads=6, mlp_ratio=4.0),
}


def create_vit_dealiaser(preset='small',
                         patch_size=None,
                         embed_dim=None,
                         depth=None,
                         num_heads=None,
                         mlp_ratio=4.0,
                         attn_drop=0.0,
                         proj_drop=0.0,
                         input_size=128,
                         classes=6):
    """
    ViTVelocityDealiaser 工廠函數。

    參數:
        preset    : 預設組態名稱 ('tiny'/'small'/'base'/'tiny16'/'small16')
                    若指定個別參數則覆蓋 preset 值
        patch_size: patch 邊長（像素），推薦 8 或 16
        embed_dim : token 維度
        depth     : Transformer block 數量
        num_heads : Attention head 數量
        mlp_ratio : FFN hidden 維度倍率
        attn_drop : attention dropout 機率
        proj_drop : projection dropout 機率
        input_size: 輸入影像邊長（預設 128）
        classes   : 分類類別數（預設 6：0=no-alias, 1=-2f, 2=-1f, 3=no-alias, 4=+1f, 5=+2f）

    回傳:
        ViTVelocityDealiaser 實例

    使用範例（替換 transfer_learning_complete.py 中的模型建立段落）:

        # 原本:
        extractor    = create_downsampler(input_channels=1, start_neurons=32)
        upsampler_cls = create_upsampler_cls(n_inputs=1, start_neurons=32, classes=6)
        upsampler_reg = create_upsampler_reg(n_inputs=1, start_neurons=32)
        model = VelocityDealiaser(extractor, upsampler_cls, upsampler_reg)

        # 改為:
        from unet_model.vit_dealias_physics import create_vit_dealiaser
        model = create_vit_dealiaser('small')          # ~18M params
        model = create_vit_dealiaser('tiny')           # ~5M params，較省 VRAM
        model = create_vit_dealiaser('tiny16')         # 最省 VRAM (patch=16, 64 tokens)
    """
    cfg = _PRESETS.get(preset, _PRESETS['small']).copy()
    if patch_size  is not None: cfg['patch_size']  = patch_size
    if embed_dim   is not None: cfg['embed_dim']   = embed_dim
    if depth       is not None: cfg['depth']       = depth
    if num_heads   is not None: cfg['num_heads']   = num_heads
    if mlp_ratio   is not None: cfg['mlp_ratio']   = mlp_ratio

    ps = cfg['patch_size']
    if ps < 2 or (ps & (ps - 1)) != 0:
        raise ValueError(
            f"patch_size 必須是 2 的冪次（如 4, 8, 16），got {ps}。"
            f"非冪次值會導致 decoder 輸出尺寸 ≠ 輸入尺寸。"
        )
    if input_size % ps != 0:
        raise ValueError(
            f"input_size ({input_size}) 必須整除 patch_size ({ps})。"
        )
    ed = cfg['embed_dim']
    nh = cfg['num_heads']
    if ed % nh != 0:
        raise ValueError(
            f"embed_dim ({ed}) 必須整除 num_heads ({nh})。"
        )

    return ViTVelocityDealiaser(
        patch_size=cfg['patch_size'],
        embed_dim=cfg['embed_dim'],
        depth=cfg['depth'],
        num_heads=cfg['num_heads'],
        mlp_ratio=cfg['mlp_ratio'],
        attn_drop=attn_drop,
        proj_drop=proj_drop,
        input_size=input_size,
        classes=classes,
    )
