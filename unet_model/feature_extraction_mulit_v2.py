import tensorflow as tf

_POOL = [
    (2,2),
    (2,2),
    (2,2),
    (2,2),
]

def polar_block(x, 
                input_channels, 
                kernel_size, 
                pool_size, 
                output_channels,
                alpha=0.1):
    if input_channels != output_channels:
        x1 = tf.keras.layers.Conv2D(output_channels, 1, strides=(1, 1), padding='same')(x)
        x1 = tf.keras.layers.BatchNormalization()(x1) # 添加BatchNorm
    else:
        x1 = x
    x1 = tf.keras.layers.AveragePooling2D(pool_size)(x1)

    x2 = tf.keras.layers.Conv2D(output_channels,kernel_size,padding='same')(x)
    x2 = tf.keras.layers.LeakyReLU(alpha)(x2)
    x2 = tf.keras.layers.Conv2D(output_channels,kernel_size,padding='same')(x2)
    x2 = tf.keras.layers.LeakyReLU(alpha)(x2)
    x2 = tf.keras.layers.AveragePooling2D(pool_size)(x2)

    return tf.keras.layers.Add()([x1,x2])

def create_downsampler(inp=None,
                       input_channels=1,
                       start_neurons=32,
                       filters=None,  # 新增filters參數
                       pool_size=None,
                       kernel_size=None,
                       dropout_rate=0.0,
                       mixed_precision=False):
    """建立下採樣編碼器
    
    Args:
        inp: 輸入張量，預設None會自動建立
        input_channels: 輸入通道數
        start_neurons: 初始通道數，若filters有指定則忽略
        filters: 各層通道數列表，例如[64, 128, 256, 512]
        pool_size: 池化大小（如果指定）
        kernel_size: 核大小（如果指定）
        dropout_rate: 棄置率
        mixed_precision: 是否使用混合精度
    """
    # 如果提供了filters，則使用它
    if filters is not None:
        filter_counts = filters
        n_layers = len(filters)
    else:
        # 使用傳統方式
        filter_counts = [start_neurons, start_neurons*2, start_neurons*4, start_neurons*8]
        n_layers = 4
    
    # 使用預設池化大小或自定義池化大小
    if pool_size is None:
        pool_sizes = _POOL
    else:
        pool_sizes = [pool_size] * n_layers
    
    # 使用預設核大小或自定義核大小
    if kernel_size is None:
        kernel_sizes = [(7,7), (5,5), (3,3), (3,3)]
    else:
        kernel_sizes = [kernel_size] * n_layers
    
    if inp is None:
        inp = tf.keras.Input(shape=(None, None, input_channels))
    
    # 第一層
    x0 = polar_block(inp,
                    input_channels=input_channels,
                    pool_size=pool_sizes[0],
                    kernel_size=kernel_sizes[0],
                    output_channels=filter_counts[0])
    # 第二層
    x1 = polar_block(x0,
                    input_channels=filter_counts[0], 
                    pool_size=pool_sizes[1],
                    kernel_size=kernel_sizes[1],
                    output_channels=filter_counts[1])
    # 第三層
    x2 = polar_block(x1,
                    input_channels=filter_counts[1],
                    pool_size=pool_sizes[2],
                    kernel_size=kernel_sizes[2],
                    output_channels=filter_counts[2])
    # 第四層
    x3 = polar_block(x2,
                    input_channels=filter_counts[2],
                    pool_size=pool_sizes[3],
                    kernel_size=kernel_sizes[3],
                    output_channels=filter_counts[3])
    
    # 如果啟用dropout
    if dropout_rate > 0:
        x0 = tf.keras.layers.Dropout(dropout_rate)(x0)
        x1 = tf.keras.layers.Dropout(dropout_rate)(x1)
        x2 = tf.keras.layers.Dropout(dropout_rate)(x2)
        x3 = tf.keras.layers.Dropout(dropout_rate)(x3)

    return tf.keras.Model(inputs=inp, outputs=[x0, x1, x2, x3], name='PolarExtractor')

def up_block(x, output_channels, kernel_size, upsample_factor, x_skip=None):
    x = tf.keras.layers.UpSampling2D(size=upsample_factor)(x)
    if x_skip is not None:
        x = tf.keras.layers.concatenate([x_skip, x], axis=-1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(output_channels, kernel_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(output_channels, kernel_size, padding='same')(x)
    x = tf.keras.layers.Conv2D(output_channels, kernel_size, padding='same')(x)
    return x

def _base_upsampler(inputs, start_neurons=32, kernel_size=3):
    """
    基本的 decoder，輸入4層feature list: [i0, i1, i2, i3]
    最終輸出一個 feature map (不做最後的Conv2D).
    
    Args:
        inputs: 編碼器輸出的特徵圖列表 [i0, i1, i2, i3]
        start_neurons: 起始通道數
        kernel_size: 卷積核大小
    """
    i0, i1, i2, i3 = inputs

    # 多尺度卷積
    x = tf.keras.layers.Conv2D(start_neurons*8, kernel_size, padding='same')(i3) 
    x1 = tf.keras.layers.Conv2D(start_neurons*8, kernel_size+2, padding='same')(i3) 
    x2 = tf.keras.layers.Conv2D(start_neurons*8, kernel_size+4, padding='same')(i3) 
    x = tf.keras.layers.Concatenate()([x, x1, x2])

    # 上採樣塊 - 使用相同的池化大小設定
    x = up_block(x, start_neurons*8, (kernel_size, kernel_size), _POOL[3], x_skip=i2)
    x = up_block(x, start_neurons*4, (kernel_size, kernel_size), _POOL[2], x_skip=i1)
    x = up_block(x, start_neurons*2, (kernel_size, kernel_size), _POOL[1], x_skip=i0)
    x = up_block(x, start_neurons, (kernel_size, kernel_size), _POOL[0], x_skip=None)
    return x

def create_upsampler_cls(n_inputs=2, 
                         start_neurons=32, 
                         filters=None, 
                         kernel_size=3, 
                         classes=6, 
                         dropout_rate=0.0,
                         mixed_precision=False):
    """分類解碼器
    
    Args:
        n_inputs: 輸入特徵圖的數量
        start_neurons: 初始通道數
        filters: 各層通道數，例如[512, 256, 128, 64]
        kernel_size: 核大小
        classes: 輸出類別數
        dropout_rate: 棄置率
        mixed_precision: 是否使用混合精度
    """
    # 確定各層通道數
    if filters is not None:
        filter_counts = filters
    else:
        filter_counts = [start_neurons*8, start_neurons*4, start_neurons*2, start_neurons]
    
    # 輸入層
    i0 = tf.keras.Input(shape=(None, None, n_inputs*filter_counts[3]))
    i1 = tf.keras.Input(shape=(None, None, n_inputs*filter_counts[2]))
    i2 = tf.keras.Input(shape=(None, None, n_inputs*filter_counts[1]))
    i3 = tf.keras.Input(shape=(None, None, n_inputs*filter_counts[0]))
    
    # 處理編碼器輸出
    x = _base_upsampler([i0, i1, i2, i3], start_neurons=filter_counts[3])
    
    # 添加dropout
    if dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    # 最終分類層
    out_cls = tf.keras.layers.Conv2D(classes, (kernel_size, kernel_size), padding='same')(x)
    
    return tf.keras.Model(inputs=[i0, i1, i2, i3], outputs=out_cls, name="Upsampler_Cls")

def create_upsampler_reg(n_inputs=2, 
                         start_neurons=32, 
                         filters=None, 
                         kernel_size=3, 
                         dropout_rate=0.0,
                         mixed_precision=False):
    """回歸解碼器 - 用於速度預測
    
    Args:
        n_inputs: 輸入特徵圖的數量
        start_neurons: 初始通道數
        filters: 各層通道數，例如[512, 256, 128, 64]
        kernel_size: 核大小
        dropout_rate: 棄置率
        mixed_precision: 是否使用混合精度
    """
    # 確定各層通道數
    if filters is not None:
        filter_counts = filters
    else:
        filter_counts = [start_neurons*8, start_neurons*4, start_neurons*2, start_neurons]
    
    # 輸入層
    i0 = tf.keras.Input(shape=(None, None, n_inputs*filter_counts[3]))
    i1 = tf.keras.Input(shape=(None, None, n_inputs*filter_counts[2]))
    i2 = tf.keras.Input(shape=(None, None, n_inputs*filter_counts[1]))
    i3 = tf.keras.Input(shape=(None, None, n_inputs*filter_counts[0]))
    
    # 處理編碼器輸出
    x = _base_upsampler([i0, i1, i2, i3], start_neurons=filter_counts[3])
    
    # 添加dropout
    if dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    # 最終回歸層 - 單通道輸出
    out_reg = tf.keras.layers.Conv2D(1, (kernel_size, kernel_size), padding='same')(x)
    
    return tf.keras.Model(inputs=[i0, i1, i2, i3], outputs=out_reg, name="Upsampler_Reg")
