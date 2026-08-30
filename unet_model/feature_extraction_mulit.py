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
                       input_channels: int=1,
                       start_neurons: int=32):
    if inp is None:
        inp = tf.keras.Input(shape=(None,None,input_channels))

    x0 = polar_block(inp,
                     input_channels=input_channels,
                     pool_size=_POOL[0],
                     kernel_size=(7,7),
                     output_channels=start_neurons)
    x1 = polar_block(x0,
                     input_channels=start_neurons, 
                     pool_size=_POOL[1],
                     kernel_size=(5,5),
                     output_channels=2*start_neurons)
    x2 = polar_block(x1,
                     input_channels=2*start_neurons,
                     pool_size=_POOL[2],
                     kernel_size=(3,3),
                     output_channels=4*start_neurons)
    x3 = polar_block(x2,
                     input_channels=4*start_neurons,
                     pool_size=_POOL[3],
                     kernel_size=(3,3),
                     output_channels=8*start_neurons)

    return tf.keras.Model(inputs=inp, outputs=[x0,x1,x2,x3], name='PolarExtractor')

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

def _base_upsampler(inputs, start_neurons=32):
    """
    基本的 decoder，輸入4層feature list: [i0, i1, i2, i3]
    最終輸出一個 feature map (不做最後的Conv2D).
    """
    i0, i1, i2, i3 = inputs

    # 這裡可以加幾個不同 kernel size 的卷積並 concat
    x = tf.keras.layers.Conv2D(start_neurons*8,3,padding='same')(i3) 
    x1= tf.keras.layers.Conv2D(start_neurons*8,5,padding='same')(i3) 
    x2= tf.keras.layers.Conv2D(start_neurons*8,7,padding='same')(i3) 
    x = tf.keras.layers.Concatenate()([x,x1,x2])

    x = up_block(x, start_neurons*8, (3,3), _POOL[3], x_skip=i2)
    x = up_block(x, start_neurons*4, (3,3), _POOL[2], x_skip=i1)
    x = up_block(x, start_neurons*2, (3,3), _POOL[1], x_skip=i0)
    x = up_block(x, start_neurons,   (3,3), _POOL[0], x_skip=None)
    return x

def create_upsampler_cls(n_inputs=2, start_neurons=32, n_outputs=6):
    """
    分類支路: 最後輸出 shape=(..., 6)
    """
    i0 = tf.keras.Input(shape=(None,None,n_inputs*start_neurons))
    i1 = tf.keras.Input(shape=(None,None,n_inputs*start_neurons*2))
    i2 = tf.keras.Input(shape=(None,None,n_inputs*start_neurons*4))
    i3 = tf.keras.Input(shape=(None,None,n_inputs*start_neurons*8))

    x = _base_upsampler([i0,i1,i2,i3], start_neurons=start_neurons)
    # 最後一層: 分類 => 6 channels
    out_cls = tf.keras.layers.Conv2D(n_outputs, (3,3), padding='same')(x)

    return tf.keras.Model(inputs=[i0,i1,i2,i3], outputs=out_cls, name="Upsampler_Cls")

def create_upsampler_reg(n_inputs=2, start_neurons=32, n_outputs=1):
    """
    回歸支路: 最後輸出 shape=(...,1)，表示殘差(或速度)
    """
    i0 = tf.keras.Input(shape=(None,None,n_inputs*start_neurons))
    i1 = tf.keras.Input(shape=(None,None,n_inputs*start_neurons*2))
    i2 = tf.keras.Input(shape=(None,None,n_inputs*start_neurons*4))
    i3 = tf.keras.Input(shape=(None,None,n_inputs*start_neurons*8))

    x = _base_upsampler([i0,i1,i2,i3], start_neurons=start_neurons)
    # 最後一層: 回歸 => 1 channel
    out_reg = tf.keras.layers.Conv2D(n_outputs, (3,3), padding='same')(x)

    return tf.keras.Model(inputs=[i0,i1,i2,i3], outputs=out_reg, name="Upsampler_Reg")
