import tensorflow as tf

from models.mobilenet_v2 import MobileNetV2, inverted_res_block
from tensorflow.keras import layers


def get_mobilenet_model(alpha, rows):
    """
    Functions adds 2 branches to the top of MobileNet model

    :return: model with separate outputs for heatmaps and pafs
    """
    inputs, output, endpoints = MobileNetV2(
        input_shape=(rows, rows, 3),
        alpha=alpha,
        include_top=False
    )

    # get the key layers

    out28x28 = endpoints['block_5_add']
    out28x28 = layers.ReLU(6., name='out_relu_block_5')(out28x28)

    out14x14 = endpoints['block_12_add']
    out14x14 = layers.ReLU(6., name='out_relu_block_12')(out14x14)

    # upscale smaller layers

    resized_out14 = tf.image.resize(out14x14, (28, 28), name='resized_out14')

    # concatenate

    mobilenet_output = tf.concat(axis=3, values=[out28x28, resized_out14], name='mobilenet_output')

    # stage 1 L (19)

    x = inverted_res_block(mobilenet_output, filters=128, alpha=alpha, stride=1,
                           expansion=6, block_id=100, kernel_size=3)
    x = inverted_res_block(x, filters=128, alpha=alpha, stride=1,
                           expansion=6, block_id=101, kernel_size=3)
    x = inverted_res_block(x, filters=128, alpha=alpha, stride=1,
                           expansion=6, block_id=102, kernel_size=3)
    x = inverted_res_block(x, filters=512, alpha=alpha, stride=1,
                           expansion=6, block_id=103, kernel_size=1)
    l_1 = layers.Conv2D(19,
                        kernel_size=1,
                        use_bias=False,
                        activation=None,
                        name='stage1_l_out')(x)

    # stage 1 S (38)

    x = inverted_res_block(mobilenet_output, filters=128, alpha=alpha, stride=1,
                           expansion=6, block_id=104, kernel_size=3)
    x = inverted_res_block(x, filters=128, alpha=alpha, stride=1,
                           expansion=6, block_id=105, kernel_size=3)
    x = inverted_res_block(x, filters=128, alpha=alpha, stride=1,
                           expansion=6, block_id=106, kernel_size=3)
    x = inverted_res_block(x, filters=512, alpha=alpha, stride=1,
                           expansion=6, block_id=107, kernel_size=1)
    s_1 = layers.Conv2D(38,
                        kernel_size=1,
                        use_bias=False,
                        activation=None,
                        name='stage1_s_out')(x)

    stage_1_out = tf.concat(axis=3, values=[s_1, l_1, mobilenet_output], name='stage_1_out')

    # stage 2 L (19)

    x = inverted_res_block(stage_1_out, filters=128, alpha=alpha, stride=1,
                           expansion=6, block_id=200, kernel_size=7)
    x = inverted_res_block(x, filters=128, alpha=alpha, stride=1,
                           expansion=6, block_id=201, kernel_size=7)
    x = inverted_res_block(x, filters=128, alpha=alpha, stride=1,
                           expansion=6, block_id=202, kernel_size=7)
    x = inverted_res_block(x, filters=128, alpha=alpha, stride=1,
                           expansion=6, block_id=203, kernel_size=7)
    x = inverted_res_block(x, filters=128, alpha=alpha, stride=1,
                           expansion=6, block_id=204, kernel_size=7)
    x = inverted_res_block(x, filters=128, alpha=alpha, stride=1,
                           expansion=6, block_id=205, kernel_size=1)
    l_2 = layers.Conv2D(19,
                        kernel_size=1,
                        use_bias=False,
                        activation=None,
                        name='stage2_l_out')(x)

    # stage 2 S (38)

    x = inverted_res_block(stage_1_out, filters=128, alpha=alpha, stride=1,
                           expansion=6, block_id=206, kernel_size=7)
    x = inverted_res_block(x, filters=128, alpha=alpha, stride=1,
                           expansion=6, block_id=207, kernel_size=7)
    x = inverted_res_block(x, filters=128, alpha=alpha, stride=1,
                           expansion=6, block_id=208, kernel_size=7)
    x = inverted_res_block(x, filters=128, alpha=alpha, stride=1,
                           expansion=6, block_id=209, kernel_size=7)
    x = inverted_res_block(x, filters=128, alpha=alpha, stride=1,
                           expansion=6, block_id=210, kernel_size=7)
    x = inverted_res_block(x, filters=128, alpha=alpha, stride=1,
                           expansion=6, block_id=211, kernel_size=1)
    s_2 = layers.Conv2D(38,
                        kernel_size=1,
                        use_bias=False,
                        activation=None,
                        name='stage2_s_out')(x)

    model = tf.keras.Model(inputs=inputs, outputs=[s_1, l_1, s_2, l_2])

    return model
