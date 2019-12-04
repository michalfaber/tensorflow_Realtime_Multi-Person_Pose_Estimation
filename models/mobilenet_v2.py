"""MobileNet v2 models for Keras.

MobileNetV2 is a general architecture and can be used for multiple use cases.
Depending on the use case, it can use different input layer size and
different width factors. This allows different width models to reduce
the number of multiply-adds and thereby
reduce inference cost on mobile devices.

MobileNetV2 is very similar to the original MobileNet,
except that it uses inverted residual blocks with
bottlenecking features. It has a drastically lower
parameter count than the original MobileNet.
MobileNets support any input size greater
than 32 x 32, with larger image sizes
offering better performance.

The number of parameters and number of multiply-adds
can be modified by using the `alpha` parameter,
which increases/decreases the number of filters in each layer.
By altering the image size and `alpha` parameter,
all 22 models from the paper can be built, with ImageNet weights provided.

The paper demonstrates the performance of MobileNets using `alpha` values of
1.0 (also called 100 % MobileNet), 0.35, 0.5, 0.75, 1.0, 1.3, and 1.4

For each of these `alpha` values, weights for 5 different input image sizes
are provided (224, 192, 160, 128, and 96).


The following table describes the performance of
MobileNet on various input sizes:
------------------------------------------------------------------------
MACs stands for Multiply Adds

 Classification Checkpoint| MACs (M) | Parameters (M)| Top 1 Accuracy| Top 5 Accuracy
--------------------------|------------|---------------|---------|----|-------------
| [mobilenet_v2_1.4_224]  | 582 | 6.06 |          75.0 | 92.5 |
| [mobilenet_v2_1.3_224]  | 509 | 5.34 |          74.4 | 92.1 |
| [mobilenet_v2_1.0_224]  | 300 | 3.47 |          71.8 | 91.0 |
| [mobilenet_v2_1.0_192]  | 221 | 3.47 |          70.7 | 90.1 |
| [mobilenet_v2_1.0_160]  | 154 | 3.47 |          68.8 | 89.0 |
| [mobilenet_v2_1.0_128]  | 99  | 3.47 |          65.3 | 86.9 |
| [mobilenet_v2_1.0_96]   | 56  | 3.47 |          60.3 | 83.2 |
| [mobilenet_v2_0.75_224] | 209 | 2.61 |          69.8 | 89.6 |
| [mobilenet_v2_0.75_192] | 153 | 2.61 |          68.7 | 88.9 |
| [mobilenet_v2_0.75_160] | 107 | 2.61 |          66.4 | 87.3 |
| [mobilenet_v2_0.75_128] | 69  | 2.61 |          63.2 | 85.3 |
| [mobilenet_v2_0.75_96]  | 39  | 2.61 |          58.8 | 81.6 |
| [mobilenet_v2_0.5_224]  | 97  | 1.95 |          65.4 | 86.4 |
| [mobilenet_v2_0.5_192]  | 71  | 1.95 |          63.9 | 85.4 |
| [mobilenet_v2_0.5_160]  | 50  | 1.95 |          61.0 | 83.2 |
| [mobilenet_v2_0.5_128]  | 32  | 1.95 |          57.7 | 80.8 |
| [mobilenet_v2_0.5_96]   | 18  | 1.95 |          51.2 | 75.8 |
| [mobilenet_v2_0.35_224] | 59  | 1.66 |          60.3 | 82.9 |
| [mobilenet_v2_0.35_192] | 43  | 1.66 |          58.2 | 81.2 |
| [mobilenet_v2_0.35_160] | 30  | 1.66 |          55.7 | 79.1 |
| [mobilenet_v2_0.35_128] | 20  | 1.66 |          50.8 | 75.0 |
| [mobilenet_v2_0.35_96]  | 11  | 1.66 |          45.5 | 70.4 |

The weights for all 16 models are obtained and
translated from the Tensorflow checkpoints
from TensorFlow checkpoints found [here]
(https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/README.md).

# Reference

This file contains building code for MobileNetV2, based on
[MobileNetV2: Inverted Residuals and Linear Bottlenecks]
(https://arxiv.org/abs/1801.04381) (CVPR 2018)

Tests comparing this model to the existing Tensorflow model can be
found at [mobilenet_v2_keras]
(https://github.com/JonathanCMitchell/mobilenet_v2_keras)
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import warnings
import tensorflow as tf

from keras_applications.imagenet_utils import _obtain_input_shape
from keras_applications.mobilenet_v2 import _make_divisible
from tensorflow.keras import layers


def correct_pad(inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.

    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.

    # Returns
        A tuple.
    """
    img_dim = 1
    shape = inputs.get_shape()
    input_size = shape[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))


def log_endpoint(endpoints, t):
    """
    Helper small func inserting tensor into the dictionary of all endpoints
    """
    if endpoints is not None:
        endpoints[t._keras_history.layer.name] = t


def MobileNetV2(input_shape=None,
                alpha=1.0,
                include_top=True,
                weights='imagenet',
                input_tensor=None,
                pooling=None,
                classes=1000,
                **kwargs):
    """Instantiates the MobileNetV2 architecture.

    # Arguments
        input_shape: optional shape tuple, to be specified if you would
            like to use a model with an input img resolution that is not
            (224, 224, 3).
            It should have exactly 3 inputs channels (224, 224, 3).
            You can also omit this option if you would like
            to infer input_shape from an input_tensor.
            If you choose to include both input_tensor and input_shape then
            input_shape will be used if they match, if the shapes
            do not match then we will throw an error.
            E.g. `(160, 160, 3)` would be one valid value.
        alpha: controls the width of the network. This is known as the
        width multiplier in the MobileNetV2 paper, but the name is kept for
        consistency with MobileNetV1 in Keras.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape or invalid alpha, rows when
            weights='imagenet'
    """
    endpoints = {}
    default_size = 224
    channel_axis = -1

    first_block_filters = _make_divisible(32 * alpha, 8)

    input_shape = _obtain_input_shape(input_shape,
                                      default_size=default_size,
                                      min_size=32,
                                      data_format='channels_last',
                                      require_flatten=include_top,
                                      weights=weights)

    row_axis, col_axis = (0, 1)

    rows = input_shape[row_axis]
    cols = input_shape[col_axis]

    if weights == 'imagenet':
        if alpha not in [0.35, 0.50, 0.75, 1.0, 1.3, 1.4]:
            raise ValueError('If imagenet weights are being loaded, '
                             'alpha can be one of `0.35`, `0.50`, `0.75`, '
                             '`1.0`, `1.3` or `1.4` only.')

        if rows != cols or rows not in [96, 128, 160, 192, 224]:
            rows = 224
            warnings.warn('`input_shape` is undefined or non-square, '
                          'or `rows` is not in [96, 128, 160, 192, 224].'
                          ' Weights for input shape (224, 224) will be'
                          ' loaded as the default.')

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        img_input = input_tensor
    log_endpoint(endpoints, img_input)

    x = tf.math.subtract(img_input, 127.5, name='preprocessing_sub')
    x = tf.math.divide(x, 127.5, name='preprocessing_div')

    x = layers.ZeroPadding2D(padding=correct_pad(x, 3), name='Conv1_pad')(x)
    log_endpoint(endpoints, x)

    x = layers.Conv2D(first_block_filters,
                      kernel_size=3,
                      strides=(2, 2),
                      padding='valid',
                      use_bias=False,
                      name='Conv1')(x)
    log_endpoint(endpoints, x)

    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-3,
                                  momentum=0.999,
                                  name='bn_Conv1')(x)
    log_endpoint(endpoints, x)

    x = layers.ReLU(6., name='Conv1_relu')(x)
    log_endpoint(endpoints, x)

    x = inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                           expansion=1, block_id=0, endpoints=endpoints)

    x = inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                           expansion=6, block_id=1, endpoints=endpoints)
    x = inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                           expansion=6, block_id=2, endpoints=endpoints)

    x = inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                           expansion=6, block_id=3, endpoints=endpoints)
    x = inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                           expansion=6, block_id=4, endpoints=endpoints)
    x = inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                           expansion=6, block_id=5, endpoints=endpoints)       # block_5_project (None, 28, 28, 32)

    x = inverted_res_block(x, filters=64, alpha=alpha, stride=2,
                           expansion=6, block_id=6, endpoints=endpoints)
    x = inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                           expansion=6, block_id=7, endpoints=endpoints)
    x = inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                           expansion=6, block_id=8, endpoints=endpoints)
    x = inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                           expansion=6, block_id=9, endpoints=endpoints)

    x = inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                           expansion=6, block_id=10, endpoints=endpoints)
    x = inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                           expansion=6, block_id=11, endpoints=endpoints)
    x = inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                           expansion=6, block_id=12, endpoints=endpoints)      # block_12_project (None, 14, 14, 96)

    x = inverted_res_block(x, filters=160, alpha=alpha, stride=2,
                           expansion=6, block_id=13, endpoints=endpoints)
    x = inverted_res_block(x, filters=160, alpha=alpha, stride=1,
                           expansion=6, block_id=14, endpoints=endpoints)
    x = inverted_res_block(x, filters=160, alpha=alpha, stride=1,
                           expansion=6, block_id=15, endpoints=endpoints)

    x = inverted_res_block(x, filters=320, alpha=alpha, stride=1,
                           expansion=6, block_id=16, endpoints=endpoints)      # block_16_project (None, 7, 7, 320)

    # no alpha applied to last conv as stated in the paper:
    # if the width multiplier is greater than 1 we
    # increase the number of output channels
    if alpha > 1.0:
        last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280

    x = layers.Conv2D(last_block_filters,
                      kernel_size=1,
                      use_bias=False,
                      name='Conv_1')(x)
    log_endpoint(endpoints, x)

    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-3,
                                  momentum=0.999,
                                  name='Conv_1_bn')(x)
    log_endpoint(endpoints, x)

    x = layers.ReLU(6., name='out_relu')(x)
    log_endpoint(endpoints, x)

    if include_top:
        x = layers.GlobalAveragePooling2D()(x)
        log_endpoint(endpoints, x)

        x = layers.Dense(classes, activation='softmax', use_bias=True, name='Logits')(x)
        log_endpoint(endpoints, x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
            log_endpoint(endpoints, x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)
            log_endpoint(endpoints, x)

    inputs = img_input

    return inputs, x, endpoints


def inverted_res_block(inputs, expansion, stride, alpha, filters, block_id, endpoints=None, kernel_size=3):
    channel_axis = -1

    shape = inputs.get_shape().as_list()

    in_channels = shape[channel_axis]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'block_{}_'.format(block_id)

    if block_id:
        # Expand
        x = layers.Conv2D(expansion * in_channels,
                          kernel_size=1,
                          padding='same',
                          use_bias=False,
                          activation=None,
                          name=prefix + 'expand')(x)
        log_endpoint(endpoints, x)

        x = layers.BatchNormalization(axis=channel_axis,
                                      epsilon=1e-3,
                                      momentum=0.999,
                                      name=prefix + 'expand_BN')(x)
        log_endpoint(endpoints, x)

        x = layers.ReLU(6., name=prefix + 'expand_relu')(x)
        log_endpoint(endpoints, x)
    else:
        prefix = 'expanded_conv_'

    # Depthwise
    if stride == 2:
        x = layers.ZeroPadding2D(padding=correct_pad(x, kernel_size),
                                 name=prefix + 'pad')(x)
        log_endpoint(endpoints, x)

    x = layers.DepthwiseConv2D(kernel_size=kernel_size,
                               strides=stride,
                               activation=None,
                               use_bias=False,
                               padding='same' if stride == 1 else 'valid',
                               name=prefix + 'depthwise')(x)
    log_endpoint(endpoints, x)

    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-3,
                                  momentum=0.999,
                                  name=prefix + 'depthwise_BN')(x)
    log_endpoint(endpoints, x)

    x = layers.ReLU(6., name=prefix + 'depthwise_relu')(x)
    log_endpoint(endpoints, x)

    # Project
    x = layers.Conv2D(pointwise_filters,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      activation=None,
                      name=prefix + 'project')(x)
    log_endpoint(endpoints, x)

    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-3,
                                  momentum=0.999,
                                  name=prefix + 'project_BN')(x)
    log_endpoint(endpoints, x)

    if in_channels == pointwise_filters and stride == 1:
        x = layers.Add(name=prefix + 'add')([inputs, x])
        log_endpoint(endpoints, x)
        return x

    return x
