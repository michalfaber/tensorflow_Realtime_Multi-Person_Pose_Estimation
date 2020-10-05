import tensorflow as tf


def prepare_cnreg(in_chs, args):

    new_args = dict(
        name=args["name"],
        filters=args["out_chs"],
        kernel_size=args["kernel_size"],
        strides=args["strides"],
        padding='same',
        activation=args["activation"](),
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=tf.keras.regularizers.l2(5e-4),
        bias_regularizer=tf.keras.regularizers.l2(0.0)
    )
    return new_args


def prepare_cn3_args(in_chs: int, args: map):

    new_args = dict(
        kernel_size=args["kernel_size"],
        strides=args["strides"],
        filters=args["out_chs"],
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros'
    )
    return new_args
