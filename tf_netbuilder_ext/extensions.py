import tensorflow as tf
from tf_netbuilder import NetBuilderConfig
from tf_netbuilder_ext.layers.prep_funcs import prepare_cnreg, prepare_cn3_args


def register_tf_netbuilder_extensions():
    NetBuilderConfig.add_block_type("cnreg", tf.keras.layers.Conv2D, prepare_cnreg),
    NetBuilderConfig.add_block_type("cn3", tf.keras.layers.Conv2D, prepare_cn3_args),

