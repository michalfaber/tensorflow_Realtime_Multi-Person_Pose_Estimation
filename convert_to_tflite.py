import os
import importlib
import click
import tensorflow as tf

from train_singlenet_mobilenetv3 import register_tf_netbuilder_extensions
from util import probe_model


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def export_to_tflite(model, output_path):

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    #converter.experimental_new_converter = True
    #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

    tflite_model = converter.convert()

    open(output_path, "wb").write(tflite_model)


@click.command()
@click.option('--weights', required=True,
              help='Path to the folder containing weights for the model')
@click.option('--tflite-path',required=True,
              help='Path to the output tflite file')
@click.option('--create-model-fn',required=True,
              help='Name of a function to create model instance. Check available names here: .models._init__.py')
def main(weights, tflite_path, create_model_fn):
    register_tf_netbuilder_extensions()

    # load saved model

    module = importlib.import_module('models')
    create_model = getattr(module, create_model_fn)

    model = create_model()
    model.load_weights(weights)

    # first pass

    probe_model(model, test_img_path="resources/ski_224.jpg")

    # export model to tflite

    export_to_tflite(model, tflite_path)

    print("Done !!!")


if __name__ == '__main__':
    main()

