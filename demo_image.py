import tensorflow as tf
import click
import cv2
import numpy as np
import importlib

from estimation.config import get_default_configuration
from estimation.coordinates import get_coordinates
from estimation.connections import get_connections
from estimation.estimators import estimate
from estimation.renderers import draw

from train_singlenet_mobilenetv3 import register_tf_netbuilder_extensions


@click.command()
@click.option('--image', required=True,
              help='Path to the input image file')
@click.option('--output-image', required=True,
              help='Path to the output image file')
@click.option('--create-model-fn', required=True,
              help='Name of a function to create model instance. Check available names here: .models._init__.py')
@click.option('--paf-idx', default=2,
              help='Index of model''s output containing PAF')
@click.option('--heatmap-idx', default=3,
              help='Index of model''s output containing heatmap')
def main(image, output_image, create_model_fn, paf_idx, heatmap_idx):

    register_tf_netbuilder_extensions()

    module = importlib.import_module('models')
    create_model = getattr(module, create_model_fn)
    model = create_model(pretrained=True)

    img = cv2.imread(image)  # B,G,R order
    input_img = img[np.newaxis, :, :, [2, 1, 0]]
    inputs = tf.convert_to_tensor(input_img)

    outputs = model.predict(inputs)
    pafs = outputs[paf_idx][0, ...]
    heatmaps = outputs[heatmap_idx][0, ...]

    cfg = get_default_configuration()

    coordinates = get_coordinates(cfg, heatmaps)

    connections = get_connections(cfg, coordinates, pafs)

    skeletons = estimate(cfg, connections)

    output = draw(cfg, img, coordinates, skeletons, resize_fac=8)

    cv2.imwrite(output_image, output)

    print(f"Output saved: {output_image}")


if __name__ == '__main__':
    main()
