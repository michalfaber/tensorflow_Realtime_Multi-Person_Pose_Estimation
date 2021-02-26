import tensorflow as tf
import click
import cv2
import numpy as np
import importlib
import time

from estimation.config import get_default_configuration
from estimation.coordinates import get_coordinates
from estimation.connections import get_connections
from estimation.estimators import estimate
from estimation.renderers import draw

from train_singlenet_mobilenetv3 import register_tf_netbuilder_extensions


def process_frame(cropped, heatmap_idx, model, paf_idx, output_resize_factor, cfg):
    input_img = cropped[np.newaxis, ...]
    inputs = tf.convert_to_tensor(input_img)
    outputs = model.predict(inputs)
    pafs = outputs[paf_idx][0, ...]
    heatmaps = outputs[heatmap_idx][0, ...]
    coordinates = get_coordinates(cfg, heatmaps)
    connections = get_connections(cfg, coordinates, pafs)
    skeletons = estimate(cfg, connections)
    canvas = draw(cfg, cropped, coordinates, skeletons, resize_fac=output_resize_factor)

    return canvas


@click.command()
@click.option('--video', required=True,
              help='Path to the input video file')
@click.option('--output-video', required=True,
              help='Path to the output video file')
@click.option('--create-model-fn', required=True,
              help='Name of a function to create model instance. Check available names here: .models._init__.py')
@click.option('--input-size', required=True, type=int,
              help='Model''s input size ')
@click.option('--output-resize-factor', required=True, type=int,
              help='Output resize factor')
@click.option('--paf-idx', default=2, type=int,
              help='Index of model''s output containing PAF')
@click.option('--heatmap-idx', default=3, type=int,
              help='Index of model''s output containing heatmap')
@click.option('--frames-to-analyze', default=None, type=int,
              help='Number of the frames to analyze')
@click.option('--frame-ratio', default=1,
              help='Analyze every [n] frames')
def main(video, output_video, create_model_fn, input_size, output_resize_factor, paf_idx, heatmap_idx, frames_to_analyze, frame_ratio):

    register_tf_netbuilder_extensions()

    module = importlib.import_module('models')
    create_model = getattr(module, create_model_fn)
    model = create_model(pretrained=True)

    # Video reader
    cam = cv2.VideoCapture(video)
    input_fps = cam.get(cv2.CAP_PROP_FPS)
    ret_val, orig_image = cam.read()
    video_length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

    if frames_to_analyze is None:
        frames_to_analyze = video_length

    # Video writer
    output_fps = input_fps / frame_ratio
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w = orig_image.shape[1]
    h = orig_image.shape[0]
    scale = input_size / w if w < h else input_size / h
    out = cv2.VideoWriter(output_video, fourcc, output_fps, (input_size, input_size))

    # load configs
    cfg = get_default_configuration()

    i = 0
    while (cam.isOpened()) and ret_val is True and i < frames_to_analyze:
        if i % frame_ratio == 0:

            tic = time.time()

            im = cv2.resize(orig_image, (0, 0), fx=scale, fy=scale)
            new_w = im.shape[1]
            new_h = im.shape[0]
            if new_w > new_h:
                offset = (new_w - input_size) // 2
                cropped = im[0: input_size, offset: offset + input_size]
            else:
                offset = (new_h - input_size) // 2
                cropped = im[offset: offset + input_size, 0: input_size]

            canvas = process_frame(cropped, heatmap_idx, model, paf_idx, output_resize_factor, cfg)

            print('Processing frame: ', i)
            toc = time.time()
            print('processing time is %.5f' % (toc - tic))

            out.write(canvas)

        ret_val, orig_image = cam.read()

        i += 1


if __name__ == '__main__':
    main()
