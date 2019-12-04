import os
import cv2
import numpy as np
import functools

from tensorpack.dataflow import MultiProcessMapDataZMQ, TestDataSpeed
from tensorpack.dataflow.common import MapData

from dataset.augmentors import CropAug, FlipAug, ScaleAug, RotateAug, ResizeAug
from dataset.base_dataflow import CocoDataFlow, JointsLoader
from dataset.dataflow_steps import create_all_mask, augment, read_img, apply_mask, gen_mask
from dataset.label_maps import create_heatmap, create_paf


def build_sample(components, y_size):
    """
    Builds a sample for a model.

    :param components: components
    :return: list of final components of a sample.
    """
    img = components[10]
    aug_joints = components[13]

    heatmap = create_heatmap(JointsLoader.num_joints_and_bkg, y_size, y_size,
                             aug_joints, 7.0, stride=8)

    pafmap = create_paf(JointsLoader.num_connections, y_size, y_size,
                        aug_joints, 1, stride=8)

    return [img,
            pafmap,
            heatmap]


def build_sample_with_masks(components, y_size):
    """
    Builds a sample for a model.

    :param components: components
    :return: list of final components of a sample.
    """
    img = components[10]
    mask = components[11]
    aug_joints = components[13]

    if mask is None:
        mask_paf = np.repeat(np.ones((y_size, y_size, 1), dtype=np.uint8), 38, axis=2)
        mask_heatmap = np.repeat(np.ones((y_size, y_size, 1), dtype=np.uint8), 19, axis=2)
    else:
        mask_paf = create_all_mask(mask, 38, stride=8)
        mask_heatmap = create_all_mask(mask, 19, stride=8)

    heatmap = create_heatmap(JointsLoader.num_joints_and_bkg, y_size, y_size,
                             aug_joints, 7.0, stride=8)

    pafmap = create_paf(JointsLoader.num_connections, y_size, y_size,
                        aug_joints, 1, stride=8)

    return [img.astype(np.float32),
            mask_paf.astype(np.float32),
            mask_heatmap.astype(np.float32),
            pafmap,
            heatmap]


def get_dataflow_vgg(annot_path, img_dir, strict, x_size, y_size, include_outputs_masks=False):
    """
    This function initializes the tensorpack dataflow and serves generator
    for training operation.

    :param annot_path: path to the annotation file
    :param img_dir: path to the images
    :return: dataflow object
    """
    coco_crop_size = 368

    # configure augmentors

    augmentors = [
        ScaleAug(scale_min=0.5,
                 scale_max=1.1,
                 target_dist=0.6,
                 interp=cv2.INTER_CUBIC),

        RotateAug(rotate_max_deg=40,
                  interp=cv2.INTER_CUBIC,
                  border=cv2.BORDER_CONSTANT,
                  border_value=(128, 128, 128), mask_border_val=1),

        CropAug(coco_crop_size, coco_crop_size, center_perterb_max=40, border_value=128,
                mask_border_val=1),

        FlipAug(num_parts=18, prob=0.5)
    ]

    if x_size != coco_crop_size:
        augmentors.append(ResizeAug(x_size, x_size))

    # prepare augment function

    augment_func = functools.partial(augment,
                                     augmentors=augmentors)

    # prepare building sample function

    if include_outputs_masks:
        build_sample_func = functools.partial(build_sample_with_masks,
                                              y_size=y_size)
    else:
        build_sample_func = functools.partial(build_sample,
                                          y_size=y_size)

    # build the dataflow

    df = CocoDataFlow((coco_crop_size, coco_crop_size), annot_path, img_dir)
    df.prepare()
    size = df.size()
    df = MapData(df, read_img)
    df = MapData(df, augment_func)

    df = MultiProcessMapDataZMQ(df, num_proc=4, map_func=build_sample_func, buffer_size=200, strict=strict)

    return df, size


def get_dataflow_mobilenet(annot_path, img_dir, strict, x_size = 224, y_size = 28):
    """
    This function initializes the tensorpack dataflow and serves generator
    for training operation.

    :param annot_path: path to the annotation file
    :param img_dir: path to the images
    :return: dataflow object
    """
    coco_crop_size = 368

    # configure augmentors

    augmentors = [
        ScaleAug(scale_min=0.5,
                 scale_max=1.1,
                 target_dist=0.6,
                 interp=cv2.INTER_CUBIC),

        RotateAug(rotate_max_deg=40,
                  interp=cv2.INTER_CUBIC,
                  border=cv2.BORDER_CONSTANT,
                  border_value=(128, 128, 128), mask_border_val=1),

        CropAug(coco_crop_size, coco_crop_size, center_perterb_max=40, border_value=128,
                mask_border_val=1),

        FlipAug(num_parts=18, prob=0.5),

        ResizeAug(x_size, x_size)

    ]

    # prepare augment function

    augment_func = functools.partial(augment,
                                     augmentors=augmentors)

    # prepare building sample function

    build_sample_func = functools.partial(build_sample,
                                          y_size=y_size)

    # build the dataflow

    df = CocoDataFlow((coco_crop_size, coco_crop_size), annot_path, img_dir)
    df.prepare()
    size = df.size()
    df = MapData(df, read_img)
    df = MapData(df, augment_func)
    df = MultiProcessMapDataZMQ(df, num_proc=4, map_func=build_sample_func, buffer_size=200, strict=strict)

    return df, size


if __name__ == '__main__':
    """
    Run this script to check speed of generating samples. Tweak the nr_proc
    parameter of PrefetchDataZMQ. Ideally it should reflect the number of cores 
    in your hardware
    """
    batch_size = 10
    curr_dir = os.path.dirname(__file__)
    annot_path = os.path.join(curr_dir, '../../datasets/coco_2017_dataset/annotations/person_keypoints_val2017.json')
    img_dir = os.path.abspath(os.path.join(curr_dir, '../../datasets/coco_2017_dataset/val2017/'))

    df1, size1 = get_dataflow_mobilenet(annot_path, img_dir, False, x_size=224, y_size=28)
    df2, size2 = get_dataflow_vgg(annot_path, img_dir, False, x_size=368, y_size=46)

    TestDataSpeed(df1, size=100).start()
    TestDataSpeed(df2, size=100).start()
