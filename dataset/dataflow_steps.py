import cv2
import numpy as np

from pycocotools.coco import maskUtils

from dataset.augmentors import FlipTransform, joints_to_point8, point8_to_joints, AugImgMetadata

from dataset.base_dataflow import Meta


def read_img(components):
    """
    Loads image from meta.img_path. Assigns the image to
    the field img of the same meta instance.

    :param components: components
    :return: updated components
    """

    img_buf = open(components[0], 'rb').read()

    if not img_buf:
        raise Exception('image not read, path=%s' % components[0])

    arr = np.fromstring(img_buf, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    components[1], components[2] = img.shape[:2]
    components[10] = img

    return components


def gen_mask(components):
    """
    Generate masks based on the coco mask polygons.

    :param components: components
    :return: updated components
    """
    masks_segments = components[7]
    hh = components[1]
    ww = components[2]

    if masks_segments:
        mask_miss = np.ones((hh, ww), dtype=np.uint8)
        for seg in masks_segments:
            bin_mask = maskUtils.decode(seg)
            bin_mask = np.logical_not(bin_mask)
            mask_miss = np.bitwise_and(mask_miss, bin_mask)

        components[11] = mask_miss

    return components


def augment(components, augmentors):
    """
    Augmenting of images.

    :param components: components
    :return: updated components.
    """
    img_path = components[0]
    height = components[1]
    width = components[2]
    center = components[3]
    bbox = components[4]
    area = components[5]
    num_keypoints = components[6]
    masks_segments = components[7]
    scale = components[8]
    all_joints = components[9]
    img = components[10]
    mask = components[11]
    aug_center = components[12]
    aug_joints = components[13]

    meta = Meta(img_path, height, width, center, bbox,
                area, scale, num_keypoints)
    meta.masks_segments = masks_segments
    meta.all_joints = all_joints
    meta.img = img
    meta.mask = mask
    meta.aug_center = aug_center
    meta.aug_joints = aug_joints

    aug_center = meta.center.copy()
    aug_joints = joints_to_point8(meta.all_joints)

    for aug in augmentors:
        transformation = aug.get_transform(
            AugImgMetadata(img=meta.img,
                           mask=meta.mask,
                           center=aug_center,
                           scale=meta.scale))
        im, mask = transformation.apply_image(meta)

        # augment joints
        aug_joints = transformation.apply_coords(aug_joints)

        # after flipping horizontaly the left side joints and right side joints are also
        # flipped so we need to recover their orginal orientation.
        if isinstance(transformation, FlipTransform):
            aug_joints = transformation.recover_left_right(aug_joints)

        # augment center position
        aug_center = transformation.apply_coords(aug_center)

        meta.img = im
        meta.mask = mask

    meta.aug_joints = point8_to_joints(aug_joints)
    meta.aug_center = aug_center

    return [meta.img_path,
            meta.height,
            meta.width,
            meta.center,
            meta.bbox,
            meta.area,
            meta.num_keypoints,
            meta.masks_segments,
            meta.scale,
            meta.all_joints,
            meta.img,
            meta.mask,
            meta.aug_center,
            meta.aug_joints]


def apply_mask(components):
    """
    Applies the mask (if exists) to the image.

    :param components: components
    :return: updated components
    """
    img = components[10]
    mask = components[11]
    if mask is not None:
        img[:, :, 0] = img[:, :, 0] * mask
        img[:, :, 1] = img[:, :, 1] * mask
        img[:, :, 2] = img[:, :, 2] * mask
        img[img == 0] = 128
    return components


def create_all_mask(mask, num, stride):
    """
    Helper function to create a stack of scaled down mask.

    :param mask: mask image
    :param num: number of layers
    :param stride: parameter used to scale down the mask image because it has
    the same size as orginal image. We need the size of network output.
    :return:
    """
    scale_factor = 1.0 / stride
    small_mask = cv2.resize(mask, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    small_mask = small_mask[:, :, np.newaxis]
    return np.repeat(small_mask, num, axis=2)
