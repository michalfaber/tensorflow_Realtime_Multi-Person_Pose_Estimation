import numpy as np
import cv2

from tensorpack.dataflow.imgaug.base import ImageAugmentor
from tensorpack.dataflow.imgaug.transform import Transform, ResizeTransform
from tensorpack.dataflow.imgaug.geometry import WarpAffineTransform


class AugImgMetadata:
    """
    Holder for data required for augmentation - subset of metadata
    """
    __slots__ = ["img", "mask", "center", "scale"]

    def __init__(self, img, mask, center, scale):
        self.img = img
        self.mask = mask
        self.center = center
        self.scale = scale

    def update_img(self, new_img, new_mask):
        return AugImgMetadata(new_img, new_mask, self.center, self.scale)


def joints_to_point8(joints, num_p=18):
    """
    Converts joints structure to Nx2 nparray (format expected by tensorpack augmentors)
    Nx2 = floating point nparray where each row is (x, y)

    :param joints:
    :param num_p:
    :return: Nx2 nparray
    """
    segment = np.zeros((num_p * len(joints), 2), dtype=np.float32)

    for idx_all, j_list in enumerate(joints):
        for idx, k in enumerate(j_list):
            if k:
                segment[idx_all * num_p + idx, 0] = k[0]
                segment[idx_all * num_p + idx, 1] = k[1]
            else:
                segment[idx_all * num_p + idx, 0] = -1000000
                segment[idx_all * num_p + idx, 1] = -1000000

    return segment


def point8_to_joints(points, num_p=18):
    """
    Converts Nx2 nparray to the list of joints

    :param points:
    :param num_p:
    :return: list of joints [[(x1,y1), (x2,y2), ...], []]
    """
    l = points.shape[0] // num_p

    all = []
    for i in range(l):
        skel = []
        for j in range(num_p):
            idx = i * num_p + j
            x = points[idx, 0]
            y = points[idx, 1]

            if x <= 0 or y <= 0 or x > 2000 or y > 2000:
                skel.append(None)
            else:
                skel.append((x, y))

        all.append(skel)
    return all


class FlipTransform(Transform):
    def __init__(self, num_parts, doit, img_width):
        super(FlipTransform, self).__init__()
        self._init(locals())

    def apply_image(self, meta):
        img = meta.img
        mask = meta.mask

        if self.doit:
            new_img = cv2.flip(img, 1)
            if img.ndim == 3 and new_img.ndim == 2:
                new_img = new_img[:, :, np.newaxis]

            if mask is not None:
                new_mask = cv2.flip(mask, 1)
            else:
                new_mask = None

            result = (new_img, new_mask)
        else:
            result = (img, mask)

        return result

    def apply_coords(self, coords):

        if self.doit:
            coords[:, 0] = self.img_width - coords[:, 0]

        return coords

    def recover_left_right(self, coords):
        """
        Recovers a few joints. After flip operation coordinates of some parts like
        left hand would land on the right side of a person so it is
        important to recover such positions.

        :param coords:
        :param param:
        :return:
        """
        if self.doit:
            right = [2, 3, 4, 8, 9, 10, 14, 16]
            left = [5, 6, 7, 11, 12, 13, 15, 17]

            for l_idx, r_idx in zip(left, right):
                idxs = range(0, coords.shape[0], self.num_parts)
                for idx in idxs:
                    tmp = coords[l_idx + idx, [0, 1]]
                    coords[l_idx + idx, [0, 1]] = coords[r_idx + idx, [0, 1]]
                    coords[r_idx + idx, [0, 1]] = tmp

        return coords


class FlipAug(ImageAugmentor):
    """
    Flips images and coordinates
    """
    def __init__(self, num_parts, prob=0.5):
        super(FlipAug, self).__init__()
        self._init(locals())

    def get_transform(self, meta):
        doit = self._rand_range() < self.prob
        _, w = meta.img.shape[:2]

        return FlipTransform(self.num_parts, doit, w)


class CropTransform(Transform):
    def __init__(self, crop_x, crop_y, left_up, border_value, mask_border_val):
        super(CropTransform, self).__init__()
        self._init(locals())

    def apply_image(self, meta):
        img = meta.img
        mask = meta.mask

        x1, y1 = self.left_up

        npblank = np.ones((self.crop_y, self.crop_x, 3), dtype=np.uint8) * self.border_value

        if x1 < 0:
            dx = -x1
        else:
            dx = 0

        if y1 < 0:
            dy = -y1
        else:
            dy = 0

        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0

        cropped = img[y1:y1+self.crop_y-dy, x1:x1+self.crop_x-dx, :]
        cropped_h, cropped_w = cropped.shape[:2]
        npblank[dy:dy+cropped_h, dx:dx+cropped_w, :] = cropped

        if mask is not None:
            new_mask = np.ones((self.crop_y, self.crop_x), dtype=np.uint8) \
                       * self.mask_border_val
            cropped_mask = mask[y1:y1 + self.crop_y - dy, x1:x1 + self.crop_x - dx]
            cropped_h, cropped_w = cropped_mask.shape[:2]
            new_mask[dy:dy + cropped_h, dx:dx + cropped_w] = cropped_mask
        else:
            new_mask = mask

        return npblank, new_mask

    def apply_coords(self, coords):
        coords[:, 0] -= self.left_up[0]
        coords[:, 1] -= self.left_up[1]

        return coords


class CropAug(ImageAugmentor):
    """
    Crops images and coordinates
    """
    
    def __init__(self, crop_x, crop_y, center_perterb_max=40, border_value=0, mask_border_val=0):
        super(CropAug, self).__init__()
        self._init(locals())

    def get_transform(self, meta):
        center = meta.center

        x_offset = int(self._rand_range(-0.5, 0.5) * 2 * self.center_perterb_max)
        y_offset = int(self._rand_range(-0.5, 0.5) * 2 * self.center_perterb_max)

        center_x = center[0, 0] + x_offset
        center_y = center[0, 1] + y_offset

        left_up = (int(center_x - self.crop_x / 2),
                   int(center_y - self.crop_y / 2))

        return CropTransform(self.crop_x, self.crop_y, left_up, self.border_value, self.mask_border_val)


class ScaleTransform(ResizeTransform):
    """
    Resize the image.
    """
    def __init__(self, h, w, new_h, new_w, interp):
        super(ScaleTransform, self).__init__(h, w, new_h, new_w, interp)

    def apply_image(self, meta):
        new_img = super(ScaleTransform, self).apply_image(meta.img)
        if meta.mask is not None:
            new_mask = super(ScaleTransform, self).apply_image(meta.mask)
        else:
            new_mask = None

        return new_img, new_mask

    def apply_coords(self, coords):
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords


class ScaleAug(ImageAugmentor):
    def __init__(self, scale_min, scale_max, target_dist = 1.0, interp=cv2.INTER_CUBIC):
        super(ScaleAug, self).__init__()
        self._init(locals())

    def get_transform(self, meta):
        img = meta.img
        scale = meta.scale

        h, w = img.shape[:2]

        scale_multiplier = self._rand_range(self.scale_min, self.scale_max)

        scale_abs = self.target_dist / scale

        scale = scale_abs * scale_multiplier

        new_h, new_w = int(scale * h + 0.5), int(scale * w + 0.5)

        return ScaleTransform(
            h, w, new_h, new_w, self.interp)


class ResizeAug(ImageAugmentor):
    def __init__(self, new_w, new_h, interp=cv2.INTER_CUBIC):
        super(ResizeAug, self).__init__()
        self._init(locals())

    def get_transform(self, meta):
        img = meta.img

        h, w = img.shape[:2]

        return ScaleTransform(
            h, w, self.new_h, self.new_w, self.interp)


class RotateTransform(WarpAffineTransform):
    
    def __init__(self, mat, dsize, interp=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0, mask_border_val=0):
        super(RotateTransform, self).__init__(mat, dsize, interp, borderMode, borderValue)
        self.mask_border_val = mask_border_val

    def apply_image(self, meta):
        new_img = super(RotateTransform, self).apply_image(meta.img)

        if meta.mask is not None:
            self.borderValue = self.mask_border_val
            new_mask = super(RotateTransform, self).apply_image(meta.mask)
        else:
            new_mask = None

        return new_img, new_mask

    def apply_coords(self, coords):
        coords = np.concatenate((coords, np.ones((coords.shape[0], 1), dtype='f4')), axis=1)
        coords = np.dot(coords, self.mat.T)
        return coords


class RotateAug(ImageAugmentor):
    """
    Rotates images and coordinates
    """
    def __init__(self, scale=None, translate_frac=None, rotate_max_deg=0.0, shear=0.0,
                 interp=cv2.INTER_LINEAR, border=cv2.BORDER_REPLICATE, border_value=0, mask_border_val=0):

        super(RotateAug, self).__init__()
        self._init(locals())

    def get_transform(self, meta):
        img = meta.img

        # grab the rotation matrix
        (h, w) = img.shape[:2]
        (center_x, center_y) = (w // 2, h // 2)
        deg = self._rand_range(-self.rotate_max_deg, self.rotate_max_deg)
        R = cv2.getRotationMatrix2D((center_x, center_y), deg, 1.0)

        # determine bounding box
        (h, w) = img.shape[:2]
        cos = np.abs(R[0, 0])
        sin = np.abs(R[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        R[0, 2] += (new_w / 2) - center_x
        R[1, 2] += (new_h / 2) - center_y

        return RotateTransform(R, (new_w, new_h),
                               self.interp, self.border, self.border_value, self.mask_border_val)


