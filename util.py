import io
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# find connection in the specified sequence, center 29 is in the position 15
limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
           [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
           [1, 16], [16, 18], [3, 17], [6, 18]]

# the middle joints heatmap correpondence
hmapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22],
           [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52],
           [55, 56], [37, 38], [45, 46]]

# visualize
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
          [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def get_jet_color(v, vmin, vmax):
    c = np.zeros(3)
    if v < vmin:
        v = vmin
    if v > vmax:
        v = vmax
    dv = vmax - vmin
    if v < (vmin + 0.125 * dv):
        c[0] = 256 * (0.5 + (v * 4))  # B: 0.5 ~ 1
    elif v < (vmin + 0.375 * dv):
        c[0] = 255
        c[1] = 256 * (v - 0.125) * 4  # G: 0 ~ 1
    elif v < (vmin + 0.625 * dv):
        c[0] = 256 * (-4 * v + 2.5)  # B: 1 ~ 0
        c[1] = 255
        c[2] = 256 * (4 * (v - 0.375))  # R: 0 ~ 1
    elif v < (vmin + 0.875 * dv):
        c[1] = 256 * (-4 * v + 3.5)  # G: 1 ~ 0
        c[2] = 255
    else:
        c[2] = 256 * (-4 * v + 4.5)  # R: 1 ~ 0.5
    return c


def colorize(gray_img):
    out = np.zeros(gray_img.shape + (3,))
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            out[y, x, :] = get_jet_color(gray_img[y, x], 0, 1)
    return out


def pad_right_down_corner(img, stride, pad_value):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0  # up
    pad[1] = 0  # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride)  # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride)  # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :] * 0 + pad_value, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :] * 0 + pad_value, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :] * 0 + pad_value, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :] * 0 + pad_value, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad


def probe_model_singlenet(model, test_img_path):
    img = cv2.imread(test_img_path)  # B,G,R order
    input_img = img[np.newaxis, :, :, [2, 1, 0]]
    inputs = tf.convert_to_tensor(input_img)
    output_blobs = model.predict(inputs)

    paf1 = output_blobs[0]
    paf2 = output_blobs[1]
    paf3 = output_blobs[2]
    heatmap1 = output_blobs[3]

    figure = plt.figure(figsize=(10, 10))

    plt.subplot(2, 2, 1, title='stage 1 - paf')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(paf1[0, :, :, 0], cmap='gray')

    plt.subplot(2, 2, 2, title='stage 2 - paf')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(paf2[0, :, :, 0], cmap='gray')

    plt.subplot(2, 2, 3, title='stage 3 - paf')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(paf3[0, :, :, 0], cmap='gray')

    plt.subplot(2, 2, 4, title='stage 4 - heatmaps')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(heatmap1[0, :, :, 0], cmap='gray')

    return figure


def probe_model_2br_vgg(model, test_img_path):
    img = cv2.imread(test_img_path)  # B,G,R order
    input_img = img[np.newaxis, :, :, [2, 1, 0]]
    inputs = tf.convert_to_tensor(input_img)

    output_blobs = model.predict([
        inputs,
        tf.ones((1, 46, 46, 38), dtype=tf.dtypes.float32),
        tf.ones((1, 46, 46, 19), dtype=tf.dtypes.float32)
    ])

    paf1 = output_blobs[10]
    paf2 = output_blobs[8]
    heatmap1 = output_blobs[11]
    heatmap2 = output_blobs[9]

    figure = plt.figure(figsize=(10, 10))

    plt.subplot(2, 2, 1, title='stage 6 - paf')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(paf1[0, :, :, 0], cmap='gray')

    plt.subplot(2, 2, 2, title='stage 5 - paf')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(paf2[0, :, :, 0], cmap='gray')

    plt.subplot(2, 2, 3, title='stage 6 - heatmap')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(heatmap1[0, :, :, 0], cmap='gray')

    plt.subplot(2, 2, 4, title='stage 5 - heatmap')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(heatmap2[0, :, :, 0], cmap='gray')

    return figure
