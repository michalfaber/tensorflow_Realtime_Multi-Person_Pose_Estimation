import datetime

import cv2
import re
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Conv2D

from dataset.generators import get_dataset_vgg_with_masks
from optimizers.multisgd import MultiSGD
from models.cmu_model import CmuModel
from util import plot_to_image

batch_size = 10
weights_best_file = "weights.best.h5"
decay_rate = 0.333
decay_steps = 136106 / batch_size  # basically it means that dacay will be triggered every 3 epochs
base_lr = 4e-5
max_epochs = 100

from_vgg = {
    'conv1_1': 'block1_conv1',
    'conv1_2': 'block1_conv2',
    'conv2_1': 'block2_conv1',
    'conv2_2': 'block2_conv2',
    'conv3_1': 'block3_conv1',
    'conv3_2': 'block3_conv2',
    'conv3_3': 'block3_conv3',
    'conv3_4': 'block3_conv4',
    'conv4_1': 'block4_conv1',
    'conv4_2': 'block4_conv2'
}


def load_initial_vgg19_weights(model):
    """
    """
    print("Loading vgg19 weights...")

    vgg_model = VGG19(include_top=False, weights='imagenet')
    for layer in model.layers:
        if layer.name in from_vgg:
            vgg_layer_name = from_vgg[layer.name]
            layer.set_weights(vgg_model.get_layer(vgg_layer_name).get_weights())
            print("Loaded VGG19 layer: " + vgg_layer_name)


def get_lr_multipliers(model):
    lr_mult = dict()
    for layer in model.layers:
        if isinstance(layer, Conv2D):
            # stage = 1
            if re.match("conv_stage1_\d", layer.name):
                kernel_name = layer.weights[0].name
                bias_name = layer.weights[1].name
                lr_mult[kernel_name] = 1
                lr_mult[bias_name] = 2

            # stage > 1
            elif re.match("conv_stage\d_\d", layer.name):
                kernel_name = layer.weights[0].name
                bias_name = layer.weights[1].name
                lr_mult[kernel_name] = 4
                lr_mult[bias_name] = 8

            # vgg
            else:
                kernel_name = layer.weights[0].name
                bias_name = layer.weights[1].name
                lr_mult[kernel_name] = 1
                lr_mult[bias_name] = 2
    return lr_mult


def probe_model(model, test_img_path="sample_images/ski_smaller.jpg"):
    img = cv2.imread(test_img_path)  # B,G,R order
    input_img = np.transpose(np.float32(img[:, :, :, np.newaxis]), (3, 0, 1, 2))

    mask_heatmap_output = np.ones((1, 46, 46, 19), np.float32)
    mask_paf_output = np.ones((1, 46, 46, 38), np.float32)

    output_blobs = model.predict([input_img, mask_paf_output, mask_heatmap_output])

    pafs_first = output_blobs[0]
    heatmaps_first = output_blobs[1]
    pafs_last = output_blobs[10]
    heatmaps_last = output_blobs[11]

    figure = plt.figure(figsize=(10, 10))

    plt.subplot(2, 2, 1, title='stage 1, heatmap 0')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(heatmaps_first[0, :, :, 0], cmap='gray')

    plt.subplot(2, 2, 2, title='stage 1, paf 0')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(pafs_first[0, :, :, 0], cmap='gray')

    plt.subplot(2, 2, 3, title='stage 6, heatmap 0')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(heatmaps_last[0, :, :, 0], cmap='gray')

    plt.subplot(2, 2, 4, title='stage 6, paf 0')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(pafs_last[0, :, :, 0], cmap='gray')

    return figure


def eucl_loss(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.reduce_sum(tf.math.squared_difference(y_pred, y_true)) / batch_size / 2


def train_one_step(model, optimizer, x, y_true):
    with tf.GradientTape() as tape:
        y_pred = model(x)

        losses = [eucl_loss(y_true[0], y_pred[0]),
            eucl_loss(y_true[1], y_pred[1]),
            eucl_loss(y_true[2], y_pred[2]),
            eucl_loss(y_true[3], y_pred[3]),
            eucl_loss(y_true[4], y_pred[4]),
            eucl_loss(y_true[5], y_pred[5]),
            eucl_loss(y_true[6], y_pred[6]),
            eucl_loss(y_true[7], y_pred[7]),
            eucl_loss(y_true[8], y_pred[8]),
            eucl_loss(y_true[9], y_pred[9]),
            eucl_loss(y_true[10], y_pred[10]),
            eucl_loss(y_true[11], y_pred[11])]

        total_loss = tf.reduce_sum(losses)

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return losses, total_loss


def train(ds_train, ds_val, model, optimizer, ckpt, saved_weights_path, last_epoch, last_step, max_epochs, steps_per_epoch):

    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_loss_heatmap = tf.keras.metrics.Mean('train_loss_heatmap', dtype=tf.float32)
    train_loss_paf = tf.keras.metrics.Mean('train_loss_paf', dtype=tf.float32)

    val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
    val_loss_heatmap = tf.keras.metrics.Mean('val_loss_heatmap', dtype=tf.float32)
    val_loss_paf = tf.keras.metrics.Mean('val_loss_paf', dtype=tf.float32)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_log_dir = 'logs/gradient_tape/' + current_time + '/val'
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    output_paf_idx = 10
    output_heatmap_idx = 11

    # determine start epoch in case the training has been stopped manually and resumed

    resume = last_step != 0 and (steps_per_epoch - last_step) != 0
    if resume:
        start_epoch = last_epoch
    else:
        start_epoch = last_epoch + 1

    # start processing

    for epoch in range(start_epoch, max_epochs + 1, 1):

        print("Start processing epoch {}".format(epoch))

        # set the initial step index depending on if you resumed the processing

        if resume:
            step = last_step + 1
            data_iter = ds_train.skip(last_step)
            print(f"Skipping {last_step} steps (May take a few minutes)...")
            resume = False
        else:
            step = 0
            data_iter = ds_train

        # process steps

        for x, y in data_iter:

            step += 1

            losses, total_loss = train_one_step(model, optimizer, x, y)

            train_loss(total_loss)
            train_loss_heatmap(losses[output_heatmap_idx])
            train_loss_paf(losses[output_paf_idx])

            if step % 10 == 0:

                tf.print('Epoch', epoch, f'Step {step}/{steps_per_epoch}', ': Loss paf', losses[10],
                         'Loss heatmap', losses[11], 'Total loss', total_loss)

                with train_summary_writer.as_default():
                    summary_step = (epoch - 1) * steps_per_epoch + step - 1
                    tf.summary.scalar('loss', train_loss.result(), step=summary_step)
                    tf.summary.scalar('loss_heatmap', train_loss_heatmap.result(), step=summary_step)
                    tf.summary.scalar('loss_paf', train_loss_paf.result(), step=summary_step)

            if step % 100 == 0:
                figure = probe_model(model)
                with train_summary_writer.as_default():
                    tf.summary.image("Test prediction", plot_to_image(figure), step=step)

            if step % 1000 == 0:
                ckpt.step.assign(step)
                ckpt.epoch.assign(epoch)
                save_path = manager.save()
                print("Saved checkpoint for step {}: {}".format(step, save_path))

            if step >= steps_per_epoch:
                break

        print("Completed epoch {}. Saving weights...".format(epoch))
        model.save_weights(saved_weights_path, overwrite=True)

        # save checkpoint at the end of an epoch

        ckpt.step.assign(step)
        ckpt.epoch.assign(epoch)
        manager.save()

        # reset metrics every epoch

        train_loss.reset_states()
        train_loss_heatmap.reset_states()
        train_loss_paf.reset_states()

        # calculate validation loss

        print("Calculating validation losses...")
        for val_step, (x_val, y_val_true) in enumerate(ds_val):

            if val_step % 1000 == 0:
                print(f"Validation step {val_step} ...")

            y_val_pred = model(x_val)
            losses = [eucl_loss(y_val_true[0], y_val_pred[0]),
                      eucl_loss(y_val_true[1], y_val_pred[1]),
                      eucl_loss(y_val_true[2], y_val_pred[2]),
                      eucl_loss(y_val_true[3], y_val_pred[3]),
                      eucl_loss(y_val_true[4], y_val_pred[4]),
                      eucl_loss(y_val_true[5], y_val_pred[5]),
                      eucl_loss(y_val_true[6], y_val_pred[6]),
                      eucl_loss(y_val_true[7], y_val_pred[7]),
                      eucl_loss(y_val_true[8], y_val_pred[8]),
                      eucl_loss(y_val_true[9], y_val_pred[9]),
                      eucl_loss(y_val_true[10], y_val_pred[10]),
                      eucl_loss(y_val_true[11], y_val_pred[11])]
            total_loss = tf.reduce_sum(losses)
            val_loss(total_loss)
            val_loss_heatmap(losses[output_heatmap_idx])
            val_loss_paf(losses[output_paf_idx])

        val_loss_res = val_loss.result()
        val_loss_heatmap_res = val_loss_heatmap.result()
        val_loss_paf_res = val_loss_paf.result()

        print(f'Validation losses for epoch: {epoch} : Loss paf {val_loss_paf_res}, Loss heatmap '
              f'{val_loss_heatmap_res}, Total loss {val_loss_res}')

        with val_summary_writer.as_default():
            tf.summary.scalar('val_loss', val_loss_res, step=epoch)
            tf.summary.scalar('val_loss_heatmap', val_loss_heatmap_res, step=epoch)
            tf.summary.scalar('val_loss_paf', val_loss_paf_res, step=epoch)
        val_loss.reset_states()
        val_loss_heatmap.reset_states()
        val_loss_paf.reset_states()


if __name__ == '__main__':

    annot_path_train = '../datasets/coco_2017_dataset/annotations/person_keypoints_train2017.json'
    img_dir_train = '../datasets/coco_2017_dataset/train2017/'
    annot_path_val = '../datasets/coco_2017_dataset/annotations/person_keypoints_val2017.json'
    img_dir_val = '../datasets/coco_2017_dataset/val2017/'

    ds_train, ds_train_size = get_dataset_vgg_with_masks(annot_path_train, img_dir_train, batch_size)
    ds_val, ds_val_size = get_dataset_vgg_with_masks(annot_path_val, img_dir_val, batch_size, strict=True)

    print(f"Training samples: {ds_train_size} , Validation samples: {ds_val_size}")

    steps_per_epoch = ds_train_size // batch_size
    steps_per_epoch_val = ds_val_size // batch_size

    model = CmuModel(masked_outputs=True)
    model.build(input_shape=[tf.TensorShape([None, 368, 368, 3]),
            tf.TensorShape([None, 46, 46, 38]),
            tf.TensorShape([None, 46, 46, 19])])

    lr_multipliers = get_lr_multipliers(model)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        base_lr,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True)
    multisgd = MultiSGD(learning_rate=lr_schedule, momentum=0.9, decay=0.0,
                             nesterov=False, lr_multi=lr_multipliers)

    ckpt = tf.train.Checkpoint(step=tf.Variable(0), epoch=tf.Variable(0), optimizer=multisgd, net=model)
    manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    last_step = int(ckpt.step)
    last_epoch = int(ckpt.epoch)

    if manager.latest_checkpoint:
        print(f"Restored from {manager.latest_checkpoint}")
        print(f"Resumed from epoch {last_epoch}, step {last_step}")
    else:
        print("Initializing from scratch.")
        load_initial_vgg19_weights(model)

    train(ds_train, ds_val, model, multisgd, ckpt, weights_best_file, last_epoch, last_step,
          max_epochs, steps_per_epoch)
