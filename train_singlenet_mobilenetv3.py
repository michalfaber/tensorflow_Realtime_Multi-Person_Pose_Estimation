import datetime
from datetime import timedelta
from timeit import default_timer as timer

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from tf_netbuilder.files import download_checkpoint
from tf_netbuilder_ext.extensions import register_tf_netbuilder_extensions
from models import create_openpose_singlenet
from dataset.generators import get_dataset
from util import plot_to_image, probe_model_singlenet


pretrained_mobilenet_v3_url = "https://github.com/michalfaber/tf_netbuilder/releases/download/v1.0/mobilenet_v3_224_1_0.zip"
annot_path_train = '../datasets/coco_2017_dataset/annotations/person_keypoints_train2017.json'
img_dir_train = '../datasets/coco_2017_dataset/train2017/'
annot_path_val = '../datasets/coco_2017_dataset/annotations/person_keypoints_val2017.json'
img_dir_val = '../datasets/coco_2017_dataset/val2017/'
checkpoints_folder = './tf_ckpts_singlenet'
output_weights = 'output_singlenet/openpose_singlenet'
batch_size = 10
lr = 2.5e-5
max_epochs = 300


def eucl_loss(y_true, y_pred):
    return tf.reduce_sum(tf.math.squared_difference(y_pred, y_true))


@tf.function
def train_one_step(model, optimizer, x, y_true):
    with tf.GradientTape() as tape:
        y_pred = model(x)

        losses = [eucl_loss(y_true[0], y_pred[0]),
                  eucl_loss(y_true[0], y_pred[1]),
                  eucl_loss(y_true[0], y_pred[2]),
                  eucl_loss(y_true[1], y_pred[3])
                  ]

        total_loss = tf.reduce_sum(losses)

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return losses, total_loss


def train(ds_train, ds_val, model, optimizer, ckpt, last_epoch, last_step, max_epochs, steps_per_epoch):
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_loss_heatmap = tf.keras.metrics.Mean('train_loss_heatmap', dtype=tf.float32)
    train_loss_paf = tf.keras.metrics.Mean('train_loss_paf', dtype=tf.float32)

    val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
    val_loss_heatmap = tf.keras.metrics.Mean('val_loss_heatmap', dtype=tf.float32)
    val_loss_paf = tf.keras.metrics.Mean('val_loss_paf', dtype=tf.float32)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs_singlenet/gradient_tape/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_log_dir = 'logs_singlenet/gradient_tape/' + current_time + '/val'
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    output_paf_idx = 2
    output_heatmap_idx = 3

    # determine start epoch in case the training has been stopped manually and resumed

    resume = last_step != 0 and (steps_per_epoch - last_step) != 0
    if resume:
        start_epoch = last_epoch
    else:
        start_epoch = last_epoch + 1

    # start processing

    for epoch in range(start_epoch, max_epochs + 1, 1):

        start = timer()

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

                tf.print('Epoch', epoch, f'Step {step}/{steps_per_epoch}', 'Paf1', losses[0], 'Paf2', losses[1], 'Paf3', losses[2],
                         'Heatmap', losses[3], 'Total loss', total_loss)

                with train_summary_writer.as_default():
                    summary_step = (epoch - 1) * steps_per_epoch + step - 1
                    tf.summary.scalar('loss', train_loss.result(), step=summary_step)
                    tf.summary.scalar('loss_heatmap', train_loss_heatmap.result(), step=summary_step)
                    tf.summary.scalar('loss_paf', train_loss_paf.result(), step=summary_step)

            if step % 100 == 0:
                figure = probe_model_singlenet(model, test_img_path="resources/ski_224.jpg")
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
        model.save_weights(output_weights, overwrite=True)

        # save checkpoint at the end of an epoch

        ckpt.step.assign(step)
        ckpt.epoch.assign(epoch)
        manager.save()

        # reset metrics every epoch

        train_loss.reset_states()
        train_loss_heatmap.reset_states()
        train_loss_paf.reset_states()

        end = timer()

        print("Epoch training time: " + str(timedelta(seconds=end - start)))

        # calculate validation loss

        print("Calculating validation losses...")
        for val_step, (x_val, y_val_true) in enumerate(ds_val):

            if val_step % 1000 == 0:
                print(f"Validation step {val_step} ...")

            y_val_pred = model(x_val)
            losses = [eucl_loss(y_val_true[0], y_val_pred[0]),
                      eucl_loss(y_val_true[0], y_val_pred[1]),
                      eucl_loss(y_val_true[0], y_val_pred[2]),
                      eucl_loss(y_val_true[1], y_val_pred[3])]
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

    # registering custom blocks types

    register_tf_netbuilder_extensions()

    # loading datasets

    ds_train, ds_train_size = get_dataset(annot_path_train, img_dir_train, batch_size)
    ds_val, ds_val_size = get_dataset(annot_path_val, img_dir_val, batch_size, strict=True)

    print(f"Training samples: {ds_train_size} , Validation samples: {ds_val_size}")

    steps_per_epoch = ds_train_size // batch_size
    steps_per_epoch_val = ds_val_size // batch_size

    # creating model, optimizers etc

    model = create_openpose_singlenet(pretrained=False)
    optimizer = Adam(lr)

    # loading previous state if required

    ckpt = tf.train.Checkpoint(step=tf.Variable(0), epoch=tf.Variable(0), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, checkpoints_folder, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    last_step = int(ckpt.step)
    last_epoch = int(ckpt.epoch)

    if manager.latest_checkpoint:
        print(f"Restored from {manager.latest_checkpoint}")
        print(f"Resumed from epoch {last_epoch}, step {last_step}")
    else:
        print("Initializing from scratch.")

        path = download_checkpoint(pretrained_mobilenet_v3_url)
        model.load_weights(path).expect_partial()

    # training loop

    train(ds_train, ds_val, model, optimizer, ckpt, last_epoch, last_step,
          max_epochs, steps_per_epoch)

