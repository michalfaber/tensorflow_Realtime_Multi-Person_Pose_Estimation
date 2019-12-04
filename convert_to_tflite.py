import os
import tensorflow as tf

from models.mobilenet_model import get_mobilenet_model

alpha = 1.0
rows = 224


def load_from_checkpoint(checkpoint_dir, checkpoint_path=None):
    model = get_mobilenet_model(alpha, rows)
    ckpt = tf.train.Checkpoint(net=model)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=1)

    if checkpoint_path:
        ckpt.restore(checkpoint_path)
    else:
        ckpt.restore(manager.latest_checkpoint)

    return model


def load_from_weights(path):
    model = get_mobilenet_model(alpha, rows)
    model.load_weights(path)

    return model


def export_to_tflite(saved_model_dir, output_path):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir=saved_model_dir)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    open(output_path, "wb").write(tflite_model)


if __name__ == '__main__':

    prefix = '1'

    saved_model_dir = './exported_models/{}'.format(prefix)
    tflite_output_path = './exported_models/{}_tflite/pose_mobilenet.tflite'.format(prefix)

    # save model as saved_model

    model = load_from_checkpoint('./tf_ckpts_mobilenet/')
    #model = load_from_weights('./weights.best.mobilenet.h5')

    tf.saved_model.save(model, saved_model_dir)

    # export model to tflite

    os.makedirs(os.path.dirname(tflite_output_path))
    export_to_tflite(saved_model_dir, tflite_output_path)

    print("Done !!!")
