import tensorflow as tf

from dataset.dataflows import get_dataflow_vgg, get_dataflow_mobilenet


def get_dataset_vgg(annot_path, img_dir, batch_size, strict=False):
    def gen(df):
        def f():
            for i in df:
                yield tuple(i)
        return f

    df, size = get_dataflow_vgg(
        annot_path=annot_path,
        img_dir=img_dir,
        strict=strict,
        x_size=368,
        y_size=46,
        include_outputs_masks=False
    )
    df.reset_state()
    ds = tf.data.Dataset.from_generator(
        gen(df), (tf.float32, tf.float32, tf.float32),
        output_shapes=(
            tf.TensorShape([368, 368, 3]),
            tf.TensorShape([46, 46, 38]),
            tf.TensorShape([46, 46, 19])
            )
    )
    ds = ds.map(lambda x0, x1, x2 : (x0, (x1, x2, x1, x2, x1, x2, x1, x2, x1, x2, x1, x2)))
    ds = ds.batch(batch_size)

    return ds, size


def get_dataset_vgg_with_masks(annot_path, img_dir, batch_size, strict=False):
    def gen(df):
        def f():
            for i in df:
                yield tuple(i)
        return f

    df, size = get_dataflow_vgg(
        annot_path=annot_path,
        img_dir=img_dir,
        strict=strict,
        x_size=368,
        y_size=46,
        include_outputs_masks=True

    )
    df.reset_state()
    ds = tf.data.Dataset.from_generator(
        gen(df), (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
        output_shapes=(
            tf.TensorShape([368, 368, 3]),
            tf.TensorShape([46, 46, 38]),
            tf.TensorShape([46, 46, 19]),
            tf.TensorShape([46, 46, 38]),
            tf.TensorShape([46, 46, 19])
            )
    )

    ds = ds.map(lambda x0, x1, x2, x3, x4: ((x0, x1, x2), (x3, x4, x3, x4, x3, x4, x3, x4, x3, x4, x3, x4)))
    ds = ds.batch(batch_size)

    return ds, size


def get_dataset_mobilenet(annot_path, img_dir, batch_size, strict = False):
    def gen(df):
        def f():
            for i in df:
                yield tuple(i)
        return f

    df, size = get_dataflow_mobilenet(
        annot_path=annot_path,
        img_dir=img_dir,
        strict=strict,
        x_size=224,
        y_size=28
    )
    df.reset_state()

    ds = tf.data.Dataset.from_generator(
        gen(df), (tf.float32, tf.float32, tf.float32),
        output_shapes=(
            tf.TensorShape([224, 224, 3]),
            tf.TensorShape([28, 28, 38]),
            tf.TensorShape([28, 28, 19])
            )
    )

    ds = ds.map(lambda x0, x1, x2: (x0, (x1, x2, x1, x2)))
    ds = ds.batch(batch_size)

    return ds, size
