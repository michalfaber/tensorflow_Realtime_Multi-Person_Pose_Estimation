import tensorflow as tf

from dataset.dataflows import get_dataflow, get_dataflow_vgg


def get_dataset_with_masks(annot_path, img_dir, batch_size, strict=False, x_size=368, y_size=46):
    def gen(df):
        def f():
            for i in df:
                yield tuple(i)
        return f

    df, size = get_dataflow_vgg(
        annot_path=annot_path,
        img_dir=img_dir,
        strict=strict,
        x_size=x_size,
        y_size=y_size,
        include_outputs_masks=True

    )
    df.reset_state()

    ds = tf.data.Dataset.from_generator(
        gen(df), (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
        output_shapes=(
            tf.TensorShape([x_size, x_size, 3]),
            tf.TensorShape([y_size, y_size, 38]),
            tf.TensorShape([y_size, y_size, 19]),
            tf.TensorShape([y_size, y_size, 38]),
            tf.TensorShape([y_size, y_size, 19]),
            )
    )

    ds = ds.map(lambda x0, x1, x2, x3, x4: ((x0, x1, x2), (x3, x4)))
    ds = ds.batch(batch_size)

    return ds, size


def get_dataset(annot_path, img_dir, batch_size, strict=False, x_size=224, y_size=28):
    def gen(df):
        def f():
            for i in df:
                yield tuple(i)
        return f

    df, size = get_dataflow(
        annot_path=annot_path,
        img_dir=img_dir,
        strict=strict,
        x_size=x_size,
        y_size=y_size
    )
    df.reset_state()

    ds = tf.data.Dataset.from_generator(
        gen(df), (tf.float32, tf.float32, tf.float32),
        output_shapes=(
            tf.TensorShape([x_size, x_size, 3]),
            tf.TensorShape([y_size, y_size, 38]),
            tf.TensorShape([y_size, y_size, 19])
            )
    )

    ds = ds.map(lambda x0, x1, x2: (x0, (x1, x2)))
    ds = ds.batch(batch_size)

    return ds, size
