import numpy as np
import tensorflow as tf


def load_data(cfg):

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # [0,255] -> [0,1] -> [-1,1]
    x_train = (x_train/255.) * 2. - 1.

    x_train = np.expand_dims(x_train, axis=3)
    x_train = tf.cast(x_train, dtype=tf.float32)

    if cfg['model'] == 'simple_gan':
        train_ds = tf.data.Dataset.from_tensor_slices(x_train)
    else:
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(1000).batch(cfg['batch_size'])
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds
    