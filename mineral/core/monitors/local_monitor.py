"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from tensorboard import program
from mineral.core.monitors.monitor import Monitor
from mineral.core.monitors import plot_to_tensor


class LocalMonitor(Monitor):

    def __init__(
        self,
        logging_dir
    ):
        tf.io.gfile.makedirs(logging_dir)
        self.writer = tf.summary.create_file_writer(logging_dir)
        self.set_step(0)
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', logging_dir])
        self.url = tb.launch()

    def set_step(
        self,
        step
    ):
        tf.summary.experimental.set_step(step)

    def record(
        self,
        key,
        value,
    ):
        with self.writer.as_default():
            if len(tf.shape(value)) == 0:
                tf.summary.scalar(key, value)
            elif len(tf.shape(value)) == 1:
                splits = key.split(",")
                tf.summary.image(splits[0], plot_to_tensor(
                    tf.expand_dims(tf.range(tf.shape(value)[1]), 0),
                    tf.expand_dims(value, 0),
                    splits[0],
                    splits[1],
                    splits[2]))
            elif len(tf.shape(value)) == 2:
                splits = key.split(",")
                tf.summary.image(splits[0], plot_to_tensor(
                    tf.tile(tf.expand_dims(tf.range(tf.shape(value)[1]), 0), [tf.shape(value)[0], 1]),
                    value,
                    splits[0],
                    splits[1],
                    splits[2]))
            elif len(tf.shape(value)) == 3:
                tf.summary.image(key, tf.expand_dims(value, 0) * 0.5 + 0.5)
            elif len(tf.shape(value)) == 4:
                tf.summary.image(key, value * 0.5 + 0.5)
            else:
                tf.summary.scalar(key, value)
