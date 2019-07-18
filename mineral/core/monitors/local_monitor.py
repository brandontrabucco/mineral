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
        value = tf.reshape(value, [-1])
        with self.writer.as_default():
            if tf.size(value) > 1:
                splits = key.split(",")
                tf.summary.image(splits[0], plot_to_tensor(
                    tf.range(tf.size(value)),
                    value,
                    splits[0],
                    splits[1],
                    splits[2]
                ))
            else:
                tf.summary.scalar(key, value[0])