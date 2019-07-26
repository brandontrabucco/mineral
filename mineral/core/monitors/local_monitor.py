"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from tensorboard import program
from mineral.core.monitors.monitor import Monitor


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
            tf.summary.scalar(key, value)