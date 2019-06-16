"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf


class Monitor(object):

    def __init__(
        self,
        logging_dir
    ):
        self.writer = tf.summary.create_file_writer(logging_dir)
        self.set_step(0)

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