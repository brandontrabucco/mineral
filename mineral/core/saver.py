"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
import os


class Saver(object):

    def __init__(
        self,
        logging_dir,
        **models
    ):
        tf.io.gfile.makedirs(logging_dir)
        self.logging_dir = logging_dir
        self.models = models

    def __call__(
        self,
        iteration
    ):
        for name, model in self.models.items():
            model.save_weights(
                os.path.join(self.logging_dir,
                             name + ".ckpt"))

    def load(
        self,
        iteration
    ):
        for name, model in self.models.items():
            model.load_weights(
                os.path.join(self.logging_dir,
                             name + ".ckpt"))
