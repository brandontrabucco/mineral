"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.algorithms.base import Base
from jetpack.functions.vf import VF


class GAE(Base):

    def __init__(
        self,
        vf: VF,
        gamma=1.0,
        lamb=1.0,
        monitor=None,
    ):
        self.vf = vf
        self.lamb = lamb
        self.gamma = gamma
        self.iteration = 0
        self.monitor = monitor

    def gradient_update(
        self, 
        observations,
        actions,
        rewards
    ):
        if self.monitor is not None:
            self.monitor.set_step(self.iteration)
        self.iteration += 1
        weights = tf.tile([[self.gamma]], [1, tf.shape(rewards)[1]])
        weights = tf.cumprod(weights, axis=1, exclusive=True)
        returns = tf.cumsum(rewards * weights) / weights
        values = self.vf(observations)
        with tf.GradientTape() as vf_tape:
            vf_loss = tf.losses.mean_squared_error(returns, values)
            self.vf.minimize(vf_loss, vf_tape)
            if self.monitor is not None:
                self.monitor.record("vf_loss", vf_loss)

