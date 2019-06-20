"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.algorithms.base import Base
from jetpack.functions.policy import Policy


class VPG(Base):

    def __init__(
        self,
        policy: Policy,
        gamma=1.0,
        monitor=None,
    ):
        self.policy = policy
        self.gamma = gamma
        self.iteration = 0
        self.monitor = monitor
        self.get_loss_policy = tf.keras.losses.MeanSquaredError()

    def gradient_update(
        self, 
        observations,
        actions,
        rewards,
        lengths
    ):
        if self.monitor is not None:
            self.monitor.set_step(self.iteration)
        self.iteration += 1
        rewards * tf.cast(
            tf.range(tf.shape(rewards)[1])[tf.newaxis, :] < lengths[:, tf.newaxis],
            rewards.dtype)
        weights = tf.tile([[self.gamma]], [1, tf.shape(rewards)[1]])
        weights = tf.math.cumprod(weights, axis=1, exclusive=True)
        returns = tf.math.cumsum(rewards * weights) / weights
        if self.monitor is not None:
            self.monitor.record("rewards_mean", tf.reduce_mean(rewards))
            self.monitor.record("returns_mean", tf.reduce_mean(returns))
        with tf.GradientTape() as tape_policy:
            means = self.policy.get_deterministic_actions(
                observations)[:, :(-1), :]
            loss_policy = tf.reduce_mean(
                self.get_loss_policy(actions, means, sample_weight=(
                    returns[:, :, tf.newaxis])))
            self.policy.minimize(
                loss_policy,
                tape_policy)
            if self.monitor is not None:
                self.monitor.record("loss_policy", tf.reduce_mean(loss_policy))

