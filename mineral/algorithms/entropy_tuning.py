"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.algorithms.base import Base


class EntropyTuning(Base):

    def __init__(
        self,
        policy,
        target=-1.0,
        optimizer_class=tf.keras.optimizers.Adam,
        optimizer_kwargs={},
        **kwargs
    ):
        Base.__init__(self, **kwargs)
        self.policy = policy
        self.target = target
        self.optimizer = optimizer_class(**optimizer_kwargs)
        self.alpha = tf.Variable(1.0)

    def get_tuning_variable(self):
        return self.alpha

    def update_algorithm(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        def loss_function():
            policy_actions = self.policy.sample(
                observations[:, :(-1), ...])
            policy_entropy = -self.policy.get_log_probs(
                policy_actions,
                observations[:, :(-1), ...])
            loss_entropy = self.alpha * (
                policy_entropy - self.target)
            if self.monitor is not None:
                self.monitor.record(
                    "alpha",
                    self.alpha)
                self.monitor.record(
                    "entropy",
                    tf.reduce_mean(-policy_entropy))
                self.monitor.record(
                    "loss_entropy",
                    tf.reduce_mean(loss_entropy))
            return tf.reduce_mean(loss_entropy)
        self.optimizer.minimize(
            loss_function, self.alpha)
