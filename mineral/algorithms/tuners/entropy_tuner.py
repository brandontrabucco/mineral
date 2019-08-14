"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.algorithms.tuners.tuner import Tuner


class EntropyTuner(Tuner):

    def __init__(
        self,
        policy,
        **kwargs
    ):
        Tuner.__init__(self, **kwargs)
        self.policy = policy

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
            loss_entropy = self.tuning_variable * (
                policy_entropy - self.target)
            if self.monitor is not None:
                self.monitor.record(
                    "entropy_tuning_variable",
                    self.tuning_variable)
                self.monitor.record(
                    "entropy",
                    tf.reduce_mean(-policy_entropy))
                self.monitor.record(
                    "loss_entropy",
                    tf.reduce_mean(loss_entropy))
            return tf.reduce_mean(loss_entropy)
        self.optimizer.minimize(
            loss_function, self.tuning_variable)
