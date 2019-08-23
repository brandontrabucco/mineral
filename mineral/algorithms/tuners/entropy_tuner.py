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
            policy_actions = self.policy.get_expected_value(
                observations[:, :(-1), ...])
            policy_entropy = -terminals[:, :(-1)] * self.policy.get_log_probs(
                policy_actions,
                observations[:, :(-1), ...])
            entropy_error = policy_entropy - self.target
            entropy_loss = self.tuning_variable * tf.stop_gradient(entropy_error)
            self.record(
                "entropy_tuning_variable",
                self.tuning_variable)
            self.record(
                "entropy_error_mean",
                tf.reduce_mean(entropy_error))
            self.record(
                "entropy_error_max",
                tf.reduce_max(entropy_error))
            self.record(
                "entropy_error_min",
                tf.reduce_min(entropy_error))
            self.record(
                "entropy",
                tf.reduce_mean(policy_entropy))
            self.record(
                "entropy_loss",
                tf.reduce_mean(entropy_loss))
            return tf.reduce_mean(entropy_loss)
        self.optimizer.minimize(
            loss_function, [self.tuning_variable])
