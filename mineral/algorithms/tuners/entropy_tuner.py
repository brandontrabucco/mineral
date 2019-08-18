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
        self.master_policy = policy
        self.worker_policy = policy.clone()

    def update_algorithm(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        self.master_policy.copy_to(self.worker_policy)

        def loss_function():
            policy_actions = self.worker_policy.sample(
                observations[:, :(-1), ...],
                training=True)
            policy_entropy = -self.worker_policy.get_log_probs(
                policy_actions,
                observations[:, :(-1), ...],
                training=True)
            entropy_loss = self.tuning_variable * (
                policy_entropy - self.target)
            self.record(
                "entropy_tuning_variable",
                self.tuning_variable)
            self.record(
                "entropy",
                tf.reduce_mean(policy_entropy))
            self.record(
                "entropy_loss",
                tf.reduce_mean(entropy_loss))
            return tf.reduce_mean(entropy_loss)
        self.optimizer.minimize(
            loss_function, self.tuning_variable)
        self.worker_policy.copy_to(self.master_policy)

    def get_tuning_variable(self):
        return tf.exp(self.tuning_variable)
