"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.buffers.relabelers.relabeler import Relabeler


class EntropyRelabeler(Relabeler):
    
    def __init__(
        self,
        policy,
        *args,
        observation_selector=(lambda x: x["proprio_observation"]),
        alpha=1.0,
        **kwargs
    ):
        Relabeler.__init__(self, *args, **kwargs)
        self.policy = policy
        self.observation_selector = observation_selector
        self.alpha = alpha

    def relabel(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        selected_observations = self.observation_selector(
            observations)[:, 1:, ...]
        sampled_actions = self.policy.sample(selected_observations)
        entropy = -self.policy.get_log_probs(
            sampled_actions,
            selected_observations)
        relabel_condition = self.relabel_probability >= tf.random.uniform(
            tf.shape(rewards),
            maxval=1.0,
            dtype=tf.float32)
        rewards = tf.where(
            relabel_condition,
            rewards + self.alpha * entropy,
            rewards)
        return (
            observations,
            actions,
            rewards,
            terminals)
