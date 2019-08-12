"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.algorithms.critics.value_learning import ValueLearning
from mineral import discounted_sum


class GAE(ValueLearning):

    def __init__(
        self,
        *args,
        lamb=1.0,
        **kwargs
    ):
        ValueLearning.__init__(
            self,
            *args,
            **kwargs
        )
        self.lamb = lamb

    def get_advantages(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        values = self.vf.get_expected_value(observations)[:, :, 0]
        advantages = discounted_sum(
            (terminals[:, :(-1)] * (rewards - values[:, :(-1)]) +
             terminals[:, 1:] * values[:, 1:] * self.gamma),
            self.gamma
        )
        if self.monitor is not None:
            self.monitor.record(
                "advantages_mean",
                tf.reduce_mean(advantages)
            )
        return advantages

