"""Author: Brandon Trabucco, Copyright 2019"""

import tensorflow as tf
from jetpack.functions.model import Model


class TanhModel(Model):

    def __init__(
        self,
        model
    ):
        self.model = model

    def get_stochastic_observations(
        self,
        observations,
        actions
    ):
        return tf.math.softplus(
            self.model.get_stochastic_observations(
                observations,
                actions
            )
        )

    def get_deterministic_observations(
        self,
        observations,
        actions
    ):
        return tf.math.softplus(
            self.model.get_deterministic_observations(
                observations,
                actions
            )
        )

    def get_log_probs(
        self,
        observations,
        actions,
        next_observations
    ):
        next_observations = tf.clip_by_value(next_observations, -0.999, 0.999)
        correction = -1.0 * tf.reduce_sum(
            tf.math.log(1.0 - tf.math.square(next_observations)),
            axis=-1
        )
        return correction + self.model.get_log_probs(
            observations,
            actions,
            tf.math.atanh(next_observations)
        )

    def get_kl_divergence(
        self,
        other_model,
        observations,
        actions
    ):
        return self.model.get_kl_divergence(
            other_model,
            observations,
            actions
        )

    def __call__(
        self,
        observations,
        actions
    ):
        return self.model(observations, actions)

    def __getattr__(
        self,
        attr
    ):
        return getattr(self.model, attr)

