"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.algorithms.dynamics_models.dynamics_model import DynamicsModel


class OneStep(DynamicsModel):

    def update_model(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        def loss_function():
            model_log_probs = self.model.get_log_probs(
                observations[:, 1:, ...],
                observations[:, :(-1), ...],
                actions,
                training=True)
            model_loss = -1.0 * tf.reduce_mean(
                model_log_probs * terminals[:, :(-1)])
            self.record(
                "model_log_probs_mean",
                tf.reduce_mean(model_log_probs))
            self.record(
                "model_loss",
                model_loss)
            return model_loss
        self.model.minimize(
            loss_function,
            observations[:, :(-1), ...],
            actions)
