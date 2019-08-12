"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.algorithms.dynamics_models.dynamics_model import DynamicsModel


class OneStep(DynamicsModel):

    def update_algorithm(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        def loss_function():
            log_probs_model = self.model.get_log_probs(
                observations[:, 1:, ...],
                observations[:, :(-1), ...],
                actions,
                training=True
            )
            loss_model = -1.0 * tf.reduce_mean(
                log_probs_model * terminals[:, :(-1)]
            )
            if self.monitor is not None:
                self.monitor.record(
                    "log_probs_model_mean",
                    tf.reduce_mean(log_probs_model)
                )
                self.monitor.record(
                    "loss_model",
                    loss_model
                )
            return loss_model
        self.model.minimize(
            loss_function,
            observations[:, :(-1), ...],
            actions
        )
