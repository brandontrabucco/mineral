"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.algorithms.transition_dynamics.transition_dynamics import TransitionDynamics


class OneStepRegression(TransitionDynamics):

    def __init__(
        self,
        model,
        **kwargs
    ):
        TransitionDynamics.__init__(
            self,
            **kwargs
        )
        self.model = model

    def update_transition(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        def loss_function():
            log_probs_model = self.model.get_log_probs(
                observations[:, :(-1), :],
                actions,
                observations[:, 1:, :]
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
            observations[:, :(-1), :],
            actions
        )

