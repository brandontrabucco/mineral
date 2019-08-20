"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mineral.distributions.gaussians.tanh_gaussian import TanhGaussian
from mineral.networks import DensePolicy


if __name__ == "__main__":

    policy = DensePolicy(
        [4],
        optimizer_kwargs=dict(lr=0.01),
        distribution_class=TanhGaussian
    )

    obs = np.random.normal(0, 1, [32, 6]).astype(np.float32)
    x = policy.get_stochastic_actions(obs)

    max_range = 1000
    log_prob_values = []
    for i in range(max_range):

        def loss_function():
            return -1.0 * tf.reduce_mean(policy.get_log_probs(x, obs))

        log_prob_values.append(np.mean(policy.get_log_probs(x, obs)))
        policy.minimize(loss_function, obs)

        if i == max_range - 1:
            plt.title("Log Prob Values")
            plt.plot(log_prob_values)
            plt.show()
            plt.clf()
