"""Author: Brandon Trabucco, Copyright 2019"""


import numpy as np
import matplotlib.pyplot as plt
from mineral.distributions.gaussians.exp_gaussian import ExpGaussian
from mineral.networks.dense.dense_policy import DensePolicy


if __name__ == "__main__":

    policy = DensePolicy(
        [4],
        optimizer_kwargs=dict(lr=0.01),
        distribution_class=ExpGaussian,
        distribution_kwargs=dict(std=None)
    )

    x = np.linspace(0.0, 5.0, num=1000, dtype=np.float32)
    y = np.linspace(0.0, 5.0, num=1000, dtype=np.float32)
    z = np.stack(np.meshgrid(x, y), axis=(-1))
    w = np.array([[[2.0, 2.5], [3.0, 2.5], [2.5, 2.55]]]).astype(np.float32)

    obs = np.random.normal(0, 1, [1, 1, 6]).astype(np.float32)

    max_range = 1000
    for i in range(max_range):

        def loss_function():
            return -1.0 * policy.get_log_probs(w, obs)

        policy.minimize(loss_function, obs)

        if i == 0:
            plt.title("PDF Before Gradient")
            plt.imshow(policy.get_probs(z, obs), vmin=0.0, vmax=1.0)
            plt.show()
            plt.clf()

        if i == max_range - 1:
            plt.title("PDF After Gradient")
            plt.imshow(policy.get_probs(z, obs), vmin=0.0, vmax=1.0)
            plt.show()
            plt.clf()
