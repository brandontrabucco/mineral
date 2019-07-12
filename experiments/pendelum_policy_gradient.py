"""Author: Brandon Trabucco, Copyright 2019"""


import gym
from jetpack.networks.dense.dense_policy import DensePolicy
from jetpack.distributions.gaussian.tanh_gaussian_distribution import TanhGaussianDistribution
from jetpack.core.environments.normalized_environment import NormalizedEnvironment
from jetpack.buffers.path_buffer import PathBuffer
from jetpack.algorithms.actors.policy_gradient import PolicyGradient
from jetpack.core.trainers.local_trainer import LocalTrainer
from jetpack.core.monitors.local_monitor import LocalMonitor


if __name__ == "__main__":

    monitor = LocalMonitor("./")

    env = NormalizedEnvironment(
        gym.make("Pendulum-v0")
    )

    policy = DensePolicy(
        [32, 32, 2],
        distribution_class=TanhGaussianDistribution
    )

    buffer = PathBuffer(
        env,
        policy
    )

    algorithm = PolicyGradient(
        policy,
        gamma=0.99,
        monitor=monitor
    )
    
    max_size = 32
    num_warm_up_paths = 1
    num_steps = 1000
    num_paths_to_collect = 32
    max_path_length = 200
    batch_size = 32
    num_trains_per_step = 1

    trainer = LocalTrainer(
        max_size,
        num_warm_up_paths,
        num_steps,
        num_paths_to_collect,
        max_path_length,
        batch_size,
        num_trains_per_step,
        buffer,
        algorithm,
        monitor=monitor
    )

    trainer.train()
