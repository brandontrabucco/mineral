"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.algorithms.actors.actor_critic import ActorCritic
from mineral.algorithms.critics.gae import GAE
from mineral.networks.dense.dense_policy import DensePolicy
from mineral.distributions.gaussians.tanh_gaussian_distribution import TanhGaussianDistribution
from mineral.core.envs.normalized_env import NormalizedEnv
from mineral.core.envs.pointmass_env import PointmassEnv
from mineral.buffers.path_buffer import PathBuffer
from mineral.core.trainers.local_trainer import LocalTrainer
from mineral.core.monitors.local_monitor import LocalMonitor
from mineral.networks.dense.dense_value_function import DenseValueFunction
from mineral.optimizers.gradients.natural_gradient import NaturalGradient


if __name__ == "__main__":

    monitor = LocalMonitor("./pointmass/natural_actor_critic")

    max_path_length = 10

    env = NormalizedEnv(
        PointmassEnv(size=2, ord=2),
        reward_scale=(1 / max_path_length)
    )

    policy = DensePolicy(
        [32, 32, 4],
        optimizer_class=tf.keras.optimizers.Adam,
        optimizer_kwargs=dict(lr=0.0001),
        distribution_class=TanhGaussianDistribution,
        distribution_kwargs=dict(std=None)
    )

    policy = NaturalGradient(
        policy
    )

    vf = DenseValueFunction(
        [6, 6, 1],
        optimizer_kwargs={"lr": 0.01},
    )

    target_vf = DenseValueFunction(
        [6, 6, 1]
    )

    buffer = PathBuffer(
        env,
        policy
    )

    critic = GAE(
        vf,
        target_vf,
        gamma=0.99,
        lamb=0.95,
        monitor=monitor,
    )

    num_trains_per_step = 32

    algorithm = ActorCritic(
        policy,
        critic,
        gamma=0.99,
        actor_delay=num_trains_per_step,
        monitor=monitor
    )

    max_size = 32
    num_warm_up_paths = 0
    num_steps = 1000
    num_paths_to_collect = max_size
    batch_size = max_size

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