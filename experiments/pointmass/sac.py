"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.algorithms.actors.soft_actor_critic import SoftActorCritic
from mineral.algorithms.critics.soft_q_learning import SoftQLearning
from mineral.networks.dense.dense_policy import DensePolicy
from mineral.networks.dense.dense_q_function import DenseQFunction
from mineral.distributions.gaussians.tanh_gaussian_distribution import TanhGaussianDistribution
from mineral.core.envs.normalized_env import NormalizedEnv
from mineral.core.envs.pointmass_env import PointmassEnv
from mineral.buffers.path_buffer import PathBuffer
from mineral.core.trainers.local_trainer import LocalTrainer
from mineral.core.monitors.local_monitor import LocalMonitor


if __name__ == "__main__":

    monitor = LocalMonitor("./pointmass/sac")

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

    target_policy = DensePolicy(
        [32, 32, 4],
        tau=1e-2,
        optimizer_class=tf.keras.optimizers.Adam,
        optimizer_kwargs=dict(lr=0.0001),
        distribution_class=TanhGaussianDistribution,
        distribution_kwargs=dict(std=None)
    )

    qf = DenseQFunction(
        [6, 6, 1],
        optimizer_class=tf.keras.optimizers.Adam,
        optimizer_kwargs={"lr": 0.0001},
    )

    target_qf = DenseQFunction(
        [6, 6, 1],
        tau=1e-2,
        optimizer_class=tf.keras.optimizers.Adam,
        optimizer_kwargs={"lr": 0.0001},
    )

    buffer = PathBuffer(
        env,
        policy
    )

    num_trains_per_step = 32
    off_policy_updates = 4

    clip_radius = 0.2
    std = 0.1
    gamma = 0.99

    critic = SoftQLearning(
        target_policy,
        qf,
        target_qf,
        gamma=gamma,
        clip_radius=clip_radius,
        std=std,
        monitor=monitor,
    )

    algorithm = SoftActorCritic(
        policy,
        critic,
        target_policy,
        actor_delay=num_trains_per_step // off_policy_updates,
        monitor=monitor
    )
    
    max_size = 1024
    num_warm_up_paths = 1024
    num_steps = 1000
    num_paths_to_collect = 32
    batch_size = 32

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
