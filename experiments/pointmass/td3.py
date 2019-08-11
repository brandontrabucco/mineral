"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.algorithms.actors.ddpg import DDPG
from mineral.algorithms.critics.q_learning import QLearning
from mineral.algorithms.critics.twin_delayed_critic import TwinDelayedCritic
from mineral.networks.dense_network import DenseNetwork
from mineral.distributions.gaussians.tanh_gaussian_distribution import TanhGaussianDistribution
from mineral.core.envs.normalized_env import NormalizedEnv
from mineral.core.envs.pointmass_env import PointmassEnv
from mineral.buffers.path_buffer import PathBuffer
from mineral.core.trainers.local_trainer import LocalTrainer
from mineral.core.monitors.local_monitor import LocalMonitor


if __name__ == "__main__":

    monitor = LocalMonitor("./pointmass/td3")

    max_path_length = 10

    env = NormalizedEnv(
        PointmassEnv(size=2, ord=2),
        reward_scale=(1 / max_path_length)
    )

    policy = DenseNetwork(
        [32, 32, 4],
        optimizer_class=tf.keras.optimizers.Adam,
        optimizer_kwargs=dict(lr=0.0001),
        distribution_class=TanhGaussianDistribution,
        distribution_kwargs=dict(std=None)
    )

    target_policy = DenseNetwork(
        [32, 32, 4],
        tau=1e-2,
        optimizer_class=tf.keras.optimizers.Adam,
        optimizer_kwargs=dict(lr=0.0001),
        distribution_class=TanhGaussianDistribution,
        distribution_kwargs=dict(std=None)
    )

    qf1 = DenseNetwork(
        [6, 6, 1],
        optimizer_class=tf.keras.optimizers.Adam,
        optimizer_kwargs={"lr": 0.0001},
    )

    qf2 = DenseNetwork(
        [6, 6, 1],
        optimizer_class=tf.keras.optimizers.Adam,
        optimizer_kwargs={"lr": 0.0001},
    )

    target_qf1 = DenseNetwork(
        [6, 6, 1],
        tau=1e-2,
        optimizer_class=tf.keras.optimizers.Adam,
        optimizer_kwargs={"lr": 0.0001},
    )

    target_qf2 = DenseNetwork(
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

    critic1 = QLearning(
        target_policy,
        qf1,
        target_qf1,
        gamma=gamma,
        clip_radius=clip_radius,
        std=std,
        monitor=monitor,
    )

    critic2 = QLearning(
        target_policy,
        qf2,
        target_qf2,
        gamma=gamma,
        clip_radius=clip_radius,
        std=std,
        monitor=monitor,
    )

    twin_delayed_critic = TwinDelayedCritic(
        critic1,
        critic2
    )

    algorithm = DDPG(
        policy,
        twin_delayed_critic,
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
