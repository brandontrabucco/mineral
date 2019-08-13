"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.algorithms.actors.ddpg import DDPG
from mineral.algorithms.critics.q_learning import QLearning
from mineral.algorithms.critics.twin_delayed_critic import TwinDelayedCritic
from mineral.algorithms.multi_algorithm import MultiAlgorithm
from mineral.networks.dense import Dense
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

    policy = Dense(
        [32, 32, 4],
        optimizer_class=tf.keras.optimizers.Adam,
        optimizer_kwargs=dict(lr=0.0001),
        distribution_class=TanhGaussianDistribution,
        distribution_kwargs=dict(std=None)
    )

    target_policy = Dense(
        [32, 32, 4],
        tau=1e-2,
        optimizer_class=tf.keras.optimizers.Adam,
        optimizer_kwargs=dict(lr=0.0001),
        distribution_class=TanhGaussianDistribution,
        distribution_kwargs=dict(std=None)
    )

    qf1 = Dense(
        [6, 6, 1],
        optimizer_class=tf.keras.optimizers.Adam,
        optimizer_kwargs={"lr": 0.0001},
    )

    qf2 = Dense(
        [6, 6, 1],
        optimizer_class=tf.keras.optimizers.Adam,
        optimizer_kwargs={"lr": 0.0001},
    )

    target_qf1 = Dense(
        [6, 6, 1],
        tau=1e-2,
        optimizer_class=tf.keras.optimizers.Adam,
        optimizer_kwargs={"lr": 0.0001},
    )

    target_qf2 = Dense(
        [6, 6, 1],
        tau=1e-2,
        optimizer_class=tf.keras.optimizers.Adam,
        optimizer_kwargs={"lr": 0.0001},
    )

    max_size = 1024

    buffer = PathBuffer(
        env,
        policy,
        max_size=max_size,
        max_path_length=max_path_length,
        selector=(lambda x: x["proprio_observation"]),
        monitor=monitor
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
        selector=(lambda x: x["proprio_observation"]),
        monitor=monitor,
    )

    critic2 = QLearning(
        target_policy,
        qf2,
        target_qf2,
        gamma=gamma,
        clip_radius=clip_radius,
        std=std,
        selector=(lambda x: x["proprio_observation"]),
        monitor=monitor,
    )

    twin_delayed_critic = TwinDelayedCritic(
        critic1,
        critic2
    )

    actor = DDPG(
        policy,
        twin_delayed_critic,
        target_policy,
        update_every=num_trains_per_step // off_policy_updates,
        selector=(lambda x: x["proprio_observation"]),
        monitor=monitor
    )

    algorithm = MultiAlgorithm(
        twin_delayed_critic,
        actor
    )

    num_warm_up_paths = 1024
    num_steps = 1000
    num_paths_to_collect = 32
    batch_size = 32

    trainer = LocalTrainer(
        buffer,
        algorithm,
        num_warm_up_paths=num_warm_up_paths,
        num_steps=num_steps,
        num_paths_to_collect=num_paths_to_collect,
        batch_size=batch_size,
        num_trains_per_step=num_trains_per_step,
        monitor=monitor
    )

    trainer.train()
