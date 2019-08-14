"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.algorithms.actors.ddpg import DDPG
from mineral.algorithms.critics.q_learning import QLearning
from mineral.algorithms.critics.twin_delayed_critic import TwinDelayedCritic
from mineral.algorithms.multi_algorithm import MultiAlgorithm
from mineral.networks.dense import Dense
from mineral.distributions.gaussians.tanh_gaussian import TanhGaussian
from mineral.core.envs.normalized_env import NormalizedEnv
from mineral.core.envs.debug.pointmass_env import PointmassEnv
from mineral.buffers.path_buffer import PathBuffer
from mineral.core.trainers.local_trainer import LocalTrainer
from mineral.core.monitors.local_monitor import LocalMonitor
from mineral.samplers.path_sampler import PathSampler


if __name__ == "__main__":

    max_path_length = 10
    max_size = 100000
    num_warm_up_samples = 100
    num_exploration_samples = 1
    num_evaluation_samples = 100
    num_trains_per_step = 100
    update_tuner_every = 100
    update_actor_every = 100
    batch_size = 100
    num_steps = 10000

    monitor = LocalMonitor("./pointmass/td3")

    env = NormalizedEnv(
        PointmassEnv(size=2, ord=2),
        reward_scale=(1 / max_path_length))

    policy = Dense(
        [32, 32, 4],
        optimizer_class=tf.keras.optimizers.Adam,
        optimizer_kwargs=dict(lr=0.0001),
        distribution_class=TanhGaussian,
        distribution_kwargs=dict(std=None))

    target_policy = Dense(
        [32, 32, 4],
        tau=1e-2,
        optimizer_class=tf.keras.optimizers.Adam,
        optimizer_kwargs=dict(lr=0.0001),
        distribution_class=TanhGaussian,
        distribution_kwargs=dict(std=None))

    qf1 = Dense(
        [6, 6, 1],
        optimizer_class=tf.keras.optimizers.Adam,
        optimizer_kwargs={"lr": 0.0001})

    qf2 = Dense(
        [6, 6, 1],
        optimizer_class=tf.keras.optimizers.Adam,
        optimizer_kwargs={"lr": 0.0001})

    target_qf1 = Dense(
        [6, 6, 1],
        tau=1e-2,
        optimizer_class=tf.keras.optimizers.Adam,
        optimizer_kwargs={"lr": 0.0001})

    target_qf2 = Dense(
        [6, 6, 1],
        tau=1e-2,
        optimizer_class=tf.keras.optimizers.Adam,
        optimizer_kwargs={"lr": 0.0001})

    buffer = PathBuffer(
        max_size=max_size,
        max_path_length=max_path_length,
        selector=(lambda x: x["proprio_observation"]),
        monitor=monitor)

    sampler = PathSampler(
        buffer,
        env,
        policy,
        num_warm_up_samples=num_warm_up_samples,
        num_exploration_samples=num_exploration_samples,
        num_evaluation_samples=num_evaluation_samples,
        selector=(lambda x: x["proprio_observation"]),
        monitor=monitor)

    critic1 = QLearning(
        target_policy,
        qf1,
        target_qf1,
        gamma=0.99,
        clip_radius=0.2,
        std=0.1,
        batch_size=batch_size,
        monitor=monitor)

    critic2 = QLearning(
        target_policy,
        qf2,
        target_qf2,
        gamma=0.99,
        clip_radius=0.2,
        std=0.1,
        batch_size=batch_size,
        monitor=monitor)

    twin_delayed_critic = TwinDelayedCritic(
        critic1,
        critic2)

    actor = DDPG(
        policy,
        twin_delayed_critic,
        target_policy,
        update_every=update_actor_every,
        batch_size=batch_size,
        monitor=monitor)

    algorithm = MultiAlgorithm(actor, twin_delayed_critic)

    trainer = LocalTrainer(
        sampler,
        buffer,
        algorithm,
        num_steps=num_steps,
        num_trains_per_step=num_trains_per_step,
        monitor=monitor)

    trainer.train()
