"""Author: Brandon Trabucco, Copyright 2019"""


import numpy as np
import tensorflow as tf

from mineral.core.savers.local_saver import LocalSaver
from mineral.core.trainers.local_trainer import LocalTrainer
from mineral.core.monitors.local_monitor import LocalMonitor

from mineral.networks.dense import Dense
from mineral.distributions.gaussians.tanh_gaussian import TanhGaussian

from mineral.algorithms.actors.policy_gradient import PolicyGradient

from mineral.core.envs.normalized_env import NormalizedEnv

from mineral.core.buffers.path_buffer import PathBuffer
from mineral.core.samplers.parallel_sampler import ParallelSampler


pg_variant = dict(
    logging_dir="./pg",
    reward_scale=1.0,
    hidden_size=300,
    tau=0.005,
    learning_rate=0.0003,
    batch_size=32,
    gamma=0.99,
    max_size=32,
    max_path_length=100,
    num_warm_up_paths=0,
    num_exploration_paths=32,
    num_evaluation_paths=32,
    num_threads=16,
    num_steps=10000,
    num_trains_per_step=10)


def pg(
    variant,
    env_class,
    observation_key="proprio_observation",
    **env_kwargs
):

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    monitor = LocalMonitor(variant["logging_dir"])
    env = NormalizedEnv(env_class, reward_scale=variant["reward_scale"], **env_kwargs)
    action_dim = np.prod(env.action_space.shape)

    policy = Dense(
        [variant["hidden_size"], variant["hidden_size"], 2 * action_dim],
        tau=variant["tau"],
        optimizer_class=tf.keras.optimizers.Adam,
        optimizer_kwargs=dict(lr=variant["learning_rate"]),
        distribution_class=TanhGaussian,
        distribution_kwargs=dict(std=None))

    actor = PolicyGradient(
        policy,
        gamma=variant["gamma"],
        batch_size=variant["batch_size"],
        monitor=monitor)

    buffer = PathBuffer(
        max_size=variant["max_size"],
        max_path_length=variant["max_path_length"],
        selector=(lambda x: x[observation_key]),
        monitor=monitor)

    sampler = ParallelSampler(
        env, policy, buffer,
        max_path_length=variant["max_path_length"],
        num_warm_up_paths=variant["num_warm_up_paths"],
        num_exploration_paths=variant["num_exploration_paths"],
        num_evaluation_paths=variant["num_evaluation_paths"],
        num_threads=variant["num_threads"],
        selector=(lambda i, x: x[observation_key]),
        monitor=monitor)

    saver = LocalSaver(
        variant["logging_dir"],
        policy=policy)

    trainer = LocalTrainer(
        sampler,
        [buffer],
        [actor],
        num_steps=variant["num_steps"],
        num_trains_per_step=variant["num_trains_per_step"],
        saver=saver,
        monitor=monitor)

    trainer.train()
