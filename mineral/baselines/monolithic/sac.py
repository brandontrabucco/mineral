"""Author: Brandon Trabucco, Copyright 2019"""


import numpy as np
import tensorflow as tf

from mineral.core.savers.local_saver import LocalSaver
from mineral.core.trainers.local_trainer import LocalTrainer
from mineral.core.monitors.local_monitor import LocalMonitor

from mineral.networks.dense import Dense
from mineral.distributions.gaussians.tanh_gaussian import TanhGaussian

from mineral.algorithms.tuners.entropy_tuner import EntropyTuner
from mineral.algorithms.actors.soft_actor_critic import SoftActorCritic
from mineral.algorithms.critics.soft_q_network import SoftQNetwork
from mineral.algorithms.critics.twin_critic import TwinCritic

from mineral.core.envs.normalized_env import NormalizedEnv

from mineral.core.buffers.path_buffer import PathBuffer
from mineral.core.buffers.off_policy_buffer import OffPolicyBuffer
from mineral.core.samplers.parallel_sampler import ParallelSampler


sac_variant = dict(
    logging_dir="./sac",
    reward_scale=1.0,
    hidden_size=300,
    tau=0.005,
    learning_rate=0.0003,
    batch_size=128,
    gamma=0.99,
    bellman_weight=1.0,
    discount_weight=0.0,
    max_size=1000,
    max_path_length=1000,
    num_warm_up_paths=10,
    num_exploration_paths=1,
    num_evaluation_paths=10,
    num_threads=10,
    num_steps=10000,
    num_trains_per_step=1000)


def sac(
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

    qf1 = Dense(
        [variant["hidden_size"], variant["hidden_size"], 1],
        tau=variant["tau"],
        optimizer_class=tf.keras.optimizers.Adam,
        optimizer_kwargs=dict(lr=variant["learning_rate"]))

    qf2 = qf1.clone()
    target_qf1 = qf1.clone()
    target_qf2 = qf1.clone()

    tuner = EntropyTuner(
        policy,
        optimizer_class=tf.keras.optimizers.Adam,
        optimizer_kwargs=dict(lr=variant["learning_rate"]),
        target=(-action_dim),
        batch_size=variant["batch_size"],
        monitor=monitor)

    critic1 = SoftQNetwork(
        policy,
        qf1,
        target_qf1,
        gamma=variant["gamma"],
        log_alpha=tuner.get_tuning_variable(),
        bellman_weight=variant["bellman_weight"],
        discount_weight=variant["discount_weight"],
        batch_size=variant["batch_size"],
        monitor=monitor)

    critic2 = SoftQNetwork(
        policy,
        qf2,
        target_qf2,
        gamma=variant["gamma"],
        log_alpha=tuner.get_tuning_variable(),
        bellman_weight=variant["bellman_weight"],
        discount_weight=variant["discount_weight"],
        batch_size=variant["batch_size"],
        monitor=monitor)

    critic = TwinCritic(critic1, critic2)

    actor = SoftActorCritic(
        policy,
        critic,
        log_alpha=tuner.get_tuning_variable(),
        batch_size=variant["batch_size"],
        monitor=monitor)

    buffer = PathBuffer(
        max_size=variant["max_size"],
        max_path_length=variant["max_path_length"],
        selector=(lambda x: x[observation_key]),
        monitor=monitor)

    step_buffer = OffPolicyBuffer(buffer)

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
        policy=policy,
        qf1=qf1,
        target_qf1=target_qf1,
        qf2=qf2,
        target_qf2=target_qf2)

    trainer = LocalTrainer(
        sampler,
        [step_buffer, step_buffer, step_buffer],
        [actor, critic, tuner],
        num_steps=variant["num_steps"],
        num_trains_per_step=variant["num_trains_per_step"],
        saver=saver,
        monitor=monitor)

    trainer.train()
