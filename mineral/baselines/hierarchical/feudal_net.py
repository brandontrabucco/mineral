"""Author: Brandon Trabucco, Copyright 2019"""


import numpy as np
import tensorflow as tf

from mineral.core.savers.local_saver import LocalSaver
from mineral.core.trainers.local_trainer import LocalTrainer
from mineral.core.monitors.local_monitor import LocalMonitor

from mineral.networks.dense import Dense
from mineral.distributions.gaussians.tanh_gaussian import TanhGaussian

from mineral.algorithms.actors.ddpg import DDPG
from mineral.algorithms.critics.q_network import QNetwork

from mineral.core.envs.normalized_env import NormalizedEnv

from mineral.core.buffers.path_buffer import PathBuffer
from mineral.core.buffers.off_policy_buffer import OffPolicyBuffer
from mineral.core.samplers.parallel_sampler import ParallelSampler

from mineral.relabelers.goal_conditioned_relabeler import GoalConditionedRelabeler


feudal_net_variant = dict(
    logging_dir="./feudal_net",
    reward_scale=1.0,
    hidden_size=300,
    tau=0.005,
    learning_rate=0.0003,
    batch_size=128,
    gamma=0.99,
    bellman_weight=1.0,
    discount_weight=0.0,
    max_size=1000,
    time_skip=5,
    max_path_length=1000,
    num_warm_up_paths=10,
    num_exploration_paths=1,
    num_evaluation_paths=10,
    num_threads=10,
    num_steps=10000,
    num_trains_per_step=1000)


def feudal_net(
    variant,
    env_class,
    observation_key="proprio_observation",
    goal_key="goal",
    **env_kwargs
):

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    observation_selector = (
        lambda x: x[observation_key])

    goal_selector = (
        lambda x: x[goal_key])

    both_selector = (
        lambda x: np.concatenate([observation_selector(x), goal_selector(x)], -1))

    hierarchy_selector = (
        lambda i, x: observation_selector(x) if i == 1 else both_selector(x))

    monitor = LocalMonitor(variant["logging_dir"])
    env = NormalizedEnv(env_class, reward_scale=variant["reward_scale"], **env_kwargs)
    action_dim = np.prod(env.action_space.shape)
    goal_dim = np.prod(env.observation_space[observation_key].shape)

    lower_policy = Dense(
        [variant["hidden_size"], variant["hidden_size"], 2 * action_dim],
        tau=variant["tau"],
        optimizer_class=tf.keras.optimizers.Adam,
        optimizer_kwargs=dict(lr=variant["learning_rate"]),
        distribution_class=TanhGaussian,
        distribution_kwargs=dict(std=None))

    lower_qf = Dense(
        [variant["hidden_size"], variant["hidden_size"], 1],
        tau=variant["tau"],
        optimizer_class=tf.keras.optimizers.Adam,
        optimizer_kwargs=dict(lr=variant["learning_rate"]))

    lower_target_qf = lower_qf.clone()

    lower_critic = QNetwork(
        lower_policy,
        lower_qf,
        lower_target_qf,
        gamma=variant["gamma"],
        bellman_weight=variant["bellman_weight"],
        discount_weight=variant["discount_weight"],
        batch_size=variant["batch_size"],
        monitor=monitor,
        logging_prefix="lower_")

    lower_actor = DDPG(
        lower_policy,
        lower_critic,
        batch_size=variant["batch_size"],
        update_every=variant["num_trains_per_step"],
        monitor=monitor,
        logging_prefix="lower_")

    lower_buffer = GoalConditionedRelabeler(
        PathBuffer(
            max_size=variant["max_size"],
            max_path_length=variant["max_path_length"],
            selector=(lambda x: x[observation_key]),
            monitor=monitor),
        observation_selector=observation_selector,
        goal_selector=goal_selector)

    lower_buffer = OffPolicyBuffer(lower_buffer)

    upper_policy = Dense(
        [variant["hidden_size"], variant["hidden_size"], 2 * goal_dim],
        tau=variant["tau"],
        optimizer_class=tf.keras.optimizers.Adam,
        optimizer_kwargs=dict(lr=variant["learning_rate"]),
        distribution_class=TanhGaussian,
        distribution_kwargs=dict(std=None))

    upper_qf = Dense(
        [variant["hidden_size"], variant["hidden_size"], 1],
        tau=variant["tau"],
        optimizer_class=tf.keras.optimizers.Adam,
        optimizer_kwargs=dict(lr=variant["learning_rate"]))

    upper_target_qf = upper_qf.clone()

    upper_critic = QNetwork(
        upper_policy,
        upper_qf,
        upper_target_qf,
        gamma=variant["gamma"],
        bellman_weight=variant["bellman_weight"],
        discount_weight=variant["discount_weight"],
        batch_size=variant["batch_size"],
        monitor=monitor,
        logging_prefix="upper_")

    upper_actor = DDPG(
        upper_policy,
        upper_critic,
        batch_size=variant["batch_size"],
        update_every=variant["num_trains_per_step"],
        monitor=monitor,
        logging_prefix="upper_")

    upper_buffer = PathBuffer(
        max_size=variant["max_size"],
        max_path_length=variant["max_path_length"],
        selector=(lambda x: x[observation_key]),
        monitor=monitor)

    upper_buffer = OffPolicyBuffer(upper_buffer)

    sampler = ParallelSampler(
        env,
        [lower_policy, upper_policy],
        [lower_buffer, upper_buffer],
        time_skips=(1, variant["time_skip"]),
        max_path_length=variant["max_path_length"],
        num_warm_up_paths=variant["num_warm_up_paths"],
        num_exploration_paths=variant["num_exploration_paths"],
        num_evaluation_paths=variant["num_evaluation_paths"],
        num_threads=variant["num_threads"],
        selector=hierarchy_selector,
        monitor=monitor)

    saver = LocalSaver(
        variant["logging_dir"],
        lower_policy=lower_policy,
        lower_qf=lower_qf,
        lower_target_qf=lower_target_qf,
        upper_policy=upper_policy,
        upper_qf=upper_qf,
        upper_target_qf=upper_target_qf)

    trainer = LocalTrainer(
        sampler,
        [lower_buffer, lower_buffer, lower_buffer, upper_buffer, upper_buffer, upper_buffer],
        [upper_actor, upper_critic, lower_actor, lower_critic],
        num_steps=variant["num_steps"],
        num_trains_per_step=variant["num_trains_per_step"],
        saver=saver,
        monitor=monitor)

    trainer.train()
