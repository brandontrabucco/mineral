"""Author: Brandon Trabucco, Copyright 2019"""


import numpy as np
import tensorflow as tf

from mineral.core.savers.local_saver import LocalSaver
from mineral.core.trainers.local_trainer import LocalTrainer
from mineral.core.monitors.local_monitor import LocalMonitor

from mineral.networks.dense import Dense
from mineral.distributions.gaussians.tanh_gaussian import TanhGaussian

from mineral.algorithms.tuners.entropy_tuner import EntropyTuner
from mineral.algorithms.critics.soft_value_network import SoftValueNetwork
from mineral.algorithms.critics.gae import GAE
from mineral.algorithms.actors.importance_sampling import ImportanceSampling

from mineral.optimizers.gradients.natural_gradient import NaturalGradient
from mineral.optimizers.constraints.kl_constraint import KLConstraint
from mineral.optimizers.line_search import LineSearch

from mineral.core.envs.normalized_env import NormalizedEnv

from mineral.core.buffers.path_buffer import PathBuffer
from mineral.core.samplers.parallel_sampler import ParallelSampler


trpo_variant = dict(
    logging_dir="./trpo",
    reward_scale=1.0,
    hidden_size=300,
    tau=0.005,
    learning_rate=0.0003,
    batch_size=32,
    gamma=0.99,
    lamb=0.95,
    delta=0.2,
    bellman_weight=0.0,
    discount_weight=1.0,
    max_size=32,
    max_path_length=100,
    num_warm_up_paths=0,
    num_exploration_paths=32,
    num_evaluation_paths=32,
    num_threads=16,
    num_steps=10000,
    num_trains_per_step=10)


def trpo(
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

    vf = Dense(
        [variant["hidden_size"], variant["hidden_size"], 1],
        tau=variant["tau"],
        optimizer_class=tf.keras.optimizers.Adam,
        optimizer_kwargs=dict(lr=variant["learning_rate"]))

    old_policy = policy.clone()
    target_vf = vf.clone()

    policy = KLConstraint(
        LineSearch(
            NaturalGradient(
                policy, return_sAs=True
            ), use_sAs=True
        ), old_policy, delta=variant["delta"]
    )

    tuner = EntropyTuner(
        policy,
        optimizer_class=tf.keras.optimizers.Adam,
        optimizer_kwargs=dict(lr=variant["learning_rate"]),
        target=(-action_dim),
        batch_size=variant["batch_size"],
        monitor=monitor)

    critic = SoftValueNetwork(
        policy,
        vf,
        target_vf,
        gamma=variant["gamma"],
        log_alpha=tuner.get_tuning_variable(),
        bellman_weight=variant["bellman_weight"],
        discount_weight=variant["discount_weight"],
        batch_size=variant["batch_size"],
        monitor=monitor)

    critic = GAE(
        critic,
        gamma=variant["gamma"],
        lamb=variant["lamb"])

    actor = ImportanceSampling(
        policy,
        old_policy,
        critic,
        gamma=variant["gamma"],
        old_update_every=variant["num_trains_per_step"],
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
        policy=policy,
        old_policy=old_policy,
        vf=vf,
        target_vf=target_vf)

    trainer = LocalTrainer(
        sampler,
        [buffer, buffer, buffer],
        [actor, critic, tuner],
        num_steps=variant["num_steps"],
        num_trains_per_step=variant["num_trains_per_step"],
        saver=saver,
        monitor=monitor)

    trainer.train()
