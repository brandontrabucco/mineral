"""Author: Brandon Trabucco, Copyright 2019"""


import threading
import tensorflow as tf

from mineral.core.trainers.local_trainer import LocalTrainer
from mineral.core.monitors.local_monitor import LocalMonitor

from mineral.networks.dense import Dense
from mineral.distributions.gaussians.tanh_gaussian import TanhGaussian

from mineral.algorithms.actors.ppo import PPO
from mineral.algorithms.critics.gae import GAE
from mineral.algorithms.multi_algorithm import MultiAlgorithm

from mineral.core.envs.normalized_env import NormalizedEnv
from mineral.core.envs.debug.pointmass_env import PointmassEnv

from mineral.buffers.path_buffer import PathBuffer
from mineral.samplers.hierarchy_sampler import HierarchySampler


def run_experiment(variant):

    experiment_id = variant["experiment_id"]
    max_path_length = variant["max_path_length"]
    max_size = variant["max_size"]
    num_warm_up_samples = variant["num_warm_up_samples"]
    num_exploration_samples = variant["num_exploration_samples"]
    num_evaluation_samples = variant["num_evaluation_samples"]
    num_trains_per_step = variant["num_trains_per_step"]
    update_actor_every = variant["update_actor_every"]
    batch_size = variant["batch_size"]
    num_steps = variant["num_steps"]

    monitor = LocalMonitor("./pointmass/ppo/{}".format(experiment_id))

    env = NormalizedEnv(
        PointmassEnv(size=2, ord=2),
        reward_scale=(1 / max_path_length))

    policy = Dense(
        [256, 256, 4],
        optimizer_class=tf.keras.optimizers.Adam,
        optimizer_kwargs={"lr": 0.0001},
        distribution_class=TanhGaussian,
        distribution_kwargs=dict(std=None))

    old_policy = Dense(
        [256, 256, 4],
        optimizer_class=tf.keras.optimizers.Adam,
        optimizer_kwargs={"lr": 0.0001},
        distribution_class=TanhGaussian,
        distribution_kwargs=dict(std=None))

    vf = Dense(
        [256, 256, 1],
        optimizer_class=tf.keras.optimizers.Adam,
        optimizer_kwargs={"lr": 0.0001})

    target_vf = Dense(
        [256, 256, 1],
        tau=1e-2,
        optimizer_class=tf.keras.optimizers.Adam,
        optimizer_kwargs={"lr": 0.0001})

    buffer = PathBuffer(
        max_size=max_size,
        max_path_length=max_path_length,
        selector=(lambda x: x["proprio_observation"]),
        monitor=monitor)

    sampler = HierarchySampler(
        env,
        policy,
        buffer,
        time_skips=(1,),
        num_warm_up_samples=num_warm_up_samples,
        num_exploration_samples=num_exploration_samples,
        num_evaluation_samples=num_evaluation_samples,
        selector=(lambda i, x: x["proprio_observation"]),
        monitor=monitor)

    critic = GAE(
        vf,
        target_vf,
        gamma=0.99,
        lamb=0.95,
        batch_size=batch_size,
        monitor=monitor)

    actor = PPO(
        policy,
        old_policy,
        critic,
        gamma=0.99,
        epsilon=0.1,
        update_every=update_actor_every,
        old_update_every=num_trains_per_step,
        batch_size=batch_size,
        monitor=monitor)

    algorithm = MultiAlgorithm(actor, critic)

    trainer = LocalTrainer(
        sampler,
        buffer,
        algorithm,
        num_steps=num_steps,
        num_trains_per_step=num_trains_per_step,
        monitor=monitor)

    trainer.train()


if __name__ == "__main__":

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    for experiment_id in [0, 1, 2, 3, 4]:

        variant = dict(
            experiment_id=experiment_id,
            max_path_length=10,
            max_size=100,
            num_warm_up_samples=0,
            num_exploration_samples=100,
            num_evaluation_samples=100,
            num_trains_per_step=100,
            update_actor_every=10,
            batch_size=100,
            num_steps=10000)

        threading.Thread(target=run_experiment,
                         args=(variant,)).start()