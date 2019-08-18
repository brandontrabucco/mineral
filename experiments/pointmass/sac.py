"""Author: Brandon Trabucco, Copyright 2019"""


import multiprocessing
import tensorflow as tf

from mineral.core.savers.local_saver import LocalSaver
from mineral.core.trainers.local_trainer import LocalTrainer
from mineral.core.monitors.local_monitor import LocalMonitor

from mineral.networks.dense import Dense
from mineral.distributions.gaussians.tanh_gaussian import TanhGaussian

from mineral.algorithms.actors.soft_actor_critic import SoftActorCritic
from mineral.algorithms.critics.soft_q_network import SoftQNetwork
from mineral.algorithms.tuners.entropy_tuner import EntropyTuner

from mineral.core.envs.normalized_env import NormalizedEnv
from mineral.core.envs.debug.pointmass_env import PointmassEnv

from mineral.buffers.path_buffer import PathBuffer
from mineral.samplers.parallel_sampler import ParallelSampler


def run_experiment(variant):

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    experiment_id = variant["experiment_id"]
    logging_dir = "./pointmass/sac/{}".format(experiment_id)

    max_path_length = variant["max_path_length"]
    max_size = variant["max_size"]

    num_warm_up_paths = variant["num_warm_up_paths"]
    num_exploration_paths = variant["num_exploration_paths"]
    num_evaluation_paths = variant["num_evaluation_paths"]
    num_trains_per_step = variant["num_trains_per_step"]

    update_tuner_every = variant["update_tuner_every"]
    update_actor_every = variant["update_actor_every"]

    batch_size = variant["batch_size"]
    num_steps = variant["num_steps"]

    monitor = LocalMonitor(logging_dir)

    env = NormalizedEnv(
        PointmassEnv,
        reward_scale=(1 / max_path_length), size=2, ord=2)

    action_dim = env.action_space.shape[0]

    policy = Dense(
        [256, 256, 2 * action_dim],
        tau=1e-1,
        optimizer_class=tf.keras.optimizers.Adam,
        optimizer_kwargs=dict(lr=0.0001),
        distribution_class=TanhGaussian,
        distribution_kwargs=dict(std=None))

    target_policy = policy.clone()

    qf = Dense(
        [256, 256, 1],
        tau=1e-1,
        optimizer_class=tf.keras.optimizers.Adam,
        optimizer_kwargs=dict(lr=0.0001))

    target_qf = qf.clone()

    buffer = PathBuffer(
        max_size=max_size,
        max_path_length=max_path_length,
        selector=(lambda x: x["proprio_observation"]),
        monitor=monitor)

    sampler = ParallelSampler(
        env, policy, buffer,
        max_path_length=max_path_length,
        num_warm_up_paths=num_warm_up_paths,
        num_exploration_paths=num_exploration_paths,
        num_evaluation_paths=num_evaluation_paths,
        time_skips=(1,),
        num_threads=variant["num_threads"],
        selector=(lambda i, x: x["proprio_observation"]),
        monitor=monitor)

    tuner = EntropyTuner(
        policy,
        optimizer_class=tf.keras.optimizers.Adam,
        optimizer_kwargs=dict(lr=0.0001),
        target=(-action_dim),
        update_every=update_tuner_every,
        batch_size=batch_size,
        monitor=monitor)

    critic = SoftQNetwork(
        target_policy,
        qf,
        target_qf,
        gamma=0.99,
        clip_radius=0.2,
        std=0.1,
        alpha=tuner.get_tuning_variable(),
        batch_size=batch_size,
        monitor=monitor)

    actor = SoftActorCritic(
        policy,
        target_policy,
        critic,
        alpha=tuner.get_tuning_variable(),
        update_every=update_actor_every,
        batch_size=batch_size,
        monitor=monitor)

    saver = LocalSaver(
        logging_dir,
        policy=policy,
        target_policy=target_policy,
        qf=qf,
        target_qf=target_qf)

    trainer = LocalTrainer(
        sampler,
        [buffer, buffer, buffer],
        [actor, critic, tuner],
        num_steps=num_steps,
        num_trains_per_step=num_trains_per_step,
        saver=saver,
        monitor=monitor)

    trainer.train()


if __name__ == "__main__":

    num_seeds = 5

    for experiment_id in range(num_seeds):

        variant = dict(
            experiment_id=experiment_id,
            max_path_length=10,
            max_size=10000,
            num_warm_up_paths=100,
            num_exploration_paths=1,
            num_evaluation_paths=10,
            num_trains_per_step=10,
            update_tuner_every=1,
            update_actor_every=1,
            batch_size=10,
            num_threads=16,
            num_steps=100)

        multiprocessing.Process(
            target=run_experiment, args=(variant,)).start()
