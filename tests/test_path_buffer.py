"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf

from mineral.core.monitors.local_monitor import LocalMonitor

from mineral.networks.dense import Dense
from mineral.distributions.gaussians.tanh_gaussian import TanhGaussian

from mineral.core.envs.normalized_env import NormalizedEnv
from mineral.core.envs.debug.pointmass_env import PointmassEnv

from mineral.core.buffers.path_buffer import PathBuffer
from mineral.core.samplers.parallel_sampler import ParallelSampler


if __name__ == "__main__":

    variant = dict(
        experiment_id=0,
        max_path_length=3,
        max_size=2,
        num_warm_up_paths=2,
        num_exploration_paths=0,
        num_evaluation_paths=0,
        num_trains_per_step=1,
        update_tuner_every=1,
        update_actor_every=1,
        batch_size=3,
        num_threads=2,
        num_steps=1)

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    experiment_id = variant["experiment_id"]
    logging_dir = "./test/sac/{}".format(
        experiment_id)

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
        PointmassEnv, size=2, ord=2)

    policy = Dense(
        [256, 256, 4],
        tau=1e-1,
        optimizer_class=tf.keras.optimizers.Adam,
        optimizer_kwargs=dict(lr=0.0001),
        distribution_class=TanhGaussian,
        distribution_kwargs=dict(std=None))

    buffer = PathBuffer(
        max_size=variant["max_size"],
        max_path_length=variant["max_path_length"],
        selector=(lambda x: x["proprio_observation"]),
        monitor=monitor)

    sampler = ParallelSampler(
        env, policy, buffer,
        max_path_length=variant["max_path_length"],
        num_warm_up_paths=variant["num_warm_up_paths"],
        num_exploration_paths=variant["num_exploration_paths"],
        num_evaluation_paths=variant["num_evaluation_paths"],
        num_threads=variant["num_threads"],
        selector=(lambda i, x: x["proprio_observation"]),
        monitor=monitor)

    sampler.warm_up()
    sampler.explore()
    sampler.evaluate()
    print("DONE: {}".format(sampler.num_steps_collected))

    batch = buffer.sample(32)
    print(batch)

    import ipdb; ipdb.set_trace()
