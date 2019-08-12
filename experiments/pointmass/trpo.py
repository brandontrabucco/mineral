"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.algorithms.actors.importance_sampling import ImportanceSampling
from mineral.algorithms.critics.gae import GAE
from mineral.algorithms.multi_algorithm import MultiAlgorithm
from mineral.networks.dense import Dense
from mineral.distributions.gaussians.tanh_gaussian_distribution import TanhGaussianDistribution
from mineral.core.envs.normalized_env import NormalizedEnv
from mineral.core.envs.pointmass_env import PointmassEnv
from mineral.buffers.path_buffer import PathBuffer
from mineral.core.trainers.local_trainer import LocalTrainer
from mineral.core.monitors.local_monitor import LocalMonitor
from mineral.optimizers.constraints.kl_constraint import KLConstraint
from mineral.optimizers.gradients.natural_gradient import NaturalGradient
from mineral.optimizers.line_search import LineSearch


if __name__ == "__main__":

    monitor = LocalMonitor("./pointmass/trpo")

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

    old_policy = Dense(
        [32, 32, 4],
        optimizer_class=tf.keras.optimizers.Adam,
        optimizer_kwargs=dict(lr=0.0001),
        distribution_class=TanhGaussianDistribution,
        distribution_kwargs=dict(std=None)
    )

    vf = Dense(
        [6, 6, 1],
        optimizer_kwargs={"lr": 0.0001},
    )

    policy = KLConstraint(
        LineSearch(
            NaturalGradient(
                policy, return_sAs=True
            ), use_sAs=True
        ), old_policy, delta=0.1
    )

    target_vf = Dense(
        [6, 6, 1],
        optimizer_kwargs={"lr": 0.0001},
    )

    buffer = PathBuffer(
        env,
        policy
    )

    num_trains_per_step = 32
    off_policy_updates = 10

    critic = GAE(
        vf,
        target_vf,
        gamma=0.99,
        lamb=0.95,
        monitor=monitor,
    )

    actor = ImportanceSampling(
        policy,
        old_policy,
        critic,
        gamma=0.99,
        old_update_every=num_trains_per_step,
        update_every=num_trains_per_step // off_policy_updates,
        selector=(lambda x: x["proprio_observation"]),
        monitor=monitor
    )

    algorithm = MultiAlgorithm(
        critic,
        actor
    )
    
    max_size = 32
    num_warm_up_paths = 0
    num_steps = 1000
    num_paths_to_collect = max_size
    batch_size = max_size

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
