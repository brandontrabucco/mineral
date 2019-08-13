"""Author: Brandon Trabucco, Copyright 2019"""


from mineral.networks.dense import Dense
from mineral.distributions.gaussians.tanh_gaussian_distribution import TanhGaussianDistribution
from mineral.core.envs.normalized_env import NormalizedEnv
from mineral.core.envs.pointmass_env import PointmassEnv
from mineral.buffers.path_buffer import PathBuffer
from mineral.algorithms.actors.policy_gradient import PolicyGradient
from mineral.core.trainers.local_trainer import LocalTrainer
from mineral.core.monitors.local_monitor import LocalMonitor


if __name__ == "__main__":

    monitor = LocalMonitor("./pointmass/policy_gradient")

    max_path_length = 10

    env = NormalizedEnv(
        PointmassEnv(size=2, ord=2),
        reward_scale=(1 / max_path_length))

    policy = Dense(
        [32, 32, 4],
        optimizer_kwargs=dict(lr=0.0001),
        distribution_class=TanhGaussianDistribution,
        distribution_kwargs=dict(std=None))

    max_size = 32

    buffer = PathBuffer(
        env,
        policy,
        max_size=max_size,
        max_path_length=max_path_length,
        selector=(lambda x: x["proprio_observation"]),
        monitor=monitor)

    algorithm = PolicyGradient(
        policy,
        gamma=0.99,
        selector=(lambda x: x["proprio_observation"]),
        monitor=monitor)

    num_warm_up_paths = 0
    num_steps = 1000
    num_paths_to_collect = max_size
    batch_size = max_size
    num_trains_per_step = 1

    trainer = LocalTrainer(
        buffer,
        algorithm,
        num_warm_up_paths=num_warm_up_paths,
        num_steps=num_steps,
        num_paths_to_collect=num_paths_to_collect,
        batch_size=batch_size,
        num_trains_per_step=num_trains_per_step,
        monitor=monitor)

    trainer.train()
