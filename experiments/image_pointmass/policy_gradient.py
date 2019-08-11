"""Author: Brandon Trabucco, Copyright 2019"""


from mineral.networks.conv import Conv
from mineral.distributions.gaussians.tanh_gaussian_distribution import TanhGaussianDistribution
from mineral.core.envs.normalized_env import NormalizedEnv
from mineral.core.envs.image_pointmass_env import ImagePointmassEnv
from mineral.buffers.path_buffer import PathBuffer
from mineral.algorithms.actors.policy_gradient import PolicyGradient
from mineral.core.trainers.local_trainer import LocalTrainer
from mineral.core.monitors.local_monitor import LocalMonitor


if __name__ == "__main__":

    monitor = LocalMonitor("./image_pointmass/policy_gradient")

    max_path_length = 10

    env = NormalizedEnv(
        ImagePointmassEnv(image_size=48, size=2, ord=2),
        reward_scale=(1 / max_path_length)
    )

    policy = Conv(
        [8, 16, 32],
        [5, 5, 5],
        [2, 2, 2],
        [32, 32, 4],
        optimizer_kwargs=dict(lr=0.0001),
        distribution_class=TanhGaussianDistribution,
        distribution_kwargs=dict(std=None)
    )

    buffer = PathBuffer(
        env,
        policy
    )

    algorithm = PolicyGradient(
        policy,
        gamma=0.99,
        monitor=monitor
    )
    
    max_size = 32
    num_warm_up_paths = 0
    num_steps = 1000
    num_paths_to_collect = max_size
    batch_size = max_size
    num_trains_per_step = 1

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
