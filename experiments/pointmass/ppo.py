"""Author: Brandon Trabucco, Copyright 2019"""


from mineral.algorithms.actors.ppo import PPO
from mineral.algorithms.critics.gae import GAE
from mineral.algorithms.merged import Merged
from mineral.networks.dense import Dense
from mineral.distributions.gaussians.tanh_gaussian_distribution import TanhGaussianDistribution
from mineral.core.envs.normalized_env import NormalizedEnv
from mineral.core.envs.pointmass_env import PointmassEnv
from mineral.buffers.path_buffer import PathBuffer
from mineral.core.trainers.local_trainer import LocalTrainer
from mineral.core.monitors.local_monitor import LocalMonitor


if __name__ == "__main__":

    monitor = LocalMonitor("./pointmass/ppo")

    max_path_length = 10

    env = NormalizedEnv(
        PointmassEnv(size=2, ord=2),
        reward_scale=(1 / max_path_length)
    )

    policy = Dense(
        [32, 32, 1],
        optimizer_kwargs={"lr": 0.0001},
        distribution_class=TanhGaussianDistribution
    )

    old_policy = Dense(
        [32, 32, 1],
        optimizer_kwargs={"lr": 0.0001},
        distribution_class=TanhGaussianDistribution
    )

    vf = Dense(
        [6, 6, 1],
        optimizer_kwargs={"lr": 0.0001},
    )

    target_vf = Dense(
        [6, 6, 1],
        optimizer_kwargs={"lr": 0.0001},
    )

    buffer = PathBuffer(
        env,
        policy,
        selector=(lambda x: x["proprio_observation"])
    )

    num_trains_per_step = 32
    off_policy_updates = 10

    critic = GAE(
        vf,
        target_vf,
        gamma=1.0,
        lamb=1.0,
        monitor=monitor,
    )

    actor = PPO(
        policy,
        old_policy,
        critic,
        gamma=0.99,
        epsilon=0.2,
        old_update_every=num_trains_per_step,
        update_every=num_trains_per_step // off_policy_updates,
        selector=(lambda x: x["proprio_observation"]),
        monitor=monitor
    )

    algorithm = Merged(
        critic,
        actor
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
