"""Author: Brandon Trabucco, Copyright 2019"""


from mineral.algorithms.actors.actor_critic import ActorCritic
from mineral.algorithms.critics.gae import GAE
from mineral.algorithms.multi_algorithm import MultiAlgorithm
from mineral.networks.dense import Dense
from mineral.distributions.gaussians.tanh_gaussian_distribution import TanhGaussianDistribution
from mineral.core.envs.normalized_env import NormalizedEnv
from mineral.core.envs.pointmass_env import PointmassEnv
from mineral.buffers.path_buffer import PathBuffer
from mineral.core.trainers.local_trainer import LocalTrainer
from mineral.core.monitors.local_monitor import LocalMonitor


if __name__ == "__main__":

    monitor = LocalMonitor("./pointmass/actor_critic")

    max_path_length = 10

    env = NormalizedEnv(
        PointmassEnv(size=2, ord=2),
        reward_scale=(1 / max_path_length)
    )

    policy = Dense(
        [32, 32, 4],
        optimizer_kwargs={"lr": 0.0001},
        distribution_class=TanhGaussianDistribution,
        distribution_kwargs=dict(std=None)
    )

    vf = Dense(
        [6, 6, 1],
        optimizer_kwargs={"lr": 0.0001},
    )

    target_vf = Dense(
        [6, 6, 1],
        optimizer_kwargs={"lr": 0.0001},
    )

    max_size = 32

    buffer = PathBuffer(
        env,
        policy,
        max_size=max_size,
        max_path_length=max_path_length,
        selector=(lambda x: x["proprio_observation"]),
        monitor=monitor
    )

    critic = GAE(
        vf,
        target_vf,
        gamma=0.99,
        lamb=0.95,
        selector=(lambda x: x["proprio_observation"]),
        monitor=monitor,
    )

    num_trains_per_step = 32

    actor = ActorCritic(
        policy,
        critic,
        gamma=0.99,
        update_every=num_trains_per_step,
        selector=(lambda x: x["proprio_observation"]),
        monitor=monitor
    )

    algorithm = MultiAlgorithm(
        critic,
        actor
    )

    num_warm_up_paths = 0
    num_steps = 1000
    num_paths_to_collect = max_size
    batch_size = max_size

    trainer = LocalTrainer(
        buffer,
        algorithm,
        num_warm_up_paths=num_warm_up_paths,
        num_steps=num_steps,
        num_paths_to_collect=num_paths_to_collect,
        batch_size=batch_size,
        num_trains_per_step=num_trains_per_step,
        monitor=monitor
    )

    trainer.train()
