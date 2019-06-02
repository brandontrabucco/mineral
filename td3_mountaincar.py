"""Author: Brandon Trabucco, Copyright 2019"""


import gym
from jetpack.networks.fully_connected import FullyConnectedPolicy, FullyConnectedQF
from jetpack.wrappers.proxy_env import ProxyEnv
from jetpack.data.experience_replay import ExperienceReplay
from jetpack.algorithms.td3 import TD3
from jetpack.core.batch_trainer import BatchTrainer


if __name__ == "__main__":

    env = ProxyEnv(
        gym.make("MountainCarContinuous-v0")
    )

    policy = FullyConnectedPolicy(
        [16, 16, 1]
    )

    qf1 = FullyConnectedQF(
        [32, 32]
    )

    qf2 = FullyConnectedQF(
        [32, 32]
    )

    target_policy = FullyConnectedPolicy(
        [16, 16, 1]
    )

    target_qf1 = FullyConnectedQF(
        [32, 32]
    )

    target_qf2 = FullyConnectedQF(
        [32, 32]
    )

    selector = lambda x: x

    replay = ExperienceReplay(
        selector,
        env,
        policy
    )

    algorithm = TD3(
        policy,
        qf1,
        qf2,
        target_policy,
        target_qf1,
        target_qf2,
        clip_radius=1.0,
        sigma=1.0,
        gamma=1.0,
        actor_delay=10
    )
    
    max_size = 100
    num_steps = 1000
    num_paths_to_collect = 10
    max_path_length = 100
    batch_size = 32
    num_trains_per_step = 5

    trainer = BatchTrainer(
        max_size,
        num_steps,
        num_paths_to_collect,
        max_path_length,
        batch_size,
        num_trains_per_step,
        replay,
        algorithm
    )

    trainer.train()