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
        clip_radius=2.0,
        sigma=1.0,
        gamma=0.99,
        actor_delay=100
    )
    
    max_size = 1000
    num_warm_up_paths = 10
    num_steps = 1000
    num_paths_to_collect = 1
    max_path_length = 1000
    batch_size = 256
    num_trains_per_step = 100

    trainer = BatchTrainer(
        max_size,
        num_warm_up_paths,
        num_steps,
        num_paths_to_collect,
        max_path_length,
        batch_size,
        num_trains_per_step,
        replay,
        algorithm
    )

    trainer.train()