"""Author: Brandon Trabucco, Copyright 2019"""


import gym
from jetpack.networks.fully_connected import FullyConnectedPolicy, FullyConnectedQF
from jetpack.wrappers.proxy_env import ProxyEnv
from jetpack.data.experience_replay import ExperienceReplay
from jetpack.algorithms.td3 import TD3
from jetpack.core.batch_trainer import BatchTrainer
from jetpack.core.monitor import Monitor


if __name__ == "__main__":

    monitor = Monitor("./")

    env = ProxyEnv(
        gym.make("Pendulum-v0")
    )

    policy = FullyConnectedPolicy(
        [6, 6, 1],
        tau=1e-2,
        optimizer_kwargs={"lr": 0.0001}
    )

    qf1 = FullyConnectedQF(
        [6, 6],
        tau=1e-2,
        optimizer_kwargs={"lr": 0.0001}
    )

    qf2 = FullyConnectedQF(
        [6, 6],
        tau=1e-2,
        optimizer_kwargs={"lr": 0.0001}
    )

    target_policy = FullyConnectedPolicy(
        [6, 6, 1],
        tau=1e-2,
        optimizer_kwargs={"lr": 0.0001}
    )

    target_qf1 = FullyConnectedQF(
        [6, 6],
        tau=1e-2,
        optimizer_kwargs={"lr": 0.0001}
    )

    target_qf2 = FullyConnectedQF(
        [6, 6],
        tau=1e-2,
        optimizer_kwargs={"lr": 0.0001}
    )

    target_policy.set_weights(policy.get_weights())
    target_qf1.set_weights(qf1.get_weights())
    target_qf2.set_weights(qf2.get_weights())

    replay = ExperienceReplay(
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
        clip_radius=0.2,
        sigma=0.1,
        gamma=0.99,
        actor_delay=10,
        monitor=monitor
    )
    
    max_size = 1000
    num_warm_up_paths = 10
    num_steps = 20000
    num_paths_to_collect = 1
    max_path_length = 100
    batch_size = 32
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
        algorithm,
        monitor=monitor
    )

    try:
        trainer.train()

    except KeyboardInterrupt:
        replay.evaluate(1, 1000, True, {})
