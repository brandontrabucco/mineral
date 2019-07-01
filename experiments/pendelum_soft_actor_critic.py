"""Author: Brandon Trabucco, Copyright 2019"""


import gym
from jetpack.networks.policies.gaussian_policy import GaussianPolicy
from jetpack.networks.policies.tanh_policy import TanhPolicy
from jetpack.networks.dense.dense_q_function import DenseQFunction
from jetpack.envs.normalized_env import NormalizedEnv
from jetpack.data.off_policy_buffer import OffPolicyBuffer
from jetpack.algorithms.soft_actor_critic import SoftActorCritic
from jetpack.algorithms.critics.soft_q_learning import SoftQLearning
from jetpack.core.local_trainer import LocalTrainer
from jetpack.core.local_monitor import LocalMonitor


if __name__ == "__main__":

    monitor = LocalMonitor("./")

    env = NormalizedEnv(
        gym.make("Pendulum-v0")
    )

    policy = TanhPolicy(
        GaussianPolicy(
            [6, 6, 1],
            lr=0.0001
        )
    )

    qf = DenseQFunction(
        [6, 6, 1],
        lr=0.001
    )

    target_policy = TanhPolicy(
        GaussianPolicy(
            [6, 6, 1],
            lr=0.0001
        )
    )

    target_qf = DenseQFunction(
        [6, 6, 1],
        tau=1e-2,
    )

    buffer = OffPolicyBuffer(
        env,
        policy
    )

    clip_radius = 0.2
    sigma = 0.1
    gamma = 0.99
    actor_delay = 10

    q_backup = SoftQLearning(
        qf,
        target_policy,
        target_qf,
        gamma=gamma,
        clip_radius=clip_radius,
        sigma=sigma,
        monitor=monitor,
    )

    algorithm = SoftActorCritic(
        policy,
        q_backup,
        target_policy,
        actor_delay=actor_delay,
        monitor=None,
    )
    
    max_size = 1000
    num_warm_up_paths = 10
    num_steps = 20000
    num_paths_to_collect = 1
    max_path_length = 100
    batch_size = 32
    num_trains_per_step = 100

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

    try:
        trainer.train()

    except KeyboardInterrupt:
        buffer.evaluate(1, render=True)