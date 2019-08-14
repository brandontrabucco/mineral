"""Author: Brandon Trabucco, Copyright 2019"""


from mineral.algorithms.actors.importance_sampling import ImportanceSampling
from mineral.algorithms.critics.gae import GAE
from mineral.algorithms.multi_algorithm import MultiAlgorithm
from mineral.networks.dense import Dense
from mineral.distributions.gaussians.tanh_gaussian import TanhGaussian
from mineral.core.envs.normalized_env import NormalizedEnv
from mineral.core.envs.debug.pointmass_env import PointmassEnv
from mineral.buffers.path_buffer import PathBuffer
from mineral.core.trainers.local_trainer import LocalTrainer
from mineral.core.monitors.local_monitor import LocalMonitor
from mineral.optimizers.constraints.kl_constraint import KLConstraint
from mineral.optimizers.gradients.natural_gradient import NaturalGradient
from mineral.optimizers.line_search import LineSearch
from mineral.samplers.path_sampler import PathSampler


if __name__ == "__main__":

    max_path_length = 10
    max_size = 32
    num_warm_up_samples = 0
    num_exploration_samples = 32
    num_evaluation_samples = 32
    num_trains_per_step = 100
    update_actor_every = 1
    batch_size = 100
    num_steps = 10000

    monitor = LocalMonitor("./pointmass/trpo")

    env = NormalizedEnv(
        PointmassEnv(size=2, ord=2),
        reward_scale=(1 / max_path_length))

    policy = Dense(
        [32, 32, 4],
        optimizer_kwargs={"lr": 0.0001},
        distribution_class=TanhGaussian,
        distribution_kwargs=dict(std=None))

    old_policy = Dense(
        [32, 32, 4],
        optimizer_kwargs={"lr": 0.0001},
        distribution_class=TanhGaussian,
        distribution_kwargs=dict(std=None))

    policy = KLConstraint(
        LineSearch(
            NaturalGradient(
                policy, return_sAs=True), use_sAs=True), old_policy, delta=0.1)

    vf = Dense(
        [6, 6, 1],
        optimizer_kwargs={"lr": 0.0001})

    target_vf = Dense(
        [6, 6, 1],
        optimizer_kwargs={"lr": 0.0001})

    buffer = PathBuffer(
        max_size=max_size,
        max_path_length=max_path_length,
        selector=(lambda x: x["proprio_observation"]),
        monitor=monitor)

    sampler = PathSampler(
        buffer,
        env,
        policy,
        num_warm_up_samples=num_warm_up_samples,
        num_exploration_samples=num_exploration_samples,
        num_evaluation_samples=num_evaluation_samples,
        monitor=monitor)

    critic = GAE(
        vf,
        target_vf,
        gamma=0.99,
        lamb=0.95,
        batch_size=batch_size,
        monitor=monitor)

    actor = ImportanceSampling(
        policy,
        old_policy,
        critic,
        gamma=0.99,
        epsilon=0.1,
        update_every=num_trains_per_step,
        batch_size=batch_size,
        monitor=monitor)

    algorithm = MultiAlgorithm(actor, critic)

    trainer = LocalTrainer(
        sampler,
        buffer,
        algorithm,
        num_steps=num_steps,
        num_trains_per_step=num_trains_per_step,
        monitor=monitor)

    trainer.train()
