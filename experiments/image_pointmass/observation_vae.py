"""Author: Brandon Trabucco, Copyright 2019"""


from mineral.algorithms.vaes.observation_vae import ObservationVAE
from mineral.networks.conv import Conv
from mineral.networks.dense import Dense
from mineral.distributions.gaussians.tanh_gaussian import TanhGaussian
from mineral.distributions.gaussians.gaussian import Gaussian
from mineral.core.envs.normalized_env import NormalizedEnv
from mineral.core.envs.debug.image_pointmass_env import ImagePointmassEnv
from mineral.buffers.path_buffer import PathBuffer
from mineral.core.trainers.local_trainer import LocalTrainer
from mineral.core.monitors.local_monitor import LocalMonitor
from mineral.networks.conv_transpose import ConvTranspose
from mineral.networks.variational import Variational
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
    latent_size = 32

    monitor = LocalMonitor("./image_pointmass/observation_vae")

    env = NormalizedEnv(
        ImagePointmassEnv(image_size=48, size=2, ord=2),
        reward_scale=(1 / max_path_length))

    policy = Dense(
        [32, 32, 4],
        optimizer_kwargs=dict(lr=0.0001),
        distribution_class=TanhGaussian,
        distribution_kwargs=dict(std=None))

    buffer = PathBuffer(
        max_size=max_size,
        max_path_length=max_path_length,
        selector=(lambda x: x["image_observation"]),
        monitor=monitor)

    sampler = PathSampler(
        buffer,
        env,
        policy,
        num_warm_up_samples=num_warm_up_samples,
        num_exploration_samples=num_exploration_samples,
        num_evaluation_samples=num_evaluation_samples,
        selector=(lambda x: x["proprio_observation"]),
        monitor=monitor)

    encoder = Conv(
        [8, 16, 32],
        [5, 5, 5],
        [2, 2, 2],
        [2 * latent_size, 2 * latent_size],
        optimizer_kwargs=dict(lr=0.0001),
        distribution_class=Gaussian,
        distribution_kwargs=dict(std=None))

    decoder = ConvTranspose(
        [16, 8, 3],
        [5, 5, 5],
        [2, 2, 2],
        [2 * latent_size, 6 * 6 * 32],
        [6, 6, 32],
        optimizer_kwargs=dict(lr=0.0001),
        distribution_class=Gaussian,
        distribution_kwargs=dict(std=1.0))

    vae_network = Variational(
        encoder,
        decoder,
        latent_size,
        beta=1.0,
        sample_encoder=True,
        sample_decoder=False)

    algorithm = ObservationVAE(
        vae_network,
        batch_size=batch_size,
        monitor=monitor)

    trainer = LocalTrainer(
        sampler,
        buffer,
        algorithm,
        num_steps=num_steps,
        num_trains_per_step=num_trains_per_step,
        monitor=monitor)

    trainer.train()
