"""Author: Brandon Trabucco, Copyright 2019"""


from mineral.algorithms.vaes.observation_vae import ObservationVAE
from mineral.networks.conv import Conv
from mineral.networks.dense import Dense
from mineral.distributions.gaussians.tanh_gaussian_distribution import TanhGaussianDistribution
from mineral.distributions.gaussians.gaussian_distribution import GaussianDistribution
from mineral.core.envs.normalized_env import NormalizedEnv
from mineral.core.envs.image_pointmass_env import ImagePointmassEnv
from mineral.buffers.path_buffer import PathBuffer
from mineral.core.trainers.local_trainer import LocalTrainer
from mineral.core.monitors.local_monitor import LocalMonitor
from mineral.networks.conv_transpose import ConvTranspose
from mineral.networks.latent_variable import LatentVariable


if __name__ == "__main__":

    monitor = LocalMonitor("./image_pointmass/observation_vae")

    max_path_length = 10

    env = NormalizedEnv(
        ImagePointmassEnv(image_size=48, size=2, ord=2),
        reward_scale=(1 / max_path_length))

    policy = Dense(
        [32, 32, 4],
        optimizer_kwargs=dict(lr=0.0001),
        distribution_class=TanhGaussianDistribution,
        distribution_kwargs=dict(std=None))

    max_size = 5096

    buffer = PathBuffer(
        env,
        policy,
        max_size=max_size,
        max_path_length=max_path_length,
        selector=(lambda x: x["proprio_observation"]),
        monitor=monitor)

    latent_size = 32

    encoder = Conv(
        [8, 16, 32],
        [5, 5, 5],
        [2, 2, 2],
        [2 * latent_size, 2 * latent_size],
        optimizer_kwargs=dict(lr=0.0001),
        distribution_class=GaussianDistribution,
        distribution_kwargs=dict(std=None))

    decoder = ConvTranspose(
        [16, 8, 3],
        [5, 5, 5],
        [2, 2, 2],
        [2 * latent_size, 6 * 6 * 32],
        [6, 6, 32],
        optimizer_kwargs=dict(lr=0.0001),
        distribution_class=GaussianDistribution,
        distribution_kwargs=dict(std=1.0))

    vae_network = LatentVariable(
        encoder,
        decoder,
        latent_size,
        beta=1.0,
        sample_encoder=True,
        sample_decoder=False)

    algorithm = ObservationVAE(
        vae_network,
        selector=(lambda x: x["image_observation"]),
        monitor=monitor)

    num_warm_up_paths = 32
    num_steps = 1000
    num_paths_to_collect = 32
    batch_size = 32
    num_trains_per_step = 64

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
