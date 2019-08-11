"""Author: Brandon Trabucco, Copyright 2019"""


from mineral.algorithms.dynamics_models.one_step_prediction import OneStepPrediction
from mineral.networks.conv.conv_policy import ConvPolicy
from mineral.networks.conv.conv_network import ConvNetwork
from mineral.distributions.gaussians.tanh_gaussian_distribution import TanhGaussianDistribution
from mineral.distributions.gaussians.gaussian_distribution import GaussianDistribution
from mineral.core.envs.normalized_env import NormalizedEnv
from mineral.core.envs.image_pointmass_env import ImagePointmassEnv
from mineral.buffers.path_buffer import PathBuffer
from mineral.core.trainers.local_trainer import LocalTrainer
from mineral.core.monitors.local_monitor import LocalMonitor
from mineral.networks.conv_transpose.conv_transpose_network import ConvTransposeNetwork
from mineral.networks.vae.vae_forward_model import VAEForwardModel


if __name__ == "__main__":

    monitor = LocalMonitor("./image_pointmass/vae_dynamics")

    max_path_length = 10

    env = NormalizedEnv(
        ImagePointmassEnv(image_size=48, size=2, ord=2),
        reward_scale=(1 / max_path_length)
    )

    policy = ConvPolicy(
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

    latent_size = 32

    encoder = ConvNetwork(
        [8, 16, 32],
        [5, 5, 5],
        [2, 2, 2],
        [2 * latent_size, 2 * latent_size],
        optimizer_kwargs=dict(lr=0.0001),
        distribution_class=GaussianDistribution,
        distribution_kwargs=dict(std=None)
    )

    decoder = ConvTransposeNetwork(
        [16, 8, 3],
        [5, 5, 5],
        [2, 2, 2],
        [2 * latent_size, 6 * 6 * 32],
        [6, 6, 32],
        optimizer_kwargs=dict(lr=0.0001),
        distribution_class=GaussianDistribution,
        distribution_kwargs=dict(std=1.0)
    )

    model = VAEForwardModel(
        encoder,
        decoder,
        latent_size,
        beta=1.0
    )

    monitor.record("sample_from_prior", model.sample_from_prior())

    algorithm = OneStepPrediction(
        model,
        monitor=monitor
    )
    
    max_size = 512
    num_warm_up_paths = max_size
    num_steps = 100
    num_paths_to_collect = 32
    batch_size = 32
    num_trains_per_step = 64

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

    monitor.record("sample_from_prior", model.sample_from_prior())
