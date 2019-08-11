"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.networks.network import Network


class VAENetwork(Network):

    def __init__(
        self,
        encoder,
        decoder,
        latent_size,
        beta=1.0
    ):
        tf.keras.Model.__init__(self)
        self.encoder = encoder
        self.decoder = decoder
        self.latent_size = latent_size
        self.beta = beta
        self.grad_length = 0

    def call(
        self,
        *inputs,
        **kwargs
    ):
        pass

    def compute_gradients(
        self,
        loss_function,
        *inputs
    ):
        encoder_grad = self.encoder.compute_gradients(loss_function, *inputs)
        decoder_grad = self.decoder.compute_gradients(loss_function, *inputs)
        self.grad_length = len(encoder_grad)
        return encoder_grad + decoder_grad

    def apply_gradients(
        self,
        gradients
    ):
        self.encoder.apply_gradients(gradients[:self.grad_length])
        self.decoder.apply_gradients(gradients[self.grad_length:])

    def soft_update(
        self,
        weights
    ):
        self.set_weights([
            self.tau * w + (1.0 - self.tau) * w_self
            for w, w_self in zip(weights, self.get_weights())
        ])

    def get_activations(self, *inputs, **kwargs):
        pass

    def get_parameters(self, *inputs, **kwargs):
        latent_variable = self.encoder.sample(*inputs, **kwargs)
        return (self.encoder.get_parameters(*inputs, **kwargs) +
                self.decoder.get_parameters(latent_variable, **kwargs))

    def sample(self, *inputs, **kwargs):
        latent_variable = self.encoder.sample(*inputs, **kwargs)
        return self.decoder.get_expected_value(latent_variable, **kwargs)

    def sample_from_prior(self, **kwargs):
        latent_variable = tf.random.normal([1, self.latent_size])
        return self.decoder.get_expected_value(latent_variable, **kwargs)

    def get_expected_value(self, *inputs, **kwargs):
        latent_variable = self.encoder.get_expected_value(*inputs, **kwargs)
        return self.decoder.get_expected_value(latent_variable, **kwargs)

    def get_log_probs(self, *inputs, **kwargs):
        x, *inputs = inputs
        kl_divergence = self.encoder.get_kl_divergence("prior", *inputs, **kwargs)
        latent_variable = self.encoder.get_expected_value(*inputs, **kwargs)
        log_probs = self.decoder.get_log_probs(x, latent_variable, **kwargs)
        while len(log_probs.shape) > len(kl_divergence.shape):
            log_probs = tf.reduce_mean(log_probs, -1)
        return log_probs - self.beta * kl_divergence

    def get_kl_divergence(self, pi, *inputs, **kwargs):
        sample = self.sample(*inputs, **kwargs)
        other_sample = pi.sample(*inputs, **kwargs)
        log_probs = self.get_log_probs(sample, *inputs, **kwargs)
        other_log_probs = pi.get_log_probs(other_sample, *inputs, **kwargs)
        return log_probs - other_log_probs

    def get_fisher_information(self, *inputs, **kwargs):
        return (self.encoder.get_fisher_information(*inputs[:len(inputs)//2], **kwargs) +
                self.decoder.get_fisher_information(*inputs[len(inputs)//2:], **kwargs))