"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.networks.network import Network


class EncoderDecoder(Network):

    def __init__(
        self,
        encoder,
        decoder,
        latent_size,
        beta=1.0,
        sample_encoder=False,
        sample_decoder=False,
    ):
        tf.keras.Model.__init__(self)
        self.encoder = encoder
        self.decoder = decoder
        self.latent_size = latent_size
        self.beta = beta
        self.sample_encoder = sample_encoder
        self.sample_decoder = sample_decoder
        self.grad_length = 0

    def call(self, *inputs, **kwargs):
        pass

    def compute_gradients(self, loss_function, *inputs, **kwargs):
        encoder_grad = self.encoder.compute_gradients(loss_function, *inputs, **kwargs)
        decoder_grad = self.decoder.compute_gradients(loss_function, *inputs, **kwargs)
        self.grad_length = len(encoder_grad)
        return encoder_grad + decoder_grad

    def apply_gradients(self, gradients):
        self.encoder.apply_gradients(gradients[:self.grad_length])
        self.decoder.apply_gradients(gradients[self.grad_length:])

    def get_activations(self, *inputs, **kwargs):
        pass

    def get_parameters(self, *inputs, **kwargs):
        if self.sample_encoder:
            latent_variable = self.encoder.sample(*inputs, **kwargs)
        else:
            latent_variable = self.encoder.get_expected_value(*inputs, **kwargs)
        return (self.encoder.get_parameters(*inputs, **kwargs) +
                self.decoder.get_parameters(latent_variable, **kwargs))

    def sample(self, *inputs, **kwargs):
        if self.sample_encoder:
            latent_variable = self.encoder.sample(*inputs, **kwargs)
        else:
            latent_variable = self.encoder.get_expected_value(*inputs, **kwargs)
        if self.sample_decoder:
            return self.decoder.sample(latent_variable, **kwargs)
        else:
            return self.decoder.get_expected_value(latent_variable, **kwargs)

    def sample_from_prior(self, **kwargs):
        if self.sample_encoder:
            latent_variable = tf.random.normal([1, self.latent_size])
        else:
            latent_variable = tf.zeros([1, self.latent_size])
        if self.sample_decoder:
            return self.decoder.sample(latent_variable, **kwargs)
        else:
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
