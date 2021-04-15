from infogan.model.network import Generator, Discriminator
from infogan.data.generate_toy_example import generate_circle_toy_data
import torch

def test_generator_output():

    noise = torch.randn((400, 32))  # noise: 32
    bernoulli = torch.distributions.bernoulli.Bernoulli(0.5)
    discrete_code = bernoulli.sample((400, 1))  # discrete code: 1
    data = torch.cat([noise, discrete_code], dim=1)

    generator = Generator(noise_dim=32, discrete_code_dim=1, continuous_code_dim=0, out_dim=2)
    out = generator(data)
    assert out.shape == (400, 2)

def test_discriminator_output():
    data = torch.Tensor(generate_circle_toy_data())
    discriminator = Discriminator(data_dim=2, discrete_code_dim=3, continuous_code_dim=2)
    regular_gan_out, q_discrete, q_mu, q_sigma = discriminator(data)

    assert regular_gan_out.shape == (400, 1)
    assert q_discrete.shape[1] == 3
    assert q_mu.shape[1] == 2
    assert q_sigma.shape[1] == 2