from infogan.model.network import Generator, Discriminator
from infogan.data.generate_toy_example import generate_circle_toy_data
import torch

def test_generator_output():
    data = torch.randn((400, 33))  # noise: 32, code: 1
    generator = Generator(noise_dim=32, continuous_code_dim=1, discrete_code_dim=0, out_dim=2)
    out = generator(data)
    assert out.shape == (400, 2)

def test_discriminator_output():
    data = torch.Tensor(generate_circle_toy_data())
    discriminator = Discriminator(data_dim=2, continuous_code_dim=1, discrete_code_dim=0)
    out = discriminator(data)
    assert out.shape == (400, 2)
    
