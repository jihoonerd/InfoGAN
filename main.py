from infogan.data.generate_toy_example import generate_circle_toy_data
from infogan.model.network import Generator, Discriminator
from infogan.model.loss import NormalNLLLoss
import torch
import torch.nn as nn
import torch.optim as optim

def infogan():
    # set training settings
    training_epochs = 100
    noise_vector_size = 32
    discrete_code_dim = 1
    continuous_code_dim = 0
    data_dim = 2

    # load data
    data  = torch.Tensor(generate_circle_toy_data())

    # define GAN structure
    generator = Generator(noise_dim=noise_vector_size, discrete_code_dim=discrete_code_dim, continuous_code_dim=continuous_code_dim, out_dim=data_dim)
    discriminator = Discriminator(data_dim=data_dim, discrete_code_dim=discrete_code_dim, continuous_code_dim=continuous_code_dim)
    
    discriminator_loss = nn.BCELoss()
    generator_discrete_loss = nn.NLLLoss()
    generator_continuous_loss = NormalNLLLoss()

    g_optimizer = optim.Adam(generator.parameters()) 
    d_optimizer = optim.Adam(discriminator.parameters())

    fixed_noise_z = torch.randn((data.shape[0], noise_vector_size))


    noise = torch.randn((400, 32))  # noise: 32
    bernoulli = torch.distributions.bernoulli.Bernoulli(0.5)
    discrete_code = bernoulli.sample((400, 1))  # discrete code: 1
    data = torch.cat([noise, discrete_code], dim=1)

if __name__ == '__main__':
    infogan()
