import pathlib

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from infogan.data.generate_toy_example import generate_circle_toy_data
from infogan.data.utils import generate_latent_sample
from infogan.model.network import Generator


def infogan():

    # Save original data
    p = pathlib.Path('assets/original/')  # directory for original data
    p_gen = pathlib.Path('assets/generated/')  # direcdtory for generated data
    p.mkdir(parents=True, exist_ok=True)
    p_gen.mkdir(parents=True, exist_ok=True)
    data = generate_circle_toy_data()
    plt.scatter(data[:, 0], data[:, 1])
    plt.savefig('assets/original/data.png')

    # set training settings
    training_epochs = 5000
    noise_vector_dim = 32
    discrete_code_dim = 2
    continuous_code_dim = 0
    data_dim = 2

    # load data
    data  = torch.Tensor(generate_circle_toy_data())

    # define GAN structure
    generator = Generator(noise_dim=noise_vector_dim, discrete_code_dim=discrete_code_dim, continuous_code_dim=continuous_code_dim, out_dim=data_dim)

    # Only MSE is needed. No discriminator here
    mse_loss = nn.MSELoss()

    g_optimizer = optim.Adam(generator.parameters()) 

    epoch = 0
    for _ in range(training_epochs):
        epoch += 1

        g_optimizer.zero_grad()
        z_fake = torch.randn(2000, 34)
        generated_samples = generator(z_fake)

        loss = mse_loss(generated_samples, data)
        loss.backward()
        g_optimizer.step()

        print(f"EPOCH [{epoch}]: MSE Loss: {loss} / Discriminator Loss: {loss}")

        if epoch % 1000 == 0:
            z_fake, _ = generate_latent_sample(100, noise_vector_dim, discrete_code_dim, continuous_code_dim)
            out = generator(z_fake).detach().numpy()

            plt.figure()
            plt.scatter(out[:, 0], out[:, 1], color='orange')
            plt.savefig(f"assets/generated/generated_{epoch}.png")

if __name__ == '__main__':
    infogan()
