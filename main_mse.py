import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import pathlib
from infogan.data.generate_toy_example import generate_circle_toy_data
from infogan.data.utils import generate_latent_sample
from infogan.model.loss import NormalNLLLoss
from infogan.model.network import Discriminator, Generator


def infogan():

    p = pathlib.Path('fig/')
    p.mkdir(parents=True, exist_ok=True)
    data = generate_circle_toy_data()
    plt.scatter(data[:, 0], data[:, 1])
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.grid()
    plt.savefig('fig/original.png')

    # set training settings
    training_epochs = 10000
    noise_vector_dim = 32
    discrete_code_dim = 2
    continuous_code_dim = 0
    data_dim = 2

    # load data
    data  = torch.Tensor(generate_circle_toy_data())
    batch_size = data.shape[0]

    # define GAN structure
    generator = Generator(noise_dim=noise_vector_dim, discrete_code_dim=discrete_code_dim, continuous_code_dim=continuous_code_dim, out_dim=data_dim)
    
    mse_loss = nn.MSELoss()

    g_optimizer = optim.Adam(generator.parameters()) 

    for epoch in range(training_epochs):

        g_optimizer.zero_grad()
        z_fake = torch.randn(400, 34)
        generated_samples = generator(z_fake)

        loss = mse_loss(generated_samples, data)
        loss.backward()
        g_optimizer.step()

        print(f"EPOCH [{epoch}]: MSE Loss: {loss} / Discriminator Loss: {loss}")

        if epoch % 10 == 0:
            z_fake, fake_indices = generate_latent_sample(100, noise_vector_dim, discrete_code_dim, continuous_code_dim)

            z_fake_10 = z_fake.clone().detach()
            z_fake_10[:,32] = 1
            z_fake_10[:,33] = 0
            out_10 = generator(z_fake_10).detach().numpy()

            z_fake_01 = z_fake.clone().detach()
            z_fake_01[:,32] = 0
            z_fake_01[:,33] = 1
            out_01 = generator(z_fake_01).detach().numpy()

            plt.figure()
            plt.scatter(out_10[:, 0], out_10[:, 1], color='red')
            plt.scatter(out_01[:, 0], out_01[:, 1], color='green')
            plt.xlim([-10, 10])
            plt.ylim([-10, 10])
            plt.grid()
            plt.savefig("fig/generated_{0:05d}.png".format(epoch))


if __name__ == '__main__':
    infogan()
