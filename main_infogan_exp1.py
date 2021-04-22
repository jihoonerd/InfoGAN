import os
import pathlib

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from infogan.data.generate_toy_example import generate_circle_toy_data
from infogan.data.utils import generate_latent_sample
from infogan.model.loss import NormalNLLLoss
from infogan.model.network import Discriminator, Generator


def infogan():

    exp_path = 'exp_results/exp1/'
    p = pathlib.Path(exp_path)
    p.mkdir(parents=True, exist_ok=True)
    data = generate_circle_toy_data()
    plt.scatter(data[:, 0], data[:, 1])
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.grid()
    plt.savefig(os.path.join(exp_path, 'original.png'))

    # set training settings
    training_epochs = 5000
    noise_vector_dim = 32
    discrete_code_dim = 2
    continuous_code_dim = 0
    data_dim = 2

    # load data
    data  = torch.Tensor(generate_circle_toy_data())
    batch_size = data.shape[0]

    # define GAN structure
    generator = Generator(noise_dim=noise_vector_dim, discrete_code_dim=discrete_code_dim, continuous_code_dim=continuous_code_dim, out_dim=data_dim)
    discriminator = Discriminator(data_dim=data_dim, discrete_code_dim=discrete_code_dim, continuous_code_dim=continuous_code_dim)
    
    discriminator_loss = nn.BCELoss()
    generator_discrete_loss = nn.NLLLoss()
    generator_continuous_loss = NormalNLLLoss()

    g_optimizer = optim.Adam(generator.parameters()) 
    d_optimizer = optim.Adam(discriminator.parameters())


    for epoch in range(training_epochs):
        
        real_labels = torch.ones((batch_size), requires_grad=False).unsqueeze(1)
        fake_labels = torch.zeros((batch_size), requires_grad=False).unsqueeze(1)

        # Update Discriminator
        d_optimizer.zero_grad()
        real_gan_out, q_discrete, q_mu, q_var = discriminator(data)
        loss_d_real = discriminator_loss(real_gan_out, real_labels)

        z_fake, fake_indices = generate_latent_sample(batch_size, noise_vector_dim, discrete_code_dim, continuous_code_dim)
        generated_samples = generator(z_fake)
        fake_gan_out, q_discrete, q_mu, q_var = discriminator(generated_samples.detach())
        loss_d_fake = discriminator_loss(fake_gan_out, fake_labels)

        total_dl = loss_d_real + loss_d_fake
        total_dl.backward()
        d_optimizer.step()

        # Update Generator
        g_optimizer.zero_grad()
        
        fake_gan_out, q_discrete, q_mu, q_var = discriminator(generated_samples)
        
        generator_loss = discriminator_loss(fake_gan_out, real_labels)  # fake data treated as real

        discrete_code_loss = generator_discrete_loss(q_discrete, fake_indices)
        continuous_code_loss = generator_continuous_loss(z_fake[:, noise_vector_dim + discrete_code_dim:], q_mu, q_var) * 0.1

        total_gl = generator_loss + discrete_code_loss + continuous_code_loss
        total_gl.backward()
        g_optimizer.step()

        print(f"EPOCH [{epoch}]: Generator Loss: {total_gl} / Discriminator Loss: {total_dl}")

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

            gen_path = 'exp_results/exp1/generated/'
            p = pathlib.Path(gen_path)
            p.mkdir(parents=True, exist_ok=True)
            plt.figure()
            plt.scatter(out_10[:, 0], out_10[:, 1], color='red')
            plt.scatter(out_01[:, 0], out_01[:, 1], color='green')
            plt.xlim([-10, 10])
            plt.ylim([-10, 10])
            plt.grid()
            plt.savefig(os.path.join(gen_path, "generated_{0:05d}.png".format(epoch)))

        
        


if __name__ == '__main__':
    infogan()
