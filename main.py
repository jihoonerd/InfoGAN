from numpy import real
from infogan.data.generate_toy_example import generate_circle_toy_data
from infogan.model.network import Generator, Discriminator
from infogan.model.loss import NormalNLLLoss
from infogan.data.utils import generate_latent_sample
import torch
import torch.nn as nn
import torch.optim as optim

def infogan():
    # set training settings
    training_epochs = 100
    noise_vector_dim = 32
    discrete_code_dim = 1
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

        z_fake = generate_latent_sample(batch_size, noise_vector_dim, discrete_code_dim, continuous_code_dim)
        generated_samples = generator(z_fake)
        fake_gan_out, q_discrete, q_mu, q_var = discriminator(generated_samples.detach())
        loss_d_fake = discriminator_loss(fake_gan_out, fake_labels)

        total_dl = loss_d_real, loss_d_fake
        total_dl.backward()
        d_optimizer.step()

        # Update Generator
        g_optimizer.zero_grad()
        
        


if __name__ == '__main__':
    infogan()
