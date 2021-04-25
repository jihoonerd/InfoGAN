import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from infogan.data.generate_toy_example import generate_circle_toy_data_by_angle
from infogan.data.utils import vectorize_path
from infogan.model.loss import NormalNLLLoss
from infogan.model.network import Discriminator, Generator


def exp3():

    exp_path = 'exp_results/exp3/'
    p = pathlib.Path(exp_path)
    p.mkdir(parents=True, exist_ok=True)
    np.random.seed(0)
    cw_coord, ccw_coord = generate_circle_toy_data_by_angle()
    plt.scatter(cw_coord[:, 0], cw_coord[:, 1], color='red')
    plt.scatter(ccw_coord[:, 0], ccw_coord[:, 1], color='green')
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.grid()
    plt.savefig(os.path.join(exp_path, 'original.png'))

    # set training settings
    input_vec_dim = 6
    discrete_code_dim = 0
    continuous_code_dim = 0
    training_epochs = 100000
    gen_dim = 2
    disc_dim = 8

    
    x = []
    y = []
    cw_input, cw_gt = vectorize_path(cw_coord)
    ccw_input, ccw_gt = vectorize_path(ccw_coord)
    x.append(cw_input)
    y.append(cw_gt)
    x.append(ccw_input)
    y.append(ccw_gt)
    
    data_x = torch.FloatTensor(np.concatenate(x, axis=0))
    data_y = torch.FloatTensor(np.concatenate(y, axis=0))

    dataset = TensorDataset(data_x, data_y)
    loader = DataLoader(dataset, batch_size=512, shuffle=True)

    generator = Generator(noise_dim=input_vec_dim, discrete_code_dim=discrete_code_dim, continuous_code_dim=continuous_code_dim, out_dim=gen_dim)
    discriminator = Discriminator(data_dim=disc_dim, discrete_code_dim=discrete_code_dim, continuous_code_dim=continuous_code_dim)
    
    discriminator_loss = nn.BCELoss()

    g_optimizer = optim.Adam(generator.parameters()) 
    d_optimizer = optim.Adam(discriminator.parameters())

    for epoch in range(training_epochs):

        for x, y in loader:

            real_labels = torch.ones((x.shape[0]), requires_grad=False).unsqueeze(1)
            fake_labels = torch.zeros((x.shape[0]), requires_grad=False).unsqueeze(1)

            d_optimizer.zero_grad()
            y_with_context = torch.cat([x, y], dim=1)
            real_gan_out, real_q_discrete, real_q_mu, real_q_var = discriminator(y_with_context)
            loss_d_real = discriminator_loss(real_gan_out, real_labels)

            generated_samples = generator(x)
            d_gen_with_context = torch.cat([x, generated_samples], dim=1).detach()

            d_fake_gan_out, fake_q_discrete, fake_q_mu, fake_q_var = discriminator(d_gen_with_context)
            loss_d_fake = discriminator_loss(d_fake_gan_out, fake_labels)

            total_dl = loss_d_real + loss_d_fake
            total_dl.backward()
            d_optimizer.step()

            g_optimizer.zero_grad()
            g_gen_with_context = torch.cat([x, generated_samples], dim=1)
            g_fake_gan_out, fake_q_discrete, fake_q_mu, fake_q_var = discriminator(g_gen_with_context)
            generator_loss = discriminator_loss(g_fake_gan_out, real_labels)
            total_gl = generator_loss
            total_gl.backward()
            g_optimizer.step()
            
            
            print(f"EPOCH [{epoch}]: Generator Loss: {total_gl} / Discriminator Loss: {total_dl}")

    
    generated_samples = generator(x).detach().numpy()

    for i in range(x.shape[0]):
        gen_path = os.path.join(exp_path, 'generated')
        p = pathlib.Path(gen_path)
        p.mkdir(parents=True, exist_ok=True)
        plt.figure()
        circle = plt.Circle((0, 0), 1, color='grey', fill=False)
        plt.gca().add_patch(circle)
        plt.scatter(data_x[i][0], data_x[i][1], color='red')  # plot start pos
        plt.scatter(data_x[i][2], data_x[i][3], color='blue')  # plot target pos
        plt.scatter(data_x[i][4], data_x[i][5], color='black')  # plot current pos
        plt.scatter(generated_samples[i][0], generated_samples[i][1], color='purple')  # prediction
        plt.xlim([-1.5, 1.5])
        plt.ylim([-1.5, 1.5])
        plt.grid()
        plt.savefig(os.path.join(gen_path, "generated_{0:05d}.png".format(i)))

if __name__ == '__main__':
    exp3()
