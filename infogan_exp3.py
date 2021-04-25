import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from infogan.data.generate_toy_example import generate_circle_toy_data_by_angle
from infogan.data.utils import vectorize_path, append_infogan_code, compose_past_ctxt
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
    input_vec_dim = 14  # [start x, start y, target x, target y, t_x, t_y, t-1_x, t-1_y...] 4 + 2*5
    discrete_code_dim = 2
    continuous_code_dim = 0
    training_epochs = 3000
    gen_dim = 2
    disc_dim = 16

    x = []
    y = []
    cw_input, cw_gt = vectorize_path(cw_coord)
    cw_ctxt_input, cw_ctxt_gt = compose_past_ctxt(cw_input, cw_gt)
    ccw_input, ccw_gt = vectorize_path(ccw_coord)
    ccw_ctxt_input, ccw_ctxt_gt = compose_past_ctxt(ccw_input, ccw_gt)

    x.append(cw_ctxt_input)
    y.append(cw_ctxt_gt)
    x.append(ccw_ctxt_input)
    y.append(ccw_ctxt_gt)

    data_x = torch.FloatTensor(np.concatenate(x, axis=0))
    data_y = torch.FloatTensor(np.concatenate(y, axis=0))

    dataset = TensorDataset(data_x, data_y)
    loader = DataLoader(dataset, batch_size=512, shuffle=False)

    generator = Generator(noise_dim=input_vec_dim, discrete_code_dim=discrete_code_dim, continuous_code_dim=continuous_code_dim, out_dim=gen_dim)
    discriminator = Discriminator(data_dim=disc_dim, discrete_code_dim=discrete_code_dim, continuous_code_dim=continuous_code_dim)
    
    discriminator_loss = nn.BCELoss()
    generator_discrete_loss = nn.NLLLoss()

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

            code_added, fake_indices = append_infogan_code(x, discrete_code_dim, continuous_code_dim)
            generated_samples = generator(code_added)
            
            gen_with_context = torch.cat([x, generated_samples], dim=1)

            d_fake_gan_out, fake_q_discrete, fake_q_mu, fake_q_var = discriminator(gen_with_context.detach())
            loss_d_fake = discriminator_loss(d_fake_gan_out, fake_labels)

            total_dl = loss_d_real + loss_d_fake
            total_dl.backward()
            d_optimizer.step()

            g_optimizer.zero_grad()
            g_fake_gan_out, fake_q_discrete, fake_q_mu, fake_q_var = discriminator(gen_with_context)
            generator_loss = discriminator_loss(g_fake_gan_out, real_labels)

            discrete_code_loss = generator_discrete_loss(fake_q_discrete, fake_indices)
            # continuous_code_loss = generator_continuous_loss(code_added[:, input_vec_dim + discrete_code_dim:], fake_q_mu, fake_q_var) * 0.1

            total_gl = generator_loss + discrete_code_loss
            total_gl.backward()
            g_optimizer.step()
        
        print(f"EPOCH [{epoch}]: Generator Loss: {generator_loss} / Discriminator Loss: {total_dl}")    

    generated_samples = generator(code_added).detach().numpy()

    path_len = code_added.shape[0]/2
    for i in range(x.shape[0]):

        

        gen_path = os.path.join(exp_path, 'generated')
        p = pathlib.Path(gen_path)
        p.mkdir(parents=True, exist_ok=True)
        plt.figure()
        circle = plt.Circle((0, 0), 1, color='grey', fill=False)
        plt.gca().add_patch(circle)

        code_10 = code_added.clone().detach()
        code_10[:, 14] = 1
        code_10[:, 15] = 0
        out_10 = generator(code_10).detach().numpy()

        code_01 = code_added.clone().detach()
        code_01[:, 14] = 0
        code_01[:, 15] = 1
        out_01 = generator(code_01).detach().numpy()

        plt.scatter(code_added[i][0], code_added[i][1], color='red')
        plt.scatter(code_added[i][2], code_added[i][3], color='blue')
        plt.scatter(code_added[i][4:13:2], code_added[i][5:14:2], color='black')  # plot current pos
        plt.scatter(out_10[i][0], out_10[i][1], color='purple')  # For code [1, 0]
        plt.scatter(out_01[i][0], out_01[i][1], color='yellow')  # For code [0, 1]
        plt.xlim([-1.5, 1.5])
        plt.ylim([-1.5, 1.5])
        plt.grid()
        plt.savefig(os.path.join(gen_path, "generated_{0:05d}.png".format(i)))

if __name__ == '__main__':
    exp3()
