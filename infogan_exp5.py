# TODO: 1) Use dx, dy to decide next direcdtion 2) Give advantage when output is different by [1,0] and [0,1]. This loss might be scheduled

import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules import distance
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from infogan.data.generate_toy_example import generate_circle_toy_data_by_angle_random_path
from infogan.data.utils import vectorize_path, append_infogan_code, compose_past_ctxt
from infogan.model.network import Discriminator, Generator


def exp5():

    exp_path = 'exp_results/exp5/'
    weight_name = 'generator_weight_wo_noise'
    p = pathlib.Path(exp_path)
    p.mkdir(parents=True, exist_ok=True)

    # set training settings
    input_vec_dim = 6
    discrete_code_dim = 2
    continuous_code_dim = 0
    training_epochs = 5000
    gen_dim = 2
    disc_dim = 8

    x = []
    y = []

    for _ in range(128):
        cw_coord, ccw_coord = generate_circle_toy_data_by_angle_random_path()
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
    generator_discrete_loss = nn.NLLLoss()
    pdist = nn.PairwiseDistance(p=2)

    g_optimizer = optim.Adam(generator.parameters()) 
    d_optimizer = optim.Adam(discriminator.parameters())

    
    for epoch in range(training_epochs):
        cl_schedule = 0.001
        for x, y in loader:

            real_labels = torch.ones((x.shape[0]), requires_grad=False).unsqueeze(1)
            fake_labels = torch.zeros((x.shape[0]), requires_grad=False).unsqueeze(1)

            d_optimizer.zero_grad()
            y_with_context = torch.cat([x, y], dim=1)
            real_gan_out, real_q_discrete, real_q_mu, real_q_var = discriminator(y_with_context)
            loss_d_real = discriminator_loss(real_gan_out, real_labels)

            code_added, fake_indices = append_infogan_code(x, discrete_code_dim, continuous_code_dim)
            displacement = generator(code_added)
            generated_samples = code_added[:, 4:6] + displacement
            
            gen_with_context = torch.cat([x, generated_samples], dim=1)

            d_fake_gan_out, _, _, _ = discriminator(gen_with_context.detach())
            loss_d_fake = discriminator_loss(d_fake_gan_out, fake_labels)
            total_dl = loss_d_real + loss_d_fake
            total_dl.backward()
            d_optimizer.step()

            g_optimizer.zero_grad()
            g_fake_gan_out, fake_q_discrete, fake_q_mu, fake_q_var = discriminator(gen_with_context)
            generator_loss = discriminator_loss(g_fake_gan_out, real_labels)
            discrete_code_loss = generator_discrete_loss(fake_q_discrete, fake_indices)

            # Code diff loss
            distance_from_target = pdist(code_added[:, 4:6], code_added[:, 2:4])

            g_code_10 = code_added.clone()
            g_code_10[:, 6] = 1
            g_code_10[:, 7] = 0
            g_code_10_displacement = generator(g_code_10)
            g_code_10_out = x[:, 4:6] + g_code_10_displacement

            g_code_01 = code_added.clone()
            g_code_01[:, 6] = 0
            g_code_01[:, 7] = 1
            g_code_01_displacement = generator(g_code_01)
            g_code_01_out = x[:, 4:6] + g_code_01_displacement

            cl = torch.sum(pdist(g_code_10_out, g_code_01_out) * distance_from_target)

            total_gl = generator_loss + discrete_code_loss - cl * cl_schedule
            # total_gl = generator_loss + discrete_code_loss 
            total_gl.backward()
            g_optimizer.step()
        
        print(f"EPOCH [{epoch}]: Generator Loss: {generator_loss} / Discriminator Loss: {total_dl}")    

    torch.save(generator.state_dict(), weight_name)

if __name__ == '__main__':
    exp5()
