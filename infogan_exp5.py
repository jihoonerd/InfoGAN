# TODO: 1) Use dx, dy to decide next direcdtion 2) Give advantage when output is different by [1,0] and [0,1]. This loss might be scheduled

import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from infogan.data.generate_toy_example import generate_circle_toy_data_by_angle_random_path
from infogan.data.utils import vectorize_path, append_infogan_code
from infogan.model.network import Discriminator, Generator


def exp4():

    exp_path = 'exp_results/exp5/'
    p = pathlib.Path(exp_path)
    p.mkdir(parents=True, exist_ok=True)
    

    # set training settings
    input_vec_dim = 6
    discrete_code_dim = 2
    continuous_code_dim = 0
    training_epochs = 18000
    gen_dim = 2
    disc_dim = 8

    x = []
    y = []
    
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
    loader = DataLoader(dataset, batch_size=512, shuffle=False)

    generator = Generator(noise_dim=input_vec_dim, discrete_code_dim=discrete_code_dim, continuous_code_dim=continuous_code_dim, out_dim=gen_dim)
    discriminator = Discriminator(data_dim=disc_dim, discrete_code_dim=discrete_code_dim, continuous_code_dim=continuous_code_dim)
    
    discriminator_loss = nn.BCELoss()
    generator_discrete_loss = nn.NLLLoss()
    code_loss = nn.L1Loss()

    g_optimizer = optim.Adam(generator.parameters()) 
    d_optimizer = optim.Adam(discriminator.parameters())

    
    for epoch in range(training_epochs):
        cl_schedule = 0.1
        for x, y in loader:

            real_labels = torch.ones((x.shape[0]), requires_grad=False).unsqueeze(1)
            fake_labels = torch.zeros((x.shape[0]), requires_grad=False).unsqueeze(1)

            d_optimizer.zero_grad()
            y_with_context = torch.cat([x, y], dim=1)
            real_gan_out, real_q_discrete, real_q_mu, real_q_var = discriminator(y_with_context)
            loss_d_real = discriminator_loss(real_gan_out, real_labels)

            code_added, fake_indices = append_infogan_code(x, discrete_code_dim, continuous_code_dim)
            displacement = generator(code_added)
            generated_samples = x[:, 4:6] + displacement
            
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

            cl = code_loss(g_code_10_out, g_code_01_out)

            total_gl = generator_loss + discrete_code_loss - cl * cl_schedule
            cl_schedule *= 0.9
            
            total_gl.backward()
            g_optimizer.step()
        
        print(f"EPOCH [{epoch}]: Generator Loss: {generator_loss} / Discriminator Loss: {total_dl}")    


    inference_x_10 = data_x[0].clone().detach()
    inference_x_01 = data_x[0].clone().detach()
    for i in range(int(data_x.shape[0]/2)):

        gen_path = os.path.join(exp_path, 'generated')
        p = pathlib.Path(gen_path)
        p.mkdir(parents=True, exist_ok=True)
        plt.figure()
        circle = plt.Circle((0, 0), 1, color='grey', fill=False)
        plt.gca().add_patch(circle)

        code_10 = torch.Tensor([1, 0])
        coded_10 = torch.cat([inference_x_10.reshape(1, -1), code_10.reshape(1, -1)], dim=1)
        displacement_10 = generator(coded_10)
        generated_sample_10 = (inference_x_10[4:6] + displacement_10).detach().squeeze()

        code_01 = torch.Tensor([0, 1])
        coded_01 = torch.cat([inference_x_01.reshape(1, -1), code_01.reshape(1, -1)], dim=1)
        displacement_01 = generator(coded_01)
        generated_sample_01 = (inference_x_01[4:6] + displacement_01).detach().squeeze()

        plt.scatter(inference_x_10[0], inference_x_10[1], color='red')
        plt.scatter(inference_x_10[2], inference_x_10[3], color='blue')
        plt.scatter(inference_x_10[4], inference_x_10[5], color='black')  # plot current pos
        plt.scatter(generated_sample_10[0], generated_sample_10[1], color='purple')  # prediction


        plt.scatter(inference_x_01[0], inference_x_01[1], color='red')
        plt.scatter(inference_x_01[2], inference_x_01[3], color='blue')
        plt.scatter(inference_x_01[4], inference_x_01[5], color='brown')  # plot current pos
        plt.scatter(generated_sample_01[0], generated_sample_01[1], color='cyan')  # prediction

        plt.xlim([-1.5, 1.5])
        plt.ylim([-1.5, 1.5])
        plt.grid()
        plt.savefig(os.path.join(gen_path, "generated_{0:05d}.png".format(i)))

        inference_x_10 = inference_x_10.clone()
        inference_x_10[4:6] = generated_sample_10

        inference_x_01 = inference_x_01.clone()
        inference_x_01[4:6] = generated_sample_01


if __name__ == '__main__':
    exp4()
