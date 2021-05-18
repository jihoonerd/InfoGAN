
import os
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, TensorDataset

from infogan.data.generate_toy_example import \
    generate_circle_toy_data_by_angle_random_path
from infogan.data.utils import append_infogan_code, vectorize_path
from infogan.model.network import (Discriminator, Generator,
                                   InfoGANDiscriminator)


def exp():

    config = yaml.safe_load(open('./config/config.yaml', 'r').read())
    use_infogan = config['model']['infogan']
    use_noise = config['model']['noise']
    training_epochs = config['model']['epochs']
    num_circle_samples = config['model']['num_circle_samples']
    noise_weight = config['model']['noise_weight']
    use_code_dist_loss = config['model']['code_dist_loss']
    
    export_path = f'results/infogan_{use_infogan}_noise_{use_noise}_code_dl_{use_code_dist_loss}/'
    weight_name = os.path.join(export_path, 'weights')
    p = pathlib.Path(export_path)
    p.mkdir(parents=True, exist_ok=True)

    # set training settings
    input_vec_dim = 6
    if use_infogan:
        discrete_code_dim = 2
    else:
        discrete_code_dim = 0
    gen_dim = 2
    disc_dim = input_vec_dim + gen_dim

    x = []
    y = []

    for _ in range(num_circle_samples):
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
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    generator = Generator(noise_dim=input_vec_dim, discrete_code_dim=discrete_code_dim, continuous_code_dim=0, out_dim=gen_dim)
    if use_infogan:
        discriminator = InfoGANDiscriminator(data_dim=disc_dim, discrete_code_dim=discrete_code_dim, continuous_code_dim=0)
    else:
        discriminator = Discriminator(data_dim=disc_dim)

    discriminator_loss = nn.BCELoss()
    generator_discrete_loss = nn.CrossEntropyLoss()
    pdist = nn.PairwiseDistance(p=2)

    g_optimizer = optim.Adam(generator.parameters(), amsgrad=True) 
    d_optimizer = optim.Adam(discriminator.parameters(), amsgrad=True)

    
    for epoch in range(training_epochs):
        for x, y in loader:

            real_labels = torch.ones((x.shape[0]), requires_grad=False).unsqueeze(1)
            fake_labels = torch.zeros((x.shape[0]), requires_grad=False).unsqueeze(1)

            d_optimizer.zero_grad()
            y_with_context = torch.cat([x, y], dim=1)
            if use_infogan:
                real_gan_out, _ = discriminator(y_with_context)
            else:
                real_gan_out = discriminator(y_with_context)

            loss_d_real = discriminator_loss(real_gan_out, real_labels)

            code_added, fake_indices = append_infogan_code(x, discrete_code_dim, 0)

            # Add noise to current position
            if use_noise:
                from_target_dist = pdist(code_added[:, 4:6], code_added[:, 2:4])
                smoothed_weight = -torch.abs(from_target_dist - 1) + 1 # y = -|x-1| + 1 to give small noise at the both end
                noise = torch.normal(0, 0.1, size=code_added[:, 4:6].shape) * smoothed_weight.unsqueeze(1)
                code_added[:, 4:6] += noise
            displacement = generator(code_added)
            
            if use_noise:
                # Compensate added noise here
                generated_samples = code_added[:, 4:6] + displacement - noise
            else:
                generated_samples = code_added[:, 4:6] + displacement
            
            gen_with_context = torch.cat([x, generated_samples], dim=1)

            if use_infogan:
                d_fake_gan_out, _ = discriminator(gen_with_context.detach())
            else:
                d_fake_gan_out = discriminator(gen_with_context.detach())

            loss_d_fake = discriminator_loss(d_fake_gan_out, fake_labels)
 
            total_dl = loss_d_real + loss_d_fake
            total_dl.backward()
            d_optimizer.step()

            g_optimizer.zero_grad()

            if use_infogan:
                g_fake_gan_out, fake_q_discrete = discriminator(gen_with_context)
            else:
                g_fake_gan_out = discriminator(gen_with_context)

            generator_loss = discriminator_loss(g_fake_gan_out, real_labels)

            if use_infogan:
                discrete_code_loss = generator_discrete_loss(fake_q_discrete, fake_indices)

            if use_code_dist_loss:
                distance_from_target = pdist(code_added[:, 4:6], code_added[:, 2:4])

                g_code_10 = code_added.clone()
                g_code_10[:, 6] = 1
                g_code_10[:, 7] = 0
                g_code_10_displacement = generator(g_code_10)
                if use_noise:
                    g_code_10_out = x[:, 4:6] + g_code_10_displacement - noise
                else:
                    g_code_10_out = x[:, 4:6] + g_code_10_displacement

                g_code_01 = code_added.clone()
                g_code_01[:, 6] = 0
                g_code_01[:, 7] = 1
                g_code_01_displacement = generator(g_code_01)
                if use_noise:
                    g_code_01_out = x[:, 4:6] + g_code_01_displacement - noise
                else:
                    g_code_01_out = x[:, 4:6] + g_code_01_displacement

                cdl = torch.sum(pdist(g_code_10_out, g_code_01_out) * distance_from_target)

            if use_infogan and use_code_dist_loss:
                total_gl = generator_loss + discrete_code_loss - cdl * noise_weight
            if use_infogan and not use_code_dist_loss:
                total_gl = generator_loss + discrete_code_loss
            if not use_infogan:
                total_gl = generator_loss

            total_gl.backward()
            g_optimizer.step()
        
        if use_code_dist_loss:
            print(f"EPOCH [{epoch}]: Generator Loss: {generator_loss} / Discriminator Loss: {total_dl} / Code Dist Loss: {-cdl * noise_weight}")
        else:
            print(f"EPOCH [{epoch}]: Generator Loss: {generator_loss} / Discriminator Loss: {total_dl}")

    torch.save(generator.state_dict(), weight_name)


if __name__ == '__main__':
    exp()
