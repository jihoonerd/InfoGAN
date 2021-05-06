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


def play():

    exp_path = 'exp_results/field_learning/'
    weigth_path = 'exp_results/field_learning/generator_weight_w_noise'
    p = pathlib.Path(exp_path)
    p.mkdir(parents=True, exist_ok=True)

    # set training settings
    input_vec_dim = 6
    discrete_code_dim = 2
    continuous_code_dim = 0
    gen_dim = 2

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
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)

    generator = Generator(noise_dim=input_vec_dim, discrete_code_dim=discrete_code_dim, continuous_code_dim=continuous_code_dim, out_dim=gen_dim)
    generator.load_state_dict(torch.load(weigth_path))

    inference_x_10 = data_x[0].clone().detach()
    inference_x_01 = data_x[0].clone().detach()
    for i in range(30):

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
    play()
