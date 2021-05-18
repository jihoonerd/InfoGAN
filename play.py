import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.font_manager import FontProperties
from torch.utils.data import DataLoader, TensorDataset

from infogan.data.generate_toy_example import \
    generate_circle_toy_data_by_angle_random_path
from infogan.data.utils import vectorize_path
from infogan.model.network import Generator


def play():
    config = yaml.safe_load(open('./config/config.yaml', 'r').read())
    exp_path = '/home/jihoon/Repositories/InfoGAN/results/infogan_True_noise_False'
    weigth_path = os.path.join(exp_path, 'infogan_True_noise_False_training_epochs')
    use_infogan = config['model']['infogan']
    use_noise = config['model']['noise']
    training_epochs = config['model']['epochs']
    num_circle_samples = config['model']['num_circle_samples']
    noise_weight = config['model']['noise_weight']
    use_code_dist_loss = config['model']['code_dist_loss']
    p = pathlib.Path(exp_path)
    p.mkdir(parents=True, exist_ok=True)

    # set training settings
    input_vec_dim = 6
    discrete_code_dim = 2
    continuous_code_dim = 0
    gen_dim = 2

    # set training settings
    input_vec_dim = 6
    if use_infogan:
        discrete_code_dim = 2
    else:
        discrete_code_dim = 0
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

    generator = Generator(noise_dim=input_vec_dim, discrete_code_dim=discrete_code_dim, continuous_code_dim=continuous_code_dim, out_dim=gen_dim)
    generator.load_state_dict(torch.load(weigth_path))

    if use_infogan:
        inference_x_10 = data_x[0].clone().detach()
        inference_x_01 = data_x[0].clone().detach()
    else:
        inference_x = data_x[0].clone().detach()

    if use_infogan:
        past_10_x, past_10_y = [], []
        past_01_x, past_01_y = [], []
    else:
        past_x, past_y = [], []

    for i in range(30):

        gen_path = os.path.join(exp_path, 'generated')
        p = pathlib.Path(gen_path)
        p.mkdir(parents=True, exist_ok=True)
        plt.figure()
        circle = plt.Circle((0, 0), 1, color='grey', fill=False)
        plt.gca().add_patch(circle)

        if use_infogan:
            code_10 = torch.Tensor([1, 0])
            coded_10 = torch.cat([inference_x_10.reshape(1, -1), code_10.reshape(1, -1)], dim=1)
            displacement_10 = generator(coded_10)
            generated_sample_10 = (inference_x_10[4:6] + displacement_10).detach().squeeze()

            code_01 = torch.Tensor([0, 1])
            coded_01 = torch.cat([inference_x_01.reshape(1, -1), code_01.reshape(1, -1)], dim=1)
            displacement_01 = generator(coded_01)
            generated_sample_01 = (inference_x_01[4:6] + displacement_01).detach().squeeze()        
        
            start = plt.scatter(inference_x_10[0], inference_x_10[1], color='red', s=70, label='Start')
            goal = plt.scatter(inference_x_10[2], inference_x_10[3], color='blue', s=70, label='Goal')

            plt.scatter(past_10_x, past_10_y, color='lightgrey', s=70, alpha=0.97)
            cur10 = plt.scatter(inference_x_10[4], inference_x_10[5], color='forestgreen', s=70, label='Current pos $[1,0]$')  # plot current pos
            pred10 = plt.scatter(generated_sample_10[0], generated_sample_10[1], color='palegreen', s=70, label='Pred pos $[1,0]$')  # prediction
            past_10_x.append(generated_sample_10[0])
            past_10_y.append(generated_sample_10[1])
            
            plt.scatter(past_01_x, past_01_y, color='lightgrey', s=70, alpha=0.97)
            cur01 = plt.scatter(inference_x_01[4], inference_x_01[5], color='darkgoldenrod', s=70, label='Current pos $[0,1]$')  # plot current pos
            pred01= plt.scatter(generated_sample_01[0], generated_sample_01[1], color='tan', s=70, label='Pred pos $[0,1]$')  # prediction
            past_01_x.append(generated_sample_01[0])
            past_01_y.append(generated_sample_01[1])

            inference_x_10 = inference_x_10.clone()
            inference_x_10[4:6] = generated_sample_10

            inference_x_01 = inference_x_01.clone()
            inference_x_01[4:6] = generated_sample_01

        else:
            displacement = generator(inference_x)
            generated_sample = (inference_x[4:6] + displacement).detach().squeeze()
            start = plt.scatter(inference_x[0], inference_x[1], color='red', s=70, label='Start')
            goal = plt.scatter(inference_x[2], inference_x[3], color='blue', s=70, label='Goal')
            plt.scatter(past_x, past_y, color='lightgrey', s=70, alpha=0.97)
            cur = plt.scatter(inference_x[4], inference_x[5], color='forestgreen', s=70, label='Current pos')  # plot current pos
            pred = plt.scatter(generated_sample[0], generated_sample[1], color='palegreen', s=70, label='Pred pos')  # prediction
            past_x.append(generated_sample[0])
            past_y.append(generated_sample[1])

            inference_x = inference_x.clone()
            inference_x[4:6] = generated_sample


        plt.xlim([-1.5, 1.5])
        plt.ylim([-1.5, 1.5])
        plt.grid()
        plt.gca().set_aspect("equal")

        fontP = FontProperties()
        fontP.set_size('x-small')

        if not use_infogan:
            plt.legend(handles=[start, goal, cur, pred], bbox_to_anchor=(1, 1), loc='upper left', prop=fontP)
        else:
            plt.legend(handles=[start, goal, cur10, pred10, cur01, pred01], bbox_to_anchor=(1, 1), loc='upper left', prop=fontP)
        plt.xlabel('$X$')
        plt.ylabel('$Y$')

        if use_infogan:
            plt.title('InfoGAN Path Example')
        else:
            plt.title('Vanilla GAN')
        plt.savefig(os.path.join(gen_path, "generated_{0:05d}.png".format(i)), dpi=300)




if __name__ == '__main__':
    play()
