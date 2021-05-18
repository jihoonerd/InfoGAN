
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



def mse_test():

    config = yaml.safe_load(open('./config/config.yaml', 'r').read())
    training_epochs = config['model']['epochs']
    num_circle_samples = config['model']['num_circle_samples']
    
    export_path = f'results/mse/'
    weight_name = os.path.join(export_path, 'weights')
    p = pathlib.Path(export_path)
    p.mkdir(parents=True, exist_ok=True)

    # set training settings
    input_vec_dim = 6
    discrete_code_dim = 0
    gen_dim = 2

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
    mse_loss = nn.MSELoss()
    g_optimizer = optim.Adam(generator.parameters(), amsgrad=True) 
   
    for epoch in range(training_epochs):
        for x, y in loader:

            g_optimizer.zero_grad()
            displacement = generator(x)
            generated_samples = x[:, 4:6] + displacement

            loss = mse_loss(generated_samples, y)
            loss.backward()
            g_optimizer.step()
        print(f"EPOCH [{epoch}]: MSE Loss: {loss}")
            
    torch.save(generator.state_dict(), weight_name)

if __name__ == '__main__':
    mse_test()
