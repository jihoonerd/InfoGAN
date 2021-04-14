from infogan.data.generate_toy_example import generate_circle_toy_data
from infogan.model.network import Generator, Discriminator, QNetwork
import torch
import torch.nn as nn
import torch.optim as optim

def infogan():

    # set training settings
    training_epochs = 100
    noise_vector_size = 32

    # load data
    data  = torch.Tensor(generate_circle_toy_data())

    # define GAN structure
    generator = Generator(noise_dim=noise_vector_size, continuous_code_dim=1, discrete_code_dim=0, out_dim=2)
    discriminator = Discriminator(data_dim=2)
    q_network = QNetwork(data_dim=2, continuous_code_dim=1, discrete_code_dim=0)
    
    g_optimizer = optim.Adam(list(generator.parameters()) + list(q_network.parameters()))
    d_optimizer = optim.Adam(discriminator.parameters())

    g_loss = None
    d_loss = nn.BCELoss()


if __name__ == '__main__':
    infogan()
