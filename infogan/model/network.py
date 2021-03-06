import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, noise_dim, discrete_code_dim, continuous_code_dim, out_dim):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(noise_dim + continuous_code_dim + discrete_code_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        out = self.fc(x)
        return out


class Discriminator(nn.Module):
    def __init__(self, data_dim):
        super().__init__()

        self.data_dim = data_dim

        self.discriminator = nn.Sequential(
            nn.Linear(data_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 128)
        )

        self.regular_gan = nn.Sequential(
            nn.Linear(128, 1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        discriminator_out = self.discriminator(x)
        regular_gan_out = self.sigmoid(self.regular_gan(discriminator_out))    
        return regular_gan_out


class InfoGANDiscriminator(nn.Module):
    def __init__(self, data_dim, discrete_code_dim, continuous_code_dim):
        super().__init__()

        self.data_dim = data_dim
        self.discrete_code_dim = discrete_code_dim
        self.continuous_code_dim = continuous_code_dim

        self.discriminator = nn.Sequential(
            nn.Linear(data_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 128)
        )

        self.regular_gan = nn.Sequential(
            nn.Linear(128, 1)
        )

        self.infogan_q = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, discrete_code_dim + continuous_code_dim * 2)  # continuous code needs to have mu, sigma respectively
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        discriminator_out = self.discriminator(x)
        regular_gan_out = self.sigmoid(self.regular_gan(discriminator_out))
        
        q_out = self.infogan_q(discriminator_out)
        
        q_discrete = q_out[:,:self.discrete_code_dim]
        return regular_gan_out, q_discrete
