import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, noise_dim, discrete_code_dim, continuous_code_dim, out_dim):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(noise_dim + continuous_code_dim + discrete_code_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim)
        )

    def forward(self, x):
        out = self.fc(x)
        return out


class Discriminator(nn.Module):
    def __init__(self, data_dim, discrete_code_dim, continuous_code_dim):
        super().__init__()

        self.data_dim = data_dim
        self.discrete_code_dim = discrete_code_dim
        self.continuous_code_dim = continuous_code_dim

        self.discriminator = nn.Sequential(
            nn.Linear(data_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32)
        )

        self.regular_gan = nn.Sequential(
            nn.Linear(32, 1)
        )

        self.infogan_q = nn.Sequential(
            nn.Linear(32, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, discrete_code_dim + continuous_code_dim * 2)  # continuous code needs to have mu, sigma respectively
        )

        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def forward(self, x):
        discriminator_out = self.discriminator(x)
        regular_gan_out = self.sigmoid(self.regular_gan(discriminator_out))
        
        q_out = self.infogan_q(discriminator_out)
        
        q_discrete = self.softplus(q_out[:,:self.discrete_code_dim])
        
        q_continuous = q_out[:,self.discrete_code_dim:]  # e.g.) [cont_mu1, cont_mu2, cont_sigma1, cont_simga2]
        q_mu = q_continuous[:, :self.continuous_code_dim]
        q_sigma = self.softplus(q_continuous[:, self.continuous_code_dim:])  # Use softplus to make std positive value

        return regular_gan_out, q_discrete, q_mu, q_sigma
