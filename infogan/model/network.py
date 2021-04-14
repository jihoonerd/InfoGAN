import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear


class Generator(nn.Module):
    def __init__(self, noise_dim=32, continuous_code_dim=1, discrete_code_dim=0,  out_dim=2):
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
    def __init__(self, data_dim=2):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(data_dim, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out = self.fc(x)
        out = torch.sigmoid(out)
        return out


class QNetwork(nn.Module):
    def __init__(self, data_dim=2, continuous_code_dim=1, discrete_code_dim=0):
        super().__init__()

        self.fc = nn.Linear(data_dim, continuous_code_dim + discrete_code_dim)

    def forward(self, x):
        out = self.fc(x)
        return out