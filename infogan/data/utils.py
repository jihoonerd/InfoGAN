import torch


def generate_latent_sample(batch_size, noise_dim, discrete_code_dim, continuous_code_dim):
    z = torch.FloatTensor(batch_size, noise_dim).uniform_(-1, 1)

    if discrete_code_dim != 0:
        label_idx = torch.randint(low=0, high=discrete_code_dim, size=(batch_size,))
        discrete_code = torch.zeros((batch_size, discrete_code_dim))
        for i in range(batch_size):
            discrete_code[i][label_idx[i]] = 1.0

    latent_vec = torch.cat([z, discrete_code], dim=1)
    
    if continuous_code_dim != 0:
        for _ in range(continuous_code_dim):
            continuous_code = torch.FloatTensor(batch_size, 1).uniform_(-1, 1)
            latent_vec = torch.cat([latent_vec, continuous_code], dim=1)

    return latent_vec, label_idx
