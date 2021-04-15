from infogan.data.generate_toy_example import generate_circle_toy_data
from infogan.data.utils import generate_latent_sample
import torch
import matplotlib.pyplot as plt


def test_toy_circle_data():
    data = generate_circle_toy_data()
    plt.scatter(data[:, 0], data[:, 1])
    plt.show()
    

def test_generate_latent_sample():
    batch_size = 5
    noise_vec_dim = 4
    discrete_dim = 3
    continuous_dim = 2
    latent_sample_vec, indices = generate_latent_sample(batch_size, noise_vec_dim, discrete_dim, continuous_dim)
    
    assert (latent_sample_vec[:, :noise_vec_dim].abs() <= 1).all()
    assert (latent_sample_vec[:, noise_vec_dim:noise_vec_dim + discrete_dim].sum(dim=1) == torch.ones(batch_size,)).all()
    assert (latent_sample_vec[:, noise_vec_dim + discrete_dim: ].abs() <= 1).all()