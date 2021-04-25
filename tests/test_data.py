from infogan.data.generate_toy_example import generate_circle_toy_data, generate_circle_toy_data_by_angle
from infogan.data.utils import generate_latent_sample, vectorize_path, compose_past_ctxt
import torch
import matplotlib.pyplot as plt
import numpy as np

def test_toy_circle_data():
    data = generate_circle_toy_data()
    plt.scatter(data[:, 0], data[:, 1])
    plt.show()


def test_generate_circle_toy_data_by_angle():
    data = generate_circle_toy_data_by_angle()
    plt.scatter(data[0][:, 0], data[0][:, 1], color='red')
    plt.scatter(data[1][:, 0], data[1][:, 1], color='blue')
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

def test_compose_past_ctxt():
    np.random.seed(0)
    cw_coord, ccw_coord = generate_circle_toy_data_by_angle()
    cw_input, cw_gt = vectorize_path(cw_coord)
    ccw_input, ccw_gt = vectorize_path(ccw_coord)

    cw_ctxt_input, cw_ctxt_gt = compose_past_ctxt(cw_input, cw_gt)
    ccw_ctxt_input, ccw_ctxt_gt = compose_past_ctxt(ccw_input, ccw_gt)

    assert (cw_ctxt_gt[0] == cw_ctxt_input[1][4:6]).all()
    assert (cw_ctxt_gt[-2] == cw_ctxt_input[-1][4:6]).all()

    assert (ccw_ctxt_gt[0] == ccw_ctxt_input[1][4:6]).all()
    assert (ccw_ctxt_gt[-2] == ccw_ctxt_input[-1][4:6]).all()
