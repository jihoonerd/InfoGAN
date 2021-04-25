import numpy as np
import torch


def generate_latent_sample(batch_size, noise_dim, discrete_code_dim, continuous_code_dim):
    """Generate latent vector from noise"""
    z = torch.FloatTensor(batch_size, noise_dim).uniform_(-1, 1)

    if discrete_code_dim != 0:
        label_idx = torch.randint(
            low=0, high=discrete_code_dim, size=(batch_size,))
        discrete_code = torch.zeros((batch_size, discrete_code_dim))
        for i in range(batch_size):
            discrete_code[i][label_idx[i]] = 1.0

    latent_vec = torch.cat([z, discrete_code], dim=1)

    if continuous_code_dim != 0:
        for _ in range(continuous_code_dim):
            continuous_code = torch.FloatTensor(batch_size, 1).uniform_(-1, 1)
            latent_vec = torch.cat([latent_vec, continuous_code], dim=1)

    return latent_vec, label_idx


def append_infogan_code(input_data, discrete_code_dim, continuous_code_dim):
    """Append infogan latent code to given input data"""

    num_samples = input_data.shape[0]

    if discrete_code_dim != 0:
        label_idx = torch.randint(
            low=0, high=discrete_code_dim, size=(num_samples,))
        discrete_code = torch.zeros((num_samples, discrete_code_dim))
        for i in range(num_samples):
            discrete_code[i][label_idx[i]] = 1.0

    return_vec = torch.cat([input_data, discrete_code], dim=1)

    # This condition handles continuous latent codes
    if continuous_code_dim != 0:
        for _ in range(continuous_code_dim):
            continuous_code = torch.FloatTensor(num_samples, 1).uniform_(-1, 1)
            # concat generated codes
            return_vec = torch.cat([return_vec, continuous_code], dim=1)

    return return_vec, label_idx


def split_path(path_data):
    """This splits path_data into start, target, interpolating states"""
    start_state = path_data[0]
    target_state = path_data[-1]
    intermediate_state = path_data[1: -1]
    return start_state, target_state, intermediate_state


def vectorize_path(coord):
    """
    Returns vectorized input vector and ground truth

    input vector is represented as [start_x, start_y, target_x, target_y, current_x, current_y]
    gt vector is represented as [next_x, next_y]

    Since starting point and target point are given, network should figure out the direction w/o seqence data    
    """

    input_vector = []
    gt_vector = []

    start_state, target_state, intm_state = split_path(coord)

    # Initial step
    input_vector.append(np.concatenate(
        [start_state, target_state, start_state]))
    gt_vector.append(intm_state[0])

    for i in range(intm_state.shape[0]-1):
        input_vector.append(np.concatenate(
            [start_state, target_state, intm_state[i]]))
        gt_vector.append(intm_state[i+1])

    # Last step
    input_vector.append(np.concatenate(
        [start_state, target_state, intm_state[-1]]))
    gt_vector.append(target_state)

    return np.stack(input_vector), np.stack(gt_vector)


def compose_past_ctxt(vectorize_input, gt):
    """Add 4 previous inputs"""
    cw_context = np.zeros((vectorize_input.shape[0]-4, 14))
    for i in range(cw_context.shape[0]):
        cw_context[i,:6] = vectorize_input[i+4][:6]
        cw_context[i,6:8] = vectorize_input[i+3][4:6]
        cw_context[i,8:10] = vectorize_input[i+2][4:6]
        cw_context[i,10:12] = vectorize_input[i+1][4:6]
        cw_context[i,12:] = vectorize_input[i][4:6]

    return torch.Tensor(cw_context), torch.Tensor(gt[4:])