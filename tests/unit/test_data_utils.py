import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import torch
from utils.data_utils import generate_training_data

def test_generate_training_data():
    num_points = 100
    n_values = [0, 1, 2, 3, 4, 5]
    t_n = generate_training_data(num_points, n_values)

    assert isinstance(t_n, torch.Tensor), "The output should be a torch.Tensor"
    assert t_n.shape == (num_points * len(n_values), 2), "The shape of the tensor is incorrect"
    assert torch.all(t_n[:, 0] >= 0.01) and torch.all(t_n[:, 0] <= 10), "t values should be between 0.01 and 10"
    assert torch.all(t_n[:, 1].int() == torch.tensor(n_values).repeat_interleave(num_points).int()), "n values are not correctly repeated"
