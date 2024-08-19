import numpy as np
import torch

def generate_training_data(num_points, n_values):
    t = np.linspace(0.01, 10, num_points)
    t_grid, n_grid = np.meshgrid(t, n_values)
    t_flat = t_grid.flatten().reshape(-1, 1)
    n_flat = n_grid.flatten().reshape(-1, 1)
    t_n = np.hstack((t_flat, n_flat))
    return torch.tensor(t_n, dtype=torch.float32)