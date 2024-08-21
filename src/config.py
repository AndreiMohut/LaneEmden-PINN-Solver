import torch
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Model configuration
MODEL_CONFIG = {
    "input_size": 2,
    "hidden_size": 128,
    "output_size": 1,
    "activation": "Tanh",  # Options: 'Tanh', 'ReLU', 'Softplus', etc.
    "output_activation": "Softplus"  # Activation function for the last hidden layer
}

# Training configuration
TRAINING_CONFIG = {
    "num_epochs": 30001,
    "learning_rate": 0.001,
    "num_points": 100,
    "n_values": [0, 1, 2, 3, 4, 5],  # Polytropic indices for training
    "new_n_values": [1.5, 2.5, 3.5, 4.5]  # Polytropic indices for extrapolation
}

# Plotting configuration
PLOTTING_CONFIG = {
    "y_limits": [-0.5, 1.1]
}
