import torch
from pinn_architecture import PirateNet
import torch.nn as nn

def test_piratenet_output_shape():
    num_points = 100
    input_dim = 128  # Assuming input_dim is after the Fourier embedding
    num_fourier_features = 64  # Simulating Fourier features

    # Create a dummy array with shape (num_points, input_dim)
    phi_x = torch.rand(num_points, input_dim)  # Input already embedded with Fourier features

    # Instantiate the PirateNet
    num_blocks = 4
    pinn = PirateNet(input_dim=input_dim, hidden_dim=128, num_blocks=num_blocks)

    # Check if the output shape matches (100, 1) for 100 points
    output = pinn(phi_x)
    assert output.shape == (num_points, 1), f"Expected shape (100, 1), got {output.shape}"

def test_piratenet_initialization():
    input_dim = 128  # Input dimension after Fourier embedding
    num_points = 50

    # Create dummy input data
    phi_x = torch.rand(num_points, input_dim)  # Input already embedded with Fourier features

    # Instantiate the PirateNet
    num_blocks = 4
    pinn = PirateNet(input_dim=input_dim, hidden_dim=128, num_blocks=num_blocks)

    # Test initialization
    assert pinn is not None, "Model should be initialized"
    assert isinstance(pinn, nn.Module), "Model should be a PyTorch nn.Module"
