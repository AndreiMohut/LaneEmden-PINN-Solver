import torch
import numpy as np

def random_fourier_features(inputs, num_features, length_scale=1.0, seed=None):
    """
    Compute the Random Fourier Features embedding of the input data.
    
    :param inputs: Input data (temporal and spatial coordinates) of shape (N, d),
                   where N is the number of points and d is the dimensionality.
    :param num_features: Number of Fourier features to generate.
    :param length_scale: Standard deviation `s` for the Gaussian distribution.
                         This controls the random frequencies.
    :param seed: Seed for random number generation.
    :return: Transformed data with shape (N, 2 * num_features), fully differentiable with respect to inputs.
    """
    
    if seed is not None:
        torch.manual_seed(seed)

    # Extract input dimensionality
    dim = inputs.shape[1]
    
    # Sample random frequencies from a Gaussian distribution N(0, s^2)
    # Ensure w and b are created with torch tensors that allow gradient flow
    w = torch.randn(dim, num_features, requires_grad=True) * (1.0 / length_scale)  # Sample from N(0, s^2)
    b = torch.rand(num_features, requires_grad=True) * 2 * np.pi  # Uniform between 0 and 2*pi
    
    # Apply Fourier features: f(x) = cos(Bx + b), sin(Bx + b)
    projection = torch.matmul(inputs, w) + b
    rff_embedding = torch.cat([torch.cos(projection), torch.sin(projection)], dim=1)
    
    return rff_embedding

# Example usage
if __name__ == "__main__":
    # Create some synthetic input data to simulate temporal and spatial coordinates
    num_points = 100  # Number of data points
    input_dim = 2  # Input dimensionality, e.g., time + one spatial dimension
    
    # Generate random input data
    input_data = torch.rand(num_points, input_dim)  # Values between 0 and 1
    print("Input data shape:", input_data.shape)  # Check the shape of the input data
    
    # Number of random Fourier features
    num_features = 64  # Adjust based on desired complexity
    
    # Get the Fourier features embedding
    embedded_input = random_fourier_features(input_data, num_features, length_scale=1.0, seed=42)
    
    print("Embedded input shape:", embedded_input.shape)
