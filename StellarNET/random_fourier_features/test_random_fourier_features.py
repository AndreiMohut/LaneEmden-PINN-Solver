import torch
import numpy as np
from random_fourier_features.random_fourier_features import random_fourier_features

# Test 1: Output Shape
def test_random_fourier_features_output_shape():
    """
    Test whether the output of random_fourier_features has the correct shape.
    """
    inputs = torch.ones((10, 2))  # 10 points, 2 input dimensions
    num_features = 64
    
    # Get the random Fourier features
    rff_output = random_fourier_features(inputs, num_features, seed=0)
    
    # The output shape should be (10, 2 * num_features) since we concatenate cos and sin
    assert rff_output.shape == (10, 2 * num_features), "Output shape is incorrect"

# Test 2: Reproducibility
def test_random_fourier_features_reproducibility():
    """
    Test whether the random Fourier features are reproducible given the same seed.
    """
    inputs = torch.ones((10, 2))  # 10 points, 2 input dimensions
    num_features = 64
    
    # Generate output with the same seed
    rff_output_1 = random_fourier_features(inputs, num_features, seed=0)
    rff_output_2 = random_fourier_features(inputs, num_features, seed=0)
    
    # Check if the outputs are the same
    assert torch.allclose(rff_output_1, rff_output_2), "Random Fourier features should be reproducible with the same seed"

# Test 3: Value Range Check
def test_random_fourier_features_value_range():
    """
    Test whether the values of the random Fourier features are in the expected range (-1, 1)
    due to cosine and sine operations.
    """
    inputs = torch.ones((10, 2))  # 10 points, 2 input dimensions
    num_features = 64
    
    # Generate random Fourier features
    rff_output = random_fourier_features(inputs, num_features, seed=0)
    
    # Check that the values are within the range (-1, 1)
    assert torch.all(rff_output >= -1) and torch.all(rff_output <= 1), "RFF values should be in the range [-1, 1]"

# Test 4: Edge Case - Zero Features
def test_random_fourier_features_edge_case_zero_features():
    """
    Test the edge case where num_features is zero. The output should be an empty array.
    """
    inputs = torch.ones((10, 2))  # 10 points, 2 input dimensions
    num_features = 0  # Edge case: no features
    
    # Generate random Fourier features with zero features
    rff_output = random_fourier_features(inputs, num_features, seed=0)
    
    # The output should have shape (10, 0) because there are no features
    assert rff_output.shape == (10, 0), "Output shape should be (10, 0) when num_features is 0"

# Test 5: Large Input Handling
def test_random_fourier_features_large_input():
    """
    Test whether the function can handle large input data.
    """
    inputs = torch.ones((10000, 3))  # Large input: 10,000 points, 3 input dimensions
    num_features = 64
    
    # Generate random Fourier features for large input
    rff_output = random_fourier_features(inputs, num_features, seed=0)
    
    # Check that the output has the correct shape for large input
    assert rff_output.shape == (10000, 2 * num_features), "Output shape is incorrect for large input data"

# Test 6: Differentiability of Fourier Features
def test_random_fourier_features_differentiability():
    """
    Test if the random Fourier features are differentiable with respect to the input.
    """
    inputs = torch.ones((10, 2), requires_grad=True)  # 10 points, 2 input dimensions
    num_features = 64

    # Generate random Fourier features
    rff_output = random_fourier_features(inputs, num_features, seed=0)

    # Sum the output to create a scalar value (needed for calling backward)
    output_sum = rff_output.sum()

    # Perform backward pass to compute gradients with respect to inputs
    output_sum.backward()

    # Check if gradients were successfully computed
    assert inputs.grad is not None, "Gradients were not computed."
    assert torch.any(inputs.grad != 0), "Some gradients should be non-zero for differentiable input."

# Example usage
if __name__ == "__main__":
    # Run all the tests
    test_random_fourier_features_output_shape()
    test_random_fourier_features_reproducibility()
    test_random_fourier_features_value_range()
    test_random_fourier_features_edge_case_zero_features()
    test_random_fourier_features_large_input()
    test_random_fourier_features_differentiability()

    print("All Random Fourier Features tests passed!")
