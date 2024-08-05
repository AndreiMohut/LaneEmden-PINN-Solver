# LaneEmden-PINN-Solver

This repository contains a Python implementation for solving the Lane-Emden equation using a Physics-Informed Neural Network (PINN). The Lane-Emden equation is a type of differential equation that arises in the study of polytropic models in astrophysics.

## Lane-Emden Equation

The Lane-Emden equation is a dimensionless form of the Poisson equation for the gravitational potential of a self-gravitating, spherically symmetric, polytropic fluid. It is given by:

$$ \frac{1}{\xi^2} \frac{d}{d\xi} \left( \xi^2 \frac{d\theta}{d\xi} \right) + \theta^n = 0 $$

where:
- $\xi$ is the dimensionless radial coordinate,
- $\theta(\xi)$ is the dimensionless density,
- $n$ is the polytropic index.

## Overview

This implementation uses a Physics-Informed Neural Network (PINN) to solve the Lane-Emden equation. The PINN leverages neural networks to approximate the solution while incorporating the physics of the differential equation into the loss function.

### Key Features

- **Multiple Activation Functions**: The neural network model uses different activation functions for each layer.
- **Residual Calculation**: The residual of the Lane-Emden equation is computed using TensorFlow's automatic differentiation.
- **Boundary Conditions**: The loss function includes terms to enforce boundary conditions at the origin.

### Training Process

1. **Data Generation**: Generate training data points for the radial coordinate \( $\xi$ \).
2. **Residual Calculation**: Compute the residual of the Lane-Emden equation using TensorFlow's GradientTape.
3. **Loss Function**: The loss function consists of the mean squared error of the residuals and the boundary conditions:
   - Residual loss: Ensures the neural network output satisfies the Lane-Emden equation.
   - Boundary loss: Ensures the neural network output meets the specified boundary conditions.
4. **Optimization**: The model is trained using the Adam optimizer with a specified learning rate.

## Possible State-of-the-Art Improvements

1. Deep Ritz method, Deep Galerkin Method, Finite Basis PINNs (FBPINNs)
2. Adaptive Loss Weighting and Non-Dimensionalization: Utilize adaptive loss weighting schemes based on the magnitude of back-propagated gradients and Neural Tangent Kernel (NTK) theory to balance different loss terms during training.
3. Utilizing advanced optimization techniques, such as L-BFGS, or developing custom training loops, can significantly improve the convergence rates and stability of PINNs

