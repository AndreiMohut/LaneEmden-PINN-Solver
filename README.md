# LaneEmden-PINN-Solver

This repository contains a Python implementation for solving the Lane-Emden equation using a Physics-Informed Neural Network (PINN). The Lane-Emden equation is a type of differential equation that arises in the study of polytropic models in astrophysics.

## Lane-Emden Equation

The Lane-Emden equation is a dimensionless form of the Poisson equation for the gravitational potential of a self-gravitating, spherically symmetric, polytropic fluid. It is given by:

$$ \frac{1}{\xi^2} \frac{d}{d\xi} \left( \xi^2 \frac{d\theta}{d\xi} \right) + \theta^n = 0 $$

where:
- $\xi$ is the dimensionless radial coordinate,
- $\theta(\xi)$ is the dimensionless density,
- $n$ is the polytropic index.

## Known Analytical Solutions

For certain values of the polytropic index \( n \), the Lane-Emden equation has known analytical solutions:

- **For \( n = 0 \):** 
  $$ \theta(\xi) = 1 - \frac{\xi^2}{6} $$

- **For \( n = 1 \):** 
  $$ \theta(\xi) = \frac{\sin(\xi)}{\xi} $$

- **For \( n = 5 \):** 
  $$ \theta(\xi) = \frac{1}{\sqrt{1 + \frac{\xi^2}{3}}} $$

These solutions are used to validate the numerical and PINN-based methods implemented in this repository. 

## Overview

This implementation uses a Physics-Informed Neural Network (PINN) to solve the Lane-Emden equation. The PINN leverages neural networks to approximate the solution while incorporating the physics of the differential equation into the loss function.

### Key Features

- **Flexible Neural Network Architecture:** The PINN architecture can be easily customized by adjusting the model configuration in `config.py`.
- **Numerical Differentiation:** The PINN model computes derivatives using both automatic differentiation and finite difference methods for flexibility and accuracy.
- **Comprehensive Testing:** The repository includes unit and integration tests, ensuring the robustness of the implementation.
- **Boundary Conditions:** The loss function includes terms to enforce boundary conditions at the origin.

### Training Process

1. **Data Generation**: Generate training data points for the radial coordinate \( \xi \) and polytropic index \( n \).
2. **Residual Calculation**: Compute the residual of the Lane-Emden equation using both automatic differentiation and finite difference methods.
3. **Loss Function**: The loss function consists of the mean squared error of the residuals and the boundary conditions:
   - Residual loss: Ensures the neural network output satisfies the Lane-Emden equation.
   - Boundary loss: Ensures the neural network output meets the specified boundary conditions.
4. **Optimization**: The model is trained using the Adam optimizer with a specified learning rate.

### How to Run the Project

```bash
git clone https://github.com/yourusername/LaneEmden-PINN-Solver.git
cd LaneEmden-PINN-Solver
pytest tests/
python src/main.py
```
