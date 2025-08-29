import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from generate_data.generate_data import generate_input_data
from random_fourier_features.random_fourier_features import random_fourier_features
from pinn_architecture.pinn_architecture import PirateNet

# Define the ODE case you want to solve
# 1: Linear ODE (dy/dt = 2t, y(0) = 1)
# 2: Nonlinear ODE (dy/dt = sin(t) - y^2, y(0) = 1)
# 3: Harmonic Oscillator with damping (d2y/dt2 + dy/dt + w^2 y = 0, y(0) = 1, dy/dt(0) = 0)
ode_case = 3

# Harmonic oscillator parameters
omega = 1.0  # Angular frequency

# Define the ODE residual for each case
def ode_residual_case_1(t, y_pred, dydt_pred):
    return dydt_pred - (2 * t)  # Linear ODE: dy/dt = 2t

def ode_residual_case_2(t, y_pred, dydt_pred):
    return dydt_pred - (torch.sin(t) - y_pred**2)  # Nonlinear ODE: dy/dt = sin(t) - y^2

def ode_residual_case_3(t, y_pred, dydt_pred, d2ydt2_pred):
    return d2ydt2_pred + dydt_pred + omega**2 * y_pred  # Harmonic Oscillator: d2y/dt2 + dy/dt + w^2 y = 0

# Define the function for numerical solution (for Case 3 - Harmonic Oscillator with damping)
def numerical_solution_oscillator(t, y):
    return [y[1], -y[1] - omega**2 * y[0]]  # dy/dt = v, d2y/dt2 = -v - w^2y

# Solve the harmonic oscillator numerically for Case 3 using scipy's solve_ivp
def solve_harmonic_oscillator():
    t_eval = np.linspace(0, t_max, int(t_max / temporal_discretization) + 1)
    # y(0) = 1 (initial position), dy/dt(0) = 0 (initial velocity)
    sol = solve_ivp(numerical_solution_oscillator, [0, t_max], [1.0, 0.0], t_eval=t_eval)
    return sol.t, sol.y[0]

# Select the ODE case
if ode_case == 1:
    ode_residual_func = ode_residual_case_1
    y0_value = 1.0  # Initial condition y(0) = 1
    title = "True Solution vs. PINN Approximation (Linear ODE: dy/dt = 2t)"
    plot_numerical = False
elif ode_case == 2:
    ode_residual_func = ode_residual_case_2
    y0_value = 1.0  # Initial condition y(0) = 1
    title = "Numerical Solution vs. PINN Approximation (Nonlinear ODE: dy/dt = sin(t) - y^2)"
    plot_numerical = True
elif ode_case == 3:
    ode_residual_func = ode_residual_case_3
    y0_value = 1.0  # Initial condition y(0) = 1
    title = "Numerical Solution vs. PINN Approximation (Harmonic Oscillator with Damping)"
    plot_numerical = True

# Generate the temporal domain
t_max = 10.0  # Time from 0 to 10
temporal_discretization = 0.02  # Discretization step for time
num_spatial_dims = 0  # No spatial dimensions, just time
generated_data = generate_input_data(num_spatial_dims, spatial_bounds=[], spatial_discretizations=[], t_max=t_max, temporal_discretization=temporal_discretization)

# Ensure the generated data is a leaf tensor and requires gradients
generated_data = generated_data.clone().detach().requires_grad_(True)

# Step 1: Apply Random Fourier Features to the time coordinates for function approximation
num_fourier_features = 64  # Number of Fourier features to generate
phi_t = random_fourier_features(generated_data, num_fourier_features, length_scale=1.0, seed=42)

# Ensure the Fourier-transformed input is a leaf tensor
phi_t = phi_t.clone().detach().requires_grad_(True)

# Initialize the PirateNet architecture to approximate the solution of the ODE
input_dim = phi_t.shape[1]  # The input dimension now depends on the Fourier features
hidden_dim = 128
num_blocks = 4
model = PirateNet(input_dim=input_dim, hidden_dim=hidden_dim, num_blocks=num_blocks)

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Prepare initial condition
y0 = torch.tensor([[y0_value]], dtype=torch.float32)  # Initial condition based on the ODE case

# Training loop
num_epochs = 500
model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Forward pass with Fourier-transformed time coordinates for function approximation
    y_pred = model(phi_t)  # Passing the embedded time coordinates

    # Compute first derivative: dy/dt
    y_pred_deriv = model(random_fourier_features(generated_data, num_fourier_features, length_scale=1.0, seed=42))
    dydt_pred = torch.autograd.grad(
        outputs=y_pred_deriv, inputs=generated_data, 
        grad_outputs=torch.ones_like(y_pred_deriv), create_graph=True, allow_unused=True)[0]

    # Compute second derivative: d2y/dt2 (grad of grad)
    d2ydt2_pred = torch.autograd.grad(
        outputs=dydt_pred, inputs=generated_data, 
        grad_outputs=torch.ones_like(dydt_pred), create_graph=True, allow_unused=True)[0]

    # Define the ODE residual based on the selected case
    if ode_case == 3:
        ode_residual = ode_residual_func(generated_data, y_pred_deriv, dydt_pred, d2ydt2_pred)
    else:
        ode_residual = ode_residual_func(generated_data, y_pred_deriv, dydt_pred)

    # Compute the loss: MSE of the ODE residual + initial condition
    ode_loss = torch.mean(ode_residual**2)
    
    # Enforce initial conditions y(0) = y0 and dy/dt(0) = 0

    # Enforce y(0) = y0
    ic_y0_loss = (model(random_fourier_features(torch.tensor([[0.0]], dtype=torch.float32), num_fourier_features, length_scale=1.0, seed=42)) - y0).pow(2).mean()

    # Enforce dy/dt(0) = 0 (compute the first derivative at t=0)
    t0 = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True)  # Time t=0 with requires_grad=True
    y0_pred = model(random_fourier_features(t0, num_fourier_features, length_scale=1.0, seed=42))
    dydt_y0_pred = torch.autograd.grad(outputs=y0_pred, inputs=t0, grad_outputs=torch.ones_like(y0_pred), create_graph=True)[0]
    ic_dydt_y0_loss = dydt_y0_pred.pow(2).mean()  # Enforce dy/dt(0) = 0

    # Combine both initial condition losses
    ic_loss = ic_y0_loss + ic_dydt_y0_loss

    # Total loss
    loss = ode_loss + ic_loss
    
    # Backward pass
    loss.backward()

    # Update the weights
    optimizer.step()

    # Print loss every 50 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.10f}")

# Step 5: Plot the results (True/Numerical vs. Approximation)
model.eval()
with torch.no_grad():
    y_pred = model(phi_t).numpy()

# Reshape for plotting
t_vals = generated_data.detach().numpy().flatten()

# Plot Numerical Solution vs. PINN Approximation
plt.figure(figsize=(8, 5))

if plot_numerical:
    if ode_case == 3:
        # Plot the numerical solution for Case 3 (Harmonic Oscillator)
        t_numerical, y_numerical = solve_harmonic_oscillator()
        plt.plot(t_numerical, y_numerical, label="Numerical Solution", color='b')

# Always plot the PINN Approximation
plt.plot(t_vals, y_pred, label="PINN Approximation", linestyle='--', color='r')

plt.xlabel("Time t")
plt.ylabel("y(t)")
plt.title(title)
plt.legend()
plt.grid()
plt.show()
