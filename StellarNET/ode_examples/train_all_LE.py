import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from random_fourier_features.random_fourier_features import random_fourier_features
from pinn_architecture.pinn_architecture import StellarNet
from generate_data.generate_data import generate_input_data

# Define the Lane-Emden residual

def lane_emden_residual(t, n, theta_pred, dtheta_dt, d2theta_dt2):
    safe_t = torch.where(t > 1e-6, t, torch.ones_like(t) * 1e-6)
    return d2theta_dt2 + (2 / safe_t) * dtheta_dt + theta_pred**n

# Generate the spatio-temporal grid with n as a second spatial dimension
num_spatial_dims = 1  # One spatial dimension for n
spatial_bounds = [(0, 5)]  # n values range from 0 to 5
spatial_discretizations = [1]  # Discretization step for n

t_max = 10.0  # Maximum time boundary
temporal_discretization = 0.01  # Discretization step for time

generated_data = generate_input_data(num_spatial_dims, spatial_bounds, spatial_discretizations, t_max, temporal_discretization)

generated_data = generated_data.clone().detach().requires_grad_(True)

# Extract n and t from the combined data grid
n = generated_data[:, 0].view(-1, 1).clone().detach().requires_grad_(True)
t = generated_data[:, -1].view(-1, 1).clone().detach().requires_grad_(True)

# Apply Random Fourier Features to the spatio-temporal coordinates
num_fourier_features = 64
phi_nt = random_fourier_features(torch.cat([n, t], dim=1), num_fourier_features, length_scale=1.0, seed=20)

# Ensure the Fourier-transformed input is a leaf tensor
phi_nt = phi_nt.clone().detach().requires_grad_(True)

# Define the PINN architecture
input_dim = phi_nt.shape[1]
hidden_dim = 256
num_blocks = 2
model = StellarNet(input_dim=input_dim, hidden_dim=hidden_dim, num_blocks=num_blocks)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=50)

# Prepare initial condition theta(t=0, n) = 1
n_initial = n[t == 0].squeeze()  # Extract n when t = 0
n0 = torch.tensor([[1.0]] * len(n_initial), dtype=torch.float32)

# Training loop
num_epochs = 5001
model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Forward pass with Fourier-transformed spatio-temporal coordinates for function approximation
    theta_pred = model(phi_nt)

    theta_pred_deriv = model(random_fourier_features(torch.cat([n, t], dim=1), num_fourier_features, length_scale=1.0, seed=40))

    # Compute the derivatives
    dtheta_dt = torch.autograd.grad(
        outputs=theta_pred_deriv, inputs=t,  # t dimension (time)
        grad_outputs=torch.ones_like(theta_pred_deriv), create_graph=True, allow_unused=True)[0]

    d2theta_dt2 = torch.autograd.grad(
        outputs=dtheta_dt, inputs=t,  # t dimension (time)
        grad_outputs=torch.ones_like(dtheta_dt), create_graph=True, allow_unused=True)[0]
    
    ode_residual = lane_emden_residual(t, n, theta_pred, dtheta_dt, d2theta_dt2)

    # Remove all elements at t = 0 to avoid division by zero
    ode_residual = ode_residual[t != 0]

    ode_loss = torch.mean(ode_residual**2)

    # enforce the initial condition theta(t=0, n) = 1
    phi_n0 = random_fourier_features(torch.cat([n_initial.unsqueeze(1), torch.zeros_like(n_initial.unsqueeze(1))], dim=1), num_fourier_features, length_scale=1.0, seed=40)
    theta0_pred = model(phi_n0)
    ic_loss_theta = torch.mean((theta0_pred - n0)**2)

    dteta0_pred = torch.autograd.grad(
        outputs=theta0_pred, inputs=n_initial,  # n dimension
        grad_outputs=torch.ones_like(theta0_pred), create_graph=True, allow_unused=True)[0]
    ic_loss_dtheta = torch.mean(dteta0_pred**2)

    loss = 10 * ode_loss + 5 * ic_loss_theta + 5 * ic_loss_dtheta

    # Backward pass
    loss.backward(retain_graph=True)

    # Update the weights
    optimizer.step()

    scheduler.step(loss.item())

    # Print loss every 100 epochs
    if (epoch) % 100 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.16f}")


import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Analytical solutions for specific n
def analytical_solution(t, n):
    if n == 0:
        return 1 - t**2 / 6
    elif n == 1:
        return np.where(t > 0, np.sin(t) / t, 1.0)
    elif n == 5:
        return 1 / np.sqrt(1 + t**2 / 3)
    else:
        return None

# Numerical solution for arbitrary n
def numerical_solution_lane_emden(n, t_max, temporal_discretization):
    def lane_emden_ode(t, y):
        theta, dtheta_dt = y
        return [dtheta_dt, -(2 / t) * dtheta_dt - theta**n]
    
    t_eval = np.linspace(1e-6, t_max, int(t_max / temporal_discretization) + 1)
    sol = solve_ivp(lane_emden_ode, [1e-6, t_max], [1.0, 0.0], t_eval=t_eval)
    return sol.t, sol.y[0]

# Generate PINN predictions and check initial conditions
model.eval()
with torch.no_grad():
    for n_value in range(6):  # Iterate through implied values of n (0 to 5)
        # Filter data for current n value
        mask_n = (n == n_value).squeeze()
        t_values = t[mask_n].detach().numpy().flatten()
        phi_nt_n = phi_nt[mask_n]
        
        # PINN prediction
        theta_pinn = model(phi_nt_n).detach().numpy().flatten()
        
        # Analytical solution (if available)
        if n_value in [0, 1, 5]:
            theta_analytical = analytical_solution(t_values, n_value)
        else:
            theta_analytical = None

        # Numerical solution
        t_numerical, theta_numerical = numerical_solution_lane_emden(n_value, t_max, temporal_discretization)
        
        # Plot results
        plt.figure(figsize=(12, 6))
        
        # Plot PINN prediction and benchmark solution
        plt.plot(t_values, theta_pinn, '--r', label='PINN Prediction')
        if theta_analytical is not None:
            plt.plot(t_values, theta_analytical, '-b', label='Analytical Solution')
        else:
            plt.plot(t_numerical, theta_numerical, '-b', label='Numerical Solution')
        
        # Labels and legend
        plt.xlabel('Radial Coordinate (t)')
        plt.ylabel('Solution θ(t)')
        plt.title(f'Lane-Emden Solution for n={n_value}')
        plt.legend()
        plt.grid()
        
        # Save the plot
        plt.savefig(f"Lane_Emden_Solution_n{n_value}.png")
        plt.show()

        # Plot absolute error
        if theta_analytical is not None:
            benchmark_solution = theta_analytical
        else:
            benchmark_solution = theta_numerical
        
        absolute_error = np.abs(theta_pinn - benchmark_solution)
        plt.figure(figsize=(12, 6))
        plt.plot(t_values, absolute_error, '-g', label='Absolute Error')
        plt.axhline(y=np.mean(absolute_error), color='r', linestyle='--', label='Mean Absolute Error')
        plt.xlabel('Radial Coordinate (t)')
        plt.ylabel('Absolute Error')
        plt.title(f'Absolute Error for Lane-Emden Solution (n={n_value})')
        plt.legend()
        plt.grid()
        
        # Save the error plot
        plt.savefig(f"Lane_Emden_Error_n{n_value}.png")
        plt.show()

torch.save(model.state_dict(), "stellar_net_lane_emden.pth")
print("Model saved as stellar_net_lane_emden.pth")
