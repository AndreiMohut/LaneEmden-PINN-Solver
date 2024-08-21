import os
import torch
import numpy as np
from model import PINN
from train import train_model
from utils.data_utils import generate_training_data
from utils.plotting_utils import plot_results, plot_extrapolation_results
from config import MODEL_CONFIG, TRAINING_CONFIG

if __name__ == "__main__":
    # Generate training data
    t_n = generate_training_data(TRAINING_CONFIG['num_points'], TRAINING_CONFIG['n_values'])
    t = t_n[:, 0:1]  # Extract t values
    n = t_n[:, 1:2]  # Extract n values

    # Boundary conditions
    t_boundary = torch.tensor([[0.01]] * len(TRAINING_CONFIG['n_values']), dtype=torch.float32, requires_grad=True)
    n_boundary = torch.tensor([[n_val] for n_val in TRAINING_CONFIG['n_values']], dtype=torch.float32, requires_grad=True)
    theta_boundary = torch.tensor([[1.0]] * len(TRAINING_CONFIG['n_values']), dtype=torch.float32)
    dtheta_dt_boundary = torch.tensor([[0.0]] * len(TRAINING_CONFIG['n_values']), dtype=torch.float32)

    # Initialize model
    model = PINN(
        input_size=MODEL_CONFIG['input_size'],
        hidden_size=MODEL_CONFIG['hidden_size'],
        output_size=MODEL_CONFIG['output_size'],
        activation=MODEL_CONFIG['activation'],
        output_activation=MODEL_CONFIG['output_activation']
    )

    # Train model
    model = train_model(model, t_n, t_boundary, n_boundary, theta_boundary, dtheta_dt_boundary)

    # Generate a name for the saved model file
    model_filename = f"pinn_model_epochs_{TRAINING_CONFIG['num_epochs']}_lr_{TRAINING_CONFIG['learning_rate']}.pth"
    model_path = os.path.join("models", model_filename)

    # Save the model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Generate predictions
    t_test = np.linspace(0.01, 10, TRAINING_CONFIG['num_points'])
    t_test_grid, n_test_grid = np.meshgrid(t_test, TRAINING_CONFIG['n_values'])
    t_test_flat = t_test_grid.flatten().reshape(-1, 1)
    n_test_flat = n_test_grid.flatten().reshape(-1, 1)
    t_n_test = torch.tensor(np.hstack((t_test_flat, n_test_flat)), dtype=torch.float32)
    
    theta_pred = model(t_n_test).detach().numpy()

    # Plot results for training data
    plot_results(t_test, theta_pred, TRAINING_CONFIG['n_values'], TRAINING_CONFIG['num_points'])

    # Extrapolation for new n values
    t_test_grid_new, n_test_grid_new = np.meshgrid(t_test, TRAINING_CONFIG['new_n_values'])
    t_test_flat_new = t_test_grid_new.flatten().reshape(-1, 1)
    n_test_flat_new = n_test_grid_new.flatten().reshape(-1, 1)
    t_n_test_new = torch.tensor(np.hstack((t_test_flat_new, n_test_flat_new)), dtype=torch.float32)
    
    theta_pred_new = model(t_n_test_new).detach().numpy()

    # Plot extrapolation results
    plot_extrapolation_results(t_test, theta_pred_new, TRAINING_CONFIG['new_n_values'], TRAINING_CONFIG['num_points'])
