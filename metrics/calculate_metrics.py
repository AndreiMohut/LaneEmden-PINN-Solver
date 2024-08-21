import sys
import os
import torch
import numpy as np
import pandas as pd

# Adjust the path to include the src directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from model import PINN
from utils.numerical_solvers import solve_lane_emden_numerically
from config import MODEL_CONFIG, TRAINING_CONFIG
from metrics_utils import (
    mean_absolute_difference,
    max_absolute_difference,
    mean_squared_error,
    generate_comparison_plot,
    calculate_number_of_parameters,
    analytical_solution_n_0,
    analytical_solution_n_1,
    analytical_solution_n_5,
    generate_combined_plot
)

def calculate_and_save_metrics(model_path):
    # Initialize the model
    model = PINN(
        input_size=MODEL_CONFIG['input_size'],
        hidden_size=MODEL_CONFIG['hidden_size'],
        output_size=MODEL_CONFIG['output_size'],
        activation=MODEL_CONFIG['activation'],
        output_activation=MODEL_CONFIG['output_activation']
    )

    # Load the trained model
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    # Prepare to store results
    results = []
    plots_dir = os.path.join("metrics", "results", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Calculate number of parameters
    num_params = calculate_number_of_parameters(model)

    # Dictionaries to store predictions for combined plots
    predictions_dict = {}
    numerical_dict = {}

    # Calculate metrics and generate plots for each n value
    for n_val in TRAINING_CONFIG['n_values']:
        # Generate test data for the specific n value
        t_test = np.linspace(0.01, 10, TRAINING_CONFIG['num_points'])
        t_test_tensor = torch.tensor(t_test.reshape(-1, 1), dtype=torch.float32)
        n_test_tensor = torch.tensor([[n_val]] * len(t_test), dtype=torch.float32)

        t_n_test = torch.cat((t_test_tensor, n_test_tensor), dim=1)
        theta_pred = model(t_n_test).detach().numpy()

        # Get the numerical solution for comparison
        t_num, theta_num = solve_lane_emden_numerically(n=n_val, t_max=10, num_points=TRAINING_CONFIG['num_points'])

        # Store predictions for combined plots
        predictions_dict[n_val] = theta_pred
        numerical_dict[n_val] = theta_num

        # Calculate metrics
        mean_diff = mean_absolute_difference(theta_pred, theta_num)
        max_diff = max_absolute_difference(theta_pred, theta_num)
        mse = mean_squared_error(theta_pred, theta_num)

        # Store the results
        results.append([n_val, mean_diff, max_diff, mse])

        # Generate and save the comparison plot
        generate_comparison_plot(t_test, theta_pred, theta_num, n_val, plots_dir)

    # Compare with analytical solutions for n = 0, 1, 5
    analytical_results = []
    for n_val, analytical_fn in zip([0, 1, 5], [analytical_solution_n_0, analytical_solution_n_1, analytical_solution_n_5]):
        analytical_theta = analytical_fn(t_test)
        mse_pinn_vs_analytical = mean_squared_error(predictions_dict[n_val], analytical_theta)
        mse_numerical_vs_analytical = mean_squared_error(numerical_dict[n_val], analytical_theta)
        analytical_results.append([n_val, mse_pinn_vs_analytical, mse_numerical_vs_analytical])

    # Generate the combined plot
    generate_combined_plot(t_test, predictions_dict, numerical_dict, plots_dir)

    # Convert results to a DataFrame and save to CSV
    df = pd.DataFrame(results, columns=["n", "Mean Absolute Difference", "Max Absolute Difference", "Mean Squared Error"])
    os.makedirs("metrics/results", exist_ok=True)
    df.to_csv(os.path.join("metrics/results", f'numerical_vs_pinn_comparison_{TRAINING_CONFIG["num_epochs"]}_epochs.csv'), index=False)

    # Save analytical comparison to CSV
    df_analytical = pd.DataFrame(analytical_results, columns=["n", "MSE (PINN vs Analytical)", "MSE (Numerical vs Analytical)"])
    df_analytical.to_csv(os.path.join("metrics/results", f'analytical_comparison_{TRAINING_CONFIG["num_epochs"]}_epochs.csv'), index=False)

    # Generate a summary report in markdown format
    with open(os.path.join("metrics/results", f'metrics_report_{TRAINING_CONFIG["num_epochs"]}_epochs.md'), "w") as report:
        report.write(f"# Metrics Report for {TRAINING_CONFIG['num_epochs']} Epochs\n\n")
        report.write(f"**Number of Parameters**: {num_params}\n\n")
        report.write("This report summarizes the performance metrics of the PINN model compared to the numerical solutions for the Lane-Emden equation.\n\n")
        report.write(df.to_markdown(index=False))
        report.write("\n\n## Comparison with Analytical Solutions\n\n")
        report.write(df_analytical.to_markdown(index=False))
        report.write("\n\n## Plots\n\n")
        for n_val in TRAINING_CONFIG['n_values']:
            report.write(f"![Comparison for n={n_val}](plots/comparison_n_{n_val}.png)\n")
        report.write("\n## Combined Plot\n")
        report.write(f"![Combined Plot](plots/combined_comparison.png)\n")

    print("Metrics calculated and saved successfully.")

if __name__ == "__main__":
    # Define the path to the model to be evaluated
    model_path = f"models/pinn_model_epochs_{TRAINING_CONFIG['num_epochs']}_lr_{TRAINING_CONFIG['learning_rate']}.pth"
    
    # Calculate metrics and save the results
    calculate_and_save_metrics(model_path)
