import numpy as np
import matplotlib.pyplot as plt
import os
import torch

def mean_absolute_difference(predictions, actuals):
    return np.mean(np.abs(predictions.flatten() - actuals))

def max_absolute_difference(predictions, actuals):
    return np.max(np.abs(predictions.flatten() - actuals))

def mean_squared_error(predictions, actuals):
    return np.mean((predictions.flatten() - actuals) ** 2)

def generate_comparison_plot(t_values, predictions, actuals, n_value, plots_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(t_values, predictions, label='PINN Prediction', color='blue')
    plt.plot(t_values, actuals, label='Numerical Solution', color='orange', linestyle='dashed')
    plt.fill_between(t_values, predictions.flatten(), actuals, color='gray', alpha=0.2, label='Difference')
    plt.title(f"Comparison for n={n_value}")
    plt.xlabel('t')
    plt.ylabel(r'$\theta(t)$')
    plt.legend()
    plot_path = os.path.join(plots_dir, f'comparison_n_{n_value}.png')
    plt.savefig(plot_path)
    plt.close()

def calculate_number_of_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def analytical_solution_n_0(t):
    return 1 - (t**2) / 6

def analytical_solution_n_1(t):
    return np.sin(t) / t

def analytical_solution_n_5(t):
    return 1 / np.sqrt(1 + t**2 / 3)

def generate_combined_plot(t_values, predictions_dict, numerical_dict, plots_dir):
    plt.figure(figsize=(10, 6))
    for n_val, pred in predictions_dict.items():
        plt.plot(t_values, pred, label=f'PINN Prediction for n={n_val}')
        plt.plot(t_values, numerical_dict[n_val], label=f'Numerical Solution for n={n_val}', linestyle='dashed')
    plt.ylim(-0.5, 1.1)
    plt.xlabel('t')
    plt.ylabel(r'$\theta(t)$')
    plt.title('PINN vs Numerical Solutions for All n')
    plt.legend()
    plot_path = os.path.join(plots_dir, f'combined_comparison.png')
    plt.savefig(plot_path)
    plt.close()
