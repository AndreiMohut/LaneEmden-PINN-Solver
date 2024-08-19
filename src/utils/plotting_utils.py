import matplotlib.pyplot as plt
from utils.numerical_solvers import solve_lane_emden_numerically
from config import PLOTTING_CONFIG

def plot_results(t_test, theta_pred, n_values, num_points):
    plt.figure(figsize=(10, 6))
    for i, n_val in enumerate(n_values):
        t_num, theta_num = solve_lane_emden_numerically(n=n_val, t_max=10, num_points=num_points)
        theta_pred_n = theta_pred[i*num_points:(i+1)*num_points]
        
        plt.plot(t_test, theta_pred_n, label=f'PINN Prediction for n={n_val}')
        plt.plot(t_num, theta_num, label=f'Numerical Solution for n={n_val}', linestyle='dashed')
    
    plt.ylim(PLOTTING_CONFIG['y_limits'])
    plt.xlabel('t')
    plt.ylabel(f'theta(t, n)')
    plt.title(f'PINN vs Numerical Solution for Lane-Emden Equation')
    
    # Adjust the legend
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')
    plt.tight_layout()
    plt.show()

def plot_extrapolation_results(t_test, theta_pred, new_n_values, num_points):
    plt.figure(figsize=(10, 6))
    for i, n_val in enumerate(new_n_values):
        t_num, theta_num = solve_lane_emden_numerically(n=n_val, t_max=10, num_points=num_points)
        theta_pred_n = theta_pred[i*num_points:(i+1)*num_points]
        
        plt.plot(t_test, theta_pred_n, label=f'PINN Prediction for n={n_val}')
        plt.plot(t_num, theta_num, label=f'Numerical Solution for n={n_val}', linestyle='dashed')
    
    plt.ylim(PLOTTING_CONFIG['y_limits'])
    plt.xlabel('t')
    plt.ylabel(f'theta(t, n)')
    plt.title(f'Extrapolation: PINN vs Numerical Solution for Lane-Emden Equation')
    
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')
    plt.tight_layout()
    plt.show()
