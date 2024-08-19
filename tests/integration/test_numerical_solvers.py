import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import numpy as np
from utils.numerical_solvers import solve_lane_emden_numerically

def analytical_solution_n_0(t):
    return 1 - (t**2) / 6

def analytical_solution_n_1(t):
    return np.sin(t) / t

def analytical_solution_n_5(t):
    return 1 / np.sqrt(1 + t**2 / 3)

def test_numerical_solver_n_0():
    t_num, theta_num = solve_lane_emden_numerically(n=0, t_max=2, num_points=100)
    theta_analytical = analytical_solution_n_0(t_num)
    
    assert np.allclose(theta_num, theta_analytical, atol=1e-2), "Numerical solution does not match analytical solution for n=0"

def test_numerical_solver_n_1():
    t_num, theta_num = solve_lane_emden_numerically(n=1, t_max=np.pi, num_points=100)
    theta_analytical = analytical_solution_n_1(t_num)
    
    assert np.allclose(theta_num, theta_analytical, atol=1e-2), "Numerical solution does not match analytical solution for n=1"

def test_numerical_solver_n_5():
    t_num, theta_num = solve_lane_emden_numerically(n=5, t_max=10, num_points=100)
    theta_analytical = analytical_solution_n_5(t_num)
    
    assert np.allclose(theta_num, theta_analytical, atol=1e-2), "Numerical solution does not match analytical solution for n=5"
