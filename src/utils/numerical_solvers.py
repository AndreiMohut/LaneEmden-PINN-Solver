import numpy as np
from scipy.integrate import solve_ivp

def solve_lane_emden_numerically(n, t_max, num_points):
    def lane_emden_rhs(t, y, n):
        theta, theta_prime = y
        return [theta_prime, -2/t * theta_prime - theta**n]

    t_eval = np.linspace(0.01, t_max, num_points)
    sol = solve_ivp(lane_emden_rhs, [0.01, t_max], [1, 0], args=(n,), t_eval=t_eval)
    return sol.t, sol.y[0]
