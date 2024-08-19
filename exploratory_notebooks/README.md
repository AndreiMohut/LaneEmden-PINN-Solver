# Lane-Emden Equation Solver using Physics-Informed Neural Networks

This directory contains two Jupyter notebooks that illustrate how the Lane-Emden equation, a type of non-linear differential equation, can be approximated using Physics-Informed Neural Networks (PINNs).

## Overview of the Notebooks:

1) Case_n2.ipynb:

This notebook demonstrates how a neural network (NN) with multiple activation functions can be used to approximate the solution of the Lane-Emden equation for the specific case where n=2.

2) NN_n_0_1_2_3_4_5.ipynb:

This notebook extends the PINN approach to approximate the solutions for multiple values of n (ranging from n=0 to n=5). The neural network architecture includes a combination of Tanh and Softplus activation functions, which has been shown to produce reproducible results across multiple runs. This notebook also served as a milestone in the development of the code present in the /src/ directory.

Objective: Develop a robust and generalizable PINN architecture capable of solving the Lane-Emden equation for n=0,1,2,3,4,5.

Approach: A single PINN is trained iteratively for each n value, using a combination of Tanh and Softplus activation functions. The model’s performance is validated against numerical solutions.

Results: The model consistently reproduces accurate solutions for each 
n value, with results being reproducible across different runs.