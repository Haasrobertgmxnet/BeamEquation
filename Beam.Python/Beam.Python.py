import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from Beam_PINN import calculate_pinn
from Beam_FEM import calculate_fem

def exact_solution(z):
    return (1/24) * z**2 * (z**2 - 4 * z + 6)

# Beam parameters
L = 5             # Length of the beam [m]
E = 210e9         # Elastic modulus [Pa]
I = 1e-6          # Area moment of inertia [m^4]
q = 1000          # Load [N/m]

# Charakteristic quantities of the dimensionless problem
w_char = q * L**4 / (E * I)
scaling_theta = w_char / L

start = time.time()
adam_obj, lbfgs_obj = calculate_pinn(5000, 1500, 100, 1e-8)
ende = time.time()
time_pinn = ende - start

start = time.time()
v_fem, v_fem_theta = calculate_fem(99)
ende = time.time()
time_fem = ende - start

print(f"PINN Execution time: {time_pinn:.4f} seconds.")
print(f"FEM Execution time: {time_fem:.4f} seconds.")

z_eval, v_adam, loss_history = adam_obj.get_training_results()
z_eval, v_lbfgs, _ = lbfgs_obj.get_training_results()
v_exact = exact_solution(z_eval).detach().numpy().flatten()

w_exact = v_exact * w_char
w_adam = v_adam * w_char
w_lbfgs = v_lbfgs * w_char
w_fem = v_fem * w_char
w_fem_theta = v_fem_theta * scaling_theta
x_eval = z_eval.numpy().flatten() * L

def calc_errors(x, x_ref):
    rmse = np.sqrt(np.mean((x - x_ref)**2))
    rel = np.linalg.norm(x - x_ref, 2) / np.linalg.norm(x_ref, 2)
    return rmse, rel

rmse_adam, rel_adam = calc_errors(w_adam, w_exact)
rmse_lbfgs, rel_lbfgs = calc_errors(w_lbfgs, w_exact)
rmse_fem, rel_fem = calc_errors(w_fem, w_exact)

print(f"ADAM:\tRMSE: {rmse_adam:.6e}")
print(f"LBFGS:\tRMSE: {rmse_lbfgs:.6e}")
print(f"FEM:\tRMSE: {rmse_fem:.6e}")

print(f"ADAM:\trel error: {rel_adam:.6e}")
print(f"LBFGS:\trel error: {rel_lbfgs:.6e}")
print(f"FEM:\trel error: {rel_fem:.6e}")

plt_title = 'Bending line under uniform load'
plt.figure(figsize=(8,5))
plt.plot(x_eval, w_exact, label='Analytic solution')
plt.plot(x_eval, w_adam, '--', label='ADAM')
plt.xlabel('x [m]')
plt.ylabel('w(x) [m]')
plt.legend()
plt.grid(True)
plt.title(plt_title)
plt.show()

plt.figure(figsize=(8,5))
plt.plot(x_eval, w_adam, '--', label='ADAM', linewidth=2)
plt.plot(x_eval, w_lbfgs, 'o', label='LBFGS', linewidth=2)
plt.plot(x_eval, w_fem, '+', label='FEM', linewidth=2)
plt.xlabel('x [m]')
plt.ylabel('w(x) [m]')
plt.legend()
plt.grid(True, which='both', linestyle='--')
plt.title(plt_title)
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(loss_history)
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss (Adam)")
plt.grid(True)
plt.show()

quit()

