import time
import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from Beam_PINN import calculate_pinn
from Beam_FEM import calculate_fem

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def exact_solution(z):
    return (1/24) * z**2 * (z**2 - 4 * z + 6)

def main():
    # set inital value for random generator
    set_seed(35)

    # Beam parameters
    L = 5             # Length of the beam [m]
    E = 210e9         # Elastic modulus [Pa]
    I = 1e-6          # Area moment of inertia [m^4]
    q = 1000          # Load [N/m]

    # Number of sample points
    n_samples = 100

    # Charakteristic quantities of the dimensionless problem
    w_char = q * L**4 / (E * I)
    scaling_theta = w_char / L

    start = time.time()
    adam_results, time_adam, lbfgs_results, time_lbfgs = calculate_pinn(200, 300, n_samples, 1e-8)
    ende = time.time()
    time_pinn = ende - start

    start = time.time()
    v_fem, v_fem_theta = calculate_fem(n_samples)
    ende = time.time()
    time_fem = ende - start

    print(f"PINN Execution time: {time_pinn:.4f} seconds.")
    print(f"ADAM Execution time: {time_adam:.4f} seconds.")
    print(f"LBFGS Execution time: {time_lbfgs:.4f} seconds.")
    print(f"FEM Execution time: {time_fem:.4f} seconds.")

    z_eval, v_adam, loss_history = adam_results
    z_eval, v_lbfgs, _ = lbfgs_results
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

    rmse_adam_exact, rel_adam_exact = calc_errors(w_adam, w_exact)
    rmse_lbfgs_exact, rel_lbfgs_exact = calc_errors(w_lbfgs, w_exact)
    rmse_fem_exact, rel_fem_exact = calc_errors(w_fem, w_exact)
    rmse_adam_fem, rel_adam_fem = calc_errors(w_adam, w_fem)
    rmse_lbfgs_fem, rel_lbfgs_fem = calc_errors(w_lbfgs, w_fem)
    rmse_adam_lbfgs, rel_adam_lbfgs = calc_errors(w_adam, w_lbfgs)

    print("RMSE values")
    print(f"ADAM-Exact:\tRMSE: {rmse_adam_exact:.6e}")
    print(f"LBFGS-Exact:\tRMSE: {rmse_lbfgs_exact:.6e}")
    print(f"FEM-Exact\tRMSE: {rmse_fem_exact:.6e}")
    print(f"ADAM-FEM:\tRMSE: {rmse_adam_fem:.6e}")
    print(f"LBFGS-FEM:\tRMSE: {rmse_lbfgs_fem:.6e}")
    print(f"ADAM-LBFGS:\tRMSE: {rmse_adam_lbfgs:.6e}")

    print("Relative errors values")
    print(f"ADAM-Exact:\trel error: {rel_adam_exact:.6e}")
    print(f"LBFGS-Exact:\trel error: {rel_lbfgs_exact:.6e}")
    print(f"FEM-Exact\trel error: {rel_fem_exact:.6e}")
    print(f"ADAM-FEM:\trel error: {rel_adam_fem:.6e}")
    print(f"LBFGS-FEM:\trel error: {rel_lbfgs_fem:.6e}")
    print(f"ADAM-LBFGS:\trel error: {rel_adam_lbfgs:.6e}")


    plt_title = 'Deflection curve of a cantilever beam under uniform load'
    plt.figure(figsize=(8, 5))
    plt.plot(x_eval, w_lbfgs, color='orange', label='LBFGS', linewidth=2, linestyle='-')
    plt.plot(x_eval, w_adam, color='red', label='ADAM', linewidth=2, linestyle='--')
    plt.plot(x_eval, w_exact, color='navy', label='Analytic solution', linewidth=2, linestyle=':')
    plt.xlabel('x [m]')
    plt.ylabel('w(x) [m]')
    plt.legend()
    plt.grid(True)
    plt.title(plt_title)
    plt.tight_layout()
    # plt.show()

    plt_title = 'Deflection curve of a cantilever beam under uniform load'
    plt.figure(figsize=(8, 5))
    plt.plot(x_eval, w_lbfgs, color='orange', label='LBFGS', linewidth=2, linestyle='-')
    plt.plot(x_eval, w_adam, color='red', label='ADAM', linewidth=2, linestyle='--')
    plt.plot(x_eval, w_fem, color='navy', label='FEM', linewidth=2, linestyle=':')
    plt.xlabel('x [m]')
    plt.ylabel('w(x) [m]')
    plt.legend()
    plt.grid(True)
    plt.title(plt_title)
    plt.tight_layout()
    # plt.show()

    plt.figure()
    plt.plot(loss_history)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss (Adam)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

