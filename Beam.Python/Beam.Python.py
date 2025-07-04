import time
import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List

from Beam_PINN import calculate_pinn
from Beam_FEM import calculate_fem


def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility across PyTorch, NumPy, and Python.

    Args:
        seed (int): Random seed value.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def exact_solution(z: torch.Tensor) -> torch.Tensor:
    """
    Analytical solution to the dimensionless cantilever beam deflection problem.

    Args:
        z (torch.Tensor): Dimensionless spatial coordinate (0 to 1).

    Returns:
        torch.Tensor: Dimensionless deflection at point z.
    """
    return (1 / 24) * z ** 2 * (z ** 2 - 4 * z + 6)


def calc_errors(x: np.ndarray, x_ref: np.ndarray) -> Tuple[float, float]:
    """
    Calculate RMSE and relative L2 error between predictions and reference.

    Args:
        x (np.ndarray): Predicted values.
        x_ref (np.ndarray): Reference (exact or FEM) values.

    Returns:
        Tuple[float, float]: (RMSE, Relative L2 error)
    """
    rmse = np.sqrt(np.mean((x - x_ref) ** 2))
    rel = np.linalg.norm(x - x_ref, 2) / np.linalg.norm(x_ref, 2)
    return rmse, rel


def main() -> None:
    """
    Main execution function that compares PINN (ADAM + L-BFGS), FEM, and exact solution
    for a cantilever beam under uniform load. Includes training, evaluation, error analysis,
    and visualization.
    """
    set_seed(35)

    # Beam parameters
    L: float = 5.0             # Length of the beam [m]
    E: float = 210e9           # Elastic modulus [Pa]
    I: float = 1e-6            # Area moment of inertia [m^4]
    q: float = 1000.0          # Uniform load [N/m]

    # Sampling parameters
    n_samples: int = 100       # Number of evaluation/collocation points

    # Scaling factors for dimensionless form
    w_char: float = q * L**4 / (E * I)
    scaling_theta: float = w_char / L

    # PINN training
    start = time.time()
    adam_results, time_adam, lbfgs_results, time_lbfgs = calculate_pinn(
        adam_epochs=200, lbfgs_epochs=300, n_points=n_samples, loss_threshold=1e-8
    )
    time_pinn = time.time() - start

    # FEM computation
    start = time.time()
    v_fem, v_fem_theta = calculate_fem(n_samples)
    time_fem = time.time() - start

    # Log runtimes
    print(f"\nExecution Times:")
    print(f"PINN Total:     {time_pinn:.4f} s")
    print(f"  ADAM:         {time_adam:.4f} s")
    print(f"  LBFGS:        {time_lbfgs:.4f} s")
    print(f"FEM:            {time_fem:.4f} s\n")

    # Extract results
    z_eval, v_adam, loss_history = adam_results
    _, v_lbfgs, _ = lbfgs_results
    v_exact: np.ndarray = exact_solution(z_eval).detach().numpy().flatten()

    # Scale dimensionless displacements to real-world units
    w_exact = v_exact * w_char
    w_adam = v_adam * w_char
    w_lbfgs = v_lbfgs * w_char
    w_fem = v_fem * w_char
    w_fem_theta = v_fem_theta * scaling_theta
    x_eval = z_eval.numpy().flatten() * L

    # Compute errors
    rmse_adam_exact, rel_adam_exact = calc_errors(w_adam, w_exact)
    rmse_lbfgs_exact, rel_lbfgs_exact = calc_errors(w_lbfgs, w_exact)
    rmse_fem_exact, rel_fem_exact = calc_errors(w_fem, w_exact)
    rmse_adam_fem, rel_adam_fem = calc_errors(w_adam, w_fem)
    rmse_lbfgs_fem, rel_lbfgs_fem = calc_errors(w_lbfgs, w_fem)
    rmse_adam_lbfgs, rel_adam_lbfgs = calc_errors(w_adam, w_lbfgs)

    # Print error summary
    print("RMSE Errors:")
    print(f"ADAM vs Exact:\t{rmse_adam_exact:.6e}")
    print(f"LBFGS vs Exact:\t{rmse_lbfgs_exact:.6e}")
    print(f"FEM vs Exact:\t{rmse_fem_exact:.6e}")
    print(f"ADAM vs FEM:\t{rmse_adam_fem:.6e}")
    print(f"LBFGS vs FEM:\t{rmse_lbfgs_fem:.6e}")
    print(f"ADAM vs LBFGS:\t{rmse_adam_lbfgs:.6e}\n")

    print("Relative L2 Errors:")
    print(f"ADAM vs Exact:\t{rel_adam_exact:.6e}")
    print(f"LBFGS vs Exact:\t{rel_lbfgs_exact:.6e}")
    print(f"FEM vs Exact:\t{rel_fem_exact:.6e}")
    print(f"ADAM vs FEM:\t{rel_adam_fem:.6e}")
    print(f"LBFGS vs FEM:\t{rel_lbfgs_fem:.6e}")
    print(f"ADAM vs LBFGS:\t{rel_adam_lbfgs:.6e}\n")

    # --- Visualization ---

    # Plot PINN vs Exact
    plt.figure(figsize=(8, 5))
    plt.plot(x_eval, w_lbfgs, color='orange', label='LBFGS', linewidth=2, linestyle='-')
    plt.plot(x_eval, w_adam, color='red', label='ADAM', linewidth=2, linestyle='--')
    plt.plot(x_eval, w_exact, color='navy', label='Analytical', linewidth=2, linestyle=':')
    plt.xlabel('x [m]')
    plt.ylabel('w(x) [m]')
    plt.title('Beam Deflection - PINN vs Exact')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Plot PINN vs FEM
    plt.figure(figsize=(8, 5))
    plt.plot(x_eval, w_lbfgs, color='orange', label='LBFGS', linewidth=2, linestyle='-')
    plt.plot(x_eval, w_adam, color='red', label='ADAM', linewidth=2, linestyle='--')
    plt.plot(x_eval, w_fem, color='navy', label='FEM', linewidth=2, linestyle=':')
    plt.xlabel('x [m]')
    plt.ylabel('w(x) [m]')
    plt.title('Beam Deflection - PINN vs FEM')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Loss plot
    if loss_history is not None:
        plt.figure()
        plt.plot(loss_history)
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss (ADAM)")
        plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
