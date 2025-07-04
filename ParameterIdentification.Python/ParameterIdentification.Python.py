"""
Inverse problem using Physics-Informed Neural Networks (PINNs)
to estimate the elastic modulus from deflection measurements
of a simply supported beam under uniform loading.
"""

import time
import os
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from Beam_PINN_inverse import PINN_Inverse
from typing import Union

def set_seed(seed: int) -> None:
    """
    Set seeds for reproducibility.

    Args:
        seed (int): Random seed.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def exact_solution(z: torch.Tensor) -> torch.Tensor:
    """
    Compute exact analytical solution for beam deflection.

    Args:
        z (torch.Tensor): Normalized spatial coordinates.

    Returns:
        torch.Tensor: Exact deflection.
    """
    return (1 / 24) * z**2 * (z**2 - 4 * z + 6)

def bad_solution(z: torch.Tensor) -> torch.Tensor:
    """
    Create an artificially perturbed deflection profile.

    Args:
        z (torch.Tensor): Normalized spatial coordinates.

    Returns:
        torch.Tensor: Perturbed deflection.
    """
    z_eval = torch.where(z > 0.667, z - 0.667, torch.zeros_like(z))
    return exact_solution(z) + z_eval * 0.3

def add_noise(v: np.ndarray, scal: float, seed: int = 42) -> np.ndarray:
    """
    Add Gaussian noise to a signal.

    Args:
        v (np.ndarray): Original signal.
        scal (float): Noise level relative to signal mean.
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        np.ndarray: Noisy signal.
    """
    rng = np.random.default_rng(seed=seed)
    max_noise = scal * np.abs(np.mean(v))
    return v + rng.normal(loc=0.0, scale=max_noise, size=v.shape)

def generate_measured_data(w_char: float, n_points: int = 100) -> np.ndarray:
    """
    Generate noisy measured data based on exact beam deflection.

    Args:
        w_char (float): Characteristic deflection scale.
        n_points (int, optional): Number of spatial points. Defaults to 100.

    Returns:
        np.ndarray: Noisy deflection data.
    """
    z_eval = torch.linspace(0, 1, n_points).reshape(-1, 1)
    v_exact = exact_solution(z_eval).detach().numpy().flatten()
    w_exact = v_exact * w_char
    noise = 0.01 * np.max(w_exact)
    return w_char * v_exact + np.random.normal(0, noise, size=w_exact.shape)

class TrainingResults:
    """
    Container for storing PINN training results.
    """

    def __init__(
        self,
        x: np.ndarray,
        w_exact: np.ndarray,
        w_measured: np.ndarray,
        w_pred: np.ndarray,
        E_pred: np.ndarray,
        total_loss: np.ndarray,
        data_loss: torch.Tensor,
        phyiscs_loss: torch.Tensor,
        bc_loss: torch.Tensor,
        d1_e: np.ndarray
    ) -> None:
        self.x = x
        self.w_exact = w_exact
        self.w_measured = w_measured
        self.w_pred = w_pred
        self.E_pred = E_pred
        self.total_loss = total_loss
        self.data_loss = data_loss
        self.phyiscs_loss = phyiscs_loss
        self.bc_loss = bc_loss
        self.d1_e = d1_e

def calculate(seed1: int, seed2: int) -> TrainingResults:
    """
    Perform a single PINN training run with specific seeds.

    Args:
        seed1 (int): Seed for data noise.
        seed2 (int): Seed for training.

    Returns:
        TrainingResults: Results of training run.
    """
    L = 5
    E_true = 210e9
    I = 1e-6
    q = 1000
    w_char = q * L**4 / (E_true * I)
    n_points = 100

    z_eval = torch.linspace(0, 1, n_points).reshape(-1, 1)
    v_exact = exact_solution(z_eval).detach().numpy().flatten()
    v_measured = add_noise(v_exact, 0.05, seed1)
    v_measured_tensor = torch.tensor(v_measured, dtype=torch.float32).reshape(-1, 1)
    w_exact = w_char * v_exact
    w_measured = w_char * v_measured

    set_seed(seed2)
    pinn_inverse = PINN_Inverse(epochs_adam=10000, epochs_lbfgs=5000, n_points=n_points, loss_threshold=1e-6, learning_rate=0.005)
    pinn_inverse.train(v_measured_tensor, lambda_reg=1e-3)

    z_plot, v_pred, e_pred = pinn_inverse.get_results()
    print(f"Final Loss LBFGS: {pinn_inverse.final_loss_lbfgs:6e}")
    w_pred = v_pred * w_char
    e_pred = e_pred / np.max(e_pred)
    E_pred = e_pred * E_true

    return TrainingResults(
        x=z_plot * L,
        w_exact=w_exact,
        w_measured=w_measured,
        w_pred=w_pred,
        E_pred=E_pred,
        total_loss=pinn_inverse.final_loss_lbfgs.detach().numpy().flatten(),
        data_loss=pinn_inverse.loss_data,
        phyiscs_loss=pinn_inverse.loss_physics,
        bc_loss=pinn_inverse.loss_bc,
        d1_e=pinn_inverse.d1_e
    )

def is_vec_constant(vec: np.ndarray) -> bool:
    """
    Check if a vector is approximately constant.

    Args:
        vec (np.ndarray): Input vector.

    Returns:
        bool: True if vector values are approximately equal.
    """
    eps = 1e-4
    vec_max = np.max(vec)
    return (vec_max - np.min(vec)) < np.abs(vec_max) * eps

def generate_integer_pairs(N: int = 100, low: int = 0, high: int = 100, seed: Union[int, None] = None, pairs: list[tuple[int, int]] = []) -> list[tuple[int, int]]:
    """
    Generate unique random integer pairs.

    Args:
        N (int): Number of pairs.
        low (int): Minimum value (inclusive).
        high (int): Maximum value (inclusive).
        seed (int, optional): Random seed.
        pairs (list[tuple[int, int]]): Existing pairs to avoid duplicates.

    Returns:
        list[tuple[int, int]]: List of new integer pairs.
    """
    rng = np.random.default_rng(seed)
    existing = set(pairs)
    result = []

    while len(result) < N:
        candidate = tuple(rng.integers(low=low, high=high + 1, size=2))
        if candidate not in existing:
            result.append(candidate)
            existing.add(candidate)
        if len(existing) >= (high - low + 1) ** 2:
            raise ValueError("No more unique pairs available.")
    return result

def plot_diagrams(df_filtered: pd.DataFrame, k: str, v: list[int]) -> None:
    """
    Plot and save deflection and elasticity diagrams.

    Args:
        df_filtered (pd.DataFrame): Filtered dataset for one run.
        k (str): Identifier name for the plots.
        v (list[int]): Seed values.
    """
    x1 = df_filtered["x"]
    w_exact1 = df_filtered["w_exact"]
    w_measured1 = df_filtered["w_measured"]
    w_pred1 = df_filtered["w_pred"]
    E_pred1 = df_filtered["E_pred"]

    # Deflection exact
    plt.figure()
    plt.plot(x1, w_measured1, 'o', label="measured w")
    plt.plot(x1, w_exact1, '-', label="exact w(x)")
    plt.xlabel('x [m]')
    plt.ylabel('w(x) [m]')
    plt.legend()
    plt.grid(True)
    plt.title(f"{k}: Deflection and exact solution")
    plt.savefig(f"{k}Deflection_exact.png")

    # Deflection PINN
    plt.figure()
    plt.plot(x1, w_measured1, 'o', label="measured w")
    plt.plot(x1, w_pred1, '-', label="PINN w(x)")
    plt.xlabel('x [m]')
    plt.ylabel('w(x) [m]')
    plt.legend()
    plt.grid(True)
    plt.title(f"{k}: Deflection")
    plt.savefig(f"{k}Deflection_PINN.png")

    # Elastic modulus
    plt.figure()
    plt.plot(x1, E_pred1, '--', label="absolute E(x)")
    plt.xlabel('x [m]')
    plt.ylabel("E(x)")
    plt.title(f"{k}: Elastic modulus")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{k}Elastic modulus.png")

def main() -> None:
    """
    Main entry point to execute multiple training runs,
    evaluate results, store data and plots.
    """
    pairs_loaded: list[tuple[int, int]] = []
    N = 2

    try:
        with open("pairs.txt", "r") as f:
            for line in f:
                a, b = map(int, line.strip().split(","))
                pairs_loaded.append((a, b))
        pairs = generate_integer_pairs(N=N, low=0, high=100, seed=42, pairs=pairs_loaded)
    except FileNotFoundError:
        pairs = generate_integer_pairs(N=N, low=0, high=100, seed=42)

    with open("pairs.txt", "a+") as f:
        for a, b in pairs:
            f.write(f"{a},{b}\n")

    df_pointwise_data = pd.DataFrame()
    df_only_stats = pd.DataFrame()

    for pair in pairs:
        print(pair)
        training_results = calculate(pair[0], pair[1])
        E_max_diff = np.max(training_results.E_pred) - np.min(training_results.E_pred)
        R2 = r2_score(training_results.w_measured, training_results.w_pred)
        df1 = pd.DataFrame({
            "seed1": pair[0],
            "seed2": pair[1],
            "Data_Loss": training_results.data_loss.item(),
            "Physics_Loss": training_results.phyiscs_loss.item(),
            "BC_Loss": training_results.bc_loss.item(),
            "E_max_diff": E_max_diff,
            "R2": R2,
            "x": training_results.x,
            "w_exact": training_results.w_exact,
            "w_measured": training_results.w_measured,
            "w_pred": training_results.w_pred,
            "E_pred": training_results.E_pred
        })
        df2 = pd.DataFrame({
            "seed1": [pair[0]],
            "seed2": [pair[1]],
            "Data_Loss": training_results.data_loss.item(),
            "Physics_Loss": training_results.phyiscs_loss.item(),
            "BC_Loss": training_results.bc_loss.item(),
            "E_max_diff": E_max_diff,
            "R2": [R2]
        })
        df_pointwise_data = pd.concat([df_pointwise_data, df1], ignore_index=True)
        df_only_stats = pd.concat([df_only_stats, df2], ignore_index=True)

    print(df_only_stats)

    # Identify key datasets
    data_min_R2 = df_only_stats.loc[df_only_stats["R2"].idxmin()]
    data_max_data_loss = df_only_stats.loc[df_only_stats["Data_Loss"].idxmax()]
    data_max_physics_loss = df_only_stats.loc[df_only_stats["Physics_Loss"].idxmax()]
    data_max_bc_loss = df_only_stats.loc[df_only_stats["BC_Loss"].idxmax()]
    data_max_E_diff = df_only_stats.loc[df_only_stats["E_max_diff"].idxmax()]

    plot_dict = {
        "Dataset with minimal R2": [data_min_R2["seed1"], data_min_R2["seed2"]],
        "Dataset with maximal data loss": [data_max_data_loss["seed1"], data_max_data_loss["seed2"]],
        "Dataset with maximal physics loss": [data_max_physics_loss["seed1"], data_max_physics_loss["seed2"]],
        "Dataset with maximal bc loss": [data_max_bc_loss["seed1"], data_max_bc_loss["seed2"]],
        "Dataset with maximal difference in E": [data_max_E_diff["seed1"], data_max_E_diff["seed2"]],
    }

    for k, v in plot_dict.items():
        df_filtered = df_pointwise_data[(df_pointwise_data["seed1"] == v[0]) & (df_pointwise_data["seed2"] == v[1])]
        print(df_filtered)
        plot_diagrams(df_filtered, k, v)

    # Save CSVs
    csv_files = {
        "data_pointwise.csv": [df_pointwise_data, '.', ','],
        "data_only_stats.csv": [df_only_stats, '.', ','],
        "data_pointwise_xl.csv": [df_pointwise_data, ',', ';'],
        "data_only_stats_xl.csv": [df_only_stats, ',', ';'],
    }
    for k, v in csv_files.items():
        if os.path.isfile(k):
            v[0].to_csv(k, mode='a', header=False, sep=v[2], decimal=v[1], index=False)
        else:
            v[0].to_csv(k, sep=v[2], decimal=v[1], index=False)

if __name__ == "__main__":
    main()
