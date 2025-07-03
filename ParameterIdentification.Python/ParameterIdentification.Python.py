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

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def exact_solution(z):
    return (1/24) * z**2 * (z**2 - 4*z + 6)

def bad_solution(z):
    z_eval = z
    u= 0.667
    z_eval = torch.where(z_eval > u, z_eval - u, torch.zeros_like(z_eval))
    return exact_solution(z) + z_eval*0.3

def add_noise(v, scal, seed = 42):
    rng = np.random.default_rng(seed=seed)
    max_noise = scal * np.abs(np.mean(v))
    return v + rng.normal(loc=0.0,scale = max_noise, size=v.shape)

def generate_measured_data(w_char, n_points = 100):
    z_eval = torch.linspace(0, 1, n_points).reshape(-1,1)
    v_exact = exact_solution(z_eval).detach().numpy().flatten()
    # v_exact = bad_solution(z_eval).detach().numpy().flatten()

    w_exact = v_exact * w_char
    noise = 0.01 * np.max(w_exact)
    return w_char * v_exact + np.random.normal(0, noise, size=w_exact.shape)

class TrainingResults():
    def __init__(self, x, w_exact, w_measured, w_pred, E_pred, total_loss, data_loss, phyiscs_loss, bc_loss, d1_e):
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

def calculate(seed1, seed2):
    # === Physikalische Parameter ===
    L = 5
    E_true = 210e9   # Nur fuer Erzeugung der "Messdaten"
    I = 1e-6
    q = 1000

    w_char = q * L**4 / (E_true * I)

    # === Erzeuge "Messdaten" (synthetisch mit Stoergroesse) ===
    n_points = 100
    z_eval = torch.linspace(0, 1, n_points).reshape(-1,1)
    v_exact = exact_solution(z_eval).detach().numpy().flatten()
    v_measured = add_noise(v_exact, 0.05, seed1)
    v_measured_tensor = torch.tensor(v_measured , dtype=torch.float32).reshape(-1,1)

    w_exact = w_char * v_exact
    w_measured = w_char * v_measured
    
    # === PINN Trainieren ===
    # set inital value for random generator
    set_seed(seed2)
    pinn_inverse = PINN_Inverse(epochs_adam=10000, epochs_lbfgs=5000, n_points=n_points, loss_threshold = 1e-6, learning_rate= 0.005)
    pinn_inverse.train(v_measured_tensor, lambda_reg= 1e-3)
    z_plot, v_pred, e_pred = pinn_inverse.get_results()

    print(f"Final Loss LBFGS: {pinn_inverse.final_loss_lbfgs:6e}")
    # print(f"e pred max: {np.max(e_pred)}")
    # print(f"e pred min: {np.min(e_pred)}")

    w_pred = v_pred * w_char
    e_pred = e_pred / np.max(e_pred)
    E_pred = e_pred * E_true

    training_results = TrainingResults(z_plot*L, w_exact, w_measured, w_pred, E_pred, pinn_inverse.final_loss_lbfgs.detach().numpy().flatten(), pinn_inverse.loss_data, pinn_inverse.loss_physics, pinn_inverse.loss_bc, pinn_inverse.d1_e)
    return training_results

def is_vec_constant(vec):
    eps = 1e-4
    vec_max = np.max(vec)
    return (vec_max - np.min(vec)) < np.abs(vec_max)*eps

def generate_integer_pairs(N=100, low=0, high=100, seed=None, pairs=[]):
    rng = np.random.default_rng(seed)
    existing = set(pairs)  # Für schnellen Vergleich
    result = []

    while len(result) < N:
        # Generiere ein Paar
        candidate = tuple(rng.integers(low=low, high=high + 1, size=2))
        if candidate not in existing:
            result.append(candidate)
            existing.add(candidate)  # Verhindert spätere Duplikate

        if len(existing) >= (high - low + 1) ** 2:
            raise ValueError("Keine weiteren einzigartigen Paare verfügbar")

    return result

def plot_diagrams(df_filtered, k, v):
    x1 = df_filtered["x"]
    w_exact1 = df_filtered["w_exact"]
    w_measured1 = df_filtered["w_measured"]
    w_pred1 = df_filtered["w_pred"]
    E_pred1 = df_filtered["E_pred"]

    title = k + ": Deflection and exact solution"
    fname = k + "Deflection_exact.png"
    plt.figure()
    plt.plot(x1, w_measured1, 'o', label="measured w")
    plt.plot(x1, w_exact1, '-', label="exact w(x)")
    plt.xlabel('x [m]')
    plt.ylabel('w(x) [m]')
    plt.legend()
    plt.grid(True)
    plt.title(title)
    plt.savefig(fname)

    title = k + ": Deflection"
    fname = k + "Deflection_PINN.png"
    plt.figure()
    plt.plot(x1, w_measured1, 'o', label="measured w")
    plt.plot(x1, w_pred1, '-', label="PINN w(x)")
    plt.xlabel('x [m]')
    plt.ylabel('w(x) [m]')
    plt.legend()
    plt.grid(True)
    plt.title(title)
    plt.savefig(fname)

    title = k + ": Elastic modulus"
    fname = k + "Elastic modulus.png"
    plt.figure()
    plt.plot(x1, E_pred1, '--',label="absolute E(x)")
    plt.xlabel('x [m]')
    plt.ylabel("E(x)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(fname)


def main():

    pairs_loaded = []
    pairs = []
    N = 25

    try:
        with open("pairs.txt", "r") as f:
            for line in f:
                a, b = map(int, line.strip().split(","))
                pairs_loaded.append((a, b))
        pairs = generate_integer_pairs(N=N, low=0, high=100,seed=42, pairs= pairs_loaded)

    except FileNotFoundError:
        pairs = generate_integer_pairs(N=N, low=0, high=100,seed=42)

    with open("pairs.txt", "a+") as f:
        for a, b in pairs:
            f.write(f"{a},{b}\n")
    
    df_pointwise_data = pd.DataFrame()
    df_only_stats = pd.DataFrame()

    for pair in pairs:
        print(pair)
        training_results = calculate(pair[0], pair[1])
        # data_loss = training_results.data_loss
        # d1_e = training_results.d1_e
        E_max_diff = np.max(training_results.E_pred) - np.min(training_results.E_pred)
        R2 = r2_score(training_results.w_measured, training_results.w_pred)

        n = len(training_results.w_pred)
        df1 = pd.DataFrame(
            {
                "seed1" : pair[0],
                "seed2" : pair[1],
                "Data_Loss" : training_results.data_loss.item(),
                "Physics_Loss" : training_results.phyiscs_loss.item(),
                "BC_Loss" : training_results.bc_loss.item(),
                "E_max_diff" : E_max_diff,
                "R2" : R2,
                "x" : training_results.x,
                "w_exact" : training_results.w_exact,
                "w_measured" : training_results.w_measured,
                "w_pred" : training_results.w_pred,
                "E_pred" : training_results.E_pred
            }
        )

        df2 = pd.DataFrame(
            {
                "seed1" : [int(pair[0])],
                "seed2" : [int(pair[1])],
                "Data_Loss" : training_results.data_loss.item(),
                "Physics_Loss" : training_results.phyiscs_loss.item(),
                "BC_Loss" : training_results.bc_loss.item(),
                "E_max_diff" : E_max_diff,
                "R2" : [R2]
            }
        )

        df_pointwise_data = pd.concat([df_pointwise_data, df1], ignore_index=True)
        df_only_stats = pd.concat([df_only_stats, df2], ignore_index=True)

    print(df_only_stats)

    data_min_R2 = df_only_stats.loc[df_only_stats["R2"].idxmin()]
    print("Datset with minimal R2:")
    print(data_min_R2)

    data_max_data_loss = df_only_stats.loc[df_only_stats["Data_Loss"].idxmax()]
    print("Datset with maximal data loss:")
    print(data_max_data_loss)

    data_max_physics_loss = df_only_stats.loc[df_only_stats["Physics_Loss"].idxmax()]
    print("Datset with naximal physics loss:")
    print(data_max_physics_loss)

    data_max_bc_loss = df_only_stats.loc[df_only_stats["BC_Loss"].idxmax()]
    print("Datset with maximal bc data loss:")
    print(data_max_bc_loss)

    data_max_E_diff = df_only_stats.loc[df_only_stats["E_max_diff"].idxmax()]
    print("Datset with maximal difference in E:")
    print(data_max_E_diff)

    csv_files = dict({
        "data_pointwise.csv" : [df_pointwise_data, '.', ','],
        "data_only_stats.csv" : [df_only_stats, '.', ','],
        "data_pointwise_xl.csv" : [df_pointwise_data, ',', ';'],
        "data_only_stats_xl.csv" : [df_only_stats, ',', ';'],
        })

    for k,v in csv_files.items():
        if os.path.isfile(k):
            v[0].to_csv(k, mode='a', header=False, sep = v[2], decimal = v[1], index=False)
        else:
            v[0].to_csv(k, sep = v[2], decimal = v[1], index=False)
    # df_pointwise_data.to_csv("data_pointwise.csv", index=False)
    # df_only_stats.to_csv("data_only_stats.csv", index=False)
    # df_only_stats.to_csv("data_only_stats.csv", mode='a', header=False, index=False)

    plot_dict = dict({
        "Dataset with minimal R2" : [data_min_R2["seed1"], data_min_R2["seed2"]],
        "Dataset with maximal data loss" : [data_max_data_loss["seed1"], data_max_data_loss["seed2"]],
        "Dataset with maximal physics loss" : [data_max_physics_loss["seed1"], data_max_physics_loss["seed2"]],
        "Dataset with maximal bc loss" : [data_max_bc_loss["seed1"], data_max_bc_loss["seed2"]],
        "Dataset with maximal difference in E" : [data_max_E_diff["seed1"], data_max_E_diff["seed2"]]
        })

    for [k,v] in plot_dict.items():
        df_filtered = df_pointwise_data[(df_pointwise_data["seed1"] == v[0]) & (df_pointwise_data["seed2"] == v[1])]
        print(df_filtered) 
        plot_diagrams(df_filtered, k, v)

    return

if __name__ == "__main__":
    main()