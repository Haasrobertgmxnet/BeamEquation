import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from Beam_PINN_inverse import PINN_Inverse

# === Physikalische Parameter ===
L = 5
E_true = 210e9   # → Nur für Erzeugung der "Messdaten"
I = 1e-6
q = 1000

w_char = q * L**4 / (E_true * I)

# === Erzeuge "Messdaten" (synthetisch mit Störgröße) ===
def exact_solution(z):
    return (1/24) * z**2 * (z**2 - 4*z + 6)

n_points = 100
z_eval = torch.linspace(0, 1, n_points).reshape(-1,1)
v_exact = exact_solution(z_eval).detach().numpy().flatten()
w_exact = v_exact * w_char
noise = 0.01 * np.max(w_exact)
w_measured = w_exact + np.random.normal(0, noise, size=w_exact.shape)

w_measured_tensor = torch.tensor(w_measured, dtype=torch.float32).reshape(-1,1)

# === PINN Trainieren ===
pinn_inverse = PINN_Inverse(adam_epochs=5000, n_points=n_points)
pinn_inverse.train(w_measured_tensor, w_char)
z_plot, w_pred, E_pred, loss_history = pinn_inverse.get_results(w_char)

# === Plot ===
import matplotlib.pyplot as plt
plt.figure()
plt.plot(z_plot*L, w_measured, 'o', label="Messdaten")
plt.plot(z_plot*L, w_pred, '-', label="PINN w(x)")
plt.xlabel('x [m]')
plt.ylabel('w(x) [m]')
plt.legend()
plt.grid(True)
plt.title("Biegung")

plt.figure()
plt.plot(z_plot*L, E_pred / np.max(E_pred), label="relatives E(z)")
plt.xlabel('x [m]')
plt.ylabel("E(z) (relativ)")
plt.title("Geschätzte Elastizitätsmodul-Verteilung")
plt.grid(True)
plt.legend()

plt.figure()
plt.plot(loss_history)
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss History")
plt.grid(True)
plt.show()

