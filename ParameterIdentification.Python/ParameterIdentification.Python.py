import time
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from Beam_PINN_inverse import PINN_Inverse

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
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


def generate_measured_data(w_char, n_points = 100):
    z_eval = torch.linspace(0, 1, n_points).reshape(-1,1)
    v_exact = exact_solution(z_eval).detach().numpy().flatten()
    # v_exact = bad_solution(z_eval).detach().numpy().flatten()

    w_exact = v_exact * w_char
    noise = 0.01 * np.max(w_exact)
    return w_char * v_exact + np.random.normal(0, noise, size=w_exact.shape)
    # return w_exact + np.random.normal(0, noise, size=w_exact.shape)



def calculate(seed):
    # set inital value for random generator
    set_seed(seed)

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
    w_exact = v_exact * w_char

    w_measured = generate_measured_data(w_char)
    v_measured = w_measured/w_char
    v_measured_tensor = torch.tensor(v_measured , dtype=torch.float32).reshape(-1,1)

    # plt.figure()
    # plt.plot(z_eval, w_measured, 'o', label="measured w")
    # plt.plot(z_eval, w_exact, '-', label="exact w(x)")
    # plt.xlabel('x [m]')
    # plt.ylabel('w(x) [m]')
    # plt.legend()
    # plt.grid(True)
    # plt.title("Deflection")
    # plt.show()

    # === PINN Trainieren ===
    pinn_inverse = PINN_Inverse(epochs_adam=3000, epochs_lbfgs=1500, n_points=n_points, loss_threshold = 1e-6, learning_rate= 0.005)
    pinn_inverse.train(v_measured_tensor, lambda_reg= 1e-3)
    z_plot, v_pred, e_pred, loss_history, final_loss_lbfgs  = pinn_inverse.get_results()

    print(f"Final Loss LBFGS: {final_loss_lbfgs:6e}")
    print(f"e pred max: {np.max(e_pred)}")
    print(f"e pred min: {np.min(e_pred)}")

    w_pred = v_pred * w_char
    e_pred = e_pred / np.max(e_pred)
    E_pred = e_pred * E_true

    # === Plot ===
       
    # print(e_pred)
    print(f"e pred max: {np.max(e_pred)}")
    print(f"e pred min: {np.min(e_pred)}")

    print(f"E pred max: {np.max(E_pred)}")
    print(f"E pred min: {np.min(E_pred)}")

    # plt.figure()
    # plt.plot(z_eval, w_measured, 'o', label="measured w")
    # plt.plot(z_eval, w_exact, '-', label="exact w(x)")
    # plt.xlabel('x [m]')
    # plt.ylabel('w(x) [m]')
    # plt.legend()
    # plt.grid(True)
    # plt.title("Deflection")

    # plt.figure()
    # plt.plot(z_plot*L, w_measured, 'o', label="measured w")
    # plt.plot(z_plot*L, w_pred, '-', label="PINN w(x)")
    # plt.xlabel('x [m]')
    # plt.ylabel('w(x) [m]')
    # plt.legend()
    # plt.grid(True)
    # plt.title("Deflection")

    # plt.figure()
    # plt.plot(z_plot, e_pred,label="dim less e(z)")
    # plt.xlabel('z')
    # plt.ylabel("e")
    # plt.title("Estimated Elastic modulus distribution")
    # plt.grid(True)
    # plt.legend()

    # plt.figure()
    # plt.plot(z_plot*L, E_pred, ':',label="absolute E(x)")
    # plt.plot(z_plot*L, E_true*np.sign(E_pred), '--', label="constant E")
    # plt.xlabel('x [m]')
    # plt.ylabel("E(z) (abs)")
    # plt.title("Estimated Elastic modulus distribution")
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    # mse_loss = nn.MSELoss(v_measured, v_pred)
    return final_loss_lbfgs.detach().numpy().flatten(), z_plot*L, w_exact, w_measured, w_pred, E_pred
    return final_loss_lbfgs, [z_plot*L], [w_exact], [w_measured], [w_pred], [E_pred]
    quit()


    # plt.figure()
    # plt.plot(z_plot*L, E_pred / np.max(E_pred), label="relative E(x)")
    # plt.xlabel('x [m]')
    # plt.ylabel("E(z) (relativ)")
    # plt.title("Estimated Elastic modulus distribution")
    # plt.grid(True)
    # plt.legend()

    plt.figure()
    plt.plot(loss_history)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss History")
    plt.grid(True)
    # plt.show()

    plt.figure()
    plt.plot(z_plot*L, w_pred, '-', label="PINN w(x)")
    plt.plot(z_plot*L, w_char*v_exact*E_true/E_pred, '.', label="Control w(x)")
    plt.xlabel('x [m]')
    plt.ylabel("E(z) (relativ)")
    plt.title("Estimated Elastic modulus distribution")
    plt.grid(True)
    plt.legend()
    plt.show()

def main():
    seeds = [31,51,73,56,86,8,63,17,84,97,18,72,64,29,20,79,50,66,15,75]
    seeds = [51,73,56,86,8,63,17,84,97,18,72,64,29,20,79,50,66,15,75]
    seeds = [51,73,56,86,8,63,17,84,97,18,64,29,20,79,50,66,15,75]
    # seeds = range(100)
    data_losses = []
    total_losses = []
    max_data_loss = 0
    max_total_loss = 0
    max_data_loss_ = 0
    max_total_loss_ = 0

    w_pred1 = None
    E_pred1 = None
    # w_pred2 = None
    # E_pred2 = None
    for seed in seeds:
        print(f"Calculation: {seed}")
        total_loss, x, w_exact, w_measured, _w_pred, _E_pred = calculate(seed)
        print("Total loss:", " ");
        print(total_loss)
        #data_losses.append(data_loss)
        total_losses.append(total_loss)
        # if max_data_loss < data_loss:
        #     max_data_loss = data_loss
        #     w_pred1 = _w_pred
        #     E_pred1 = _E_pred
        if max_total_loss < total_loss:
            max_total_loss = total_loss
            w_pred1 = _w_pred
            E_pred1 = _E_pred

    # max_data_loss_ = np.max(data_loss)
    # max_total_loss_ = np.max(total_loss)

    # print(f"data_losses: {data_losses}")
    # print(f"max_data_loss: {max_data_loss}")
    # print(f"max_data_loss_: {max_data_loss_}")

    print(f"total_losses: {total_losses}")
    print(f"max_total_loss: {max_total_loss}")
    # print(f"max_total_loss_: {max_total_loss_}")

    plt.figure()
    plt.plot(x, w_measured, 'o', label="measured w")
    plt.plot(x, w_exact, '-', label="exact w(x)")
    plt.xlabel('x [m]')
    plt.ylabel('w(x) [m]')
    plt.legend()
    plt.grid(True)
    plt.title("Deflection")

    plt.figure()
    plt.plot(x, w_measured, 'o', label="measured w")
    plt.plot(x, w_pred1, '-', label="PINN w(x)")
    plt.xlabel('x [m]')
    plt.ylabel('w(x) [m]')
    plt.legend()
    plt.grid(True)
    plt.title("Deflection")

    plt.figure()
    plt.plot(x, E_pred1, '--',label="absolute E(x)")
    # plt.plot(x, E_true*np.sign(E_pred1), '--', label="constant E")
    plt.xlabel('x [m]')
    plt.ylabel("E(z) (abs)")
    plt.title("Estimated Elastic modulus distribution")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()