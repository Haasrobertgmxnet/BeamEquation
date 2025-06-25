import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class DNN(nn.Module):
    def __init__(self, layers, activation=nn.Tanh):
        super(DNN, self).__init__()
        layer_list = []
        for i in range(len(layers) - 1):
            layer_list.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                layer_list.append(activation())
        self.model = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.model(x)

class E_Net(nn.Module):
    def __init__(self, layers, activation=nn.Tanh):
        super(E_Net, self).__init__()
        layer_list = []
        for i in range(len(layers) - 1):
            layer_list.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                layer_list.append(activation())
        layer_list.append(nn.Softplus())  # Damit E(z) > 0 bleibt
        self.model = nn.Sequential(*layer_list)

    def forward(self, z):
        return self.model(z) + 1e-3

class PINN_Inverse():
    def __init__(self, adam_epochs, n_points=100, loss_threshold=1e-6, learning_rate=0.01, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.device = torch.device('cpu')
        self.x = torch.linspace(0, 1, n_points, requires_grad=True).reshape(-1,1).to(self.device)

        self.net_v = DNN([1, 10, 10, 1]).to(self.device)
        self.net_E = E_Net([1, 10, 10, 1]).to(self.device)

        self.loss_fn = nn.MSELoss()
        self.epochs = adam_epochs
        self.learning_rate = learning_rate
        self.loss_threshold = loss_threshold
        self.loss_history = []

    def loss_function(self, w_measured, w_char, q_norm=1.0, lambda_reg=1e-6):
        v = self.net_v(self.x)
        E = self.net_E(self.x)

        d1 = torch.autograd.grad(v, self.x, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        d2 = torch.autograd.grad(d1, self.x, grad_outputs=torch.ones_like(d1), retain_graph=True, create_graph=True)[0]

        # Residual of the ODE: d/dz ( E(z) * d²v/dz² ) = q_norm
        inner = E * d2
        d_inner = torch.autograd.grad(inner, self.x, grad_outputs=torch.ones_like(inner), create_graph=True)[0]
        residual = d_inner - q_norm

        # Loss 1: ODE residual
        loss_phys = self.loss_fn(residual, torch.zeros_like(residual))

        # Loss 2: Data loss
        w_predicted = v * w_char
        loss_data = self.loss_fn(w_predicted, w_measured)

        # Loss 3: Regularization of E(z)
        l2_reg = torch.mean(E**2)

        total_loss = loss_phys + 1.0 * loss_data + lambda_reg * l2_reg
        return total_loss

    def train(self, w_measured, w_char):
        optimizer = torch.optim.Adam(list(self.net_v.parameters()) + list(self.net_E.parameters()), lr=self.learning_rate)

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            loss = self.loss_function(w_measured, w_char)
            loss.backward()
            optimizer.step()

            self.loss_history.append(loss.item())
            if epoch % 500 == 0:
                print(f"Epoch {epoch} | Loss: {loss.item():.3e}")

            if loss.item() < self.loss_threshold:
                print(f"Training stopped at Epoch {epoch} | Loss: {loss.item():.3e}")
                break

    def get_results(self, w_char):
        with torch.no_grad():
            z_eval = self.x.cpu().numpy().flatten()
            w_pred = (self.net_v(self.x) * w_char).cpu().numpy().flatten()
            E_pred = self.net_E(self.x).cpu().numpy().flatten()
        return z_eval, w_pred, E_pred, self.loss_history

