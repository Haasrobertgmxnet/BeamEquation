import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple


class PINN(nn.Module):
    """
    Physics-Informed Neural Network for approximating v(z).

    Parameters
    ----------
    layers : List[int]
        List of neuron counts per layer including input and output.
    activation : type, optional
        Activation function to use (default: nn.Tanh).
    """

    def __init__(self, layers: List[int], activation: type = nn.Tanh):
        super(PINN, self).__init__()
        layer_list = []
        for i in range(len(layers) - 1):
            layer_list.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                layer_list.append(activation())
        self.model = nn.Sequential(*layer_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Network output.
        """
        return self.model(x)


class E_Net(nn.Module):
    """
    Neural network for approximating E(z), with Softplus to ensure positivity.

    Parameters
    ----------
    layers : List[int]
        Layer definition.
    activation : type, optional
        Activation function to use (default: nn.Tanh).
    """

    def __init__(self, layers: List[int], activation: type = nn.Tanh):
        super(E_Net, self).__init__()
        layer_list = []
        for i in range(len(layers) - 1):
            layer_list.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                layer_list.append(activation())
        layer_list.append(nn.Softplus())  # Ensure E(z) > 0
        self.model = nn.Sequential(*layer_list)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to predict E(z).

        Parameters
        ----------
        z : torch.Tensor
            Input position(s).

        Returns
        -------
        torch.Tensor
            Predicted E(z), always positive.
        """
        return self.model(z) + 1e-3


class PINN_Inverse:
    """
    Inverse problem solver using Physics-Informed Neural Networks (PINNs).

    Parameters
    ----------
    epochs_adam : int
        Number of Adam optimizer epochs.
    epochs_lbfgs : int
        Number of L-BFGS optimizer iterations.
    n_points : int, optional
        Number of collocation points (default: 100).
    loss_threshold : float, optional
        Loss value for early stopping (default: 1e-6).
    learning_rate : float, optional
        Learning rate for Adam optimizer (default: 0.1).
    """

    def __init__(
        self,
        epochs_adam: int,
        epochs_lbfgs: int,
        n_points: int = 100,
        loss_threshold: float = 1e-6,
        learning_rate: float = 0.1,
    ):
        self.device = torch.device("cpu")
        self.z = torch.linspace(0, 1, n_points, requires_grad=True).reshape(-1, 1).to(self.device)

        self.net_v = PINN([1, 20, 20, 20, 1]).to(self.device)
        self.net_e = E_Net([1, 35, 35, 35, 1]).to(self.device)

        self.loss_fn = nn.MSELoss()
        self.epochs_adam = epochs_adam
        self.epochs_lbfgs = epochs_lbfgs
        self.learning_rate = learning_rate
        self.loss_threshold = loss_threshold
        self.loss_history: List[float] = []

    def simple_loss(self, v_measured: torch.Tensor, lambda_reg: float = 0) -> torch.Tensor:
        """
        Computes the total loss combining data, physics, boundary, and regularization losses.

        Parameters
        ----------
        v_measured : torch.Tensor
            Measured displacement values v(z).
        lambda_reg : float, optional
            L2 regularization coefficient (default: 0).

        Returns
        -------
        torch.Tensor
            Total loss.
        """
        v = self.net_v(self.z)
        e = self.net_e(self.z)

        # Data loss
        loss_data = self.loss_fn(v, v_measured)

        # Physics loss: compute fourth derivative of E(z) * d²v/dz²
        d1 = torch.autograd.grad(v, self.z, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        d2 = torch.autograd.grad(d1, self.z, grad_outputs=torch.ones_like(d1), create_graph=True)[0]

        inner = e * d2
        d1_inner = torch.autograd.grad(inner, self.z, grad_outputs=torch.ones_like(inner), retain_graph=True, create_graph=True)[0]
        d2_inner = torch.autograd.grad(d1_inner, self.z, grad_outputs=torch.ones_like(d1_inner), create_graph=True)[0]

        residual = d2_inner - 1.0
        loss_physics = self.loss_fn(residual, torch.zeros_like(residual))

        # Boundary condition loss
        z0 = torch.zeros(1, 1, requires_grad=True)
        z1 = torch.ones(1, 1, requires_grad=True)

        v0 = self.net_v(z0)
        dv0 = torch.autograd.grad(v0, z0, grad_outputs=torch.ones_like(v0), create_graph=True)[0]

        v1 = self.net_v(z1)
        dv1 = torch.autograd.grad(v1, z1, grad_outputs=torch.ones_like(v1), create_graph=True)[0]
        ddv1 = torch.autograd.grad(dv1, z1, grad_outputs=torch.ones_like(dv1), create_graph=True)[0]
        dddv1 = torch.autograd.grad(ddv1, z1, grad_outputs=torch.ones_like(ddv1), create_graph=True)[0]

        loss_bc = (
            self.loss_fn(v0, torch.zeros_like(v0)) +
            self.loss_fn(dv0, torch.zeros_like(dv0)) +
            self.loss_fn(ddv1, torch.zeros_like(ddv1)) +
            self.loss_fn(dddv1, torch.zeros_like(dddv1))
        )

        # Regularization: first derivative of E(z)
        d1_e = torch.autograd.grad(e, self.z, grad_outputs=torch.ones_like(e), create_graph=True)[0]
        loss_d1_e = self.loss_fn(d1_e, torch.zeros_like(d1_e))

        # Optional L2 regularization
        l2_reg = 0.0
        if lambda_reg > 0:
            l2_reg = lambda_reg * sum(torch.norm(param, 2)**2 for param in self.net_e.parameters())

        # Save for inspection
        self.loss_data = loss_data
        self.loss_bc = loss_bc
        self.loss_physics = loss_physics
        self.d1_e = d1_e

        return 100 * loss_data + 1.5 * loss_physics + 2 * loss_bc + 1e-5 * loss_d1_e + l2_reg

    def train(self, v_measured: torch.Tensor, lambda_reg: float = 0) -> None:
        """
        Trains the model using Adam followed by L-BFGS.

        Parameters
        ----------
        v_measured : torch.Tensor
            Measured displacement values.
        lambda_reg : float, optional
            L2 regularization coefficient (default: 0).
        """
        optimizer = torch.optim.Adam(
            list(self.net_v.parameters()) + list(self.net_e.parameters()), 
            lr=self.learning_rate
        )

        for epoch in range(self.epochs_adam):
            optimizer.zero_grad()
            loss = self.simple_loss(v_measured, lambda_reg)
            loss.backward()
            optimizer.step()

            self.loss_history.append(loss.item())
            if epoch % 500 == 0:
                print(f"Epoch {epoch} | Loss: {loss.item():.3e}")
            if loss.item() < self.loss_threshold:
                print(f"Early stopping at epoch {epoch} | Loss: {loss.item():.3e}")
                break

        # L-BFGS optimizer (only on net_v parameters here)
        optimizer_lbfgs = torch.optim.LBFGS(
            self.net_v.parameters(), 
            max_iter=self.epochs_lbfgs, 
            line_search_fn="strong_wolfe"
        )

        def closure() -> torch.Tensor:
            optimizer_lbfgs.zero_grad()
            loss = self.simple_loss(v_measured, lambda_reg=0)
            loss.backward()
            return loss

        print("Starting L-BFGS optimization ...")
        optimizer_lbfgs.step(closure)
        self.final_loss_lbfgs = self.simple_loss(v_measured, lambda_reg)

    def get_results(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the predicted values for z, v(z), and E(z).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Discretized z values, predicted displacement v(z), and predicted modulus E(z).
        """
        with torch.no_grad():
            z_eval = self.z.cpu().numpy().flatten()
            v_pred = self.net_v(self.z).cpu().numpy().flatten()
            e_pred = self.net_e(self.z).cpu().numpy().flatten()
        return z_eval, v_pred, e_pred
