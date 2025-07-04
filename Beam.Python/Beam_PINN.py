import torch
import torch.nn as nn
import numpy as np
import time
from typing import Optional, Tuple, List


class NN_Solver:
    """
    Base class for training Physics-Informed Neural Networks (PINNs)
    to solve the dimensionless beam bending problem using the ODE d⁴v/dz⁴ = 1.
    """

    def __init__(self, net: nn.Module, epochs: int, n_points: int = 100,
                 loss_threshold: float = 1e-6, learning_rate: float = 0.001) -> None:
        """
        Initialize the solver with training parameters and problem setup.

        Args:
            net (nn.Module): Neural network model to train.
            epochs (int): Number of training epochs.
            n_points (int): Number of collocation points for residual evaluation.
            loss_threshold (float): Threshold for early stopping.
            learning_rate (float): Learning rate for the optimizer.
        """
        self.net = net
        self.epochs = epochs
        self.n_points = n_points
        self.loss_threshold = loss_threshold
        self.learning_rate = learning_rate
        self.x: torch.Tensor = torch.linspace(0, 1, self.n_points, requires_grad=True).reshape(-1, 1)
        self.loss_fn = nn.MSELoss()
        self.loss_history: Optional[List[float]] = None

    def loss_function(self, lambda_reg: float) -> torch.Tensor:
        """
        Compute total loss including PDE residual, boundary conditions and optional L2 regularization.

        Args:
            lambda_reg (float): Regularization strength.

        Returns:
            torch.Tensor: Total loss scalar.
        """
        v = self.net(self.x)

        # 4th derivative of v with respect to z
        d1 = torch.autograd.grad(v, self.x, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        d2 = torch.autograd.grad(d1, self.x, grad_outputs=torch.ones_like(d1), create_graph=True)[0]
        d3 = torch.autograd.grad(d2, self.x, grad_outputs=torch.ones_like(d2), create_graph=True)[0]
        d4 = torch.autograd.grad(d3, self.x, grad_outputs=torch.ones_like(d3), create_graph=True)[0]

        residual = d4 - 1.0
        loss_residual = self.loss_fn(residual, torch.zeros_like(residual))

        # Boundary conditions
        z0 = torch.zeros(1, 1, requires_grad=True)
        z1 = torch.ones(1, 1, requires_grad=True)

        v0 = self.net(z0)
        dv0 = torch.autograd.grad(v0, z0, grad_outputs=torch.ones_like(v0), create_graph=True)[0]

        v1 = self.net(z1)
        dv1 = torch.autograd.grad(v1, z1, grad_outputs=torch.ones_like(v1), create_graph=True)[0]
        ddv1 = torch.autograd.grad(dv1, z1, grad_outputs=torch.ones_like(dv1), create_graph=True)[0]
        dddv1 = torch.autograd.grad(ddv1, z1, grad_outputs=torch.ones_like(ddv1), create_graph=True)[0]

        bc_loss = self.loss_fn(v0, torch.zeros_like(v0)) + \
                  self.loss_fn(dv0, torch.zeros_like(dv0)) + \
                  self.loss_fn(ddv1, torch.zeros_like(ddv1)) + \
                  self.loss_fn(dddv1, torch.zeros_like(dddv1))

        l2_reg = lambda_reg * sum(torch.norm(param, 2)**2 for param in self.net.parameters()) if lambda_reg > 0 else 0

        total_loss = loss_residual + bc_loss + l2_reg
        return total_loss

    def getNet(self) -> nn.Module:
        """Return the trained neural network."""
        return self.net

    def get_training_results(self) -> Tuple[torch.Tensor, np.ndarray, Optional[List[float]]]:
        """
        Evaluate the trained model at collocation points.

        Returns:
            Tuple[torch.Tensor, np.ndarray, Optional[List[float]]]:
                - z_eval: Evaluation points (torch tensor).
                - v_pred: Predicted deflections (NumPy array).
                - loss_history: Training loss history.
        """
        z_eval = torch.linspace(0, 1, self.n_points).reshape(-1, 1)
        with torch.no_grad():
            v_pred = np.array(self.net(z_eval).detach().numpy().flatten())
        return z_eval, v_pred, self.loss_history

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        """
        Predict the deflection using the trained model.

        Args:
            z (torch.Tensor): Evaluation points.

        Returns:
            torch.Tensor: Predicted values.
        """
        return self.net(z)


class Adam_Solver(NN_Solver):
    """Solver using the Adam optimizer."""

    def __init__(self, net: nn.Module, epochs: int, n_points: int = 100,
                 loss_threshold: float = 1e-6, learning_rate: float = 0.001) -> None:
        super().__init__(net, epochs, n_points, loss_threshold, learning_rate)

    def train(self, lambda_reg: float = 0.0) -> None:
        """
        Train the network using the Adam optimizer.

        Args:
            lambda_reg (float): L2 regularization parameter.
        """
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.loss_history = []

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            loss = self.loss_function(lambda_reg)
            loss.backward()
            optimizer.step()

            self.loss_history.append(loss.item())
            if epoch % 500 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item():.6e}')

            if loss.item() < self.loss_threshold:
                print(f"Training interrupted at Epoch {epoch} with Loss: {loss.item():.6e}")
                break

        self.final_loss_adam: float = loss.item()


class LBFGS_Solver(NN_Solver):
    """Solver using the L-BFGS optimizer."""

    def __init__(self, net: nn.Module, epochs: int, n_points: int = 100,
                 loss_threshold: float = 1e-6, learning_rate: float = 1.0) -> None:
        super().__init__(net, epochs, n_points, loss_threshold, learning_rate)

    def train(self, lambda_reg: float = 0.0) -> None:
        """
        Train the network using the L-BFGS optimizer.

        Args:
            lambda_reg (float): L2 regularization parameter.
        """
        optimizer_lbfgs = torch.optim.LBFGS(
            self.net.parameters(), max_iter=self.epochs, line_search_fn="strong_wolfe"
        )

        def closure() -> torch.Tensor:
            optimizer_lbfgs.zero_grad()
            loss = self.loss_function(lambda_reg)
            loss.backward()
            return loss

        print("Start LBFGS ...")
        optimizer_lbfgs.step(closure)
        self.final_loss_lbfgs: torch.Tensor = self.loss_function(lambda_reg)


def calculate_pinn(adam_epochs: int, lbfgs_epochs: int, n_points: int = 100,
                   loss_threshold: float = 1e-6, learning_rate: float = 0.1
                   ) -> Tuple[
                       Tuple[torch.Tensor, np.ndarray, Optional[List[float]]],
                       float,
                       Tuple[torch.Tensor, np.ndarray, Optional[List[float]]],
                       float
                   ]:
    """
    Train and evaluate the PINN using Adam and optionally L-BFGS.

    Args:
        adam_epochs (int): Epochs for Adam optimizer.
        lbfgs_epochs (int): Epochs for LBFGS optimizer.
        n_points (int): Number of collocation points.
        loss_threshold (float): Threshold for early stopping.
        learning_rate (float): Initial learning rate.

    Returns:
        Tuple containing:
            - Adam results (z_eval, v_pred, loss_history)
            - Training time (Adam)
            - LBFGS results (z_eval, v_pred, loss_history)
            - Training time (LBFGS)
    """
    class PINN(nn.Module):
        def __init__(self, layers: List[int]) -> None:
            super().__init__()
            layer_list = []
            for i in range(len(layers) - 1):
                linear_layer = nn.Linear(layers[i], layers[i+1])
                self.init_weights(linear_layer)
                layer_list.append(linear_layer)
                if i < len(layers) - 2:
                    layer_list.append(nn.Tanh())
            self.model = nn.Sequential(*layer_list)

        def init_weights(self, layer: nn.Module) -> None:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x)

    net = PINN(layers=[1, 5, 5, 1])

    start = time.time()
    adam_solver = Adam_Solver(net, adam_epochs, n_points, loss_threshold, learning_rate)
    adam_solver.train()
    time_adam = time.time() - start

    adam_results = adam_solver.get_training_results()

    lbfgs_solver = None
    time_lbfgs = 0.0
    lbfgs_results = adam_results

    if adam_solver.final_loss_adam > loss_threshold:
        start = time.time()
        lbfgs_solver = LBFGS_Solver(net, lbfgs_epochs, n_points, loss_threshold, learning_rate * 0.1)
        lbfgs_solver.train()
        time_lbfgs = time.time() - start
        lbfgs_results = lbfgs_solver.get_training_results()

    return adam_results, time_adam, lbfgs_results, time_lbfgs
