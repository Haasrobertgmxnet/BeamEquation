import torch
import torch.nn as nn
import numpy as np
import time

class NN_Solver():
    def __init__(self, net, epochs, n_points = 100, loss_threshold = 1e-6, learning_rate = 0.001):
        self.net = net
        self.epochs = epochs
        self.n_points = n_points
        self.loss_threshold = loss_threshold
        self.learning_rate = learning_rate
        self.x = torch.linspace(0, 1, self.n_points, requires_grad=True).reshape(-1,1)
        self.loss_fn = nn.MSELoss()
        self.loss_history = None

    def loss_function(self, lambda_reg):
        v = self.net(self.x)

        # Calculation of the 4th derivative of v(z) wrt z
        d1 = torch.autograd.grad(v, self.x, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        d2 = torch.autograd.grad(d1, self.x, grad_outputs=torch.ones_like(d1), create_graph=True)[0]
        d3 = torch.autograd.grad(d2, self.x, grad_outputs=torch.ones_like(d2), create_graph=True)[0]
        d4 = torch.autograd.grad(d3, self.x, grad_outputs=torch.ones_like(d3), create_graph=True)[0]

        # Residual of the dimensionless ODE d4v/dz4 = 1
        residual = d4 - 1.0
        loss_residual = self.loss_fn(residual, torch.zeros_like(residual))

        # Boundary conditions (all of dimensionless form z in [0,1])
        z0 = torch.zeros(1,1, requires_grad=True)
        z1 = torch.ones(1,1, requires_grad=True)

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
    
        # L2-Regularisation optional
        l2_reg = 0
        if lambda_reg > 0:
            l2_reg = lambda_reg * sum(torch.norm(param, 2)**2 for param in self.net.parameters())

        total_loss = loss_residual + bc_loss + l2_reg
        return total_loss

    def getNet(self):
        return self.net

    def get_training_results(self):
        # Evaluation
        z_eval = torch.linspace(0, 1, self.n_points).reshape(-1,1)
        with torch.no_grad():
            v_pred = np.array(self.net(z_eval).detach().numpy().flatten())
        return z_eval, v_pred, self.loss_history

    def predict(self, z):
        return self.net(z)


class Adam_Solver(NN_Solver):
    def __init__(self, net, epochs, n_points = 100, loss_threshold = 1e-6, learning_rate = 0.001):
        super(Adam_Solver, self).__init__(net, epochs, n_points, loss_threshold, learning_rate)

    def train(self, lambda_reg = 0.0):
        optimizer = torch.optim.Adam(self.net.parameters(), lr = self.learning_rate)
        self.loss_history = []
        
        for epoch in range(self.epochs):
            optimizer.zero_grad()

            loss = self.loss_function(lambda_reg)
            loss.backward()
            optimizer.step()

            self.loss_history.append(loss.item())
            if epoch % 500 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item():.6e}')

            # Early stopping
            if loss.item() < self.loss_threshold:
                print(f"Training interrupted at Epoch {epoch} mit Loss: {loss.item():.6e}")
                break
        self.final_loss_adam = loss.item()

class LBFGS_Solver(NN_Solver):
    def __init__(self, net, epochs, n_points = 100, loss_threshold = 1e-6, learning_rate = 1.0):
        super(LBFGS_Solver, self).__init__(net, epochs, n_points, loss_threshold, learning_rate)

    def train(self, lambda_reg = 0.0):
        optimizer_lbfgs = torch.optim.LBFGS(self.net.parameters(), max_iter= self.epochs, line_search_fn="strong_wolfe")

        def closure():
            optimizer_lbfgs.zero_grad()
            loss = self.loss_function(lambda_reg)
            loss.backward()
            return loss

        print("Start LBFGS ...")
        optimizer_lbfgs.step(closure)
        self.final_loss_lbfgs = self.loss_function(lambda_reg)

def calculate_pinn(adam_epochs, lbfgs_epochs, n_points = 100, loss_threshold = 1e-6, learning_rate = 0.1):
    # Neural net
    class PINN(nn.Module):
        def __init__(self, layers):
            super(PINN, self).__init__()
            layer_list = []
            for i in range(len(layers) - 1):
                linear_layer = nn.Linear(layers[i], layers[i+1])
                self.init_weights(linear_layer)  # Initialisierung
                layer_list.append(linear_layer)
                if i < len(layers) - 2:
                    layer_list.append(nn.Tanh())
            self.model = nn.Sequential(*layer_list)

        def init_weights(self, layer):
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

        def forward(self, x):
            return self.model(x)


    start = time.time()
    net = PINN(layers = [1, 5, 5, 1])
    adam_solver = Adam_Solver(net, adam_epochs, n_points, loss_threshold, learning_rate)
    adam_solver.train()
    ende = time.time()
    time_adam = ende - start
    
    adam_results = adam_solver.get_training_results()

    lbfgs_solver = None
    if adam_solver.final_loss_adam > loss_threshold:
        start = time.time()
        lbfgs_solver = LBFGS_Solver(net, lbfgs_epochs, n_points, loss_threshold, learning_rate*0.1)
        lbfgs_solver.train()
        ende = time.time()
        time_lbfgs = ende - start

    lbfgs_results = lbfgs_solver.get_training_results()
    return adam_results, time_adam, lbfgs_results, time_lbfgs


