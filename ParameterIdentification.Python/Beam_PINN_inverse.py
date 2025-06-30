import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class PINN(nn.Module):
    def __init__(self, layers, activation=nn.Tanh):
        super(PINN, self).__init__()
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
    def __init__(self, epochs_adam, epochs_lbfgs, n_points=100, loss_threshold=1e-6, learning_rate=0.1):

        self.device = torch.device('cpu')
        self.z = torch.linspace(0, 1, n_points, requires_grad=True).reshape(-1,1).to(self.device)

        self.net_v = PINN([1, 20, 20, 20, 1]).to(self.device)
        self.net_e = E_Net([1, 35, 35, 35, 1]).to(self.device)

        self.loss_fn = nn.MSELoss()
        self.epochs_adam = epochs_adam
        self.epochs_lbfgs = epochs_lbfgs
        self.learning_rate = learning_rate
        self.loss_threshold = loss_threshold
        self.loss_history = []


    def simple_loss(self, v_measured, lambda_reg = 0):
        v = self.net_v(self.z)
        e = self.net_e(self.z)

        # Data loss
        loss_data = self.loss_fn(v, v_measured)

        # Calculation of the derivatives of v(z) wrt z
        d1 = torch.autograd.grad(v, self.z, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        d2 = torch.autograd.grad(d1, self.z, grad_outputs=torch.ones_like(d1), create_graph=True)[0]

        inner = e*d2
        d1_inner = torch.autograd.grad(inner, self.z, grad_outputs=torch.ones_like(inner), retain_graph=True, create_graph=True)[0]
        d2_inner = torch.autograd.grad(d1_inner, self.z, grad_outputs=torch.ones_like(d1_inner), create_graph=True)[0]

        residual = d2_inner - 1.0
        loss_phyiscs = self.loss_fn(residual, torch.zeros_like(residual))

        # Boundary conditions (all of dimensionless form z in [0,1])
        z0 = torch.zeros(1,1, requires_grad=True)
        z1 = torch.ones(1,1, requires_grad=True)

        v0 = self.net_v(z0)
        dv0 = torch.autograd.grad(v0, z0, grad_outputs=torch.ones_like(v0), create_graph=True)[0]

        v1 = self.net_v(z1)
        dv1 = torch.autograd.grad(v1, z1, grad_outputs=torch.ones_like(v1), create_graph=True)[0]
        ddv1 = torch.autograd.grad(dv1, z1, grad_outputs=torch.ones_like(dv1), create_graph=True)[0]
        dddv1 = torch.autograd.grad(ddv1, z1, grad_outputs=torch.ones_like(ddv1), create_graph=True)[0]

        loss_bc = self.loss_fn(v0, torch.zeros_like(v0)) + \
                  self.loss_fn(dv0, torch.zeros_like(dv0)) + \
                  self.loss_fn(ddv1, torch.zeros_like(ddv1)) + \
                  self.loss_fn(dddv1, torch.zeros_like(dddv1))

        # regularisation
        d1_e = torch.autograd.grad(e, self.z, grad_outputs=torch.ones_like(e), create_graph=True)[0]

        loss_d1_e = self.loss_fn(d1_e, torch.zeros_like(d1_e))

        # L2-Regularisation optional
        l2_reg = 0
        if lambda_reg > 0:
            l2_reg = lambda_reg * sum(torch.norm(param, 2)**2 for param in self.net_e.parameters())

        return 100 * loss_data + 1.5 * loss_phyiscs + 2 * loss_bc + 1e-5 * loss_d1_e + l2_reg

    def loss_function(self, v_measured, lambda_reg = 1e-9):
        v = self.net_v(self.z)
        e = self.net_e(self.z)

        # Calculation of the 4th derivative of v(z) wrt z
        d1 = torch.autograd.grad(v, self.z, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        d2 = torch.autograd.grad(d1, self.z, grad_outputs=torch.ones_like(d1), create_graph=True)[0]

        inner = e*d2
        d1_inner = torch.autograd.grad(inner, self.z, grad_outputs=torch.ones_like(inner), retain_graph=True, create_graph=True)[0]
        d2_inner = torch.autograd.grad(d1_inner, self.z, grad_outputs=torch.ones_like(d1_inner), create_graph=True)[0]

        d1_e = torch.autograd.grad(e, self.z, grad_outputs=torch.ones_like(inner), retain_graph=True, create_graph=True)[0]
        # Residual of the dimensionless ODE d4v/dz4 = 1
        residual = d2_inner - 1.0
        loss_residual = self.loss_fn(residual, torch.zeros_like(residual))

        # Boundary conditions (all of dimensionless form z in [0,1])
        z0 = torch.zeros(1,1, requires_grad=True)
        z1 = torch.ones(1,1, requires_grad=True)

        v0 = self.net_v(z0)
        dv0 = torch.autograd.grad(v0, z0, grad_outputs=torch.ones_like(v0), create_graph=True)[0]

        v1 = self.net_v(z1)
        dv1 = torch.autograd.grad(v1, z1, grad_outputs=torch.ones_like(v1), create_graph=True)[0]
        ddv1 = torch.autograd.grad(dv1, z1, grad_outputs=torch.ones_like(dv1), create_graph=True)[0]
        dddv1 = torch.autograd.grad(ddv1, z1, grad_outputs=torch.ones_like(ddv1), create_graph=True)[0]

        bc_loss = self.loss_fn(v0, torch.zeros_like(v0)) + \
                  self.loss_fn(dv0, torch.zeros_like(dv0)) + \
                  self.loss_fn(ddv1, torch.zeros_like(ddv1)) + \
                  self.loss_fn(dddv1, torch.zeros_like(dddv1))

        # Loss 3: Regularization of e(z)
        l2_reg_d1_e = torch.mean(d1_e**2)
        l2_reg_weights = sum(torch.norm(param, 2)**2 for param in self.net_e.parameters())

        loss_data = self.loss_fn(v, v_measured)

        total_loss = loss_data + 1e-2*bc_loss + 1e-2*loss_residual + 0.0 * (10.0*l2_reg_d1_e * l2_reg_weights)
        return total_loss

        # Calculation of the 4th derivative of v(z) wrt z
        d1 = torch.autograd.grad(v, self.z, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        d2 = torch.autograd.grad(d1, self.z, grad_outputs=torch.ones_like(d1), create_graph=True)[0]
        d3 = torch.autograd.grad(d2, self.z, grad_outputs=torch.ones_like(d2), create_graph=True)[0]
        d4 = torch.autograd.grad(d3, self.z, grad_outputs=torch.ones_like(d3), create_graph=True)[0]

        # Residual of the dimensionless ODE d4v/dz4 = 1
        residual = d4 - 1.0
        loss_residual = self.loss_fn(residual, torch.zeros_like(residual))

        # Boundary conditions (all of dimensionless form z in [0,1])
        z0 = torch.zeros(1,1, requires_grad=True)
        z1 = torch.ones(1,1, requires_grad=True)

        v0 = self.net_v(z0)
        dv0 = torch.autograd.grad(v0, z0, grad_outputs=torch.ones_like(v0), create_graph=True)[0]

        v1 = self.net_v(z1)
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
            l2_reg = lambda_reg * sum(torch.norm(param, 2)**2 for param in self.net_v.parameters())

        total_loss = loss_residual + bc_loss + l2_reg
        return total_loss, v, d1

    def loss_function_(self, v_measured, lambda_reg=1e-5):
        v = self.net_v(self.z)
        e = self.net_e(self.z)

        # Calculation of the 4th derivative of v(z) wrt z
        d1 = torch.autograd.grad(v, self.z, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        d2 = torch.autograd.grad(d1, self.z, grad_outputs=torch.ones_like(d1), create_graph=True)[0]
        d3 = torch.autograd.grad(d2, self.z, grad_outputs=torch.ones_like(d2), create_graph=True)[0]
        d4 = torch.autograd.grad(d3, self.z, grad_outputs=torch.ones_like(d3), create_graph=True)[0]

        # Residual of the dimensionless ODE d4v/dz4 = 1
        residual = d4 - 1.0
        loss_residual = self.loss_fn(residual, torch.zeros_like(residual))

        d1 = torch.autograd.grad(v, self.z, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        d2 = torch.autograd.grad(d1, self.z, grad_outputs=torch.ones_like(d1), retain_graph=True, create_graph=True)[0]

        # Residual of the ODE: 
        # inner = e * d2
        inner = d2
        d1_inner = torch.autograd.grad(inner, self.z, grad_outputs=torch.ones_like(inner), retain_graph=True, create_graph=True)[0]
        d2_inner = torch.autograd.grad(d1_inner, self.z, grad_outputs=torch.ones_like(d1_inner), create_graph=True)[0]

        # 4. Residuum:
        # residual = d2_inner - 1.0

        # Loss 1: ODE residual
        loss_phys = self.loss_fn(residual, torch.zeros_like(residual))

        # Loss 2: Data loss
        loss_data = self.loss_fn(v, v_measured)

        # Loss 3: Regularization of e(z)
        l2_reg = torch.mean(e**2)

        total_loss = loss_phys + loss_data + lambda_reg * l2_reg
        return total_loss

    def train(self, v_measured, lambda_reg = 0):
        optimizer = torch.optim.Adam(list(self.net_v.parameters()) + list(self.net_e.parameters()), lr=self.learning_rate)

        for epoch in range(self.epochs_adam):
            optimizer.zero_grad()
            # loss= self.loss_function(v_measured)
            loss = self.simple_loss(v_measured, lambda_reg)
            loss.backward()
            optimizer.step()

            self.loss_history.append(loss.item())
            if epoch % 500 == 0:
                print(f"Epoch {epoch} | Loss: {loss.item():.3e}")

            if loss.item() < self.loss_threshold:
                print(f"Training stopped at Epoch {epoch} | Loss: {loss.item():.3e}")
                break

        optimizer_lbfgs = torch.optim.LBFGS(self.net_v.parameters(), max_iter= self.epochs_adam, line_search_fn="strong_wolfe")

        def closure():
            optimizer_lbfgs.zero_grad()
            loss = self.simple_loss(v_measured, lambda_reg = 0)
            loss.backward()
            return loss

        print("Start LBFGS ...")
        optimizer_lbfgs.step(closure)
        self.final_loss_lbfgs = self.simple_loss(v_measured, lambda_reg = 0)

    def get_results(self):
        with torch.no_grad():
            z_eval = self.z.cpu().numpy().flatten()
            v_pred = self.net_v(self.z).cpu().numpy().flatten()
            e_pred = self.net_e(self.z).cpu().numpy().flatten()
        return z_eval, v_pred, e_pred, self.loss_history, self.final_loss_lbfgs

