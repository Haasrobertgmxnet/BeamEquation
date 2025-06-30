import numpy as np
import matplotlib.pyplot as plt

class FEM_Solver():
    def __init__(self, n_elems):
        self.n_elems = n_elems
        self.n_nodes = self.n_elems + 1
        self.x = np.linspace(0, 1, self.n_nodes)
        self.element_stiffness = np.array([
            [12, 6, -12, 6],
            [6, 4, -6, 2],
            [-12, -6, 12, -6],
            [6, 2, -6, 4]
        ])
        self.element_load = 1 / 2 * np.array([1, 1/6, 1, -1/6])

    def solve(self):
        # Gesamte Systemsteifigkeitsmatrix und Lastvektor aufbauen
        K = np.zeros((2*self.n_nodes, 2*self.n_nodes))
        f = np.zeros(2*self.n_nodes)

        for e in range(self.n_elems):
            ke = self.element_stiffness
            fe = self.element_load
    
            dofs = np.array([2*e, 2*e+1, 2*e+2, 2*e+3])
    
            K[np.ix_(dofs, dofs)] += ke
            f[dofs] += fe

        np.set_printoptions(precision=3, suppress=False)

        # Randbedingungen: Fest eingespannt bei x=0 Verschiebung und Rotation = 0
        fixed_dofs = [0, 1]
        free_dofs = np.setdiff1d(np.arange(2*self.n_nodes), fixed_dofs)

        # Solve the system of equations
        K_ff = K[np.ix_(free_dofs, free_dofs)]
        f_f = f[free_dofs]

        u_f = np.linalg.solve(K_ff, f_f)

        # Assemble overall solution
        u = np.zeros(2*self.n_nodes)
        u[free_dofs] = u_f

        # Ausgabe: Vertikale Verschiebung (jede zweite Komponente)
        v = u[0::2] / self.n_elems**4          # Vertikale Verschiebungen
        theta = u[1::2] / self.n_elems**3     # Rotationen (Biegewinkel in rad)

        return v, theta

def calculate_fem(n_nodes = 100): 
    n_elem = n_nodes - 1
    fem_solver = FEM_Solver(n_elem)
    return fem_solver.solve()
