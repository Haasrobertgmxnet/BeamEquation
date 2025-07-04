import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


class FEM_Solver:
    """
    Finite Element Method (FEM) solver for a 1D beam problem under uniform loading.
    Solves the deflection and rotation using beam theory with Hermite elements.
    """

    def __init__(self, n_elems: int) -> None:
        """
        Initialize the FEM solver.

        Args:
            n_elems (int): Number of finite elements (number of nodes = n_elems + 1).
        """
        self.n_elems: int = n_elems
        self.n_nodes: int = self.n_elems + 1
        self.x: np.ndarray = np.linspace(0, 1, self.n_nodes)
        self.element_stiffness: np.ndarray = np.array([
            [12, 6, -12, 6],
            [6, 4, -6, 2],
            [-12, -6, 12, -6],
            [6, 2, -6, 4]
        ], dtype=float)
        self.element_load: np.ndarray = 0.5 * np.array([1, 1/6, 1, -1/6], dtype=float)

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assemble and solve the global FEM system for displacements and rotations.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - v: Vertical deflection at each node, normalized.
                - theta: Rotation (slope) at each node, normalized.
        """
        K: np.ndarray = np.zeros((2 * self.n_nodes, 2 * self.n_nodes))
        f: np.ndarray = np.zeros(2 * self.n_nodes)

        for e in range(self.n_elems):
            ke = self.element_stiffness
            fe = self.element_load
            dofs = np.array([2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3])
            K[np.ix_(dofs, dofs)] += ke
            f[dofs] += fe

        # Apply boundary conditions: clamped at x=0 (displacement and rotation)
        fixed_dofs = [0, 1]
        free_dofs = np.setdiff1d(np.arange(2 * self.n_nodes), fixed_dofs)

        K_ff = K[np.ix_(free_dofs, free_dofs)]
        f_f = f[free_dofs]

        u_f = np.linalg.solve(K_ff, f_f)

        # Construct the full displacement vector
        u = np.zeros(2 * self.n_nodes)
        u[free_dofs] = u_f

        # Extract vertical displacements and rotations
        v: np.ndarray = u[0::2] / self.n_elems**4      # Normalize displacement
        theta: np.ndarray = u[1::2] / self.n_elems**3  # Normalize rotations

        return v, theta


def calculate_fem(n_nodes: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the FEM solver and return displacements and rotations.

    Args:
        n_nodes (int, optional): Number of nodes (default is 100).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - v: Vertical deflections.
            - theta: Rotations.
    """
    n_elem: int = n_nodes - 1
    fem_solver = FEM_Solver(n_elem)
    return fem_solver.solve()
