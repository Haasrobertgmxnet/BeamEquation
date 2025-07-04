# Numerical Study of the 1D Euler-Bernoulli Equation Using PINNs

This repository presents a numerical study of the 1D Euler-Bernoulli equation for a deflected beam. The Euler-Bernoulli equation is explored in various forms, which are then used in subsequent calculations.

## Overview

Physics-Informed Neural Networks (PINNs) have been applied to solve the Euler-Bernoulli equation. The study consists of two main parts:

1. **Direct Solution via PINN**  
   The equation is solved directly using a PINN. The solution is compared with a classical Finite Element (FE) solution. As expected, due to the high optimization of FE solvers for traditional linear engineering problems, the FE approach significantly outperforms the PINN in this case.

2. **Elastic Modulus Estimation from Deflection Data**  
   Simulated deflection measurements are used to approximate a solution that enables the estimation of the unknown elastic modulus. Two PINNs are employed for this data-driven task. With careful tuning of the calculation parameters, highly satisfactory results can be achieved.

## Key Concepts

- **Euler-Bernoulli Beam Theory**
- **Physics-Informed Neural Networks (PINNs)**
- **Finite Element Method (FEM)**
- **Data-Driven Inverse Problems**

## Conclusion

This study demonstrates both the potential and limitations of PINNs in solving differential equations and inverse problems in structural mechanics. While traditional FE methods currently outperform PINNs in direct solution tasks, PINNs prove to be powerful tools in data-driven inverse applications such as parameter estimation.

---

Feel free to explore the code, contribute, or raise issues.
