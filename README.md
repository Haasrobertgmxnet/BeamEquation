# 1D Euler–Bernoulli Beam Deflection: PINN vs FEM 🚀

This repository contains a comprehensive numerical study of the 1D Euler–Bernoulli equation for a deflected beam, comparing Physics-Informed Neural Networks (PINNs) with classical Finite Element Method (FEM) approaches.

---

## 📘 Table of Contents

- [Overview](#overview)  
- [Repository Structure](#repository-structure)  
- [Key Concepts](#key-concepts)  
- [Usage](#usage)  
- [Installation](#installation)  
- [Results Summary](#results-summary)  
- [References](#references)  
- [License](#license)

---

## Overview

The Euler–Bernoulli beam equation is formulated in several equivalent forms and solved via:

1. **Direct PINN Solution**  
   - PINN solves the boundary-value problem.  
   - Results are benchmarked against a classical FEM solver.  
   - FEM, optimized for linear engineering systems, outperforms the PINN in speed and accuracy.

2. **Inverse PINN for Elastic Modulus Estimation**  
   - Synthetic deflection measurements are used.  
   - Two PINNs infer the unknown Young’s modulus by fitting deflection data.  
   - With careful hyperparameter tuning, the PINNs achieve highly accurate modulus estimates.

---

## Repository Structure

```
BeamEquation/
├── BeamBending/
│   ├── Documentation.pdf # Full article
├── Beam.Python                              # Python code for solving the beam equation with FE and PINN
│   ├── Beam.Python.py                       # Main Python file with the calculation pipeline
│   ├── Beam_FEM.py                          # Infrastructure and helper classes to support the FE calculation
│   ├── Beam_PINN.py                         # Infrastructure and helper classes to support the PINN calculation
├── Parameteridentification.Python           # Python code for estimating the elastic moduuls with PINN
│   ├── Parameteridentification.Python.py    # Main Python file with the calculation pipeline
│   ├── Beam_PINN_inverse.py                 # Infrastructure and helper classes to support the PINN calculation
└── README.md                                # (this file)
```

---

## Key Concepts

- **Euler–Bernoulli Beam Theory** (thin beam bending equations)  
- **Physics-Informed Neural Networks (PINNs)**  
- **Finite Element Method (FEM)**  
- **Inverse Parameter Estimation** (data-driven determination of material properties)

---

## Installation

Ensure you have Python 3.8+ and pip installed.

```bash
git clone https://github.com/Haasrobertgmxnet/BeamEquation.git
cd BeamEquation/BeamBending

# Optional: create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

---

## Usage

### 1. Direct PINN vs FEM comparison

```bash
py Beam.Python.py --mode direct
```

or

```bash
python Beam.Python.py --mode direct
```
This script trains the PINN, solves the beam deflection, and compares results to FEM.

### 2. Inverse PINN for Young’s Modulus Estimation

```bash
py Parameteridentification.Python.py --mode inverse
```

or

```bash
python Parameteridentification.Python.py --mode inverse
```

This runs the inverse PINN workflow, estimating Young’s modulus from synthetic deflection data.

### Jupyter Notebooks

There are no Notebooks yet ...

---

## Results Summary

- **Direct PINN vs FEM**: FEM achieves greater accuracy and computation speed in solving forward boundary-value problems.
- **Inverse PINN**: The data-driven PINN approach precisely recovers Young’s modulus when properly tuned, demonstrating effectiveness for parameter estimation.

For detailed metrics, plots, and discussion, see the complete article: [Documentation.pdf](BeamBending/Documentation.pdf).

---

## References

- Robert Haas, *Numerical Study of the 1D Euler–Bernoulli Equation Using PINNs*, 2025.  
  Full text: [Documentation.pdf](BeamBending/Documentation.pdf)  
- Github Repository: https://github.com/Haasrobertgmxnet/BeamEquation

---

## License

<!-- Distributed under the MIT License. See `LICENSE` for details. -->

---

## Questions or Support

For any issues or questions, please open an issue on the [GitHub repository](https://github.com/Haasrobertgmxnet/BeamEquation/issues) or contact the maintainers directly.

---

*Thank you for exploring this repository — we hope you find it insightful and valuable!*
