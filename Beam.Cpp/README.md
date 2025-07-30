# PINN Euler-Bernoulli Beam Solver

A C++ implementation using PyTorch C++ API to solve the Euler-Bernoulli beam equation for deflection analysis using Physics-Informed Neural Networks (PINNs).

## Overview

This project implements a Physics-Informed Neural Network (PINN) approach to solve the fourth-order Euler-Bernoulli beam equation that governs the deflection of beams under various loading conditions. The implementation leverages PyTorch's C++ API for efficient computation and automatic differentiation.

The Euler-Bernoulli beam equation is given by:
```
∂⁴w/∂x⁴ = q(x)/(EI)
```
where:
- `w(x)` is the beam deflection
- `q(x)` is the distributed load
- `E` is Young's modulus
- `I` is the second moment of area

## Features

- **Physics-Informed Neural Networks**: Incorporates physical laws directly into the loss function
- **Fourth-order PDE solving**: Handles the complexity of the Euler-Bernoulli equation
- **Flexible boundary conditions**: Supports various beam configurations (simply supported, cantilever, fixed-fixed)
- **Multiple loading scenarios**: Handles distributed loads, point loads, and combinations
- **C++ performance**: Utilizes PyTorch C++ API for efficient computation
- **Automatic differentiation**: Leverages PyTorch's autograd for computing higher-order derivatives

## Requirements

### Dependencies
- **PyTorch C++** (LibTorch) >= 1.12.0
- **CMake** >= 3.14
- **C++17** compatible compiler
- **Eigen3** (optional, for matrix operations)

### System Requirements
- Linux/macOS/Windows
- CUDA support (optional, for GPU acceleration)

## Installation

### 1. Install LibTorch

Download the appropriate LibTorch distribution from [PyTorch website](https://pytorch.org/get-started/locally/):

```bash
# For CPU-only version
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip
unzip libtorch-cxx11-abi-shared-with-deps-latest.zip

# For CUDA version (adjust CUDA version as needed)
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-latest.zip
unzip libtorch-cxx11-abi-shared-with-deps-latest.zip
```

### 2. Clone and Build

```bash
git clone https://github.com/yourusername/pinn-euler-bernoulli-beam.git
cd pinn-euler-bernoulli-beam
mkdir build && cd build

# Configure with CMake (adjust LibTorch path)
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..

# Build the project
make -j4
```

## Usage

### Basic Example

```cpp
#include "pinn_beam_solver.h"

int main() {
    // Define beam parameters
    BeamParameters params;
    params.length = 1.0;          // Beam length (m)
    params.youngs_modulus = 200e9; // Young's modulus (Pa)
    params.moment_of_inertia = 1e-6; // Second moment of area (m^4)
    
    // Define loading
    auto uniform_load = [](double x) { return -1000.0; }; // N/m
    
    // Create and configure PINN solver
    PINNBeamSolver solver(params);
    solver.set_distributed_load(uniform_load);
    solver.set_boundary_conditions(BoundaryType::SIMPLY_SUPPORTED);
    
    // Train the network
    solver.train(epochs=5000, learning_rate=0.001);
    
    // Evaluate deflection at specific points
    std::vector<double> x_eval = {0.0, 0.25, 0.5, 0.75, 1.0};
    auto deflections = solver.evaluate_deflection(x_eval);
    
    // Output results
    for (size_t i = 0; i < x_eval.size(); ++i) {
        std::cout << "x = " << x_eval[i] << ", w = " << deflections[i] << std::endl;
    }
    
    return 0;
}
```

### Command Line Interface

```bash
# Run with default parameters
./pinn_beam_solver

# Specify custom parameters
./pinn_beam_solver --length 2.0 --load_magnitude 500 --boundary_type cantilever

# Load configuration from file
./pinn_beam_solver --config config/beam_config.json
```

## Network Architecture

The PINN implementation uses a fully connected neural network with the following default architecture:

- **Input Layer**: 1 neuron (position x)
- **Hidden Layers**: 4 layers with 50 neurons each
- **Activation Function**: Hyperbolic tangent (tanh)
- **Output Layer**: 1 neuron (deflection w)

The physics constraints are enforced through the loss function:

```
L_total = L_pde + λ_bc * L_boundary + λ_data * L_data
```

where:
- `L_pde`: Physics loss from the Euler-Bernoulli equation
- `L_boundary`: Boundary condition loss
- `L_data`: Data fitting loss (if available)

## Configuration

### Beam Parameters

```json
{
  "beam": {
    "length": 1.0,
    "youngs_modulus": 200e9,
    "moment_of_inertia": 1e-6,
    "density": 7850
  },
  "loading": {
    "type": "uniform",
    "magnitude": -1000.0
  },
  "boundary_conditions": {
    "type": "simply_supported"
  },
  "network": {
    "hidden_layers": 4,
    "neurons_per_layer": 50,
    "activation": "tanh"
  },
  "training": {
    "epochs": 5000,
    "learning_rate": 0.001,
    "batch_size": 256
  }
}
```

## Supported Boundary Conditions

1. **Simply Supported**: `w(0) = 0, w''(0) = 0, w(L) = 0, w''(L) = 0`
2. **Cantilever**: `w(0) = 0, w'(0) = 0`
3. **Fixed-Fixed**: `w(0) = 0, w'(0) = 0, w(L) = 0, w'(L) = 0`
4. **Free-Free**: No displacement constraints

## Validation

The solver includes validation against analytical solutions for:
- Uniformly loaded simply supported beam
- Point load on cantilever beam
- Multiple point loads on continuous beam

Run validation tests:
```bash
./test_validation
```

## Results Visualization

The project includes Python scripts for result visualization:

```bash
# Generate plots for deflection, moment, and shear
python scripts/plot_results.py --input results/beam_results.csv

# Compare with analytical solution
python scripts/compare_analytical.py --config config/validation.json
```

## Performance

Typical performance on modern hardware:
- **CPU (Intel i7-10700K)**: ~30 seconds for 5000 epochs
- **GPU (RTX 3080)**: ~8 seconds for 5000 epochs
- **Memory usage**: ~100MB for standard network size

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

### Development Guidelines

- Follow C++17 standards
- Use meaningful variable names
- Add unit tests for new features
- Document public APIs with Doxygen comments
- Ensure compatibility with both CPU and GPU execution

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.

2. Timoshenko, S. P., & Gere, J. M. (1972). *Mechanics of Materials*. Van Nostrand Reinhold Company.

3. Haghighat, E., Raissi, M., Moure, A., Gomez, H., & Juanes, R. (2021). A physics-informed deep learning framework for inversion and surrogate modeling in solid mechanics. *Computer Methods in Applied Mechanics and Engineering*, 379, 113741.

## Acknowledgments

- PyTorch team for the excellent C++ API
- Research community working on Physics-Informed Neural Networks
- Contributors to the structural mechanics and computational physics communities

## Contact

For questions or support, please open an issue on GitHub or contact [your-email@example.com].

---

**Note**: This is an academic/research implementation. For production structural analysis, please consult with qualified structural engineers and use validated commercial software.