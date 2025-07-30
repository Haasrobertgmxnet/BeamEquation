# BeamEquation: Physics-Informed Neural Network (PINN) for Euler-Bernoulli Beam

This repository implements a Physics-Informed Neural Network (PINN) to solve the Euler-Bernoulli beam equation. It combines machine learning (LibTorch/C++) with physical constraints to estimate the deflection of a beam on the domain \([0, 1]\).

## Contents

- `main.cpp`: Training and inference routine
- `Beam.h/cpp`: PINN model definition and loss logic
- `Global.h`: Global constants and options
- `Timer.h`: Runtime measurement helper
- `plot.py`: Python script to visualize model output
- `test_beam.cpp`: Unit tests with Catch2

---

## Getting Started

### Requirements

- Visual Studio 2022
- CMake 3.20+
- vcpkg for dependencies (see below)
- LibTorch (installed via vcpkg)

### Setup

1. **Clone Repository**

```bash
git clone https://github.com/yourusername/BeamEquation.git
```

2. **Install Dependencies**

```bash
# Optional: set VCPKG_ROOT if not global
git clone https://github.com/microsoft/vcpkg.git
./vcpkg/bootstrap-vcpkg.sh
./vcpkg/vcpkg install libtorch catch2
```

3. **Build Project**

```bash
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/Users/haasr/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build . --config Release
```

---

## Training

Run the executable (in Release mode):

```bash
./BeamEquation.exe > result.txt
```

The model will:

- Train using Adam (coarse fit)
- Refine with LBFGS
- Output deflection `u(x)` over [0, 1]

---

## Plotting Results

### Requirements

- Python 3.8+
- `matplotlib`

```bash
pip install matplotlib
```

### Usage

```bash
python plot.py result.txt
```

This script plots the predicted beam deflection over \([0,1]\).

---

## Testing

We use [Catch2](https://github.com/catchorg/Catch2) for unit testing.

### Build Tests

```bash
cmake .. -DBUILD_TESTING=ON
cmake --build . --config Debug
ctest
```

Or run directly:

```bash
./test_beam
```

---

## License

MIT License

---

## Acknowledgments

- LibTorch (PyTorch C++ API)
- Microsoft vcpkg
- Catch2 Testing Framework

