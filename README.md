# Matrix Library

[![Docs](https://img.shields.io/badge/docs-LinAlgKit%20Site-brightgreen)](https://SciComputeOrg.github.io/LinAlgKit/)
[![CI](https://github.com/SciComputeOrg/LinAlgKit/actions/workflows/ci.yml/badge.svg)](https://github.com/SciComputeOrg/LinAlgKit/actions/workflows/ci.yml)
[![Wheels](https://github.com/SciComputeOrg/LinAlgKit/actions/workflows/release.yml/badge.svg)](https://github.com/SciComputeOrg/LinAlgKit/actions/workflows/release.yml)
[![PyPI](https://img.shields.io/pypi/v/LinAlgKit.svg)](https://pypi.org/project/LinAlgKit/)

A high-performance C++ matrix library with support for various matrix operations, including addition, subtraction, multiplication, transposition, determinant calculation, and more.

## Features

- Support for different numeric types (int, float, double)
- Common matrix operations (addition, subtraction, multiplication)
- Scalar operations
- Matrix transposition
- Determinant calculation
- Matrix inversion (for 2x2 matrices in this version)
- Identity, zeros, and ones matrix generation
- Google Test for unit testing
- Google Benchmark for performance testing
- CMake build system

## Requirements

- C++17 or later
- CMake 3.10 or later
- Git (for downloading dependencies)
- Google Test (automatically downloaded by CMake)
- Google Benchmark (automatically downloaded by CMake, optional for benchmarks)

## Building the Project

### Prerequisites

Make sure you have the following installed on your system:

- CMake (3.10 or later)
- A C++ compiler with C++17 support (GCC, Clang, or MSVC)
- Git
- Make or Ninja (for Unix-like systems) or Visual Studio (for Windows)

### Build Instructions

#### Linux/macOS

```bash
# Create a build directory
mkdir -p build && cd build

# Configure the project
cmake ..

# Build the project
cmake --build .

# Run tests
ctest --output-on-failure

# Run the example
./bin/examples/example

# Run benchmarks (if enabled)
./bin/benchmarks/matrix_benchmarks
```

#### Windows

```cmd
# Create a build directory
mkdir build
cd build

# Configure the project (using Visual Studio 2019 as an example)
cmake .. -G "Visual Studio 16 2019" -A x64

# Build the project
cmake --build . --config Release

# Run tests
ctest -C Release --output-on-failure

# Run the example
.\bin\Release\examples\example.exe

# Run benchmarks (if enabled)
.\bin\Release\benchmarks\matrix_benchmarks.exe
```

## Usage

Here's a quick example of how to use the matrix library:

```cpp
#include <iostream>
#include "matrixlib.h"

int main() {
    using namespace matrixlib;
    
    // Create matrices
    Matrixi a = {{1, 2}, {3, 4}};
    Matrixi b = {{5, 6}, {7, 8}};
    
    // Matrix operations
    auto sum = a + b;
    auto product = a * b;
    auto transposed = a.transpose();
    
    // Output results
    std::cout << "A + B =\n" << sum << "\n\n";
    std::cout << "A * B =\n" << product << "\n\n";
    std::cout << "A^T =\n" << transposed << "\n";
    
    return 0;
}
```

## Running Tests

Tests are built automatically when you build the project. You can run them using CTest:

```bash
# From the build directory
ctest --output-on-failure
```

## Running Benchmarks

Benchmarks are built when the `BUILD_BENCHMARKS` option is enabled (enabled by default). You can run them using:

```bash
# From the build directory
./bin/benchmarks/matrix_benchmarks
```

## Python Package: LinAlgKit

LinAlgKit is the Python package name for this library, built with pybind11 and scikit-build-core.

### Quick install (editable)

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-dev python3-pip build-essential cmake
pip install -U pip
pip install -U scikit-build-core pybind11 numpy

mkdir -p ~/linalgkit_build && cd ~/linalgkit_build
pip install -e /mnt/c/z_open_source/matrixlib
```

### Python usage examples

```python
import numpy as np
import LinAlgKit as lk

A = lk.Matrix.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]]))
print("A =\n", A.to_numpy())
print("det(A) =", A.determinant())
print("trace(A) =", A.trace())

B = lk.MatrixF.from_numpy(np.ones((3, 3), dtype=np.float32))
print("B rows, cols:", B.rows, B.cols)

C = lk.MatrixI.from_numpy(np.array([[2, 0], [0, 2]], dtype=np.int32))
print("C trace:", C.trace())
```

### NumPy interop

- `Matrix/MatrixF/MatrixI.from_numpy(ndarray)` creates a matrix from a 2D numpy array (copy).
- `.to_numpy()` returns a 2D numpy array (copy).
- Future: zero-copy views require contiguous internal storage redesign.

### WSL/Windows notes

- Prefer building in your WSL home directory (e.g., `~/linalgkit_build`) and point CMake to the Windows source directory to avoid locks on `/mnt/c`.
- Example:

```bash
mkdir -p ~/matrixlib_build && cd ~/matrixlib_build
cmake -G "Unix Makefiles" -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON -DPYTHON_EXECUTABLE=$(which python3) /mnt/c/z_open_source/matrixlib
make -j
```

## Determinant: optimized vs naive

- `determinant()` uses the Bareiss fraction-free LU algorithm (with partial pivoting), O(n^3).
- `determinant_naive()` keeps the recursive Laplace expansion for tiny matrices/testing.
- Inverse is implemented for 2x2; larger inverse will use LU in future.

## Benchmarks

Build with benchmarks enabled and run:

```bash
cd build
./bin/benchmarks/matrix_benchmarks
```

Benchmarks include:
- Addition, multiplication, transpose
- Determinant (optimized LU/Bareiss)
- Determinant (naive) — tiny sizes only

## Roadmap

- Python packaging polish (sdist/wheel, manylinux/musllinux wheels)
- Sparse matrices (CSR): construction (from COO), mat-vec, sparse-dense matmul, optional SciPy interop
- Blocked matrix multiplication for cache efficiency
- Extended tests and property-based testing
- Expanded benchmarks, profiling, and a research paper write-up
- GitHub CI: C++ (GTest), Python wheels, import tests

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open issues and pull requests. For larger features (e.g., sparse matrices), open a design discussion first.
