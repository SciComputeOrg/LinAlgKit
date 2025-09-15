# LinAlgKit Documentation

Welcome to the LinAlgKit docs. This project provides a C++ linear algebra core with Python bindings via pybind11.

- Source: `matrixlib/`
- Python package: `LinAlgKit` in `python_pkg/LinAlgKit/`
- Build system: CMake + scikit-build-core

## Contents

- Getting Started
- Python Usage
- C++ API Overview
- Performance Notes
- Benchmarks
- Releasing

## Getting Started

See the project `README.md` for quick install and build instructions. For Python usage with editable install:

```bash
pip install -U pip scikit-build-core pybind11 numpy
mkdir -p ~/linalgkit_build && cd ~/linalgkit_build
pip install -e /path/to/repo
```

## Python Usage

```python
import numpy as np
import LinAlgKit as lk

A = lk.Matrix.from_numpy(np.array([[1.0, 2.0],[3.0, 4.0]]))
print(A.determinant())
B = A.transpose()
print(B.to_numpy())
```

## C++ API Overview

- `matrixlib::Matrix<T>` for arithmetic types (`int`, `float`, `double`).
- Operations: `+`, `-`, `*` (matrix-matrix, scalar), `transpose()`, `trace()`, `determinant()`.
- Static constructors: `identity(n)`, `zeros(r, c)`, `ones(r, c)`.

See `include/matrixlib.h` for the full interface.

## Performance Notes

- Determinant uses Bareiss fraction-free LU (O(n^3)).
- Naive recursive determinant kept for tests/tiny sizes.
- Current storage uses `std::vector<std::vector<T>>` for simplicity. A contiguous layout would enable zero-copy NumPy views and faster BLAS-like ops.

## Benchmarks

See `benchmarks/benchmark_matrix.cpp`. Run after building with benchmarks enabled:

```bash
./bin/benchmarks/matrix_benchmarks
```

## Releasing

See `docs/releasing.md`.
