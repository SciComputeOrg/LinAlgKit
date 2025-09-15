# LinAlgKit Documentation

Welcome to the LinAlgKit docs. This project provides a simple, Python-first linear algebra API built on NumPy.

- Python package: `LinAlgKit` in `python_pkg/LinAlgKit/`
- Pure Python packaging with `setuptools`

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

## Python API Overview

- `LinAlgKit.Matrix`, `LinAlgKit.MatrixF`, `LinAlgKit.MatrixI`
- Operations: `+`, `-`, `*` (matrix-matrix, scalar), `transpose()`, `trace()`, `determinant()`
- Constructors: `identity(n)`, `zeros(r, c)`, `ones(r, c)`

## Performance Notes

- Uses NumPy under the hood for core routines.
- Determinant and operations leverage `numpy.linalg` where applicable.

## Benchmarks

See `benchmarks/benchmark_matrix.cpp`. Run after building with benchmarks enabled:

```bash
./bin/benchmarks/matrix_benchmarks
```

## Releasing

See `docs/releasing.md`.
