# LinAlgKit

[![Docs](https://img.shields.io/badge/docs-LinAlgKit%20Site-brightgreen)](https://SciComputeOrg.github.io/LinAlgKit/)
[![CI](https://github.com/SciComputeOrg/LinAlgKit/actions/workflows/ci.yml/badge.svg)](https://github.com/SciComputeOrg/LinAlgKit/actions/workflows/ci.yml)
[![Wheels](https://github.com/SciComputeOrg/LinAlgKit/actions/workflows/release.yml/badge.svg)](https://github.com/SciComputeOrg/LinAlgKit/actions/workflows/release.yml)
[![PyPI](https://img.shields.io/pypi/v/LinAlgKit.svg)](https://pypi.org/project/LinAlgKit/)

![LinAlgKit Banner](https://dummyimage.com/1200x280/0e1116/ffffff&text=LinAlgKit%20%E2%80%94%20Lightweight%20Linear%20Algebra%20for%20Python)

LinAlgKit is a lightweight, NumPy-powered linear algebra toolkit for Python. It offers a minimal, clean API for matrices with scientific computing essentials: construction, arithmetic, transpose, trace, and determinant — all with first-class NumPy interoperability.

## Features

- Supports multiple numeric dtypes: int, float32, float64
- Clean matrix API: `+`, `-`, `*` (matrix and scalar), `transpose()`, `trace()`, `determinant()`
- Constructors: `identity(n)`, `zeros(r, c)`, `ones(r, c)`
- NumPy interop: `.from_numpy(ndarray)`, `.to_numpy()`
- Pure Python packaging — quick `pip install` on any platform

## Installation

```bash
pip install -U pip
pip install LinAlgKit
```

Editable install for development:

```bash
pip install -U pip
pip install -e .
```

## Quickstart

```python
import numpy as np
import LinAlgKit as lk

# Construct from NumPy
A = lk.Matrix.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]]))
B = lk.Matrix.identity(2)

# Core ops
C = A + B
AT = A.transpose()
detA = A.determinant()

print("C =\n", C.to_numpy())
print("AT =\n", AT.to_numpy())
print("det(A) =", detA)
```

## Python API overview

- `Matrix`, `MatrixF`, `MatrixI` classes with common operations
- Functional helpers: `array`, `zeros`, `ones`, `eye`, `matmul`, `transpose`, `trace`, `det`
- NumPy interop by design (copy in both directions for safety)

## Design Philosophy

- Vectorization-first: prefer NumPy operations and shapes that compose well.
- Minimal surface area: focus on the 80% of linear algebra tasks used daily.
- Explicit data flow: `.from_numpy()` and `.to_numpy()` are copy-based and clear.
- Pythonic ergonomics: a small, predictable API that reads like the math.
- Interop-ready: functions also accept NumPy arrays where sensible.

## Why LinAlgKit? (vs. raw NumPy)

- Matrix-first API with clear semantics (`Matrix`, `transpose()`, `determinant()`), helpful for pedagogy and readability.
- Convenience constructors (`identity`, `zeros`, `ones`) aligned with matrix mental models.
- Gentle learning curve for users coming from linear algebra courses before diving into broader NumPy idioms.
- Clean separation between object API and functional helpers so you can mix OO and vectorized styles.

## Testing

```bash
python -m pip install -U pytest
pytest -q
```

## NumPy interop

- `Matrix/MatrixF/MatrixI.from_numpy(array)` constructs from a 2D `ndarray` (copy)
- `.to_numpy()` returns a 2D `ndarray` (copy)
- For vectorized workflows, you can also use the functional API (`matmul`, `trace`, `det`) directly on NumPy arrays

## Examples

```python
from LinAlgKit import array, eye, matmul, det

A = array([[1., 2.], [3., 4.]])
I = eye(2)
print(det(A))         # -> -2.0
print(matmul(A, I))   # -> A
```

## Scientific notes

- Determinant, trace, and matmul are delegated to NumPy/`numpy.linalg` where applicable
- API emphasizes clarity and composability; best used together with NumPy idioms

## Benchmarks

For research-grade benchmarking, consider `asv` or simple scripts using `timeit` with NumPy arrays. A basic harness can be added in `scripts/` if needed.

## Roadmap

- Convenience APIs (slicing helpers, broadcasting-aware ops)
- Optional SciPy interop (sparse CSR constructors)
- Expanded tests and property-based testing
- Example notebooks and gallery in `docs/`

## Citation

If you use LinAlgKit in academic work, please cite this repository:

```bibtex
@software{linalgkit2025,
  author  = {SciComputeOrg},
  title   = {LinAlgKit: A Lightweight Linear Algebra Toolkit for Python},
  year    = {2025},
  url     = {https://github.com/SciComputeOrg/LinAlgKit}
}
```

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open issues and pull requests. For larger features (e.g., sparse matrices), open a design discussion first.
