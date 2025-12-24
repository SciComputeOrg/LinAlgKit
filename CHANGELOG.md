# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.1] - 2025-12-24

### Added

- **High-Performance `fast` Module**
  - Numba JIT-compiled activation functions: `fast_sigmoid`, `fast_relu`, `fast_leaky_relu`, `fast_elu`, `fast_gelu`, `fast_swish`, `fast_tanh`, `fast_softplus`
  - JIT-compiled loss functions: `fast_mse_loss`, `fast_mae_loss` (up to **13x faster**)
  - `fast_normalize` for L2 normalization
  - `HAS_NUMBA` flag to check if Numba is available

- **In-Place Matrix Operations**
  - `add_(other)` — In-place addition
  - `sub_(other)` — In-place subtraction
  - `mul_(scalar)` — In-place scalar multiplication
  - `hadamard_(other)` — In-place element-wise product

- **Zero-Copy Access**
  - `.T` property — Transpose view (no memory copy)
  - `to_numpy_view()` — Direct access to underlying array
  - `transpose(copy=False)` — Optional zero-copy transpose

- **Documentation**
  - `docs/releases.md` — Comprehensive release notes
  - `scripts/benchmark_fast.py` — Performance benchmark script

### Performance Improvements

| Function | Speedup |
|----------|---------|
| `mae_loss` | **13.1x** |
| `mse_loss` | **12.0x** |
| `leaky_relu` | **4.4x** |
| `gelu` | **2.6x** |
| `tanh` | **2.4x** |

### Dependencies

- Optional: `numba>=0.57.0` for JIT acceleration

---

## [0.2.0] - 2025-12-24

### Added

- **Matrix Decompositions**
  - `lu()` — LU decomposition with partial pivoting
  - `qr()` — QR decomposition using Householder reflections
  - `cholesky()` — Cholesky decomposition for positive-definite matrices
  - `svd()` — Singular Value Decomposition

- **Eigenvalue Methods**
  - `eig()` — Eigenvalues and eigenvectors
  - `eigvals()` — Eigenvalues only (faster)
  - `eigh()` — Symmetric/Hermitian eigenvalue decomposition

- **Linear System Solvers**
  - `solve()` — Solve linear system Ax = b
  - `inv()` — Matrix inverse
  - `pinv()` — Moore-Penrose pseudoinverse
  - `lstsq()` — Least-squares solution

- **Matrix Analysis**
  - `norm()` — Frobenius, 1, 2, inf norms
  - `cond()` — Condition number
  - `rank()` — Matrix rank

- **Activation Functions** (Deep Learning)
  - `sigmoid`, `relu`, `leaky_relu`, `elu`, `gelu`, `swish`
  - `softplus`, `tanh`, `softmax`, `log_softmax`
  - Derivative functions for backpropagation

- **Loss Functions**
  - `mse_loss`, `mae_loss`, `huber_loss`
  - `cross_entropy_loss`, `binary_cross_entropy`

- **Normalization**
  - `batch_norm`, `layer_norm`, `instance_norm`

- **Convolution Operations**
  - `conv2d` — 2D convolution
  - `max_pool2d`, `avg_pool2d`, `global_avg_pool2d`

- **Weight Initialization**
  - `xavier_uniform`, `xavier_normal`
  - `he_uniform`, `he_normal`

- **Utility Functions**
  - `dropout`, `one_hot`, `clip`, `flatten`, `reshape`
  - `normalize`, `cosine_similarity`, `euclidean_distance`
  - `pairwise_distances`, `numerical_gradient`
  - `outer`, `inner`, `dot`, `cross`

- **Documentation**
  - `docs/deep_learning.md` — Comprehensive deep learning functions guide
  - `EXPANSION_PLAN.md` — Full roadmap through v1.0.0
  - Updated API documentation with new methods

- **Tests**
  - 23 new tests for deep learning functions
  - 15 tests for matrix decompositions and solvers
  - Total: 41 tests passing

### Changed
- Version bumped to 0.2.0
- Updated `__init__.py` to export 50+ new functions

---

## [0.1.0] - 2025-12-24

### Added
- **Core Matrix Classes**
  - `Matrix` — Double-precision (float64) matrix class
  - `MatrixF` — Single-precision (float32) matrix class  
  - `MatrixI` — Integer matrix class
  
- **Static Constructors**
  - `Matrix.identity(size)` — Create identity matrix
  - `Matrix.zeros(rows, cols)` — Create zero-filled matrix
  - `Matrix.ones(rows, cols)` — Create ones-filled matrix

- **Matrix Operations**
  - `transpose()` — Matrix transposition
  - `trace()` — Sum of diagonal elements
  - `determinant()` — LU decomposition-based determinant (O(n³))
  - `determinant_naive()` — Recursive cofactor expansion for small matrices

- **Arithmetic Operators**
  - `+` — Element-wise matrix addition
  - `-` — Element-wise matrix subtraction
  - `*` — Matrix multiplication (matrix × matrix) or scalar multiplication
  - Scalar × Matrix left multiplication via `__rmul__`

- **Element Access**
  - `A[i, j]` — Get element at row i, column j
  - `A[i, j] = value` — Set element at row i, column j

- **NumPy Interoperability**
  - `Matrix.from_numpy(ndarray)` — Construct matrix from 2D NumPy array
  - `to_numpy()` — Convert matrix to 2D NumPy array
  - All conversions are copy-based for data safety

- **Functional API** (NumPy-compatible helpers)
  - `array(data)` — Create NumPy array from nested lists
  - `zeros(shape)` — Create zero-filled NumPy array
  - `ones(shape)` — Create ones-filled NumPy array
  - `eye(n)` — Create identity matrix as NumPy array
  - `matmul(a, b)` — Matrix multiplication on arrays
  - `transpose(a)` — Transpose NumPy array
  - `trace(a)` — Compute trace of NumPy array
  - `det(a)` — Compute determinant of NumPy array

- **Properties**
  - `rows` — Number of rows (read-only)
  - `cols` — Number of columns (read-only)
  - `BACKEND` — Backend identifier ("python")

- **Project Infrastructure**
  - Pure Python implementation with NumPy backend
  - CI/CD with GitHub Actions for testing on Linux, macOS, Windows
  - Automated PyPI publishing on tag push
  - MkDocs documentation setup
  - pytest test suite
  - Benchmark scripts in `scripts/`

### Fixed
- **Critical Bug**: Fixed `from_numpy()` class method that incorrectly passed numpy arrays to `__init__()`
- **Critical Bug**: Fixed arithmetic operators (`__add__`, `__sub__`, `__mul__`, `__rmul__`) that had the same issue
- **Critical Bug**: Fixed `transpose()` method to properly wrap result arrays
- Added `_wrap()` helper method for safe matrix construction from numpy arrays

### Technical Details
- Added `__version__ = "0.1.0"` for programmatic version access
- Package structure: `python_pkg/LinAlgKit/`
- Requires Python ≥ 3.8 and NumPy ≥ 1.22

---

## [0.0.1] - 2025-09-15 (Pre-release)

### Added
- Initial project structure
- C++ `Matrix<T>` template with pybind11 bindings (later replaced with pure Python)
- Basic matrix operations prototype
- CMake build system setup

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 0.1.0 | 2025-12-24 | First stable release, pure Python, all core features |
| 0.0.1 | 2025-09-15 | Initial development version |

---

## Upgrade Notes

### Upgrading to 0.1.0

No breaking changes from pre-release versions. If you were using the development version:

1. Update your installation:
   ```bash
   pip install --upgrade LinAlgKit
   ```

2. All existing code should work without modifications.

---

## Contributors

- SciComputeOrg Development Team
- Community Contributors

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute.
