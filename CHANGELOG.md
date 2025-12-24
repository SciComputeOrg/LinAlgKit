# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Comprehensive API documentation with detailed examples (`docs/api.md`)
- `FUTURE_UPDATES.md` roadmap document
- `__version__` attribute for programmatic version checking

### Changed
- Improved `pyproject.toml` metadata with keywords and classifiers for better PyPI discoverability
- Updated release workflow to use token-based authentication

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
