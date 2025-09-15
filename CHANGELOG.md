# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-09-15
### Added
- Initial public release of LinAlgKit.
- C++ `Matrix<T>` with operations: add, sub, mul, transpose, trace, determinant.
- Optimized determinant using Bareiss fraction-free LU; retained naive recursive determinant for tests.
- PyBind11 Python bindings exposing `Matrix` (double), `MatrixF` (float), `MatrixI` (int).
- NumPy interop: `from_numpy` and `to_numpy` for all exposed types.
- Benchmarks: addition, multiplication, transpose, determinant (optimized vs naive).
- CI: C++ build/tests, Python smoke import.
- Release workflow: cibuildwheel for multi-platform wheels; auto-publish to TestPyPI/PyPI on tags.

