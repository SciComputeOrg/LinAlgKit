"""
LinAlgKit Python package

This package provides Python bindings for the C++ matrix library via pybind11.

Classes:
- Matrix (double)
- MatrixF (float)
- MatrixI (int)
"""

# The compiled extension is named `matrixlib_py` and is built into this package
try:
    from .matrixlib_py import Matrix, MatrixF, MatrixI  # type: ignore
except Exception as e:
    raise ImportError(
        "Failed to import compiled extension 'matrixlib_py'. "
        "Make sure the package is built (e.g., `pip install -e .`)."
    ) from e
