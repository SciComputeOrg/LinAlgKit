"""
LinAlgKit — fast C++ backend for Python via pybind11.

This package requires the compiled extension. Prebuilt wheels are provided for
common platforms. If the extension is missing, installation must build from
source (requires a C++17 compiler and CMake).
"""

try:
    from .matrixlib_py import Matrix, MatrixF, MatrixI  # type: ignore
except Exception as e:
    raise ImportError(
        "LinAlgKit requires the compiled C++ extension 'matrixlib_py'. "
        "Install prebuilt wheels from PyPI or ensure a C++17 compiler and CMake are available to build from source."
    ) from e

BACKEND = "compiled"

__all__ = [
    "Matrix",
    "MatrixF",
    "MatrixI",
    "BACKEND",
]
