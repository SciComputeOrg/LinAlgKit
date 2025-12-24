"""
LinAlgKit — simple Python-first linear algebra API.

Pure-Python + NumPy implementation for easy installs and a clean API.
"""

from .pylib import (
    Matrix,
    MatrixF,
    MatrixI,
    array,
    zeros,
    ones,
    eye,
    matmul,
    transpose,
    trace,
    det,
)

__version__ = "0.1.0"
BACKEND = "python"

__all__ = [
    "Matrix",
    "MatrixF",
    "MatrixI",
    "array",
    "zeros",
    "ones",
    "eye",
    "matmul",
    "transpose",
    "trace",
    "det",
    "BACKEND",
    "__version__",
]
