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
]
