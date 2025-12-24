"""
Pure-Python + NumPy backend for LinAlgKit.

This module provides a simple, Pythonic API that works out-of-the-box without
compilation. If the compiled extension is available, the package's __init__
will prefer it automatically; otherwise this backend is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Iterable, Union
import numpy as np

Number = Union[int, float]


@dataclass
class _BaseMatrix:
    _data: np.ndarray

    # --- construction helpers ---
    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> "_BaseMatrix":
        a = np.asarray(arr)
        if a.ndim != 2:
            raise ValueError("expected a 2D array")
        obj = object.__new__(cls)
        obj._data = a.copy()
        return obj

    @classmethod
    def _wrap(cls, arr: np.ndarray) -> "_BaseMatrix":
        """Wrap a numpy array in a Matrix instance without calling __init__."""
        obj = object.__new__(cls)
        obj._data = arr
        return obj

    # --- properties ---
    @property
    def rows(self) -> int:
        return int(self._data.shape[0])

    @property
    def cols(self) -> int:
        return int(self._data.shape[1])

    # --- conversion ---
    def to_numpy(self) -> np.ndarray:
        return self._data.copy()

    # --- ops ---
    def transpose(self) -> "_BaseMatrix":
        return self._wrap(self._data.T.copy())

    def trace(self) -> Number:
        return float(np.trace(self._data)) if self._data.dtype.kind == 'f' else int(np.trace(self._data))

    def determinant(self) -> Number:
        det = float(np.linalg.det(self._data))
        return det

    # naive determinant for tiny matrices (for parity with C++ API)
    def determinant_naive(self) -> Number:
        a = self._data
        n, m = a.shape
        if n != m:
            raise ValueError("determinant is only defined for square matrices")
        if n == 1:
            return float(a[0, 0])
        if n == 2:
            return float(a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0])
        det = 0.0
        for j in range(n):
            sub = np.delete(np.delete(a, 0, axis=0), j, axis=1)
            sign = 1.0 if (j % 2 == 0) else -1.0
            det += sign * float(a[0, j]) * Matrix.from_numpy(sub).determinant_naive()
        return det

    # arithmetic
    def __add__(self, other: "_BaseMatrix") -> "_BaseMatrix":
        return self._wrap(self._data + other._data)

    def __sub__(self, other: "_BaseMatrix") -> "_BaseMatrix":
        return self._wrap(self._data - other._data)

    def __mul__(self, other: Union["_BaseMatrix", Number]) -> "_BaseMatrix":
        if isinstance(other, _BaseMatrix):
            return self._wrap(self._data @ other._data)
        else:
            return self._wrap(self._data * other)

    def __rmul__(self, scalar: Number) -> "_BaseMatrix":
        return self._wrap(scalar * self._data)

    # indexing helpers (row, col)
    def __getitem__(self, idx: Tuple[int, int]):
        r, c = idx
        return self._data[r, c].item()

    def __setitem__(self, idx: Tuple[int, int], value: Number) -> None:
        r, c = idx
        self._data[r, c] = value


class Matrix(_BaseMatrix):
    def __init__(self, rows: int | None = None, cols: int | None = None, value: Number = 0.0):
        if rows is None and cols is None:
            super().__init__(np.zeros((0, 0), dtype=float))
        else:
            super().__init__(np.full((int(rows), int(cols)), float(value), dtype=float))

    # static constructors
    @staticmethod
    def identity(size: int) -> "Matrix":
        return Matrix.from_numpy(np.eye(int(size), dtype=float))  # type: ignore[arg-type]

    @staticmethod
    def zeros(rows: int, cols: int) -> "Matrix":
        return Matrix.from_numpy(np.zeros((int(rows), int(cols)), dtype=float))  # type: ignore[arg-type]

    @staticmethod
    def ones(rows: int, cols: int) -> "Matrix":
        return Matrix.from_numpy(np.ones((int(rows), int(cols)), dtype=float))  # type: ignore[arg-type]


class MatrixF(_BaseMatrix):
    def __init__(self, rows: int | None = None, cols: int | None = None, value: Number = 0.0):
        if rows is None and cols is None:
            super().__init__(np.zeros((0, 0), dtype=np.float32))
        else:
            super().__init__(np.full((int(rows), int(cols)), float(value), dtype=np.float32))

    @staticmethod
    def identity(size: int) -> "MatrixF":
        return MatrixF.from_numpy(np.eye(int(size), dtype=np.float32))  # type: ignore[arg-type]

    @staticmethod
    def zeros(rows: int, cols: int) -> "MatrixF":
        return MatrixF.from_numpy(np.zeros((int(rows), int(cols)), dtype=np.float32))  # type: ignore[arg-type]

    @staticmethod
    def ones(rows: int, cols: int) -> "MatrixF":
        return MatrixF.from_numpy(np.ones((int(rows), int(cols)), dtype=np.float32))  # type: ignore[arg-type]


class MatrixI(_BaseMatrix):
    def __init__(self, rows: int | None = None, cols: int | None = None, value: Number = 0):
        if rows is None and cols is None:
            super().__init__(np.zeros((0, 0), dtype=int))
        else:
            super().__init__(np.full((int(rows), int(cols)), int(value), dtype=int))

    @staticmethod
    def identity(size: int) -> "MatrixI":
        return MatrixI.from_numpy(np.eye(int(size), dtype=int))  # type: ignore[arg-type]

    @staticmethod
    def zeros(rows: int, cols: int) -> "MatrixI":
        return MatrixI.from_numpy(np.zeros((int(rows), int(cols)), dtype=int))  # type: ignore[arg-type]

    @staticmethod
    def ones(rows: int, cols: int) -> "MatrixI":
        return MatrixI.from_numpy(np.ones((int(rows), int(cols)), dtype=int))  # type: ignore[arg-type]


# Functional API (thin wrappers over NumPy)

def array(a: Iterable[Iterable[Number]], dtype: str | None = None):
    return np.array(a, dtype=dtype)


def zeros(shape: Tuple[int, int], dtype: str | None = None):
    return np.zeros(shape, dtype=dtype)


def ones(shape: Tuple[int, int], dtype: str | None = None):
    return np.ones(shape, dtype=dtype)


def eye(n: int, dtype: str | None = None):
    return np.eye(n, dtype=dtype)


def matmul(a, b):
    return np.matmul(a, b)


def transpose(a):
    return np.transpose(a)


def trace(a) -> Number:
    return float(np.trace(a))


def det(a) -> Number:
    return float(np.linalg.det(a))
