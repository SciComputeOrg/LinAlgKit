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

    # --- Phase 1: Matrix Decompositions ---
    
    def lu(self) -> Tuple["_BaseMatrix", "_BaseMatrix", "_BaseMatrix"]:
        """
        LU decomposition with partial pivoting.
        
        Returns:
            P, L, U: Permutation, Lower, Upper matrices such that P @ A = L @ U
        """
        from scipy.linalg import lu as scipy_lu
        P, L, U = scipy_lu(self._data)
        return self._wrap(P), self._wrap(L), self._wrap(U)
    
    def qr(self) -> Tuple["_BaseMatrix", "_BaseMatrix"]:
        """
        QR decomposition using Householder reflections.
        
        Returns:
            Q, R: Orthogonal and upper triangular matrices such that A = Q @ R
        """
        Q, R = np.linalg.qr(self._data)
        return self._wrap(Q), self._wrap(R)
    
    def cholesky(self) -> "_BaseMatrix":
        """
        Cholesky decomposition for positive-definite matrices.
        
        Returns:
            L: Lower triangular matrix such that A = L @ L.T
        
        Raises:
            LinAlgError: If matrix is not positive-definite
        """
        L = np.linalg.cholesky(self._data)
        return self._wrap(L)
    
    def svd(self, full_matrices: bool = True) -> Tuple["_BaseMatrix", np.ndarray, "_BaseMatrix"]:
        """
        Singular Value Decomposition.
        
        Args:
            full_matrices: If True, return full U and Vt; if False, return reduced
        
        Returns:
            U, S, Vt: Unitary matrices and singular values such that A = U @ diag(S) @ Vt
        """
        U, S, Vt = np.linalg.svd(self._data, full_matrices=full_matrices)
        return self._wrap(U), S, self._wrap(Vt)
    
    # --- Phase 1: Eigenvalue Problems ---
    
    def eig(self) -> Tuple[np.ndarray, "_BaseMatrix"]:
        """
        Compute eigenvalues and right eigenvectors.
        
        Returns:
            eigenvalues: 1D array of eigenvalues
            eigenvectors: Matrix where column i is eigenvector for eigenvalue i
        """
        eigenvalues, eigenvectors = np.linalg.eig(self._data)
        return eigenvalues, self._wrap(eigenvectors)
    
    def eigvals(self) -> np.ndarray:
        """
        Compute eigenvalues only (faster than eig()).
        
        Returns:
            eigenvalues: 1D array of eigenvalues
        """
        return np.linalg.eigvals(self._data)
    
    def eigh(self) -> Tuple[np.ndarray, "_BaseMatrix"]:
        """
        Eigenvalue decomposition for symmetric/Hermitian matrices.
        More efficient and numerically stable than eig() for symmetric matrices.
        
        Returns:
            eigenvalues: 1D array of real eigenvalues (sorted ascending)
            eigenvectors: Matrix of orthonormal eigenvectors
        """
        eigenvalues, eigenvectors = np.linalg.eigh(self._data)
        return eigenvalues, self._wrap(eigenvectors)
    
    # --- Phase 1: Linear System Solvers ---
    
    def solve(self, b: Union["_BaseMatrix", np.ndarray]) -> "_BaseMatrix":
        """
        Solve linear system Ax = b.
        
        Args:
            b: Right-hand side (matrix or array)
        
        Returns:
            x: Solution such that A @ x = b
        """
        if isinstance(b, _BaseMatrix):
            b_arr = b._data
        else:
            b_arr = np.asarray(b)
        x = np.linalg.solve(self._data, b_arr)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        return self._wrap(x)
    
    def inv(self) -> "_BaseMatrix":
        """
        Compute the matrix inverse.
        
        Returns:
            A_inv: Inverse matrix such that A @ A_inv = I
        
        Raises:
            LinAlgError: If matrix is singular
        """
        return self._wrap(np.linalg.inv(self._data))
    
    def pinv(self, rcond: float = 1e-15) -> "_BaseMatrix":
        """
        Compute the Moore-Penrose pseudoinverse.
        
        Args:
            rcond: Cutoff for small singular values
        
        Returns:
            A_pinv: Pseudoinverse of the matrix
        """
        return self._wrap(np.linalg.pinv(self._data, rcond=rcond))
    
    def lstsq(self, b: Union["_BaseMatrix", np.ndarray]) -> Tuple["_BaseMatrix", np.ndarray, int, np.ndarray]:
        """
        Solve least-squares problem min ||Ax - b||.
        
        Args:
            b: Right-hand side
        
        Returns:
            x: Least-squares solution
            residuals: Sum of squared residuals
            rank: Effective rank of A
            s: Singular values of A
        """
        if isinstance(b, _BaseMatrix):
            b_arr = b._data
        else:
            b_arr = np.asarray(b)
        x, residuals, rank, s = np.linalg.lstsq(self._data, b_arr, rcond=None)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        return self._wrap(x), residuals, rank, s
    
    # --- Phase 1: Matrix Norms & Conditions ---
    
    def norm(self, ord: Union[int, float, str, None] = None) -> float:
        """
        Compute matrix norm.
        
        Args:
            ord: Order of the norm:
                - None or 'fro': Frobenius norm (default)
                - 1: Maximum column sum
                - 2: Spectral norm (largest singular value)
                - inf or 'inf': Maximum row sum
                - -1, -2, -inf: Minimum versions
        
        Returns:
            norm: The computed norm value
        """
        if ord == 'inf':
            ord = np.inf
        elif ord == '-inf':
            ord = -np.inf
        return float(np.linalg.norm(self._data, ord=ord))
    
    def cond(self, p: Union[int, float, str, None] = None) -> float:
        """
        Compute the condition number.
        
        Args:
            p: Order of the norm (default: 2)
        
        Returns:
            cond: Condition number (ratio of largest to smallest singular value for p=2)
        """
        if p == 'inf':
            p = np.inf
        return float(np.linalg.cond(self._data, p=p))
    
    def rank(self, tol: float = None) -> int:
        """
        Compute the matrix rank.
        
        Args:
            tol: Threshold below which singular values are considered zero
        
        Returns:
            rank: Number of linearly independent rows/columns
        """
        return int(np.linalg.matrix_rank(self._data, tol=tol))


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
