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


# =============================================================================
# Deep Learning Mathematical Functions
# =============================================================================

# --- Activation Functions ---

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function: σ(x) = 1 / (1 + exp(-x))
    
    Args:
        x: Input array
    
    Returns:
        Activated values in range (0, 1)
    """
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid: σ'(x) = σ(x) * (1 - σ(x))"""
    s = sigmoid(x)
    return s * (1 - s)


def relu(x: np.ndarray) -> np.ndarray:
    """
    Rectified Linear Unit: ReLU(x) = max(0, x)
    
    Args:
        x: Input array
    
    Returns:
        Activated values (non-negative)
    """
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU: 1 if x > 0, else 0"""
    return (x > 0).astype(x.dtype)


def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    Leaky ReLU: f(x) = x if x > 0, else alpha * x
    
    Args:
        x: Input array
        alpha: Slope for negative values (default: 0.01)
    
    Returns:
        Activated values
    """
    return np.where(x > 0, x, alpha * x)


def leaky_relu_derivative(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Derivative of Leaky ReLU"""
    return np.where(x > 0, 1, alpha)


def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Exponential Linear Unit: f(x) = x if x > 0, else alpha * (exp(x) - 1)
    
    Args:
        x: Input array
        alpha: Scale for negative values (default: 1.0)
    
    Returns:
        Activated values
    """
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def elu_derivative(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Derivative of ELU"""
    return np.where(x > 0, 1, alpha * np.exp(x))


def gelu(x: np.ndarray) -> np.ndarray:
    """
    Gaussian Error Linear Unit (used in transformers like BERT, GPT).
    GELU(x) = x * Φ(x) where Φ is the CDF of standard normal.
    
    Args:
        x: Input array
    
    Returns:
        Activated values
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def swish(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """
    Swish activation (self-gated): f(x) = x * sigmoid(beta * x)
    
    Args:
        x: Input array
        beta: Scaling parameter (default: 1.0)
    
    Returns:
        Activated values
    """
    return x * sigmoid(beta * x)


def softplus(x: np.ndarray) -> np.ndarray:
    """
    Softplus: f(x) = log(1 + exp(x)), smooth approximation of ReLU
    
    Args:
        x: Input array
    
    Returns:
        Activated values (always positive)
    """
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def tanh(x: np.ndarray) -> np.ndarray:
    """
    Hyperbolic tangent activation.
    
    Args:
        x: Input array
    
    Returns:
        Activated values in range (-1, 1)
    """
    return np.tanh(x)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of tanh: 1 - tanh²(x)"""
    return 1 - np.tanh(x)**2


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Softmax function: converts logits to probabilities.
    softmax(x)_i = exp(x_i) / Σ exp(x_j)
    
    Args:
        x: Input array (typically logits)
        axis: Axis along which to compute softmax (default: -1)
    
    Returns:
        Probability distribution (sums to 1 along axis)
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)  # Numerical stability
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Log-softmax: numerically stable log of softmax.
    
    Args:
        x: Input array
        axis: Axis along which to compute (default: -1)
    
    Returns:
        Log probabilities
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    return x - x_max - np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))


# --- Loss Functions ---

def mse_loss(predictions: np.ndarray, targets: np.ndarray, reduction: str = 'mean') -> float:
    """
    Mean Squared Error loss.
    
    Args:
        predictions: Model predictions
        targets: Ground truth values
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        Loss value
    """
    diff = predictions - targets
    loss = diff ** 2
    if reduction == 'mean':
        return float(np.mean(loss))
    elif reduction == 'sum':
        return float(np.sum(loss))
    return loss


def mae_loss(predictions: np.ndarray, targets: np.ndarray, reduction: str = 'mean') -> float:
    """
    Mean Absolute Error loss.
    
    Args:
        predictions: Model predictions
        targets: Ground truth values
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        Loss value
    """
    loss = np.abs(predictions - targets)
    if reduction == 'mean':
        return float(np.mean(loss))
    elif reduction == 'sum':
        return float(np.sum(loss))
    return loss


def huber_loss(predictions: np.ndarray, targets: np.ndarray, delta: float = 1.0, reduction: str = 'mean') -> float:
    """
    Huber loss: quadratic for small errors, linear for large errors.
    
    Args:
        predictions: Model predictions
        targets: Ground truth values
        delta: Threshold for switching between quadratic and linear
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        Loss value
    """
    diff = np.abs(predictions - targets)
    loss = np.where(diff <= delta, 0.5 * diff**2, delta * (diff - 0.5 * delta))
    if reduction == 'mean':
        return float(np.mean(loss))
    elif reduction == 'sum':
        return float(np.sum(loss))
    return loss


def cross_entropy_loss(predictions: np.ndarray, targets: np.ndarray, epsilon: float = 1e-12) -> float:
    """
    Cross-entropy loss for multi-class classification.
    
    Args:
        predictions: Predicted probabilities (after softmax)
        targets: One-hot encoded targets or class indices
        epsilon: Small value for numerical stability
    
    Returns:
        Loss value
    """
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    if targets.ndim == 1:
        # targets are class indices
        n_samples = predictions.shape[0]
        return -np.sum(np.log(predictions[np.arange(n_samples), targets])) / n_samples
    else:
        # targets are one-hot encoded
        return -np.mean(np.sum(targets * np.log(predictions), axis=-1))


def binary_cross_entropy(predictions: np.ndarray, targets: np.ndarray, epsilon: float = 1e-12) -> float:
    """
    Binary cross-entropy loss.
    
    Args:
        predictions: Predicted probabilities (after sigmoid)
        targets: Binary targets (0 or 1)
        epsilon: Small value for numerical stability
    
    Returns:
        Loss value
    """
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))


# --- Normalization Functions ---

def batch_norm(x: np.ndarray, gamma: np.ndarray = None, beta: np.ndarray = None, 
               epsilon: float = 1e-5, axis: int = 0) -> np.ndarray:
    """
    Batch normalization.
    
    Args:
        x: Input array (batch_size, features)
        gamma: Scale parameter (default: 1)
        beta: Shift parameter (default: 0)
        epsilon: Small value for numerical stability
        axis: Axis along which to normalize (default: 0 for batch)
    
    Returns:
        Normalized array
    """
    mean = np.mean(x, axis=axis, keepdims=True)
    var = np.var(x, axis=axis, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + epsilon)
    
    if gamma is not None:
        x_norm = x_norm * gamma
    if beta is not None:
        x_norm = x_norm + beta
    
    return x_norm


def layer_norm(x: np.ndarray, gamma: np.ndarray = None, beta: np.ndarray = None,
               epsilon: float = 1e-5) -> np.ndarray:
    """
    Layer normalization (normalizes across features).
    
    Args:
        x: Input array
        gamma: Scale parameter
        beta: Shift parameter
        epsilon: Small value for numerical stability
    
    Returns:
        Normalized array
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + epsilon)
    
    if gamma is not None:
        x_norm = x_norm * gamma
    if beta is not None:
        x_norm = x_norm + beta
    
    return x_norm


def instance_norm(x: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """
    Instance normalization (for style transfer and image applications).
    Normalizes each sample in the batch independently.
    
    Args:
        x: Input array (batch, channels, height, width)
        epsilon: Small value for numerical stability
    
    Returns:
        Normalized array
    """
    mean = np.mean(x, axis=(2, 3), keepdims=True)
    var = np.var(x, axis=(2, 3), keepdims=True)
    return (x - mean) / np.sqrt(var + epsilon)


# --- Convolution Operations ---

def conv2d(x: np.ndarray, kernel: np.ndarray, stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    2D convolution operation.
    
    Args:
        x: Input array (batch, channels, height, width) or (height, width)
        kernel: Convolution kernel (out_channels, in_channels, kH, kW) or (kH, kW)
        stride: Stride of convolution (default: 1)
        padding: Zero-padding added to both sides (default: 0)
    
    Returns:
        Convolved output
    """
    # Handle 2D input (single image, single channel)
    if x.ndim == 2:
        x = x[np.newaxis, np.newaxis, :, :]
    elif x.ndim == 3:
        x = x[np.newaxis, :, :, :]
    
    if kernel.ndim == 2:
        kernel = kernel[np.newaxis, np.newaxis, :, :]
    elif kernel.ndim == 3:
        kernel = kernel[np.newaxis, :, :, :]
    
    batch, in_channels, H, W = x.shape
    out_channels, _, kH, kW = kernel.shape
    
    # Add padding
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    
    H_out = (H + 2 * padding - kH) // stride + 1
    W_out = (W + 2 * padding - kW) // stride + 1
    
    output = np.zeros((batch, out_channels, H_out, W_out))
    
    for i in range(H_out):
        for j in range(W_out):
            h_start, w_start = i * stride, j * stride
            patch = x[:, :, h_start:h_start+kH, w_start:w_start+kW]
            for c in range(out_channels):
                output[:, c, i, j] = np.sum(patch * kernel[c], axis=(1, 2, 3))
    
    return output.squeeze()


def max_pool2d(x: np.ndarray, kernel_size: int = 2, stride: int = None) -> np.ndarray:
    """
    2D max pooling operation.
    
    Args:
        x: Input array (batch, channels, height, width)
        kernel_size: Size of the pooling window
        stride: Stride of pooling (default: kernel_size)
    
    Returns:
        Pooled output
    """
    if stride is None:
        stride = kernel_size
    
    if x.ndim == 2:
        x = x[np.newaxis, np.newaxis, :, :]
    elif x.ndim == 3:
        x = x[np.newaxis, :, :, :]
    
    batch, channels, H, W = x.shape
    H_out = (H - kernel_size) // stride + 1
    W_out = (W - kernel_size) // stride + 1
    
    output = np.zeros((batch, channels, H_out, W_out))
    
    for i in range(H_out):
        for j in range(W_out):
            h_start, w_start = i * stride, j * stride
            output[:, :, i, j] = np.max(x[:, :, h_start:h_start+kernel_size, w_start:w_start+kernel_size], axis=(2, 3))
    
    return output.squeeze()


def avg_pool2d(x: np.ndarray, kernel_size: int = 2, stride: int = None) -> np.ndarray:
    """
    2D average pooling operation.
    
    Args:
        x: Input array (batch, channels, height, width)
        kernel_size: Size of the pooling window
        stride: Stride of pooling (default: kernel_size)
    
    Returns:
        Pooled output
    """
    if stride is None:
        stride = kernel_size
    
    if x.ndim == 2:
        x = x[np.newaxis, np.newaxis, :, :]
    elif x.ndim == 3:
        x = x[np.newaxis, :, :, :]
    
    batch, channels, H, W = x.shape
    H_out = (H - kernel_size) // stride + 1
    W_out = (W - kernel_size) // stride + 1
    
    output = np.zeros((batch, channels, H_out, W_out))
    
    for i in range(H_out):
        for j in range(W_out):
            h_start, w_start = i * stride, j * stride
            output[:, :, i, j] = np.mean(x[:, :, h_start:h_start+kernel_size, w_start:w_start+kernel_size], axis=(2, 3))
    
    return output.squeeze()


def global_avg_pool2d(x: np.ndarray) -> np.ndarray:
    """
    Global average pooling (reduces spatial dimensions to 1x1).
    
    Args:
        x: Input array (batch, channels, height, width)
    
    Returns:
        Pooled output (batch, channels)
    """
    if x.ndim == 2:
        return np.mean(x)
    elif x.ndim == 3:
        return np.mean(x, axis=(1, 2))
    else:
        return np.mean(x, axis=(2, 3))


# --- Utility Functions ---

def clip(x: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Clip values to a range."""
    return np.clip(x, min_val, max_val)


def dropout(x: np.ndarray, p: float = 0.5, training: bool = True) -> np.ndarray:
    """
    Dropout regularization.
    
    Args:
        x: Input array
        p: Probability of dropping a unit (default: 0.5)
        training: If True, apply dropout; if False, return input unchanged
    
    Returns:
        Output with dropout applied (scaled by 1/(1-p) during training)
    """
    if not training or p == 0:
        return x
    mask = np.random.binomial(1, 1 - p, x.shape) / (1 - p)
    return x * mask


def one_hot(indices: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert class indices to one-hot encoding.
    
    Args:
        indices: Array of class indices
        num_classes: Total number of classes
    
    Returns:
        One-hot encoded array
    """
    indices = np.asarray(indices).astype(int)
    result = np.zeros((indices.size, num_classes))
    result[np.arange(indices.size), indices.flatten()] = 1
    return result.reshape(indices.shape + (num_classes,))


def flatten(x: np.ndarray, start_dim: int = 0) -> np.ndarray:
    """
    Flatten array from start_dim to end.
    
    Args:
        x: Input array
        start_dim: First dimension to flatten (default: 0)
    
    Returns:
        Flattened array
    """
    shape = x.shape[:start_dim] + (-1,)
    return x.reshape(shape)


def reshape(x: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Reshape array to given shape."""
    return x.reshape(shape)


# --- Initialization Functions ---

def xavier_uniform(shape: Tuple[int, ...], gain: float = 1.0) -> np.ndarray:
    """
    Xavier/Glorot uniform initialization.
    
    Args:
        shape: Shape of the weight tensor (fan_in, fan_out)
        gain: Scaling factor
    
    Returns:
        Initialized weights
    """
    fan_in, fan_out = shape[0], shape[1] if len(shape) > 1 else shape[0]
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    bound = np.sqrt(3.0) * std
    return np.random.uniform(-bound, bound, shape)


def xavier_normal(shape: Tuple[int, ...], gain: float = 1.0) -> np.ndarray:
    """
    Xavier/Glorot normal initialization.
    
    Args:
        shape: Shape of the weight tensor
        gain: Scaling factor
    
    Returns:
        Initialized weights
    """
    fan_in, fan_out = shape[0], shape[1] if len(shape) > 1 else shape[0]
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.normal(0, std, shape)


def he_uniform(shape: Tuple[int, ...]) -> np.ndarray:
    """
    He/Kaiming uniform initialization (for ReLU networks).
    
    Args:
        shape: Shape of the weight tensor
    
    Returns:
        Initialized weights
    """
    fan_in = shape[0]
    bound = np.sqrt(6.0 / fan_in)
    return np.random.uniform(-bound, bound, shape)


def he_normal(shape: Tuple[int, ...]) -> np.ndarray:
    """
    He/Kaiming normal initialization (for ReLU networks).
    
    Args:
        shape: Shape of the weight tensor
    
    Returns:
        Initialized weights
    """
    fan_in = shape[0]
    std = np.sqrt(2.0 / fan_in)
    return np.random.normal(0, std, shape)


# --- Gradient Operations ---

def numerical_gradient(f, x: np.ndarray, epsilon: float = 1e-7) -> np.ndarray:
    """
    Compute numerical gradient using central difference.
    
    Args:
        f: Function to differentiate
        x: Point at which to evaluate gradient
        epsilon: Small perturbation
    
    Returns:
        Gradient array
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old_val = x[idx]
        
        x[idx] = old_val + epsilon
        fxh1 = f(x)
        
        x[idx] = old_val - epsilon
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2) / (2 * epsilon)
        x[idx] = old_val
        it.iternext()
    
    return grad


# --- Advanced Math Functions ---

def outer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Outer product of two vectors."""
    return np.outer(a, b)


def inner(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Inner product of two arrays."""
    return np.inner(a, b)


def dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Dot product."""
    return np.dot(a, b)


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross product of 3D vectors."""
    return np.cross(a, b)


def norm(x: np.ndarray, ord: Union[int, float, str] = None, axis: int = None) -> np.ndarray:
    """Compute vector/matrix norm."""
    return np.linalg.norm(x, ord=ord, axis=axis)


def normalize(x: np.ndarray, axis: int = -1, epsilon: float = 1e-12) -> np.ndarray:
    """L2 normalize along axis."""
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + epsilon)


def cosine_similarity(a: np.ndarray, b: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute cosine similarity between arrays."""
    a_norm = normalize(a, axis=axis)
    b_norm = normalize(b, axis=axis)
    return np.sum(a_norm * b_norm, axis=axis)


def euclidean_distance(a: np.ndarray, b: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute Euclidean distance between arrays."""
    return np.linalg.norm(a - b, axis=axis)


def pairwise_distances(X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
    """
    Compute pairwise Euclidean distances.
    
    Args:
        X: First set of points (n_samples_X, n_features)
        Y: Second set of points (n_samples_Y, n_features), defaults to X
    
    Returns:
        Distance matrix (n_samples_X, n_samples_Y)
    """
    if Y is None:
        Y = X
    XX = np.sum(X**2, axis=1)[:, np.newaxis]
    YY = np.sum(Y**2, axis=1)[np.newaxis, :]
    distances = XX + YY - 2 * np.dot(X, Y.T)
    return np.sqrt(np.maximum(distances, 0))

