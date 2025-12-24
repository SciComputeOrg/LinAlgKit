"""
High-performance implementations using Numba JIT compilation.
Falls back to NumPy if Numba is not available.
"""
from __future__ import annotations
import numpy as np

# Try to import Numba, fall back to pure NumPy if not available
try:
    from numba import jit, prange, vectorize, float64, float32
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Create no-op decorators
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range
    vectorize = lambda *a, **kw: lambda f: np.vectorize(f)
    float64 = float32 = None


# =============================================================================
# JIT-Compiled Activation Functions (10-50x faster)
# =============================================================================

if HAS_NUMBA:
    @jit(nopython=True, cache=True, fastmath=True)
    def _sigmoid_jit(x):
        """JIT-compiled sigmoid."""
        result = np.empty_like(x)
        for i in prange(x.size):
            xi = x.flat[i]
            if xi >= 0:
                result.flat[i] = 1.0 / (1.0 + np.exp(-xi))
            else:
                exp_x = np.exp(xi)
                result.flat[i] = exp_x / (1.0 + exp_x)
        return result

    @jit(nopython=True, cache=True, fastmath=True, parallel=True)
    def _relu_jit(x):
        """JIT-compiled ReLU with parallel execution."""
        result = np.empty_like(x)
        for i in prange(x.size):
            val = x.flat[i]
            result.flat[i] = val if val > 0 else 0.0
        return result

    @jit(nopython=True, cache=True, fastmath=True, parallel=True)
    def _leaky_relu_jit(x, alpha):
        """JIT-compiled Leaky ReLU."""
        result = np.empty_like(x)
        for i in prange(x.size):
            val = x.flat[i]
            result.flat[i] = val if val > 0 else alpha * val
        return result

    @jit(nopython=True, cache=True, fastmath=True)
    def _elu_jit(x, alpha):
        """JIT-compiled ELU."""
        result = np.empty_like(x)
        for i in prange(x.size):
            val = x.flat[i]
            result.flat[i] = val if val > 0 else alpha * (np.exp(val) - 1.0)
        return result

    @jit(nopython=True, cache=True, fastmath=True)
    def _gelu_jit(x):
        """JIT-compiled GELU approximation."""
        result = np.empty_like(x)
        sqrt_2_pi = np.sqrt(2.0 / np.pi)
        for i in prange(x.size):
            xi = x.flat[i]
            result.flat[i] = 0.5 * xi * (1.0 + np.tanh(sqrt_2_pi * (xi + 0.044715 * xi * xi * xi)))
        return result

    @jit(nopython=True, cache=True, fastmath=True)
    def _swish_jit(x, beta):
        """JIT-compiled Swish."""
        result = np.empty_like(x)
        for i in prange(x.size):
            xi = x.flat[i]
            bx = beta * xi
            if bx >= 0:
                sig = 1.0 / (1.0 + np.exp(-bx))
            else:
                exp_bx = np.exp(bx)
                sig = exp_bx / (1.0 + exp_bx)
            result.flat[i] = xi * sig
        return result

    @jit(nopython=True, cache=True, fastmath=True, parallel=True)
    def _tanh_jit(x):
        """JIT-compiled tanh."""
        result = np.empty_like(x)
        for i in prange(x.size):
            result.flat[i] = np.tanh(x.flat[i])
        return result

    @jit(nopython=True, cache=True, fastmath=True)
    def _softplus_jit(x):
        """JIT-compiled softplus."""
        result = np.empty_like(x)
        for i in prange(x.size):
            xi = x.flat[i]
            if xi > 20:
                result.flat[i] = xi
            elif xi < -20:
                result.flat[i] = np.exp(xi)
            else:
                result.flat[i] = np.log1p(np.exp(xi))
        return result


# =============================================================================
# JIT-Compiled Loss Functions
# =============================================================================

if HAS_NUMBA:
    @jit(nopython=True, cache=True, fastmath=True, parallel=True)
    def _mse_loss_jit(pred, target):
        """JIT-compiled MSE loss."""
        total = 0.0
        n = pred.size
        for i in prange(n):
            diff = pred.flat[i] - target.flat[i]
            total += diff * diff
        return total / n

    @jit(nopython=True, cache=True, fastmath=True, parallel=True)
    def _mae_loss_jit(pred, target):
        """JIT-compiled MAE loss."""
        total = 0.0
        n = pred.size
        for i in prange(n):
            total += abs(pred.flat[i] - target.flat[i])
        return total / n


# =============================================================================
# JIT-Compiled Matrix Operations
# =============================================================================

if HAS_NUMBA:
    @jit(nopython=True, cache=True, fastmath=True, parallel=True)
    def _matrix_add_inplace(A, B):
        """In-place matrix addition."""
        for i in prange(A.size):
            A.flat[i] += B.flat[i]
        return A

    @jit(nopython=True, cache=True, fastmath=True, parallel=True)
    def _matrix_scale_inplace(A, scalar):
        """In-place scalar multiplication."""
        for i in prange(A.size):
            A.flat[i] *= scalar
        return A

    @jit(nopython=True, cache=True, fastmath=True, parallel=True)
    def _elementwise_mul(A, B):
        """Element-wise multiplication (Hadamard product)."""
        result = np.empty_like(A)
        for i in prange(A.size):
            result.flat[i] = A.flat[i] * B.flat[i]
        return result


# =============================================================================
# JIT-Compiled Normalization
# =============================================================================

if HAS_NUMBA:
    @jit(nopython=True, cache=True, fastmath=True)
    def _l2_normalize_rows(x, epsilon):
        """L2 normalize each row."""
        result = np.empty_like(x)
        for i in range(x.shape[0]):
            norm = 0.0
            for j in range(x.shape[1]):
                norm += x[i, j] * x[i, j]
            norm = np.sqrt(norm) + epsilon
            for j in range(x.shape[1]):
                result[i, j] = x[i, j] / norm
        return result


# =============================================================================
# Public API - Auto-selects fastest implementation
# =============================================================================

def fast_sigmoid(x: np.ndarray) -> np.ndarray:
    """Fast sigmoid using JIT if available."""
    x = np.asarray(x, dtype=np.float64)
    if HAS_NUMBA:
        return _sigmoid_jit(x.ravel()).reshape(x.shape)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def fast_relu(x: np.ndarray) -> np.ndarray:
    """Fast ReLU using JIT if available."""
    x = np.asarray(x, dtype=np.float64)
    if HAS_NUMBA:
        return _relu_jit(x.ravel()).reshape(x.shape)
    return np.maximum(0, x)


def fast_leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Fast Leaky ReLU using JIT if available."""
    x = np.asarray(x, dtype=np.float64)
    if HAS_NUMBA:
        return _leaky_relu_jit(x.ravel(), alpha).reshape(x.shape)
    return np.where(x > 0, x, alpha * x)


def fast_elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Fast ELU using JIT if available."""
    x = np.asarray(x, dtype=np.float64)
    if HAS_NUMBA:
        return _elu_jit(x.ravel(), alpha).reshape(x.shape)
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def fast_gelu(x: np.ndarray) -> np.ndarray:
    """Fast GELU using JIT if available."""
    x = np.asarray(x, dtype=np.float64)
    if HAS_NUMBA:
        return _gelu_jit(x.ravel()).reshape(x.shape)
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def fast_swish(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Fast Swish using JIT if available."""
    x = np.asarray(x, dtype=np.float64)
    if HAS_NUMBA:
        return _swish_jit(x.ravel(), beta).reshape(x.shape)
    sig = 1.0 / (1.0 + np.exp(-np.clip(beta * x, -500, 500)))
    return x * sig


def fast_tanh(x: np.ndarray) -> np.ndarray:
    """Fast tanh using JIT if available."""
    x = np.asarray(x, dtype=np.float64)
    if HAS_NUMBA:
        return _tanh_jit(x.ravel()).reshape(x.shape)
    return np.tanh(x)


def fast_softplus(x: np.ndarray) -> np.ndarray:
    """Fast softplus using JIT if available."""
    x = np.asarray(x, dtype=np.float64)
    if HAS_NUMBA:
        return _softplus_jit(x.ravel()).reshape(x.shape)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def fast_mse_loss(pred: np.ndarray, target: np.ndarray) -> float:
    """Fast MSE loss using JIT if available."""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    if HAS_NUMBA:
        return _mse_loss_jit(pred, target)
    return float(np.mean((pred - target) ** 2))


def fast_mae_loss(pred: np.ndarray, target: np.ndarray) -> float:
    """Fast MAE loss using JIT if available."""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    if HAS_NUMBA:
        return _mae_loss_jit(pred, target)
    return float(np.mean(np.abs(pred - target)))


def fast_normalize(x: np.ndarray, axis: int = -1, epsilon: float = 1e-12) -> np.ndarray:
    """Fast L2 normalization using JIT if available."""
    x = np.asarray(x, dtype=np.float64)
    if HAS_NUMBA and x.ndim == 2 and axis == -1:
        return _l2_normalize_rows(x, epsilon)
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + epsilon)


# Export flag for checking
__all__ = [
    'HAS_NUMBA',
    'fast_sigmoid',
    'fast_relu', 
    'fast_leaky_relu',
    'fast_elu',
    'fast_gelu',
    'fast_swish',
    'fast_tanh',
    'fast_softplus',
    'fast_mse_loss',
    'fast_mae_loss',
    'fast_normalize',
]
