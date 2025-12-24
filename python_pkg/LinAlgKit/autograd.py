"""
LinAlgKit Autograd Engine

Pure Python + Numba implementation of automatic differentiation.
Supports reverse-mode autodiff with computation graph tracking.
"""
from __future__ import annotations
from typing import Optional, Tuple, List, Union, Callable
import numpy as np

try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


class Function:
    """
    Base class for autograd functions.
    
    Each operation (add, mul, matmul, etc.) subclasses this and implements
    forward() and backward() methods.
    """
    
    @staticmethod
    def forward(ctx: 'Context', *args, **kwargs) -> np.ndarray:
        """Compute forward pass and save tensors for backward."""
        raise NotImplementedError
    
    @staticmethod
    def backward(ctx: 'Context', grad_output: np.ndarray) -> Tuple[Optional[np.ndarray], ...]:
        """Compute gradients with respect to inputs."""
        raise NotImplementedError


class Context:
    """Context object to save tensors for backward pass."""
    
    def __init__(self):
        self.saved_tensors: List[np.ndarray] = []
        self.saved_values: dict = {}
    
    def save_for_backward(self, *tensors):
        """Save tensors for use in backward pass."""
        self.saved_tensors = list(tensors)
    
    def save(self, **kwargs):
        """Save arbitrary values for backward."""
        self.saved_values.update(kwargs)


# =============================================================================
# Tensor Operations (Autograd Functions)
# =============================================================================

class Add(Function):
    """Element-wise addition: a + b"""
    
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + b
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return grad_output, grad_output


class Sub(Function):
    """Element-wise subtraction: a - b"""
    
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a - b
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return grad_output, -grad_output


class Mul(Function):
    """Element-wise multiplication: a * b"""
    
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(a, b)
        return a * b
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a, b = ctx.saved_tensors
        return grad_output * b, grad_output * a


class Div(Function):
    """Element-wise division: a / b"""
    
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(a, b)
        return a / b
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a, b = ctx.saved_tensors
        return grad_output / b, -grad_output * a / (b ** 2)


class MatMul(Function):
    """Matrix multiplication: a @ b"""
    
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(a, b)
        return a @ b
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a, b = ctx.saved_tensors
        # grad_a = grad_output @ b.T
        # grad_b = a.T @ grad_output
        grad_a = grad_output @ b.T
        grad_b = a.T @ grad_output
        return grad_a, grad_b


class Transpose(Function):
    """Matrix transpose"""
    
    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        return a.T
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        return (grad_output.T,)


class Sum(Function):
    """Sum reduction"""
    
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
        ctx.save_for_backward(a)
        ctx.save(axis=axis, keepdims=keepdims, shape=a.shape)
        return np.sum(a, axis=axis, keepdims=keepdims)
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        shape = ctx.saved_values['shape']
        axis = ctx.saved_values['axis']
        keepdims = ctx.saved_values['keepdims']
        
        if axis is not None and not keepdims:
            grad_output = np.expand_dims(grad_output, axis=axis)
        
        return (np.broadcast_to(grad_output, shape).copy(),)


class Mean(Function):
    """Mean reduction"""
    
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
        ctx.save_for_backward(a)
        ctx.save(axis=axis, keepdims=keepdims, shape=a.shape)
        return np.mean(a, axis=axis, keepdims=keepdims)
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        shape = ctx.saved_values['shape']
        axis = ctx.saved_values['axis']
        keepdims = ctx.saved_values['keepdims']
        
        if axis is not None:
            n = shape[axis]
            if not keepdims:
                grad_output = np.expand_dims(grad_output, axis=axis)
        else:
            n = np.prod(shape)
        
        return (np.broadcast_to(grad_output / n, shape).copy(),)


class Pow(Function):
    """Power: a ** n"""
    
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, n: float) -> np.ndarray:
        ctx.save_for_backward(a)
        ctx.save(n=n)
        return a ** n
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray, None]:
        a, = ctx.saved_tensors
        n = ctx.saved_values['n']
        return (n * (a ** (n - 1)) * grad_output, None)


class Neg(Function):
    """Negation: -a"""
    
    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        return -a
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        return (-grad_output,)


class Exp(Function):
    """Exponential: exp(a)"""
    
    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        result = np.exp(a)
        ctx.save_for_backward(result)
        return result
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        result, = ctx.saved_tensors
        return (grad_output * result,)


class Log(Function):
    """Natural logarithm: log(a)"""
    
    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(a)
        return np.log(a)
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        a, = ctx.saved_tensors
        return (grad_output / a,)


# =============================================================================
# Activation Functions (Autograd)
# =============================================================================

class ReLU(Function):
    """ReLU activation"""
    
    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(a)
        return np.maximum(0, a)
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        a, = ctx.saved_tensors
        return ((a > 0).astype(grad_output.dtype) * grad_output,)


class Sigmoid(Function):
    """Sigmoid activation"""
    
    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        result = 1.0 / (1.0 + np.exp(-np.clip(a, -500, 500)))
        ctx.save_for_backward(result)
        return result
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        result, = ctx.saved_tensors
        return (grad_output * result * (1 - result),)


class Tanh(Function):
    """Tanh activation"""
    
    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        result = np.tanh(a)
        ctx.save_for_backward(result)
        return result
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        result, = ctx.saved_tensors
        return (grad_output * (1 - result ** 2),)


class Softmax(Function):
    """Softmax activation"""
    
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, axis: int = -1) -> np.ndarray:
        # Numerically stable softmax
        a_max = np.max(a, axis=axis, keepdims=True)
        exp_a = np.exp(a - a_max)
        result = exp_a / np.sum(exp_a, axis=axis, keepdims=True)
        ctx.save_for_backward(result)
        ctx.save(axis=axis)
        return result
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        result, = ctx.saved_tensors
        axis = ctx.saved_values['axis']
        # Jacobian-vector product for softmax
        s = result
        return (grad_output * s - s * np.sum(grad_output * s, axis=axis, keepdims=True),)


class LeakyReLU(Function):
    """Leaky ReLU activation"""
    
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        ctx.save_for_backward(a)
        ctx.save(alpha=alpha)
        return np.where(a > 0, a, alpha * a)
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray, None]:
        a, = ctx.saved_tensors
        alpha = ctx.saved_values['alpha']
        return (np.where(a > 0, grad_output, alpha * grad_output), None)


class GELU(Function):
    """GELU activation (approximation)"""
    
    @staticmethod
    def forward(ctx: Context, a: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(a)
        return 0.5 * a * (1 + np.tanh(np.sqrt(2 / np.pi) * (a + 0.044715 * a ** 3)))
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        a, = ctx.saved_tensors
        sqrt_2_pi = np.sqrt(2 / np.pi)
        x = sqrt_2_pi * (a + 0.044715 * a ** 3)
        sech2 = 1 - np.tanh(x) ** 2
        grad = 0.5 * (1 + np.tanh(x)) + 0.5 * a * sech2 * sqrt_2_pi * (1 + 3 * 0.044715 * a ** 2)
        return (grad_output * grad,)


# =============================================================================
# Loss Functions (Autograd)
# =============================================================================

class MSELoss(Function):
    """Mean Squared Error Loss"""
    
    @staticmethod
    def forward(ctx: Context, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(predictions, targets)
        ctx.save(n=predictions.size)
        return np.mean((predictions - targets) ** 2)
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray, None]:
        predictions, targets = ctx.saved_tensors
        n = ctx.saved_values['n']
        return (2 * (predictions - targets) * grad_output / n, None)


class CrossEntropyLoss(Function):
    """Cross-Entropy Loss (expects logits, not probabilities)"""
    
    @staticmethod
    def forward(ctx: Context, logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
        # Compute softmax
        logits_max = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        ctx.save_for_backward(probs, targets)
        
        # Cross-entropy: -sum(targets * log(probs))
        if targets.ndim == 1:
            # Integer labels
            batch_size = logits.shape[0]
            log_probs = np.log(probs + 1e-12)
            loss = -log_probs[np.arange(batch_size), targets].mean()
        else:
            # One-hot labels
            loss = -np.mean(np.sum(targets * np.log(probs + 1e-12), axis=-1))
        
        return np.array(loss)
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray, None]:
        probs, targets = ctx.saved_tensors
        batch_size = probs.shape[0]
        
        if targets.ndim == 1:
            # Integer labels
            grad = probs.copy()
            grad[np.arange(batch_size), targets] -= 1
            grad = grad * grad_output / batch_size
        else:
            # One-hot labels
            grad = (probs - targets) * grad_output / batch_size
        
        return (grad, None)


# =============================================================================
# Reshape Operations
# =============================================================================

class Reshape(Function):
    """Reshape tensor"""
    
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        ctx.save(original_shape=a.shape)
        return a.reshape(shape)
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray, None]:
        original_shape = ctx.saved_values['original_shape']
        return (grad_output.reshape(original_shape), None)


class Flatten(Function):
    """Flatten tensor"""
    
    @staticmethod
    def forward(ctx: Context, a: np.ndarray, start_dim: int = 0) -> np.ndarray:
        ctx.save(original_shape=a.shape, start_dim=start_dim)
        new_shape = a.shape[:start_dim] + (-1,)
        return a.reshape(new_shape)
    
    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> Tuple[np.ndarray, None]:
        original_shape = ctx.saved_values['original_shape']
        return (grad_output.reshape(original_shape), None)


# Export all functions
__all__ = [
    'Function', 'Context',
    'Add', 'Sub', 'Mul', 'Div', 'MatMul', 'Transpose',
    'Sum', 'Mean', 'Pow', 'Neg', 'Exp', 'Log',
    'ReLU', 'Sigmoid', 'Tanh', 'Softmax', 'LeakyReLU', 'GELU',
    'MSELoss', 'CrossEntropyLoss',
    'Reshape', 'Flatten',
]
