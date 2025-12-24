"""
LinAlgKit Tensor Class

A NumPy-backed tensor with automatic differentiation support.
Supports gradient tracking, computation graphs, and backward propagation.
"""
from __future__ import annotations
from typing import Optional, Tuple, List, Union, Callable
import numpy as np

from . import autograd as ag

# Type aliases
ArrayLike = Union[np.ndarray, list, float, int, 'Tensor']


class Tensor:
    """
    A multi-dimensional array with automatic differentiation support.
    
    Similar to PyTorch's Tensor, but backed by NumPy with optional Numba acceleration.
    
    Examples:
        >>> x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        >>> y = x * 2
        >>> z = y.sum()
        >>> z.backward()
        >>> print(x.grad)  # [2.0, 2.0, 2.0]
    """
    
    __slots__ = ('data', 'grad', 'requires_grad', '_grad_fn', '_ctx', '_children', '_is_leaf')
    
    def __init__(
        self,
        data: ArrayLike,
        requires_grad: bool = False,
        dtype: Optional[np.dtype] = None,
    ):
        """
        Create a new Tensor.
        
        Args:
            data: Array-like data
            requires_grad: If True, gradients will be computed for this tensor
            dtype: Data type (default: float32)
        """
        if isinstance(data, Tensor):
            data = data.data
        
        if dtype is None:
            dtype = np.float32
        
        self.data = np.asarray(data, dtype=dtype)
        self.grad: Optional[np.ndarray] = None
        self.requires_grad = requires_grad
        
        # Autograd graph
        self._grad_fn: Optional[type] = None  # Function that created this tensor
        self._ctx: Optional[ag.Context] = None  # Context with saved tensors
        self._children: List[Tensor] = []  # Input tensors
        self._is_leaf = True  # True if created by user (not by an operation)
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape
    
    @property
    def ndim(self) -> int:
        return self.data.ndim
    
    @property
    def size(self) -> int:
        return self.data.size
    
    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype
    
    @property
    def T(self) -> 'Tensor':
        return self.transpose()
    
    def item(self) -> float:
        """Get scalar value."""
        return self.data.item()
    
    def numpy(self) -> np.ndarray:
        """Convert to NumPy array."""
        return self.data.copy()
    
    def detach(self) -> 'Tensor':
        """Return a new tensor detached from computation graph."""
        return Tensor(self.data.copy(), requires_grad=False)
    
    def clone(self) -> 'Tensor':
        """Create a copy of this tensor."""
        t = Tensor(self.data.copy(), requires_grad=self.requires_grad)
        return t
    
    # =========================================================================
    # Autograd
    # =========================================================================
    
    def _apply_op(
        self,
        op_class: type,
        *inputs: 'Tensor',
        **kwargs
    ) -> 'Tensor':
        """Apply an autograd operation and track in computation graph."""
        ctx = ag.Context()
        
        # Get raw data from tensors
        raw_inputs = [t.data if isinstance(t, Tensor) else t for t in inputs]
        
        # Forward pass
        result_data = op_class.forward(ctx, *raw_inputs, **kwargs)
        
        # Determine if result needs grad
        needs_grad = any(
            isinstance(t, Tensor) and t.requires_grad 
            for t in inputs
        )
        
        # Create result tensor
        result = Tensor(result_data, requires_grad=needs_grad)
        
        if needs_grad:
            result._grad_fn = op_class
            result._ctx = ctx
            result._children = [t for t in inputs if isinstance(t, Tensor)]
            result._is_leaf = False
        
        return result
    
    def backward(self, grad: Optional['Tensor'] = None):
        """
        Compute gradients via backpropagation.
        
        Args:
            grad: Gradient of the loss with respect to this tensor.
                  If None, assumes this is a scalar and uses 1.0.
        """
        if not self.requires_grad:
            return
        
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError(
                    "grad can only be implicitly created for scalar outputs"
                )
            grad = Tensor(np.ones_like(self.data))
        
        # Initialize gradient
        grad_data = grad.data if isinstance(grad, Tensor) else grad
        
        # Topological sort
        topo_order = []
        visited = set()
        
        def build_topo(tensor: Tensor):
            if id(tensor) not in visited:
                visited.add(id(tensor))
                for child in tensor._children:
                    build_topo(child)
                topo_order.append(tensor)
        
        build_topo(self)
        
        # Initialize gradient for this tensor
        self.grad = grad_data
        
        # Backward pass in reverse topological order
        for tensor in reversed(topo_order):
            if tensor._grad_fn is None:
                continue
            
            if tensor.grad is None:
                continue
            
            # Compute gradients for inputs
            grads = tensor._grad_fn.backward(tensor._ctx, tensor.grad)
            
            # Distribute gradients to children
            for child, child_grad in zip(tensor._children, grads):
                if child_grad is None:
                    continue
                if not child.requires_grad:
                    continue
                
                # Handle broadcasting
                child_grad = _unbroadcast(child_grad, child.shape)
                
                # Accumulate gradients
                if child.grad is None:
                    child.grad = child_grad
                else:
                    child.grad = child.grad + child_grad
    
    def zero_grad(self):
        """Zero out the gradient."""
        self.grad = None
    
    # =========================================================================
    # Operators
    # =========================================================================
    
    def __add__(self, other: ArrayLike) -> 'Tensor':
        other = _ensure_tensor(other)
        return self._apply_op(ag.Add, self, other)
    
    def __radd__(self, other: ArrayLike) -> 'Tensor':
        return self.__add__(other)
    
    def __sub__(self, other: ArrayLike) -> 'Tensor':
        other = _ensure_tensor(other)
        return self._apply_op(ag.Sub, self, other)
    
    def __rsub__(self, other: ArrayLike) -> 'Tensor':
        other = _ensure_tensor(other)
        return self._apply_op(ag.Sub, other, self)
    
    def __mul__(self, other: ArrayLike) -> 'Tensor':
        other = _ensure_tensor(other)
        return self._apply_op(ag.Mul, self, other)
    
    def __rmul__(self, other: ArrayLike) -> 'Tensor':
        return self.__mul__(other)
    
    def __truediv__(self, other: ArrayLike) -> 'Tensor':
        other = _ensure_tensor(other)
        return self._apply_op(ag.Div, self, other)
    
    def __rtruediv__(self, other: ArrayLike) -> 'Tensor':
        other = _ensure_tensor(other)
        return self._apply_op(ag.Div, other, self)
    
    def __neg__(self) -> 'Tensor':
        return self._apply_op(ag.Neg, self)
    
    def __pow__(self, n: float) -> 'Tensor':
        return self._apply_op(ag.Pow, self, n=n)
    
    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        return self._apply_op(ag.MatMul, self, other)
    
    # =========================================================================
    # Math Operations
    # =========================================================================
    
    def sum(self, axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
        """Sum of elements."""
        return self._apply_op(ag.Sum, self, axis=axis, keepdims=keepdims)
    
    def mean(self, axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
        """Mean of elements."""
        return self._apply_op(ag.Mean, self, axis=axis, keepdims=keepdims)
    
    def exp(self) -> 'Tensor':
        """Exponential."""
        return self._apply_op(ag.Exp, self)
    
    def log(self) -> 'Tensor':
        """Natural logarithm."""
        return self._apply_op(ag.Log, self)
    
    def transpose(self, *axes) -> 'Tensor':
        """Transpose."""
        return self._apply_op(ag.Transpose, self)
    
    def reshape(self, *shape) -> 'Tensor':
        """Reshape tensor."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return self._apply_op(ag.Reshape, self, shape=shape)
    
    def flatten(self, start_dim: int = 0) -> 'Tensor':
        """Flatten tensor."""
        return self._apply_op(ag.Flatten, self, start_dim=start_dim)
    
    # =========================================================================
    # Activation Functions
    # =========================================================================
    
    def relu(self) -> 'Tensor':
        """ReLU activation."""
        return self._apply_op(ag.ReLU, self)
    
    def sigmoid(self) -> 'Tensor':
        """Sigmoid activation."""
        return self._apply_op(ag.Sigmoid, self)
    
    def tanh(self) -> 'Tensor':
        """Tanh activation."""
        return self._apply_op(ag.Tanh, self)
    
    def softmax(self, axis: int = -1) -> 'Tensor':
        """Softmax activation."""
        return self._apply_op(ag.Softmax, self, axis=axis)
    
    def leaky_relu(self, alpha: float = 0.01) -> 'Tensor':
        """Leaky ReLU activation."""
        return self._apply_op(ag.LeakyReLU, self, alpha=alpha)
    
    def gelu(self) -> 'Tensor':
        """GELU activation."""
        return self._apply_op(ag.GELU, self)
    
    # =========================================================================
    # Representation
    # =========================================================================
    
    def __repr__(self) -> str:
        grad_fn_str = f", grad_fn=<{self._grad_fn.__name__}>" if self._grad_fn else ""
        req_grad_str = ", requires_grad=True" if self.requires_grad and not self._grad_fn else ""
        return f"Tensor({self.data}{req_grad_str}{grad_fn_str})"
    
    def __str__(self) -> str:
        return str(self.data)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx):
        return Tensor(self.data[idx], requires_grad=self.requires_grad)
    
    def __setitem__(self, idx, value):
        if isinstance(value, Tensor):
            value = value.data
        self.data[idx] = value


# =============================================================================
# Helper Functions
# =============================================================================

def _ensure_tensor(x: ArrayLike) -> Tensor:
    """Convert to Tensor if not already."""
    if isinstance(x, Tensor):
        return x
    return Tensor(x, requires_grad=False)


def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """
    Reduce gradient to match original shape after broadcasting.
    """
    if grad.shape == shape:
        return grad
    
    # Sum over broadcasted dimensions
    ndim_added = grad.ndim - len(shape)
    for _ in range(ndim_added):
        grad = grad.sum(axis=0)
    
    # Sum over dimensions that were broadcast
    for i, (g_dim, s_dim) in enumerate(zip(grad.shape, shape)):
        if s_dim == 1 and g_dim > 1:
            grad = grad.sum(axis=i, keepdims=True)
    
    return grad


# =============================================================================
# Functional API
# =============================================================================

def tensor(data: ArrayLike, requires_grad: bool = False, dtype=None) -> Tensor:
    """Create a tensor."""
    return Tensor(data, requires_grad=requires_grad, dtype=dtype)


def zeros(*shape, requires_grad: bool = False, dtype=np.float32) -> Tensor:
    """Create a tensor of zeros."""
    return Tensor(np.zeros(shape, dtype=dtype), requires_grad=requires_grad)


def ones(*shape, requires_grad: bool = False, dtype=np.float32) -> Tensor:
    """Create a tensor of ones."""
    return Tensor(np.ones(shape, dtype=dtype), requires_grad=requires_grad)


def randn(*shape, requires_grad: bool = False, dtype=np.float32) -> Tensor:
    """Create a tensor with random normal values."""
    return Tensor(np.random.randn(*shape).astype(dtype), requires_grad=requires_grad)


def rand(*shape, requires_grad: bool = False, dtype=np.float32) -> Tensor:
    """Create a tensor with random uniform values [0, 1)."""
    return Tensor(np.random.rand(*shape).astype(dtype), requires_grad=requires_grad)


def from_numpy(arr: np.ndarray, requires_grad: bool = False) -> Tensor:
    """Create a tensor from a NumPy array."""
    return Tensor(arr, requires_grad=requires_grad)


# =============================================================================
# Loss Functions (Functional API)
# =============================================================================

def mse_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """Mean Squared Error loss."""
    ctx = ag.Context()
    result = ag.MSELoss.forward(ctx, predictions.data, targets.data)
    
    out = Tensor(result, requires_grad=predictions.requires_grad)
    if predictions.requires_grad:
        out._grad_fn = ag.MSELoss
        out._ctx = ctx
        out._children = [predictions, targets]
        out._is_leaf = False
    
    return out


def cross_entropy_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Cross-Entropy loss (expects logits)."""
    ctx = ag.Context()
    targets_data = targets.data if isinstance(targets, Tensor) else targets
    result = ag.CrossEntropyLoss.forward(ctx, logits.data, targets_data)
    
    out = Tensor(result, requires_grad=logits.requires_grad)
    if logits.requires_grad:
        out._grad_fn = ag.CrossEntropyLoss
        out._ctx = ctx
        out._children = [logits, _ensure_tensor(targets)]
        out._is_leaf = False
    
    return out


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'Tensor',
    'tensor', 'zeros', 'ones', 'randn', 'rand', 'from_numpy',
    'mse_loss', 'cross_entropy_loss',
]
