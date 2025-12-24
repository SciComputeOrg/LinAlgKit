"""
LinAlgKit Neural Network Layers

Core layer implementations: Linear, Conv2d, etc.
"""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

from ..tensor import Tensor, randn
from .module import Module, Parameter


class Linear(Module):
    """
    Applies a linear transformation: y = xW^T + b
    
    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
        bias: If True, adds a learnable bias. Default: True
    
    Shape:
        - Input: (*, in_features)
        - Output: (*, out_features)
    
    Example:
        >>> layer = nn.Linear(10, 5)
        >>> x = Tensor(randn(32, 10))
        >>> output = layer(x)  # shape: (32, 5)
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights with Xavier/Glorot initialization
        k = np.sqrt(1.0 / in_features)
        self.weight = Parameter(Tensor(
            np.random.uniform(-k, k, (out_features, in_features)).astype(np.float32)
        ))
        
        if bias:
            self.bias = Parameter(Tensor(
                np.random.uniform(-k, k, (out_features,)).astype(np.float32)
            ))
        else:
            self.bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        # x @ W.T + b
        output = x @ self.weight.T
        if self.bias is not None:
            output = output + self.bias
        return output
    
    def __repr__(self) -> str:
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"


class Dropout(Module):
    """
    Randomly zeroes some elements during training.
    
    Args:
        p: Probability of an element to be zeroed. Default: 0.5
    
    Shape:
        - Input: (*)
        - Output: (*) same shape as input
    """
    
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return x
        
        # Create dropout mask
        mask = (np.random.rand(*x.shape) > self.p).astype(x.dtype)
        # Scale by 1/(1-p) to maintain expected value
        scale = 1.0 / (1.0 - self.p)
        
        result = Tensor(x.data * mask * scale, requires_grad=x.requires_grad)
        if x.requires_grad:
            result._children = [x]
        return result
    
    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"


class Flatten(Module):
    """
    Flattens a contiguous range of dims into a tensor.
    
    Args:
        start_dim: First dim to flatten. Default: 1
        end_dim: Last dim to flatten. Default: -1
    """
    
    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
    
    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(self.start_dim)
    
    def __repr__(self) -> str:
        return f"Flatten(start_dim={self.start_dim}, end_dim={self.end_dim})"


# =============================================================================
# Activation Layers
# =============================================================================

class ReLU(Module):
    """ReLU activation layer."""
    
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()
    
    def __repr__(self) -> str:
        return "ReLU()"


class Sigmoid(Module):
    """Sigmoid activation layer."""
    
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()
    
    def __repr__(self) -> str:
        return "Sigmoid()"


class Tanh(Module):
    """Tanh activation layer."""
    
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()
    
    def __repr__(self) -> str:
        return "Tanh()"


class Softmax(Module):
    """Softmax activation layer."""
    
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: Tensor) -> Tensor:
        return x.softmax(axis=self.dim)
    
    def __repr__(self) -> str:
        return f"Softmax(dim={self.dim})"


class LeakyReLU(Module):
    """Leaky ReLU activation layer."""
    
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope
    
    def forward(self, x: Tensor) -> Tensor:
        return x.leaky_relu(alpha=self.negative_slope)
    
    def __repr__(self) -> str:
        return f"LeakyReLU(negative_slope={self.negative_slope})"


class GELU(Module):
    """GELU activation layer."""
    
    def forward(self, x: Tensor) -> Tensor:
        return x.gelu()
    
    def __repr__(self) -> str:
        return "GELU()"


# Export
__all__ = [
    'Linear', 'Dropout', 'Flatten',
    'ReLU', 'Sigmoid', 'Tanh', 'Softmax', 'LeakyReLU', 'GELU',
]
