"""
LinAlgKit — Comprehensive Python-first linear algebra and deep learning library.

Features:
- Automatic differentiation (autograd)
- Neural network modules (nn.Module, nn.Linear, etc.)
- Optimizers (SGD, Adam, AdamW)
- High-performance Numba JIT acceleration

For autograd:
    from LinAlgKit import Tensor, nn, optim
    
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = x * 2 + 1
    y.sum().backward()
    print(x.grad)  # [2.0, 2.0]
"""

# Import fast module for HAS_NUMBA flag
from . import fast as _fast
HAS_NUMBA = _fast.HAS_NUMBA

# Autograd Tensor
from .tensor import (
    Tensor, tensor, zeros as tensor_zeros, ones as tensor_ones,
    randn, rand, from_numpy, mse_loss as tensor_mse_loss, cross_entropy_loss as tensor_cross_entropy,
)

# Neural Network modules
from . import nn
from . import optim

from .pylib import (
    # Matrix Classes
    Matrix,
    MatrixF,
    MatrixI,
    
    # Basic Array Functions
    array,
    zeros,
    ones,
    eye,
    matmul,
    transpose,
    trace,
    det,
    
    # Activation Functions
    sigmoid,
    sigmoid_derivative,
    relu,
    relu_derivative,
    leaky_relu,
    leaky_relu_derivative,
    elu,
    elu_derivative,
    gelu,
    swish,
    softplus,
    tanh,
    tanh_derivative,
    softmax,
    log_softmax,
    
    # Loss Functions
    mse_loss,
    mae_loss,
    huber_loss,
    cross_entropy_loss,
    binary_cross_entropy,
    
    # Normalization
    batch_norm,
    layer_norm,
    instance_norm,
    
    # Convolution Operations
    conv2d,
    max_pool2d,
    avg_pool2d,
    global_avg_pool2d,
    
    # Utility Functions
    clip,
    dropout,
    one_hot,
    flatten,
    reshape,
    
    # Weight Initialization
    xavier_uniform,
    xavier_normal,
    he_uniform,
    he_normal,
    
    # Gradient Operations
    numerical_gradient,
    
    # Advanced Math
    outer,
    inner,
    dot,
    cross,
    norm,
    normalize,
    cosine_similarity,
    euclidean_distance,
    pairwise_distances,
)

__version__ = "0.2.1"
BACKEND = "python"

__all__ = [
    # Matrix Classes
    "Matrix",
    "MatrixF",
    "MatrixI",
    
    # Basic Array Functions
    "array",
    "zeros",
    "ones",
    "eye",
    "matmul",
    "transpose",
    "trace",
    "det",
    
    # Activation Functions
    "sigmoid",
    "sigmoid_derivative",
    "relu",
    "relu_derivative",
    "leaky_relu",
    "leaky_relu_derivative",
    "elu",
    "elu_derivative",
    "gelu",
    "swish",
    "softplus",
    "tanh",
    "tanh_derivative",
    "softmax",
    "log_softmax",
    
    # Loss Functions
    "mse_loss",
    "mae_loss",
    "huber_loss",
    "cross_entropy_loss",
    "binary_cross_entropy",
    
    # Normalization
    "batch_norm",
    "layer_norm",
    "instance_norm",
    
    # Convolution Operations
    "conv2d",
    "max_pool2d",
    "avg_pool2d",
    "global_avg_pool2d",
    
    # Utility Functions
    "clip",
    "dropout",
    "one_hot",
    "flatten",
    "reshape",
    
    # Weight Initialization
    "xavier_uniform",
    "xavier_normal",
    "he_uniform",
    "he_normal",
    
    # Gradient Operations
    "numerical_gradient",
    
    # Advanced Math
    "outer",
    "inner",
    "dot",
    "cross",
    "norm",
    "normalize",
    "cosine_similarity",
    "euclidean_distance",
    "pairwise_distances",
    
    # Module Info
    "BACKEND",
    "__version__",
]
