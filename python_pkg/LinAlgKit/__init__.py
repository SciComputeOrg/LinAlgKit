"""
LinAlgKit — Comprehensive Python-first linear algebra and deep learning math library.

Pure-Python + NumPy implementation for easy installs and a clean API.
Includes matrices, decompositions, activation functions, loss functions, and more.
"""

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

__version__ = "0.2.0"
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
