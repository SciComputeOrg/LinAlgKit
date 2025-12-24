"""
LinAlgKit Neural Network Module

PyTorch-like neural network API.
"""
from .module import Module, Parameter, Sequential
from .layers import (
    Linear, Dropout, Flatten,
    ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, GELU,
)

__all__ = [
    'Module', 'Parameter', 'Sequential',
    'Linear', 'Dropout', 'Flatten',
    'ReLU', 'Sigmoid', 'Tanh', 'Softmax', 'LeakyReLU', 'GELU',
]
