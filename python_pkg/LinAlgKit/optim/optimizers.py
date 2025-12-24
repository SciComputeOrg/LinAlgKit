"""
LinAlgKit Optimizers

SGD, Adam, AdamW, and learning rate schedulers.
"""
from __future__ import annotations
from typing import Iterator, Optional, Tuple, List, Callable
import numpy as np

from ..tensor import Tensor
from ..nn.module import Parameter


class Optimizer:
    """
    Base class for all optimizers.
    
    Args:
        parameters: Iterable of parameters to optimize
        lr: Learning rate
    """
    
    def __init__(self, parameters: Iterator[Parameter], lr: float = 0.001):
        self.parameters: List[Parameter] = list(parameters)
        self.lr = lr
    
    def zero_grad(self):
        """Zero out gradients for all parameters."""
        for param in self.parameters:
            param.grad = None
    
    def step(self):
        """Perform a single optimization step."""
        raise NotImplementedError
    
    def state_dict(self) -> dict:
        """Returns the state of the optimizer as a dict."""
        return {'lr': self.lr}
    
    def load_state_dict(self, state_dict: dict):
        """Loads the optimizer state."""
        self.lr = state_dict.get('lr', self.lr)


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer with momentum and Nesterov.
    
    Args:
        parameters: Iterable of parameters to optimize
        lr: Learning rate (default: 0.01)
        momentum: Momentum factor (default: 0)
        weight_decay: Weight decay (L2 regularization) (default: 0)
        nesterov: Enables Nesterov momentum (default: False)
    
    Example:
        >>> optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
    """
    
    def __init__(
        self,
        parameters: Iterator[Parameter],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        
        # Momentum buffers
        self.velocities = [np.zeros_like(p.data) for p in self.parameters]
    
    def step(self):
        """Perform a single optimization step."""
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            grad = param.grad
            
            # Weight decay (L2 regularization)
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
            
            # Momentum
            if self.momentum != 0:
                v = self.velocities[i]
                v[:] = self.momentum * v + grad
                
                if self.nesterov:
                    grad = grad + self.momentum * v
                else:
                    grad = v
            
            # Update parameters
            param.data = param.data - self.lr * grad
    
    def state_dict(self) -> dict:
        return {
            'lr': self.lr,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'velocities': [v.copy() for v in self.velocities],
        }
    
    def load_state_dict(self, state_dict: dict):
        super().load_state_dict(state_dict)
        self.momentum = state_dict.get('momentum', self.momentum)
        self.weight_decay = state_dict.get('weight_decay', self.weight_decay)
        if 'velocities' in state_dict:
            self.velocities = [v.copy() for v in state_dict['velocities']]


class Adam(Optimizer):
    """
    Adam optimizer.
    
    Args:
        parameters: Iterable of parameters to optimize
        lr: Learning rate (default: 0.001)
        betas: Coefficients for computing running averages (default: (0.9, 0.999))
        eps: Term added to denominator for numerical stability (default: 1e-8)
        weight_decay: Weight decay (L2 regularization) (default: 0)
    
    Reference:
        Adam: A Method for Stochastic Optimization
        https://arxiv.org/abs/1412.6980
    
    Example:
        >>> optimizer = optim.Adam(model.parameters(), lr=0.001)
    """
    
    def __init__(
        self,
        parameters: Iterator[Parameter],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(parameters, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Moment estimates
        self.m = [np.zeros_like(p.data) for p in self.parameters]  # First moment
        self.v = [np.zeros_like(p.data) for p in self.parameters]  # Second moment
        self.t = 0  # Timestep
    
    def step(self):
        """Perform a single optimization step."""
        self.t += 1
        beta1, beta2 = self.betas
        
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            grad = param.grad
            
            # Weight decay (L2 regularization) - added to gradient
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
            
            # Update biased first moment estimate
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * grad
            
            # Update biased second moment estimate
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * (grad ** 2)
            
            # Bias correction
            m_hat = self.m[i] / (1 - beta1 ** self.t)
            v_hat = self.v[i] / (1 - beta2 ** self.t)
            
            # Update parameters
            param.data = param.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
    
    def state_dict(self) -> dict:
        return {
            'lr': self.lr,
            'betas': self.betas,
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            't': self.t,
            'm': [m.copy() for m in self.m],
            'v': [v.copy() for v in self.v],
        }
    
    def load_state_dict(self, state_dict: dict):
        super().load_state_dict(state_dict)
        self.betas = state_dict.get('betas', self.betas)
        self.eps = state_dict.get('eps', self.eps)
        self.weight_decay = state_dict.get('weight_decay', self.weight_decay)
        self.t = state_dict.get('t', self.t)
        if 'm' in state_dict:
            self.m = [m.copy() for m in state_dict['m']]
        if 'v' in state_dict:
            self.v = [v.copy() for v in state_dict['v']]


class AdamW(Optimizer):
    """
    AdamW optimizer with decoupled weight decay.
    
    Args:
        parameters: Iterable of parameters to optimize
        lr: Learning rate (default: 0.001)
        betas: Coefficients for computing running averages (default: (0.9, 0.999))
        eps: Term added to denominator for numerical stability (default: 1e-8)
        weight_decay: Decoupled weight decay (default: 0.01)
    
    Reference:
        Decoupled Weight Decay Regularization
        https://arxiv.org/abs/1711.05101
    
    Note:
        The key difference from Adam is that weight decay is applied
        directly to the parameters, not to the gradients.
    """
    
    def __init__(
        self,
        parameters: Iterator[Parameter],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        super().__init__(parameters, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Moment estimates
        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]
        self.t = 0
    
    def step(self):
        """Perform a single optimization step."""
        self.t += 1
        beta1, beta2 = self.betas
        
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            grad = param.grad
            
            # Update biased first moment estimate
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * grad
            
            # Update biased second moment estimate
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * (grad ** 2)
            
            # Bias correction
            m_hat = self.m[i] / (1 - beta1 ** self.t)
            v_hat = self.v[i] / (1 - beta2 ** self.t)
            
            # Update parameters with Adam step
            param.data = param.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            
            # Apply decoupled weight decay directly to parameters
            if self.weight_decay != 0:
                param.data = param.data - self.lr * self.weight_decay * param.data


class RMSprop(Optimizer):
    """
    RMSprop optimizer.
    
    Args:
        parameters: Iterable of parameters to optimize
        lr: Learning rate (default: 0.01)
        alpha: Smoothing constant (default: 0.99)
        eps: Term added to denominator for numerical stability (default: 1e-8)
        weight_decay: Weight decay (default: 0)
        momentum: Momentum factor (default: 0)
    """
    
    def __init__(
        self,
        parameters: Iterator[Parameter],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
    ):
        super().__init__(parameters, lr)
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        
        # Running average of squared gradients
        self.v = [np.zeros_like(p.data) for p in self.parameters]
        # Momentum buffer
        self.buffer = [np.zeros_like(p.data) for p in self.parameters]
    
    def step(self):
        """Perform a single optimization step."""
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            grad = param.grad
            
            # Weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
            
            # Update running average
            self.v[i] = self.alpha * self.v[i] + (1 - self.alpha) * (grad ** 2)
            
            # Compute update
            avg = np.sqrt(self.v[i]) + self.eps
            
            if self.momentum > 0:
                self.buffer[i] = self.momentum * self.buffer[i] + grad / avg
                param.data = param.data - self.lr * self.buffer[i]
            else:
                param.data = param.data - self.lr * grad / avg


# =============================================================================
# Learning Rate Schedulers
# =============================================================================

class LRScheduler:
    """Base class for learning rate schedulers."""
    
    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.step_count = 0
    
    def step(self):
        """Update the learning rate."""
        self.step_count += 1
        self._update_lr()
    
    def _update_lr(self):
        raise NotImplementedError
    
    def get_lr(self) -> float:
        return self.optimizer.lr


class StepLR(LRScheduler):
    """
    Decays the learning rate by gamma every step_size epochs.
    
    Args:
        optimizer: Wrapped optimizer
        step_size: Period of learning rate decay
        gamma: Multiplicative factor of learning rate decay (default: 0.1)
    """
    
    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma
    
    def _update_lr(self):
        if self.step_count % self.step_size == 0:
            self.optimizer.lr = self.optimizer.lr * self.gamma


class ExponentialLR(LRScheduler):
    """
    Decays the learning rate by gamma every epoch.
    
    Args:
        optimizer: Wrapped optimizer
        gamma: Multiplicative factor of learning rate decay
    """
    
    def __init__(self, optimizer: Optimizer, gamma: float):
        super().__init__(optimizer)
        self.gamma = gamma
    
    def _update_lr(self):
        self.optimizer.lr = self.base_lr * (self.gamma ** self.step_count)


class CosineAnnealingLR(LRScheduler):
    """
    Set the learning rate using a cosine annealing schedule.
    
    Args:
        optimizer: Wrapped optimizer
        T_max: Maximum number of iterations
        eta_min: Minimum learning rate (default: 0)
    """
    
    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0):
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min
    
    def _update_lr(self):
        self.optimizer.lr = self.eta_min + (self.base_lr - self.eta_min) * (
            1 + np.cos(np.pi * self.step_count / self.T_max)
        ) / 2


# Export
__all__ = [
    'Optimizer', 'SGD', 'Adam', 'AdamW', 'RMSprop',
    'LRScheduler', 'StepLR', 'ExponentialLR', 'CosineAnnealingLR',
]
