"""
LinAlgKit Neural Network Module System

Provides nn.Module, nn.Parameter, and base layer classes.
"""
from __future__ import annotations
from typing import Optional, Dict, Iterator, Tuple, List, Any, Callable
from collections import OrderedDict
import numpy as np

from ..tensor import Tensor, randn


class Parameter(Tensor):
    """
    A Tensor subclass that is automatically registered as a parameter
    when assigned to a Module attribute.
    
    Parameters are always require_grad=True by default.
    
    Example:
        >>> w = Parameter(randn(10, 5))
        >>> w.requires_grad
        True
    """
    
    def __init__(self, data: Tensor, requires_grad: bool = True):
        if isinstance(data, Tensor):
            super().__init__(data.data, requires_grad=requires_grad)
        else:
            super().__init__(data, requires_grad=requires_grad)
    
    def __repr__(self) -> str:
        return f"Parameter containing:\n{super().__repr__()}"


class Module:
    """
    Base class for all neural network modules.
    
    Your models should subclass this class.
    
    Modules can contain other Modules, and a tree of them creates a
    neural network.
    
    Example:
        >>> class Model(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.linear = nn.Linear(10, 5)
        ...     
        ...     def forward(self, x):
        ...         return self.linear(x)
    """
    
    def __init__(self):
        self._parameters: Dict[str, Parameter] = OrderedDict()
        self._modules: Dict[str, 'Module'] = OrderedDict()
        self.training: bool = True
    
    def forward(self, *args, **kwargs) -> Tensor:
        """Define the forward pass. Must be implemented by subclasses."""
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs) -> Tensor:
        """Call the forward method."""
        return self.forward(*args, **kwargs)
    
    def __setattr__(self, name: str, value: Any):
        """Register Parameters and Modules when assigned as attributes."""
        # Handle special attributes
        if name in ('training', '_parameters', '_modules'):
            object.__setattr__(self, name, value)
            return
        
        # Initialize containers if needed
        if not hasattr(self, '_parameters'):
            object.__setattr__(self, '_parameters', OrderedDict())
        if not hasattr(self, '_modules'):
            object.__setattr__(self, '_modules', OrderedDict())
        
        # Register Parameter
        if isinstance(value, Parameter):
            self._parameters[name] = value
        # Register Module
        elif isinstance(value, Module):
            self._modules[name] = value
        
        object.__setattr__(self, name, value)
    
    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """
        Returns an iterator over module parameters.
        
        Args:
            recurse: If True, yields parameters of this module and all submodules.
        """
        for param in self._parameters.values():
            yield param
        
        if recurse:
            for module in self._modules.values():
                yield from module.parameters(recurse=True)
    
    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        """
        Returns an iterator over module parameters, yielding both the
        name of the parameter and the parameter itself.
        """
        for name, param in self._parameters.items():
            full_name = f"{prefix}.{name}" if prefix else name
            yield full_name, param
        
        if recurse:
            for mod_name, module in self._modules.items():
                subprefix = f"{prefix}.{mod_name}" if prefix else mod_name
                yield from module.named_parameters(prefix=subprefix, recurse=True)
    
    def modules(self) -> Iterator['Module']:
        """Returns an iterator over all modules in the network."""
        yield self
        for module in self._modules.values():
            yield from module.modules()
    
    def named_modules(self, prefix: str = '') -> Iterator[Tuple[str, 'Module']]:
        """Returns an iterator over all modules, yielding name and module."""
        yield prefix, self
        for name, module in self._modules.items():
            subprefix = f"{prefix}.{name}" if prefix else name
            yield from module.named_modules(prefix=subprefix)
    
    def train(self, mode: bool = True) -> 'Module':
        """Set the module in training mode."""
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self
    
    def eval(self) -> 'Module':
        """Set the module in evaluation mode."""
        return self.train(False)
    
    def zero_grad(self):
        """Zero out gradients for all parameters."""
        for param in self.parameters():
            param.zero_grad()
    
    def state_dict(self) -> Dict[str, np.ndarray]:
        """
        Returns a dictionary containing the state of the module.
        """
        state = OrderedDict()
        for name, param in self.named_parameters():
            state[name] = param.data.copy()
        return state
    
    def load_state_dict(self, state_dict: Dict[str, np.ndarray], strict: bool = True):
        """
        Load parameters from a state dictionary.
        
        Args:
            state_dict: Dictionary mapping parameter names to values
            strict: If True, requires that keys match exactly
        """
        own_state = dict(self.named_parameters())
        
        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            unexpected = set(state_dict.keys()) - set(own_state.keys())
            if missing:
                raise KeyError(f"Missing keys: {missing}")
            if unexpected:
                raise KeyError(f"Unexpected keys: {unexpected}")
        
        for name, param in own_state.items():
            if name in state_dict:
                param.data = state_dict[name].copy()
    
    def __repr__(self) -> str:
        """Pretty print the module structure."""
        lines = [f"{self.__class__.__name__}("]
        for name, module in self._modules.items():
            module_str = repr(module).replace('\n', '\n  ')
            lines.append(f"  ({name}): {module_str}")
        lines.append(")")
        return '\n'.join(lines)
    
    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count the number of parameters."""
        total = 0
        for param in self.parameters():
            if trainable_only and not param.requires_grad:
                continue
            total += param.size
        return total


class Sequential(Module):
    """
    A sequential container of modules.
    
    Modules will be called in the order they are passed to the constructor.
    
    Example:
        >>> model = nn.Sequential(
        ...     nn.Linear(10, 20),
        ...     nn.ReLU(),
        ...     nn.Linear(20, 5)
        ... )
        >>> output = model(input)
    """
    
    def __init__(self, *modules: Module):
        super().__init__()
        for i, module in enumerate(modules):
            setattr(self, f'_{i}', module)
        self._order = list(range(len(modules)))
    
    def forward(self, x: Tensor) -> Tensor:
        for i in self._order:
            x = getattr(self, f'_{i}')(x)
        return x
    
    def __getitem__(self, idx: int) -> Module:
        return getattr(self, f'_{idx}')
    
    def __len__(self) -> int:
        return len(self._order)


# Export
__all__ = ['Module', 'Parameter', 'Sequential']
