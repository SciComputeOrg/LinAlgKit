# LinAlgKit Documentation

Welcome to the LinAlgKit docs. This project provides a comprehensive, Python-first linear algebra and deep learning math library built on NumPy.

**Version: 0.2.1** | [GitHub](https://github.com/SciComputeOrg/LinAlgKit) | [PyPI](https://pypi.org/project/LinAlgKit/)

## Features

- 🔢 **Matrix Operations** - Full matrix algebra with decompositions
- 🧠 **Deep Learning Functions** - Activations, losses, normalization
- ⚡ **High Performance** - Numba JIT acceleration up to 13x faster
- 🐍 **Pythonic API** - Clean, intuitive interface

## Quick Install

```bash
pip install LinAlgKit
```

For high-performance functions:
```bash
pip install LinAlgKit numba
```

## Quick Start

```python
import LinAlgKit as lk
import numpy as np

# Create matrices
A = lk.Matrix.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]]))

# Matrix operations
print(A.determinant())  # -2.0
print(A.T.to_numpy())   # Transpose

# Decompositions
L, U, P = A.lu()
Q, R = A.qr()

# Deep learning functions
x = np.random.randn(100, 10)
output = lk.relu(x)
probs = lk.softmax(x)
loss = lk.cross_entropy_loss(probs, targets)
```

## Documentation

| Section | Description |
|---------|-------------|
| [Getting Started](tutorial.md) | Installation, basic usage, examples |
| [API Reference](api.md) | Complete API documentation |
| [Deep Learning](deep_learning.md) | Activations, losses, normalization |
| [Performance](performance.md) | Benchmarks and optimization |
| [Release Notes](releases.md) | Version history and changelog |

## What's New in v0.2.1

### High-Performance `fast` Module
```python
from LinAlgKit import fast

# Up to 13x faster with Numba JIT
loss = fast.fast_mse_loss(pred, target)
output = fast.fast_relu(x)
```

### Performance Improvements

| Function | Speedup |
|----------|---------|
| `mae_loss` | **13.1x** |
| `mse_loss` | **12.0x** |
| `leaky_relu` | **4.4x** |
| `gelu` | **2.6x** |

### In-Place Operations
```python
A.add_(B)      # No memory allocation
A.mul_(2.0)    # Faster than A = A * 2
```

## License

MIT License - see [LICENSE](https://github.com/SciComputeOrg/LinAlgKit/blob/main/LICENSE)
