# Autograd & Neural Networks

LinAlgKit v1.0 includes a complete automatic differentiation engine and PyTorch-like neural network API.

## Quick Start

```python
from LinAlgKit import Tensor, nn, optim

# Create a simple model
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for x, y in dataloader:
    optimizer.zero_grad()
    
    pred = model(x)
    loss = lk.cross_entropy_loss(pred, y)
    loss.backward()
    
    optimizer.step()
```

---

## Tensor with Autograd

The `Tensor` class supports automatic differentiation:

```python
from LinAlgKit import Tensor

# Create tensor with gradient tracking
x = Tensor([1.0, 2.0, 3.0], requires_grad=True)

# Operations build computation graph
y = x * 2 + 1
z = y.sum()

# Backpropagation
z.backward()
print(x.grad)  # [2.0, 2.0, 2.0]
```

### Tensor Operations

| Operation | Forward | Backward |
|-----------|---------|----------|
| `+`, `-`, `*`, `/` | Element-wise ops | ✅ |
| `@` (matmul) | Matrix multiplication | ✅ |
| `.sum()` | Sum reduction | ✅ |
| `.mean()` | Mean reduction | ✅ |
| `.relu()` | ReLU activation | ✅ |
| `.sigmoid()` | Sigmoid activation | ✅ |
| `.tanh()` | Tanh activation | ✅ |
| `.softmax()` | Softmax | ✅ |
| `.gelu()` | GELU activation | ✅ |
| `.exp()`, `.log()` | Exponential, log | ✅ |
| `.reshape()` | Reshape | ✅ |
| `.flatten()` | Flatten | ✅ |

### Create Tensors

```python
from LinAlgKit import Tensor, randn, zeros, ones

# From data
x = Tensor([[1, 2], [3, 4]], requires_grad=True)

# Random tensors
x = randn(32, 784)  # Normal distribution
x = zeros(10, 10)   # All zeros
x = ones(5, 5)      # All ones
```

---

## Neural Network Modules

### nn.Module

Base class for all layers:

```python
import LinAlgKit.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.fc1(x).relu()
        x = self.fc2(x)
        return x

model = MyModel()
print(model.num_parameters())  # Count parameters
```

### nn.Sequential

Chain modules together:

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
```

### Available Layers

| Layer | Description |
|-------|-------------|
| `nn.Linear(in, out)` | Fully connected layer |
| `nn.ReLU()` | ReLU activation |
| `nn.Sigmoid()` | Sigmoid activation |
| `nn.Tanh()` | Tanh activation |
| `nn.Softmax(dim)` | Softmax activation |
| `nn.LeakyReLU(slope)` | Leaky ReLU |
| `nn.GELU()` | GELU activation |
| `nn.Dropout(p)` | Dropout regularization |
| `nn.Flatten()` | Flatten tensor |

### Save & Load Models

```python
# Save
state = model.state_dict()

# Load
model.load_state_dict(state)

# Example: save to file
import numpy as np
np.savez('model.npz', **state)
state = dict(np.load('model.npz'))
model.load_state_dict(state)
```

---

## Optimizers

### SGD

```python
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=True
)
```

### Adam

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0
)
```

### AdamW

Decoupled weight decay (recommended for transformers):

```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01
)
```

### RMSprop

```python
optimizer = optim.RMSprop(
    model.parameters(),
    lr=0.01,
    alpha=0.99,
    momentum=0.9
)
```

---

## Learning Rate Schedulers

```python
# Step decay every 30 epochs
scheduler = optim.StepLR(optimizer, step_size=30, gamma=0.1)

# Exponential decay
scheduler = optim.ExponentialLR(optimizer, gamma=0.95)

# Cosine annealing
scheduler = optim.CosineAnnealingLR(optimizer, T_max=100)

# Usage
for epoch in range(100):
    train(...)
    scheduler.step()
```

---

## Loss Functions

```python
from LinAlgKit import mse_loss, cross_entropy_loss

# MSE for regression
loss = lk.tensor_mse_loss(predictions, targets)

# Cross-entropy for classification (logits, not probabilities)
loss = lk.tensor_cross_entropy(logits, labels)
```

---

## Complete Training Example

```python
import numpy as np
from LinAlgKit import Tensor, nn, optim

# XOR problem
X = Tensor([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y = Tensor([[0], [1], [1], [0]], dtype=np.float32)

# Model
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)

optimizer = optim.Adam(model.parameters(), lr=0.1)

# Training
for epoch in range(500):
    optimizer.zero_grad()
    
    pred = model(X)
    loss = ((pred - y) ** 2).mean()
    loss.backward()
    
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Test
print("Predictions:", model(X).data.round(2))
# [[0.], [1.], [1.], [0.]]
```

---

## Performance Tips

1. **Use Numba JIT** for activation functions:
   ```python
   from LinAlgKit.fast import fast_relu, fast_sigmoid
   ```

2. **In-place operations** on Matrix class:
   ```python
   A.add_(B)  # No memory allocation
   ```

3. **Batch operations** - process multiple samples at once

4. **Disable gradients** for inference:
   ```python
   model.eval()  # Set to evaluation mode
   ```
