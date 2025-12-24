# Deep Learning Functions Reference

LinAlgKit provides comprehensive mathematical functions for building neural networks and deep learning applications.

---

## Table of Contents

1. [Activation Functions](#activation-functions)
2. [Loss Functions](#loss-functions)
3. [Normalization](#normalization-functions)
4. [Convolution Operations](#convolution-operations)
5. [Weight Initialization](#weight-initialization)
6. [Utility Functions](#utility-functions)
7. [Advanced Math](#advanced-math-functions)
8. [Examples](#examples)

---

## Activation Functions

### sigmoid(x)

Sigmoid activation: σ(x) = 1 / (1 + exp(-x))

```python
import LinAlgKit as lk
import numpy as np

x = np.array([-2, -1, 0, 1, 2])
output = lk.sigmoid(x)
# [0.119, 0.269, 0.5, 0.731, 0.881]
```

**Properties:**
- Output range: (0, 1)
- Used for: Binary classification, gates in LSTMs

---

### relu(x)

Rectified Linear Unit: ReLU(x) = max(0, x)

```python
x = np.array([-2, -1, 0, 1, 2])
output = lk.relu(x)
# [0, 0, 0, 1, 2]
```

**Properties:**
- Output range: [0, ∞)
- Fast to compute
- Can cause "dying ReLU" problem

---

### leaky_relu(x, alpha=0.01)

Leaky ReLU: f(x) = x if x > 0, else α*x

```python
x = np.array([-2, -1, 0, 1, 2])
output = lk.leaky_relu(x, alpha=0.1)
# [-0.2, -0.1, 0, 1, 2]
```

**Properties:**
- Prevents dying ReLU
- α typically 0.01 or 0.1

---

### elu(x, alpha=1.0)

Exponential Linear Unit: f(x) = x if x > 0, else α*(exp(x) - 1)

```python
x = np.array([-2, -1, 0, 1, 2])
output = lk.elu(x)
# [-0.865, -0.632, 0, 1, 2]
```

**Properties:**
- Smooth for negative values
- Mean activations closer to zero

---

### gelu(x)

Gaussian Error Linear Unit (used in BERT, GPT):

```python
x = np.array([-2, -1, 0, 1, 2])
output = lk.gelu(x)
# [-0.045, -0.158, 0, 0.841, 1.955]
```

**Properties:**
- Smooth, differentiable everywhere
- Current state-of-the-art for transformers

---

### swish(x, beta=1.0)

Self-gated activation: f(x) = x * sigmoid(β*x)

```python
output = lk.swish(x, beta=1.0)
```

**Properties:**
- Smooth, non-monotonic
- Outperforms ReLU in deep networks

---

### softmax(x, axis=-1)

Converts logits to probabilities:

```python
logits = np.array([[2.0, 1.0, 0.1]])
probs = lk.softmax(logits)
# [[0.659, 0.242, 0.099]]  (sums to 1)
```

**Properties:**
- Output sums to 1
- Used for multi-class classification

---

### log_softmax(x, axis=-1)

Numerically stable log of softmax:

```python
log_probs = lk.log_softmax(logits)
```

**Use case:** Computing cross-entropy loss efficiently

---

### softplus(x)

Smooth approximation of ReLU: f(x) = log(1 + exp(x))

```python
output = lk.softplus(x)
```

---

### tanh(x)

Hyperbolic tangent:

```python
output = lk.tanh(x)
# Range: (-1, 1)
```

---

## Loss Functions

### mse_loss(predictions, targets, reduction='mean')

Mean Squared Error for regression:

```python
pred = np.array([1.0, 2.0, 3.0])
target = np.array([1.1, 2.2, 2.8])
loss = lk.mse_loss(pred, target)
# 0.03
```

---

### mae_loss(predictions, targets, reduction='mean')

Mean Absolute Error (L1 loss):

```python
loss = lk.mae_loss(pred, target)
```

---

### huber_loss(predictions, targets, delta=1.0, reduction='mean')

Robust loss combining MSE and MAE:

```python
loss = lk.huber_loss(pred, target, delta=1.0)
```

**Properties:**
- Quadratic for small errors
- Linear for large errors (robust to outliers)

---

### cross_entropy_loss(predictions, targets, epsilon=1e-12)

Cross-entropy for multi-class classification:

```python
probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
targets = np.array([0, 1])  # Class indices
loss = lk.cross_entropy_loss(probs, targets)
```

---

### binary_cross_entropy(predictions, targets, epsilon=1e-12)

Binary cross-entropy for binary classification:

```python
probs = np.array([0.9, 0.1, 0.8])
targets = np.array([1, 0, 1])
loss = lk.binary_cross_entropy(probs, targets)
```

---

## Normalization Functions

### batch_norm(x, gamma=None, beta=None, epsilon=1e-5, axis=0)

Batch normalization:

```python
# x shape: (batch_size, features)
x_norm = lk.batch_norm(x, gamma=scale, beta=shift)
```

**Properties:**
- Normalizes across batch dimension
- Reduces internal covariate shift

---

### layer_norm(x, gamma=None, beta=None, epsilon=1e-5)

Layer normalization (used in transformers):

```python
# Normalizes across feature dimension
x_norm = lk.layer_norm(x)
```

**Properties:**
- Normalizes across features, not batch
- Works with any batch size

---

### instance_norm(x, epsilon=1e-5)

Instance normalization (for style transfer):

```python
# x shape: (batch, channels, height, width)
x_norm = lk.instance_norm(x)
```

---

## Convolution Operations

### conv2d(x, kernel, stride=1, padding=0)

2D convolution:

```python
# Input: (batch, channels, H, W) or (H, W)
# Kernel: (out_channels, in_channels, kH, kW) or (kH, kW)
image = np.random.randn(1, 1, 28, 28)
kernel = np.random.randn(32, 1, 3, 3)
output = lk.conv2d(image, kernel, stride=1, padding=1)
# Output shape: (1, 32, 28, 28)
```

---

### max_pool2d(x, kernel_size=2, stride=None)

Max pooling:

```python
output = lk.max_pool2d(x, kernel_size=2)
# Reduces spatial dimensions by half
```

---

### avg_pool2d(x, kernel_size=2, stride=None)

Average pooling:

```python
output = lk.avg_pool2d(x, kernel_size=2)
```

---

### global_avg_pool2d(x)

Global average pooling:

```python
# Input: (batch, channels, H, W)
# Output: (batch, channels)
output = lk.global_avg_pool2d(x)
```

---

## Weight Initialization

### xavier_uniform(shape, gain=1.0)

Xavier/Glorot uniform initialization (for tanh/sigmoid):

```python
weights = lk.xavier_uniform((784, 256))
```

---

### xavier_normal(shape, gain=1.0)

Xavier/Glorot normal initialization:

```python
weights = lk.xavier_normal((784, 256))
```

---

### he_uniform(shape)

He/Kaiming uniform initialization (for ReLU):

```python
weights = lk.he_uniform((784, 256))
```

---

### he_normal(shape)

He/Kaiming normal initialization:

```python
weights = lk.he_normal((784, 256))
```

---

## Utility Functions

### dropout(x, p=0.5, training=True)

Dropout regularization:

```python
# During training (randomly zeros elements)
x_dropped = lk.dropout(x, p=0.5, training=True)

# During inference (returns unchanged)
x_out = lk.dropout(x, p=0.5, training=False)
```

---

### one_hot(indices, num_classes)

One-hot encoding:

```python
labels = np.array([0, 2, 1])
encoded = lk.one_hot(labels, num_classes=3)
# [[1, 0, 0],
#  [0, 0, 1],
#  [0, 1, 0]]
```

---

### clip(x, min_val, max_val)

Clip values to a range:

```python
x_clipped = lk.clip(x, -1.0, 1.0)
```

---

### flatten(x, start_dim=0)

Flatten tensor:

```python
# Input: (batch, C, H, W)
# Output: (batch, C*H*W) if start_dim=1
x_flat = lk.flatten(x, start_dim=1)
```

---

### reshape(x, shape)

Reshape array:

```python
x_reshaped = lk.reshape(x, (batch_size, -1))
```

---

## Advanced Math Functions

### normalize(x, axis=-1, epsilon=1e-12)

L2 normalize along axis:

```python
x_normalized = lk.normalize(x)
# ||x|| = 1 along specified axis
```

---

### cosine_similarity(a, b, axis=-1)

Cosine similarity:

```python
similarity = lk.cosine_similarity(a, b)
# Range: [-1, 1]
```

---

### euclidean_distance(a, b, axis=-1)

Euclidean distance:

```python
distance = lk.euclidean_distance(a, b)
```

---

### pairwise_distances(X, Y=None)

Compute all pairwise distances:

```python
# X: (n, features), Y: (m, features)
# Output: (n, m) distance matrix
distances = lk.pairwise_distances(X, Y)
```

---

### numerical_gradient(f, x, epsilon=1e-7)

Compute numerical gradient:

```python
def loss_fn(w):
    return np.sum(w ** 2)

grad = lk.numerical_gradient(loss_fn, weights)
```

---

### outer(a, b)

Outer product:

```python
result = lk.outer(a, b)  # a[:, None] * b[None, :]
```

---

### inner(a, b)

Inner product:

```python
result = lk.inner(a, b)
```

---

### dot(a, b)

Dot product:

```python
result = lk.dot(a, b)
```

---

### cross(a, b)

Cross product (3D vectors):

```python
result = lk.cross(a, b)
```

---

## Examples

### Example 1: Simple Neural Network Forward Pass

```python
import LinAlgKit as lk
import numpy as np

# Initialize weights
W1 = lk.he_normal((784, 128))
W2 = lk.he_normal((128, 10))

# Forward pass
def forward(x):
    # Layer 1
    h1 = lk.relu(x @ W1)
    h1 = lk.dropout(h1, p=0.2, training=True)
    
    # Layer 2
    logits = h1 @ W2
    probs = lk.softmax(logits)
    return probs

# Example input
x = np.random.randn(32, 784)
output = forward(x)
print(f"Output shape: {output.shape}")  # (32, 10)
```

### Example 2: Convolutional Layer

```python
import LinAlgKit as lk
import numpy as np

# Input image batch
images = np.random.randn(16, 3, 32, 32)  # (batch, channels, H, W)

# Convolution kernel
kernel = lk.he_normal((64, 3, 3, 3))  # (out_ch, in_ch, kH, kW)

# Forward pass
conv_out = lk.conv2d(images, kernel, stride=1, padding=1)
conv_out = lk.batch_norm(conv_out)
conv_out = lk.relu(conv_out)
pooled = lk.max_pool2d(conv_out, kernel_size=2)

print(f"After conv: {conv_out.shape}")  # (16, 64, 32, 32)
print(f"After pool: {pooled.shape}")    # (16, 64, 16, 16)
```

### Example 3: Training Step with Loss

```python
import LinAlgKit as lk
import numpy as np

# Predictions and targets
logits = np.random.randn(32, 10)
targets = np.random.randint(0, 10, size=32)

# Compute loss
probs = lk.softmax(logits)
loss = lk.cross_entropy_loss(probs, targets)
print(f"Cross-entropy loss: {loss:.4f}")

# For regression
predictions = np.random.randn(32, 1)
regression_targets = np.random.randn(32, 1)
mse = lk.mse_loss(predictions, regression_targets)
print(f"MSE loss: {mse:.4f}")
```

---

## Function Reference Table

| Category | Functions |
|----------|-----------|
| **Activations** | `sigmoid`, `relu`, `leaky_relu`, `elu`, `gelu`, `swish`, `softplus`, `tanh`, `softmax`, `log_softmax` |
| **Derivatives** | `sigmoid_derivative`, `relu_derivative`, `leaky_relu_derivative`, `elu_derivative`, `tanh_derivative` |
| **Losses** | `mse_loss`, `mae_loss`, `huber_loss`, `cross_entropy_loss`, `binary_cross_entropy` |
| **Normalization** | `batch_norm`, `layer_norm`, `instance_norm` |
| **Convolution** | `conv2d`, `max_pool2d`, `avg_pool2d`, `global_avg_pool2d` |
| **Initialization** | `xavier_uniform`, `xavier_normal`, `he_uniform`, `he_normal` |
| **Utilities** | `dropout`, `one_hot`, `clip`, `flatten`, `reshape` |
| **Math** | `normalize`, `cosine_similarity`, `euclidean_distance`, `pairwise_distances`, `numerical_gradient`, `outer`, `inner`, `dot`, `cross`, `norm` |

---

*For matrix operations, see [API Reference](api.md).*
