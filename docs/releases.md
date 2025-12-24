# Release Notes

This document contains detailed release notes for each version of LinAlgKit.

---

## v0.2.1 - Performance Optimizations (2025-12-24)

### 🚀 Performance Improvements

This release introduces significant performance optimizations using Numba JIT compilation.

#### Benchmark Results (5M elements)

| Function | Standard | Fast (JIT) | Speedup |
|----------|----------|------------|---------|
| `mse_loss` | 33.75ms | 2.82ms | **12.0x** |
| `mae_loss` | 34.86ms | 2.67ms | **13.1x** |
| `leaky_relu` | 30.16ms | 6.81ms | **4.4x** |
| `gelu` | 200.78ms | 76.44ms | **2.6x** |
| `tanh` | 37.33ms | 15.29ms | **2.4x** |
| `swish` | 100.27ms | 54.62ms | **1.8x** |
| `elu` | 65.14ms | 40.06ms | **1.6x** |
| `sigmoid` | 80.98ms | 54.71ms | **1.5x** |
| `softplus` | 108.80ms | 70.88ms | **1.5x** |

### New Features

#### `fast` Module
A new high-performance module with Numba JIT-compiled functions:

```python
from LinAlgKit import fast

# 12x faster loss computation
loss = fast.fast_mse_loss(predictions, targets)

# 4.4x faster activation
output = fast.fast_leaky_relu(x, alpha=0.01)
```

Available fast functions:
- `fast_sigmoid`, `fast_relu`, `fast_leaky_relu`, `fast_elu`
- `fast_gelu`, `fast_swish`, `fast_tanh`, `fast_softplus`
- `fast_mse_loss`, `fast_mae_loss`, `fast_normalize`

#### In-Place Operations
New in-place methods that avoid memory allocation:

```python
A.add_(B)      # A += B in-place
A.sub_(B)      # A -= B in-place
A.mul_(2.0)    # A *= 2.0 in-place
A.hadamard_(B) # Element-wise multiply in-place
```

#### Zero-Copy Access
- `.T` property - Transpose view (no copy)
- `to_numpy_view()` - Direct array access (no copy)
- `transpose(copy=False)` - Optional zero-copy transpose

### Dependencies
- Optional: `numba>=0.57.0` (for JIT acceleration)

---

## v0.2.0 - Deep Learning Expansion (2025-12-24)

### 🧠 Deep Learning Functions

Major expansion with 50+ mathematical functions for deep learning.

### Activation Functions
- `sigmoid`, `relu`, `leaky_relu`, `elu`, `gelu`, `swish`
- `softplus`, `tanh`, `softmax`, `log_softmax`
- Derivative functions for backpropagation

### Loss Functions
- `mse_loss` - Mean Squared Error
- `mae_loss` - Mean Absolute Error
- `huber_loss` - Robust loss
- `cross_entropy_loss` - Multi-class classification
- `binary_cross_entropy` - Binary classification

### Normalization
- `batch_norm` - Batch normalization
- `layer_norm` - Layer normalization (transformers)
- `instance_norm` - Instance normalization (style transfer)

### Convolution Operations
- `conv2d` - 2D convolution
- `max_pool2d`, `avg_pool2d` - Pooling
- `global_avg_pool2d` - Global average pooling

### Weight Initialization
- `xavier_uniform`, `xavier_normal` - For tanh/sigmoid
- `he_uniform`, `he_normal` - For ReLU networks

### Utility Functions
- `dropout`, `one_hot`, `clip`, `flatten`, `reshape`
- `normalize`, `cosine_similarity`, `euclidean_distance`
- `pairwise_distances`, `numerical_gradient`

### Matrix Decompositions
- `lu()` - LU decomposition with partial pivoting
- `qr()` - QR decomposition
- `cholesky()` - Cholesky decomposition
- `svd()` - Singular Value Decomposition

### Eigenvalue Methods
- `eig()` - Eigenvalues and eigenvectors
- `eigvals()` - Eigenvalues only
- `eigh()` - Symmetric eigenvalue decomposition

### Linear Solvers
- `solve()` - Solve Ax = b
- `inv()` - Matrix inverse
- `pinv()` - Moore-Penrose pseudoinverse
- `lstsq()` - Least-squares solution

### Matrix Analysis
- `norm()` - Multiple norm types
- `cond()` - Condition number
- `rank()` - Matrix rank

---

## v0.1.0 - Initial Release (2025-12-24)

### Core Features

#### Matrix Classes
- `Matrix` - Double-precision (float64)
- `MatrixF` - Single-precision (float32)
- `MatrixI` - Integer matrix

#### Static Constructors
```python
Matrix.identity(3)   # 3x3 identity
Matrix.zeros(2, 3)   # 2x3 zeros
Matrix.ones(4, 4)    # 4x4 ones
```

#### Matrix Operations
- `transpose()` - Transposition
- `trace()` - Sum of diagonal
- `determinant()` - LU-based O(n³)
- `determinant_naive()` - Recursive (small matrices)

#### Arithmetic Operators
- `A + B` - Element-wise addition
- `A - B` - Element-wise subtraction
- `A * B` - Matrix multiplication
- `2 * A` - Scalar multiplication

#### NumPy Interoperability
```python
# From NumPy
A = Matrix.from_numpy(np_array)

# To NumPy
np_array = A.to_numpy()
```

#### Element Access
```python
val = A[i, j]    # Get element
A[i, j] = 5.0    # Set element
```

---

## Upgrade Guide

### From v0.1.0 to v0.2.0

No breaking changes. All v0.1.0 code works unchanged.

**New imports available:**
```python
import LinAlgKit as lk

# New activation functions
output = lk.relu(x)
output = lk.softmax(logits)

# New matrix methods
L = A.cholesky()
eigenvalues, eigenvectors = A.eig()
x = A.solve(b)
```

### From v0.2.0 to v0.2.1

No breaking changes. Install `numba` for automatic acceleration:

```bash
pip install numba
```

**Use fast module for best performance:**
```python
from LinAlgKit import fast

# Check if Numba is available
import LinAlgKit as lk
print(lk.HAS_NUMBA)  # True if numba installed
```

---

## Roadmap

See [EXPANSION_PLAN.md](https://github.com/SciComputeOrg/LinAlgKit/blob/main/EXPANSION_PLAN.md) for the complete roadmap.

### Upcoming Features

**v0.3.0** (Q1 2025)
- Sparse matrix support (CSR, COO)
- Iterative solvers (CG, GMRES)
- Matrix functions (expm, logm, sqrtm)

**v0.4.0** (Q2 2025)
- C++/Rust native core
- BLAS/LAPACK integration
- GPU acceleration (CUDA)

**v1.0.0** (2027)
- Production-ready
- Multi-backend support
- Distributed computing
