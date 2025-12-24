# LinAlgKit Tutorial

A comprehensive guide to using LinAlgKit for linear algebra operations in Python.

---

## Table of Contents

1. [Installation](#installation)
2. [Getting Started](#getting-started)
3. [Creating Matrices](#creating-matrices)
4. [Basic Operations](#basic-operations)
5. [Matrix Analysis](#matrix-analysis)
6. [Working with NumPy](#working-with-numpy)
7. [Practical Examples](#practical-examples)
8. [Tips and Best Practices](#tips-and-best-practices)

---

## Installation

### From PyPI (Recommended)

```bash
pip install LinAlgKit
```

### From Source

```bash
git clone https://github.com/SciComputeOrg/LinAlgKit.git
cd LinAlgKit
pip install -e .
```

### Verify Installation

```python
import LinAlgKit as lk
print(f"LinAlgKit version: {lk.__version__}")
print(f"Backend: {lk.BACKEND}")
```

---

## Getting Started

LinAlgKit provides a clean, Pythonic interface for matrix operations:

```python
import LinAlgKit as lk
import numpy as np

# Create a 2x2 matrix
A = lk.Matrix.from_numpy([[1, 2], [3, 4]])

# Create an identity matrix
I = lk.Matrix.identity(2)

# Perform operations
result = A + I
print(result.to_numpy())
# [[2. 2.]
#  [3. 5.]]
```

---

## Creating Matrices

### Method 1: From Dimensions

Create a matrix with specified dimensions and optional fill value:

```python
import LinAlgKit as lk

# 3x3 zero matrix
zeros = lk.Matrix(3, 3)
print(zeros.to_numpy())
# [[0. 0. 0.]
#  [0. 0. 0.]
#  [0. 0. 0.]]

# 2x4 matrix filled with 5.0
filled = lk.Matrix(2, 4, 5.0)
print(filled.to_numpy())
# [[5. 5. 5. 5.]
#  [5. 5. 5. 5.]]
```

### Method 2: From NumPy Arrays

Import existing NumPy arrays directly:

```python
import numpy as np
import LinAlgKit as lk

# From a NumPy array
arr = np.array([[1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0]])
A = lk.Matrix.from_numpy(arr)

# From a Python list (converted automatically)
B = lk.Matrix.from_numpy([[1, 2], [3, 4]])
```

### Method 3: Static Constructors

Use convenient factory methods:

```python
import LinAlgKit as lk

# Identity matrix
I = lk.Matrix.identity(3)
print(I.to_numpy())
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

# Zero matrix
Z = lk.Matrix.zeros(2, 3)
print(Z.to_numpy())
# [[0. 0. 0.]
#  [0. 0. 0.]]

# Ones matrix
O = lk.Matrix.ones(2, 3)
print(O.to_numpy())
# [[1. 1. 1.]
#  [1. 1. 1.]]
```

### Choosing the Right Type

LinAlgKit offers three matrix types:

```python
import LinAlgKit as lk
import numpy as np

# Matrix - double precision (float64) - default choice
A = lk.Matrix.from_numpy([[1.0, 2.0], [3.0, 4.0]])

# MatrixF - single precision (float32) - memory efficient
B = lk.MatrixF.from_numpy(np.array([[1, 2], [3, 4]], dtype=np.float32))

# MatrixI - integer - for discrete math
C = lk.MatrixI.from_numpy([[1, 0], [0, 1]])
```

| Type | When to Use |
|------|-------------|
| `Matrix` | General purpose, high precision needed |
| `MatrixF` | Large matrices, GPU compatibility |
| `MatrixI` | Graph algorithms, counting, indexing |

---

## Basic Operations

### Matrix Addition and Subtraction

```python
import LinAlgKit as lk

A = lk.Matrix.from_numpy([[1, 2], [3, 4]])
B = lk.Matrix.from_numpy([[5, 6], [7, 8]])

# Addition
C = A + B
print("A + B =")
print(C.to_numpy())
# [[ 6.  8.]
#  [10. 12.]]

# Subtraction  
D = A - B
print("A - B =")
print(D.to_numpy())
# [[-4. -4.]
#  [-4. -4.]]
```

### Matrix Multiplication

```python
import LinAlgKit as lk

A = lk.Matrix.from_numpy([[1, 2], [3, 4]])
B = lk.Matrix.from_numpy([[5, 6], [7, 8]])

# Matrix multiplication (A @ B)
C = A * B
print("A × B =")
print(C.to_numpy())
# [[19. 22.]
#  [43. 50.]]
```

### Scalar Multiplication

```python
import LinAlgKit as lk

A = lk.Matrix.from_numpy([[1, 2], [3, 4]])

# Scalar on right
B = A * 3
print("A × 3 =")
print(B.to_numpy())
# [[ 3.  6.]
#  [ 9. 12.]]

# Scalar on left
C = 2 * A
print("2 × A =")
print(C.to_numpy())
# [[2. 4.]
#  [6. 8.]]
```

### Transpose

```python
import LinAlgKit as lk

A = lk.Matrix.from_numpy([[1, 2, 3],
                          [4, 5, 6]])
print("A =")
print(A.to_numpy())
# [[1. 2. 3.]
#  [4. 5. 6.]]

AT = A.transpose()
print("A^T =")
print(AT.to_numpy())
# [[1. 4.]
#  [2. 5.]
#  [3. 6.]]
```

---

## Matrix Analysis

### Trace

The trace is the sum of diagonal elements:

```python
import LinAlgKit as lk

A = lk.Matrix.from_numpy([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]])

tr = A.trace()
print(f"trace(A) = {tr}")  # 15.0 (1 + 5 + 9)
```

### Determinant

The determinant indicates whether a matrix is invertible:

```python
import LinAlgKit as lk

# Invertible matrix (det ≠ 0)
A = lk.Matrix.from_numpy([[1, 2], [3, 4]])
print(f"det(A) = {A.determinant()}")  # -2.0

# Singular matrix (det = 0)
B = lk.Matrix.from_numpy([[1, 2], [2, 4]])
print(f"det(B) = {B.determinant()}")  # 0.0

# 3x3 matrix
C = lk.Matrix.from_numpy([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]])
print(f"det(C) = {C.determinant()}")  # ~0.0 (singular)
```

### Element Access

```python
import LinAlgKit as lk

A = lk.Matrix.from_numpy([[1, 2, 3],
                          [4, 5, 6]])

# Get element (0-indexed)
print(A[0, 0])  # 1.0
print(A[1, 2])  # 6.0

# Set element
A[0, 0] = 10.0
print(A.to_numpy())
# [[10.  2.  3.]
#  [ 4.  5.  6.]]
```

### Matrix Properties

```python
import LinAlgKit as lk

A = lk.Matrix.from_numpy([[1, 2, 3],
                          [4, 5, 6]])

print(f"Rows: {A.rows}")     # 2
print(f"Columns: {A.cols}")  # 3
```

---

## Working with NumPy

### Converting Between LinAlgKit and NumPy

```python
import numpy as np
import LinAlgKit as lk

# NumPy → LinAlgKit
np_array = np.random.rand(3, 3)
lk_matrix = lk.Matrix.from_numpy(np_array)

# LinAlgKit → NumPy
back_to_numpy = lk_matrix.to_numpy()

# Verify
print(np.allclose(np_array, back_to_numpy))  # True
```

### Using LinAlgKit with NumPy Functions

```python
import numpy as np
import LinAlgKit as lk

A = lk.Matrix.from_numpy([[4, -2], [-2, 4]])

# Use NumPy's advanced functions on LinAlgKit matrices
eigenvalues = np.linalg.eigvals(A.to_numpy())
print(f"Eigenvalues: {eigenvalues}")  # [6. 2.]

rank = np.linalg.matrix_rank(A.to_numpy())
print(f"Rank: {rank}")  # 2
```

### Functional API

LinAlgKit provides NumPy-compatible functional helpers:

```python
from LinAlgKit import array, zeros, ones, eye, matmul, transpose, trace, det

# Create arrays
A = array([[1, 2], [3, 4]])
I = eye(2)
Z = zeros((3, 3))

# Operations
B = matmul(A, I)
print(transpose(A))
print(f"trace: {trace(A)}")
print(f"det: {det(A)}")
```

---

## Practical Examples

### Example 1: Computing Powers of a Matrix

```python
import LinAlgKit as lk

def matrix_power(A, n):
    """Compute A^n"""
    result = lk.Matrix.identity(A.rows)
    for _ in range(n):
        result = result * A
    return result

A = lk.Matrix.from_numpy([[1, 1], [1, 0]])
A5 = matrix_power(A, 5)
print("A^5 =")
print(A5.to_numpy())
# [[8. 5.]
#  [5. 3.]]  - Fibonacci numbers!
```

### Example 2: Gram-Schmidt Orthogonalization

```python
import numpy as np
import LinAlgKit as lk

def gram_schmidt(A):
    """Orthogonalize columns of A"""
    m, n = A.rows, A.cols
    Q = lk.Matrix.zeros(m, n)
    np_A = A.to_numpy()
    np_Q = Q.to_numpy()
    
    for j in range(n):
        v = np_A[:, j].copy()
        for i in range(j):
            v -= np.dot(np_Q[:, i], np_A[:, j]) * np_Q[:, i]
        np_Q[:, j] = v / np.linalg.norm(v)
    
    return lk.Matrix.from_numpy(np_Q)

A = lk.Matrix.from_numpy([[1, 1], [0, 1], [1, 0]])
Q = gram_schmidt(A)
print("Orthogonal Q:")
print(Q.to_numpy())
```

### Example 3: Checking Matrix Properties

```python
import numpy as np
import LinAlgKit as lk

def analyze_matrix(A):
    """Print various properties of a matrix"""
    np_A = A.to_numpy()
    
    print(f"Shape: {A.rows} × {A.cols}")
    print(f"Trace: {A.trace()}")
    
    if A.rows == A.cols:
        print(f"Determinant: {A.determinant()}")
        
        # Check symmetry
        AT = A.transpose().to_numpy()
        is_symmetric = np.allclose(np_A, AT)
        print(f"Symmetric: {is_symmetric}")
        
        # Check if identity
        I = np.eye(A.rows)
        is_identity = np.allclose(np_A, I)
        print(f"Is Identity: {is_identity}")

# Analyze a matrix
A = lk.Matrix.from_numpy([[4, -2, 0],
                          [-2, 4, -2],
                          [0, -2, 4]])
analyze_matrix(A)
```

### Example 4: Building a Transformation Matrix

```python
import numpy as np
import LinAlgKit as lk

def rotation_matrix_2d(theta):
    """Create a 2D rotation matrix for angle theta (radians)"""
    c, s = np.cos(theta), np.sin(theta)
    return lk.Matrix.from_numpy([[c, -s], [s, c]])

def scale_matrix_2d(sx, sy):
    """Create a 2D scaling matrix"""
    return lk.Matrix.from_numpy([[sx, 0], [0, sy]])

# 45-degree rotation
R = rotation_matrix_2d(np.pi / 4)
print("Rotation (45°):")
print(R.to_numpy())

# Scale by 2x horizontally, 0.5x vertically
S = scale_matrix_2d(2, 0.5)
print("Scale:")
print(S.to_numpy())

# Combined transformation (scale then rotate)
T = R * S
print("Combined:")
print(T.to_numpy())
```

---

## Tips and Best Practices

### 1. Use Static Constructors for Common Matrices

```python
# Good - clear and efficient
I = lk.Matrix.identity(3)
Z = lk.Matrix.zeros(2, 4)

# Less clear
I = lk.Matrix.from_numpy(np.eye(3))
Z = lk.Matrix(2, 4, 0)
```

### 2. Check Dimensions Before Operations

```python
def safe_multiply(A, B):
    if A.cols != B.rows:
        raise ValueError(f"Cannot multiply {A.rows}×{A.cols} by {B.rows}×{B.cols}")
    return A * B
```

### 3. Use the Right Matrix Type

```python
# For precision-critical calculations
A = lk.Matrix.from_numpy(precise_data)  # float64

# For large datasets where memory matters
B = lk.MatrixF.from_numpy(large_data)  # float32

# For counting/graph operations
C = lk.MatrixI.from_numpy(adjacency)  # int
```

### 4. Leverage NumPy for Advanced Operations

LinAlgKit focuses on the essentials. For advanced operations, convert to NumPy:

```python
import numpy as np
import LinAlgKit as lk

A = lk.Matrix.from_numpy([[1, 2], [3, 4]])

# SVD (not in LinAlgKit yet)
U, S, Vt = np.linalg.svd(A.to_numpy())

# Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(A.to_numpy())
```

### 5. Remember Copy Semantics

`from_numpy()` and `to_numpy()` both create copies:

```python
arr = np.array([[1, 2], [3, 4]])
A = lk.Matrix.from_numpy(arr)

# Modifying arr does NOT affect A
arr[0, 0] = 999
print(A[0, 0])  # Still 1.0

# Modifying the returned array does NOT affect A
result = A.to_numpy()
result[0, 0] = 888
print(A[0, 0])  # Still 1.0
```

---

## Next Steps

- Read the full [API Reference](api.md) for detailed method documentation
- Check the [Performance Guide](performance.md) for optimization tips
- See [FUTURE_UPDATES.md](https://github.com/SciComputeOrg/LinAlgKit/blob/main/FUTURE_UPDATES.md) for upcoming features
- Contribute on [GitHub](https://github.com/SciComputeOrg/LinAlgKit)

---

*Happy computing! 🧮*
