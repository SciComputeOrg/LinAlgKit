# LinAlgKit API Reference

A comprehensive guide to all classes, methods, and functions in LinAlgKit.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Matrix Classes](#matrix-classes)
   - [Matrix (float64)](#matrix-float64)
   - [MatrixF (float32)](#matrixf-float32)
   - [MatrixI (int)](#matrixi-int)
4. [Matrix Methods](#matrix-methods)
   - [Constructors](#constructors)
   - [Static Constructors](#static-constructors)
   - [Properties](#properties)
   - [NumPy Interop](#numpy-interoperability)
   - [Arithmetic Operations](#arithmetic-operations)
   - [Matrix Operations](#matrix-operations)
   - [Element Access](#element-access)
5. [Functional API](#functional-api)
6. [Examples](#examples)
7. [Error Handling](#error-handling)

---

## Installation

```bash
pip install LinAlgKit
```

**Requirements:**
- Python 3.8+
- NumPy >= 1.22

---

## Quick Start

```python
import LinAlgKit as lk
import numpy as np

# Create a matrix from a NumPy array
A = lk.Matrix.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]]))

# Create an identity matrix
I = lk.Matrix.identity(2)

# Perform operations
C = A + I                    # Matrix addition
AT = A.transpose()           # Transpose
det_A = A.determinant()      # Determinant
tr_A = A.trace()             # Trace

# Convert back to NumPy
result = C.to_numpy()
print(result)
# [[2. 2.]
#  [3. 5.]]
```

---

## Matrix Classes

LinAlgKit provides three matrix classes optimized for different numeric types:

| Class | NumPy dtype | Use Case |
|-------|-------------|----------|
| `Matrix` | `float64` | General-purpose, high precision |
| `MatrixF` | `float32` | Memory-efficient, GPU-compatible |
| `MatrixI` | `int` | Integer arithmetic, indexing matrices |

All three classes share the same API, differing only in their underlying data type.

---

### Matrix (float64)

Double-precision floating-point matrix. This is the default and recommended class for most linear algebra operations.

```python
import LinAlgKit as lk

# Create a 3x3 matrix filled with zeros
A = lk.Matrix(3, 3)

# Create a 2x4 matrix filled with value 5.0
B = lk.Matrix(2, 4, 5.0)
```

**Precision:** 64-bit floating point (~15-17 significant decimal digits)

---

### MatrixF (float32)

Single-precision floating-point matrix. Useful for memory-constrained applications or GPU interoperability.

```python
import LinAlgKit as lk

# Create a single-precision matrix
A = lk.MatrixF(3, 3, 1.0)
```

**Precision:** 32-bit floating point (~6-9 significant decimal digits)

---

### MatrixI (int)

Integer matrix. Ideal for discrete mathematics, graph adjacency matrices, or counting operations.

```python
import LinAlgKit as lk

# Create an integer matrix
A = lk.MatrixI(3, 3, 1)
```

**Precision:** Platform-dependent integer (typically 32 or 64 bits)

---

## Matrix Methods

### Constructors

#### `Matrix(rows, cols, value=0.0)`

Creates a new matrix with the specified dimensions, filled with a constant value.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rows` | `int` | Required | Number of rows |
| `cols` | `int` | Required | Number of columns |
| `value` | `float` | `0.0` | Fill value for all elements |

**Returns:** `Matrix` — A new matrix instance

**Example:**
```python
import LinAlgKit as lk

# 3x3 zero matrix
zeros = lk.Matrix(3, 3)

# 2x5 matrix filled with 7.5
filled = lk.Matrix(2, 5, 7.5)

# Empty matrix (0x0)
empty = lk.Matrix()
```

**Time Complexity:** O(rows × cols)

---

### Static Constructors

#### `Matrix.identity(size)`

Creates a square identity matrix of the specified size.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `size` | `int` | Dimension of the square matrix |

**Returns:** `Matrix` — Identity matrix with 1s on diagonal, 0s elsewhere

**Mathematical Definition:**
```
I[i,j] = 1 if i == j
I[i,j] = 0 if i != j
```

**Example:**
```python
I = lk.Matrix.identity(3)
print(I.to_numpy())
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]
```

**Time Complexity:** O(n²) where n = size

---

#### `Matrix.zeros(rows, cols)`

Creates a matrix filled with zeros.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `rows` | `int` | Number of rows |
| `cols` | `int` | Number of columns |

**Returns:** `Matrix` — Zero matrix

**Example:**
```python
Z = lk.Matrix.zeros(2, 3)
print(Z.to_numpy())
# [[0. 0. 0.]
#  [0. 0. 0.]]
```

**Time Complexity:** O(rows × cols)

---

#### `Matrix.ones(rows, cols)`

Creates a matrix filled with ones.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `rows` | `int` | Number of rows |
| `cols` | `int` | Number of columns |

**Returns:** `Matrix` — Matrix of ones

**Example:**
```python
O = lk.Matrix.ones(2, 3)
print(O.to_numpy())
# [[1. 1. 1.]
#  [1. 1. 1.]]
```

**Time Complexity:** O(rows × cols)

---

### Properties

#### `rows`

Returns the number of rows in the matrix.

**Type:** `int` (read-only)

**Example:**
```python
A = lk.Matrix(3, 5)
print(A.rows)  # 3
```

---

#### `cols`

Returns the number of columns in the matrix.

**Type:** `int` (read-only)

**Example:**
```python
A = lk.Matrix(3, 5)
print(A.cols)  # 5
```

---

### NumPy Interoperability

#### `Matrix.from_numpy(arr)`

Creates a matrix from a 2D NumPy array.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `arr` | `numpy.ndarray` | 2D NumPy array |

**Returns:** `Matrix` — New matrix containing a copy of the array data

**Raises:**
- `ValueError` — If the array is not 2-dimensional

**Example:**
```python
import numpy as np
import LinAlgKit as lk

# From a NumPy array
arr = np.array([[1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0]])
A = lk.Matrix.from_numpy(arr)

# Works with any array-like
B = lk.Matrix.from_numpy([[1, 2], [3, 4]])
```

**Note:** This method creates a **copy** of the data. Modifications to the original array will not affect the matrix.

**Time Complexity:** O(rows × cols)

---

#### `to_numpy()`

Converts the matrix to a 2D NumPy array.

**Returns:** `numpy.ndarray` — A copy of the matrix data as a 2D array

**Example:**
```python
A = lk.Matrix.identity(2)
arr = A.to_numpy()
print(type(arr))  # <class 'numpy.ndarray'>
print(arr)
# [[1. 0.]
#  [0. 1.]]
```

**Note:** This method returns a **copy**. Modifications to the returned array will not affect the original matrix.

**Time Complexity:** O(rows × cols)

---

### Arithmetic Operations

#### `__add__(other)` — Matrix Addition

Adds two matrices element-wise.

**Operator:** `A + B`

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `other` | `Matrix` | Matrix to add (must have same dimensions) |

**Returns:** `Matrix` — Element-wise sum

**Mathematical Definition:**
```
C[i,j] = A[i,j] + B[i,j]
```

**Example:**
```python
A = lk.Matrix.from_numpy([[1, 2], [3, 4]])
B = lk.Matrix.from_numpy([[5, 6], [7, 8]])
C = A + B
print(C.to_numpy())
# [[ 6.  8.]
#  [10. 12.]]
```

**Time Complexity:** O(rows × cols)

---

#### `__sub__(other)` — Matrix Subtraction

Subtracts one matrix from another element-wise.

**Operator:** `A - B`

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `other` | `Matrix` | Matrix to subtract (must have same dimensions) |

**Returns:** `Matrix` — Element-wise difference

**Mathematical Definition:**
```
C[i,j] = A[i,j] - B[i,j]
```

**Example:**
```python
A = lk.Matrix.from_numpy([[5, 6], [7, 8]])
B = lk.Matrix.from_numpy([[1, 2], [3, 4]])
C = A - B
print(C.to_numpy())
# [[4. 4.]
#  [4. 4.]]
```

**Time Complexity:** O(rows × cols)

---

#### `__mul__(other)` — Matrix Multiplication / Scalar Multiplication

Performs matrix multiplication (if `other` is a Matrix) or scalar multiplication (if `other` is a number).

**Operator:** `A * B` or `A * scalar`

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `other` | `Matrix` or `Number` | Matrix or scalar to multiply |

**Returns:** `Matrix` — Product

**Matrix Multiplication (A * B):**
```
C[i,j] = Σ(k=0 to n-1) A[i,k] * B[k,j]
```
- Requires: A.cols == B.rows
- Result shape: (A.rows, B.cols)

**Scalar Multiplication (A * s):**
```
C[i,j] = A[i,j] * s
```

**Example:**
```python
# Matrix multiplication
A = lk.Matrix.from_numpy([[1, 2], [3, 4]])
B = lk.Matrix.from_numpy([[5, 6], [7, 8]])
C = A * B
print(C.to_numpy())
# [[19. 22.]
#  [43. 50.]]

# Scalar multiplication
D = A * 2
print(D.to_numpy())
# [[2. 4.]
#  [6. 8.]]
```

**Time Complexity:**
- Matrix multiplication: O(n³) for n×n matrices
- Scalar multiplication: O(rows × cols)

---

#### `__rmul__(scalar)` — Left Scalar Multiplication

Multiplies a matrix by a scalar on the left.

**Operator:** `scalar * A`

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `scalar` | `Number` | Scalar multiplier |

**Returns:** `Matrix` — Scaled matrix

**Example:**
```python
A = lk.Matrix.from_numpy([[1, 2], [3, 4]])
B = 3 * A
print(B.to_numpy())
# [[3. 6.]
#  [9. 12.]]
```

**Time Complexity:** O(rows × cols)

---

### Matrix Operations

#### `transpose()`

Returns the transpose of the matrix.

**Returns:** `Matrix` — Transposed matrix

**Mathematical Definition:**
```
B[i,j] = A[j,i]
```

**Example:**
```python
A = lk.Matrix.from_numpy([[1, 2, 3],
                          [4, 5, 6]])
AT = A.transpose()
print(AT.to_numpy())
# [[1. 4.]
#  [2. 5.]
#  [3. 6.]]
```

**Time Complexity:** O(rows × cols)

---

#### `trace()`

Computes the trace (sum of diagonal elements) of a matrix.

**Returns:** `Number` — Sum of diagonal elements

**Mathematical Definition:**
```
trace(A) = Σ(i=0 to min(rows,cols)-1) A[i,i]
```

**Example:**
```python
A = lk.Matrix.from_numpy([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]])
print(A.trace())  # 15.0 (1 + 5 + 9)
```

**Time Complexity:** O(min(rows, cols))

---

#### `determinant()`

Computes the determinant of a square matrix using LU decomposition.

**Returns:** `Number` — Determinant value

**Raises:**
- Undefined behavior for non-square matrices

**Mathematical Properties:**
- det(I) = 1 (identity matrix)
- det(AB) = det(A) × det(B)
- det(A^T) = det(A)
- det(cA) = c^n × det(A) for n×n matrix

**Example:**
```python
A = lk.Matrix.from_numpy([[1, 2],
                          [3, 4]])
print(A.determinant())  # -2.0

# Singular matrix
B = lk.Matrix.from_numpy([[1, 2],
                          [2, 4]])
print(B.determinant())  # 0.0
```

**Time Complexity:** O(n³) using LU decomposition

---

#### `determinant_naive()`

Computes the determinant using recursive cofactor expansion. Provided for educational purposes and small matrices.

**Returns:** `Number` — Determinant value

**Raises:**
- `ValueError` — If matrix is not square

**Warning:** This method has O(n!) time complexity. Use `determinant()` for matrices larger than 4×4.

**Example:**
```python
A = lk.Matrix.from_numpy([[1, 2],
                          [3, 4]])
print(A.determinant_naive())  # -2.0
```

**Time Complexity:** O(n!) — factorial, use only for small matrices

---

### Element Access

#### `__getitem__(idx)` — Get Element

Access a single element by row and column index.

**Operator:** `A[row, col]`

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `idx` | `Tuple[int, int]` | (row, column) indices, 0-indexed |

**Returns:** `Number` — Element value

**Example:**
```python
A = lk.Matrix.from_numpy([[1, 2, 3],
                          [4, 5, 6]])
print(A[0, 0])  # 1.0
print(A[1, 2])  # 6.0
```

---

#### `__setitem__(idx, value)` — Set Element

Set a single element by row and column index.

**Operator:** `A[row, col] = value`

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `idx` | `Tuple[int, int]` | (row, column) indices, 0-indexed |
| `value` | `Number` | New value |

**Example:**
```python
A = lk.Matrix.zeros(2, 2)
A[0, 0] = 1.0
A[1, 1] = 1.0
print(A.to_numpy())
# [[1. 0.]
#  [0. 1.]]
```

---

## Functional API

LinAlgKit also provides NumPy-compatible functional helpers for working directly with arrays.

### `array(a, dtype=None)`

Creates a NumPy array from nested lists.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | `Iterable[Iterable[Number]]` | Nested list of numbers |
| `dtype` | `str` or `None` | Optional data type |

**Returns:** `numpy.ndarray`

**Example:**
```python
from LinAlgKit import array
A = array([[1, 2], [3, 4]])
```

---

### `zeros(shape, dtype=None)`

Creates a zero-filled NumPy array.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `shape` | `Tuple[int, int]` | (rows, cols) |
| `dtype` | `str` or `None` | Optional data type |

**Returns:** `numpy.ndarray`

**Example:**
```python
from LinAlgKit import zeros
Z = zeros((3, 3))
```

---

### `ones(shape, dtype=None)`

Creates a ones-filled NumPy array.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `shape` | `Tuple[int, int]` | (rows, cols) |
| `dtype` | `str` or `None` | Optional data type |

**Returns:** `numpy.ndarray`

**Example:**
```python
from LinAlgKit import ones
O = ones((2, 4))
```

---

### `eye(n, dtype=None)`

Creates an identity matrix as a NumPy array.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Size of the square matrix |
| `dtype` | `str` or `None` | Optional data type |

**Returns:** `numpy.ndarray`

**Example:**
```python
from LinAlgKit import eye
I = eye(3)
```

---

### `matmul(a, b)`

Performs matrix multiplication on NumPy arrays.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | `array-like` | Left matrix |
| `b` | `array-like` | Right matrix |

**Returns:** `numpy.ndarray`

**Example:**
```python
from LinAlgKit import array, matmul
A = array([[1, 2], [3, 4]])
B = array([[5, 6], [7, 8]])
C = matmul(A, B)
print(C)
# [[19 22]
#  [43 50]]
```

---

### `transpose(a)`

Transposes a NumPy array.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | `array-like` | Input matrix |

**Returns:** `numpy.ndarray`

**Example:**
```python
from LinAlgKit import array, transpose
A = array([[1, 2, 3], [4, 5, 6]])
print(transpose(A))
# [[1 4]
#  [2 5]
#  [3 6]]
```

---

### `trace(a)`

Computes the trace of a NumPy array.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | `array-like` | Input matrix |

**Returns:** `float`

**Example:**
```python
from LinAlgKit import array, trace
A = array([[1, 2], [3, 4]])
print(trace(A))  # 5.0
```

---

### `det(a)`

Computes the determinant of a NumPy array.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | `array-like` | Input square matrix |

**Returns:** `float`

**Example:**
```python
from LinAlgKit import array, det
A = array([[1, 2], [3, 4]])
print(det(A))  # -2.0
```

---

## Examples

### Example 1: Solving a Linear System

```python
import LinAlgKit as lk
import numpy as np

# Solve Ax = b using NumPy's solver with LinAlgKit matrices
A = lk.Matrix.from_numpy([[3, 1], [1, 2]])
b = np.array([9, 8])

# Convert to NumPy for solving
x = np.linalg.solve(A.to_numpy(), b)
print(f"Solution: {x}")  # [2. 3.]
```

### Example 2: Matrix Powers

```python
import LinAlgKit as lk

A = lk.Matrix.from_numpy([[1, 1], [1, 0]])

# Compute A^n for Fibonacci numbers
def matrix_power(M, n):
    result = lk.Matrix.identity(M.rows)
    for _ in range(n):
        result = result * M
    return result

A8 = matrix_power(A, 8)
print(A8.to_numpy())
# [[34. 21.]
#  [21. 13.]]
# Fibonacci: 34 is F(9), 21 is F(8)
```

### Example 3: Checking Matrix Properties

```python
import LinAlgKit as lk
import numpy as np

A = lk.Matrix.from_numpy([[4, -2], [-2, 4]])

# Check if symmetric
AT = A.transpose()
is_symmetric = np.allclose(A.to_numpy(), AT.to_numpy())
print(f"Symmetric: {is_symmetric}")  # True

# Check if positive definite (all eigenvalues > 0)
eigenvalues = np.linalg.eigvals(A.to_numpy())
is_positive_definite = all(eigenvalues > 0)
print(f"Positive definite: {is_positive_definite}")  # True
```

---

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `ValueError: expected a 2D array` | Passed 1D or 3D+ array to `from_numpy()` | Reshape to 2D: `arr.reshape(m, n)` |
| `ValueError: determinant is only defined for square matrices` | Called `determinant_naive()` on non-square matrix | Use only on square matrices |
| Broadcasting errors | Mismatched dimensions in arithmetic | Ensure matrices have compatible shapes |

### Debugging Tips

```python
# Check matrix dimensions
print(f"Shape: {A.rows} x {A.cols}")

# View underlying data
print(A.to_numpy())

# Check data type
print(A.to_numpy().dtype)
```

---

## Version Information

```python
import LinAlgKit
print(LinAlgKit.__version__)  # "0.1.0"
print(LinAlgKit.BACKEND)      # "python"
```
