# Tutorial: Python usage with NumPy interop

This tutorial shows how to use LinAlgKit from Python, including NumPy interop.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip numpy
# If not installed already, install LinAlgKit from source or PyPI
# pip install LinAlgKit
```

## Creating matrices from NumPy

```python
import numpy as np
import LinAlgKit as lk

A = lk.Matrix.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]]))
B = lk.Matrix.from_numpy(np.eye(2))
print("A:\n", A.to_numpy())
print("B:\n", B.to_numpy())
```

## Basic operations

```python
C = A + B
D = A - B
E = A * B  # matrix-matrix
F = A * 2.0  # scalar on right
G = 3.0 * A  # scalar on left via __rmul__

print("C = A + B:\n", C.to_numpy())
print("D = A - B:\n", D.to_numpy())
print("E = A * B:\n", E.to_numpy())
print("F = A * 2:\n", F.to_numpy())
print("G = 3 * A:\n", G.to_numpy())
```

## Transpose, trace, determinant

```python
AT = A.transpose()
print("A^T:\n", AT.to_numpy())
print("trace(A):", A.trace())
print("det(A):", A.determinant())  # Bareiss (fraction-free LU)
```

## Integer and float matrices

```python
Ai = lk.MatrixI.from_numpy(np.array([[2, 0], [0, 2]], dtype=np.int32))
Af = lk.MatrixF.from_numpy(np.ones((3, 3), dtype=np.float32))
print("Ai trace:", Ai.trace())
print("Af shape:", Af.rows, Af.cols)
```

## Converting back to NumPy

```python
npA = A.to_numpy()
print(type(npA), npA.shape)
```

## Notes

- `from_numpy` and `to_numpy` copy data in both directions.
- Currently, zero‑copy views are not supported due to the internal storage layout.
- `inverse()` is implemented for 2x2 matrices; larger sizes will use LU in a future update.
