# API Reference (Python)

This reference documents the Python bindings exposed by `LinAlgKit` (via pybind11).

## Modules

- `LinAlgKit`
  - Exposes classes: `Matrix` (double), `MatrixF` (float), `MatrixI` (int)

## Common methods

All matrix types support the following operations unless noted.

- Constructor
  - `Matrix(rows: int, cols: int, value: number = 0)`
- Properties
  - `rows: int`
  - `cols: int`
- Basic ops
  - `transpose() -> Matrix`
  - `trace() -> number`
  - `determinant() -> number` (Bareiss/LU, O(n^3))
  - `determinant_naive() -> number` (tiny matrices only)
- Arithmetic
  - `__add__(other: Matrix) -> Matrix`
  - `__sub__(other: Matrix) -> Matrix`
  - `__mul__(other: Matrix) -> Matrix` (matrix multiplication)
  - `__rmul__(scalar: number) -> Matrix` (scalar on left)
  - `scale(scalar: number) -> Matrix` (alias for scalar multiply)
- Static constructors
  - `identity(size: int) -> Matrix`
  - `zeros(rows: int, cols: int) -> Matrix`
  - `ones(rows: int, cols: int) -> Matrix`

## NumPy interop

- `Matrix.from_numpy(arr: numpy.ndarray) -> Matrix`
  - Accepts 2D arrays; copies data into a new matrix.
- `Matrix.to_numpy() -> numpy.ndarray`
  - Returns a 2D array copy.

## Types

### Matrix (double)
Double-precision matrix. Exposed as `LinAlgKit.Matrix`.

### MatrixF (float)
Single-precision matrix. Exposed as `LinAlgKit.MatrixF`.

### MatrixI (int)
Integer matrix. Exposed as `LinAlgKit.MatrixI`.

## Notes

- `inverse()` is implemented for 2x2 matrices. Larger sizes will use LU in a future release.
- Current storage is not contiguous; NumPy zero-copy views are not supported.
