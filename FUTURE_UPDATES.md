# Future Updates Roadmap

This document outlines planned features, improvements, and long-term goals for LinAlgKit.

---

## 🎯 Version 0.2.0 (Planned)

**Target:** Q1 2025

### New Features

#### Matrix Inverse
- `inverse()` — Compute matrix inverse using LU decomposition
- `solve(b)` — Solve linear system Ax = b
- Pseudo-inverse for non-square matrices

#### Matrix Decompositions
- `lu()` — LU decomposition with partial pivoting
- `qr()` — QR decomposition using Householder reflections
- `cholesky()` — Cholesky decomposition for positive-definite matrices

#### Eigenvalue Operations
- `eigenvalues()` — Compute eigenvalues
- `eigenvectors()` — Compute eigenvectors
- `eig()` — Combined eigenvalue/eigenvector computation

### Enhancements

#### Broadcasting Support
```python
# Planned: element-wise operations with broadcasting
A = lk.Matrix(3, 3, 1.0)
B = A + 5  # Add scalar to all elements
```

#### Slicing Support
```python
# Planned: NumPy-style slicing
A = lk.Matrix.identity(5)
sub = A[1:4, 1:4]  # Extract 3x3 submatrix
```

#### In-place Operations
```python
# Planned: memory-efficient in-place operations
A += B  # In-place addition
A *= 2  # In-place scalar multiplication
```

---

## 🎯 Version 0.3.0 (Planned)

**Target:** Q2 2025

### Sparse Matrix Support

#### New Classes
- `SparseMatrix` — CSR (Compressed Sparse Row) format
- `SparseCOO` — Coordinate format for construction
- `SparseDiag` — Diagonal sparse matrix

#### Features
```python
# Planned API
S = lk.SparseMatrix.from_csr(data, indices, indptr, shape)
S = lk.SparseMatrix.from_dense(dense_matrix, threshold=1e-10)
dense = S.to_dense()
```

### Matrix Norms

#### Frobenius Norm
```python
# Planned
norm_f = A.norm('fro')  # sqrt(sum of squares)
```

#### Operator Norms
```python
# Planned
norm_1 = A.norm(1)      # Maximum column sum
norm_inf = A.norm('inf')  # Maximum row sum
norm_2 = A.norm(2)      # Spectral norm (largest singular value)
```

### Condition Number
```python
# Planned
cond = A.condition_number()  # Ratio of largest to smallest singular value
```

---

## 🎯 Version 0.4.0 (Planned)

**Target:** Q3 2025

### Singular Value Decomposition (SVD)

```python
# Planned API
U, S, Vt = A.svd()
U, S, Vt = A.svd(full_matrices=False)  # Reduced SVD
```

### Low-Rank Approximation

```python
# Planned
A_approx = A.low_rank_approximation(rank=5)
```

### Matrix Functions

#### Matrix Exponential
```python
# Planned
exp_A = A.exp()  # e^A using Padé approximation
```

#### Matrix Logarithm
```python
# Planned
log_A = A.log()
```

#### Matrix Power
```python
# Planned
A_n = A.power(5)      # A^5
A_half = A.power(0.5)  # Matrix square root
```

---

## 🎯 Version 0.5.0 (Planned)

**Target:** Q4 2025

### GPU Acceleration (Optional Backend)

```python
# Planned API
import LinAlgKit as lk
lk.set_backend('cuda')  # Use GPU

A = lk.Matrix.from_numpy(large_array)
B = A * A  # Computed on GPU
```

#### Supported Operations
- All basic arithmetic
- Matrix multiplication
- Decompositions (LU, QR, SVD)
- Eigenvalue computation

### Parallel CPU Operations

```python
# Planned
lk.set_threads(8)  # Use 8 CPU threads for operations
```

---

## 🎯 Version 1.0.0 (Long-term Goal)

**Target:** 2026

### Feature Complete Release

- All planned features implemented and stable
- Comprehensive test coverage (>95%)
- Performance benchmarks against NumPy/SciPy
- Complete documentation with tutorials

### API Stability
- Stable API with backwards compatibility guarantees
- Semantic versioning strictly enforced
- Deprecation warnings for any breaking changes

### Additional Goals
- SciPy interoperability
- Jupyter notebook integration
- Interactive documentation

---

## 🔧 Technical Improvements (Ongoing)

### Performance
- [ ] Optimize memory layout for cache efficiency
- [ ] Implement SIMD operations for element-wise functions
- [ ] Add optional Numba JIT compilation
- [ ] Benchmark suite with automated performance regression tests

### Testing
- [ ] Property-based testing with Hypothesis
- [ ] Numerical accuracy tests against reference implementations
- [ ] Cross-platform CI (ARM, M1/M2 Mac)
- [ ] Memory leak detection

### Documentation
- [ ] Interactive API examples (Jupyter notebooks)
- [ ] Video tutorials
- [ ] Comparison guides (vs NumPy, SciPy, TensorFlow)
- [ ] Academic paper / citation

### Developer Experience
- [ ] Type stubs for mypy
- [ ] VS Code extension for matrix visualization
- [ ] Pre-commit hooks
- [ ] Contribution guidelines expansion

---

## 💡 Community Requested Features

Have a feature request? Open an issue on GitHub:
https://github.com/SciComputeOrg/LinAlgKit/issues

### Under Consideration
- Complex number support (`MatrixC`)
- Quaternion matrices for 3D graphics
- Symbolic matrix operations
- Interval arithmetic for verified computing
- Integration with SymPy

---

## 📊 Version Comparison

| Feature | 0.1.0 | 0.2.0 | 0.3.0 | 0.4.0 | 0.5.0 | 1.0.0 |
|---------|-------|-------|-------|-------|-------|-------|
| Basic ops | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| NumPy interop | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Inverse/Solve | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Decompositions | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Sparse matrices | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| SVD | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| GPU support | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| Stable API | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Priority areas:**
1. Bug fixes and documentation improvements
2. Test coverage expansion
3. Performance optimizations
4. Feature implementations from this roadmap

---

*Last updated: December 2025*
