# LinAlgKit Expansion Plan

**Goal**: Transform LinAlgKit into a comprehensive, high-performance scientific computing library that outperforms existing alternatives.

---

## Executive Summary

This plan outlines the expansion of LinAlgKit from a basic linear algebra toolkit to a full-featured scientific computing library with:
- **100+ mathematical functions** (basic → advanced)
- **Performance optimizations** targeting 2-5x speedup over NumPy for key operations
- **Multiple backend support** (CPU, GPU, distributed)
- **Memory efficiency** rivaling or exceeding competitors

---

## Phase 1: Core Linear Algebra (v0.2.0 - v0.3.0)

### 1.1 Matrix Decompositions

| Function | Description | Complexity | Priority |
|----------|-------------|------------|----------|
| `lu()` | LU decomposition with partial pivoting | O(n³) | HIGH |
| `qr()` | QR decomposition (Householder) | O(mn²) | HIGH |
| `cholesky()` | Cholesky for positive-definite | O(n³/3) | HIGH |
| `svd()` | Singular Value Decomposition | O(min(mn², m²n)) | HIGH |
| `schur()` | Schur decomposition | O(n³) | MEDIUM |
| `hessenberg()` | Hessenberg form | O(n³) | MEDIUM |

### 1.2 Eigenvalue Problems

| Function | Description | Priority |
|----------|-------------|----------|
| `eig()` | Eigenvalues and eigenvectors | HIGH |
| `eigvals()` | Eigenvalues only (faster) | HIGH |
| `eigh()` | Hermitian/symmetric eigenvalue | HIGH |
| `eigsh()` | Top-k eigenvalues (iterative) | MEDIUM |

### 1.3 Linear System Solvers

| Function | Description | Priority |
|----------|-------------|----------|
| `solve()` | Direct solver (LU-based) | HIGH |
| `lstsq()` | Least-squares solution | HIGH |
| `inv()` | Matrix inverse | HIGH |
| `pinv()` | Moore-Penrose pseudoinverse | HIGH |
| `solve_triangular()` | Fast triangular solver | MEDIUM |
| `solve_banded()` | Banded matrix solver | MEDIUM |

### 1.4 Matrix Norms & Conditions

| Function | Description | Priority |
|----------|-------------|----------|
| `norm()` | Frobenius, 1, 2, inf norms | HIGH |
| `cond()` | Condition number | HIGH |
| `rank()` | Matrix rank | HIGH |
| `nullspace()` | Null space basis | MEDIUM |
| `orth()` | Orthonormal basis | MEDIUM |

---

## Phase 2: Advanced Linear Algebra (v0.4.0 - v0.5.0)

### 2.1 Sparse Matrix Support

- `SparseCSR` - Compressed Sparse Row format
- `SparseCOO` - Coordinate format for construction
- `SparseDiag` - Diagonal matrix (O(n) storage)

### 2.2 Iterative Solvers

- `cg()` - Conjugate Gradient (SPD matrices)
- `gmres()` - General non-symmetric
- `bicgstab()` - Bi-Conjugate Gradient Stabilized
- `minres()` - Symmetric indefinite

### 2.3 Matrix Functions

- `expm()` - Matrix exponential
- `logm()` - Matrix logarithm
- `sqrtm()` - Matrix square root
- `funm()` - General matrix function

---

## Phase 3: Calculus & Optimization (v0.6.0 - v0.7.0)

- Numerical differentiation (gradient, jacobian, hessian)
- Numerical integration (quad, simps, trapz)
- Optimization (minimize, least_squares, linprog)
- Interpolation (interp1d, spline, rbf)

---

## Phase 4: Statistics & Random (v0.8.0)

- Descriptive statistics
- Probability distributions
- Statistical tests
- Random number generation

---

## Phase 5: Signal Processing (v0.9.0)

- FFT (1D, 2D, N-D)
- Digital filtering
- Spectral analysis

---

## Performance Optimization Strategy

1. **SIMD Vectorization** - 4-8x speedup for element-wise operations
2. **Cache-Optimized Algorithms** - Blocked matrix multiply
3. **Multi-threading** - OpenMP parallel execution
4. **Memory Layout Optimization** - Row/column major selection
5. **Lazy Evaluation** - Expression templates
6. **GPU Acceleration** - CUDA/ROCm backends

---

## Implementation Roadmap

| Version | Target | Key Features |
|---------|--------|--------------|
| v0.2.0 | Q1 2025 | LU, QR, Cholesky, solve, inv, norms |
| v0.3.0 | Q2 2025 | Eigenvalues, SVD, lstsq, pinv |
| v0.4.0 | Q3 2025 | Sparse matrices, iterative solvers |
| v0.5.0 | Q4 2025 | C++ core, BLAS integration, benchmarks |
| v0.6.0 | Q1 2026 | Calculus, optimization |
| v0.7.0 | Q2 2026 | Interpolation, fitting |
| v0.8.0 | Q3 2026 | Statistics |
| v0.9.0 | Q4 2026 | Signal processing |
| v1.0.0 | 2027 | GPU, distributed, production ready |

---

## Contributing

We welcome contributions! Priority areas:
1. Implementing functions from Phase 1
2. Writing comprehensive tests
3. Performance benchmarking
4. Documentation improvements

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
