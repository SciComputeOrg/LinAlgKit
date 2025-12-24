"""
Tests for Phase 1 linear algebra operations.
Tests decompositions, eigenvalues, solvers, and norms.
"""
import numpy as np
import pytest
import LinAlgKit as lk


class TestDecompositions:
    """Test matrix decomposition methods."""
    
    def test_lu_decomposition(self):
        """Test LU decomposition with partial pivoting."""
        A = lk.Matrix.from_numpy([[2., 1., 1.],
                                   [4., 3., 3.],
                                   [8., 7., 9.]])
        P, L, U = A.lu()
        
        # Verify A = P @ L @ U (scipy.linalg.lu returns P, L, U such that A = P @ L @ U)
        reconstructed = P.to_numpy() @ L.to_numpy() @ U.to_numpy()
        np.testing.assert_allclose(reconstructed, A.to_numpy(), atol=1e-10)
        
        # L should be lower triangular with ones on diagonal
        L_np = L.to_numpy()
        assert np.allclose(np.tril(L_np), L_np)
        
        # U should be upper triangular
        U_np = U.to_numpy()
        assert np.allclose(np.triu(U_np), U_np)
    
    def test_qr_decomposition(self):
        """Test QR decomposition."""
        A = lk.Matrix.from_numpy([[1., 2.],
                                   [3., 4.],
                                   [5., 6.]])
        Q, R = A.qr()
        
        # Verify A = Q @ R
        reconstructed = Q.to_numpy() @ R.to_numpy()
        np.testing.assert_allclose(reconstructed, A.to_numpy(), atol=1e-10)
        
        # Q should be orthogonal (Q.T @ Q = I)
        Q_np = Q.to_numpy()
        np.testing.assert_allclose(Q_np.T @ Q_np, np.eye(Q_np.shape[1]), atol=1e-10)
        
        # R should be upper triangular
        R_np = R.to_numpy()
        assert np.allclose(np.triu(R_np), R_np)
    
    def test_cholesky_decomposition(self):
        """Test Cholesky decomposition for positive-definite matrices."""
        # Create a positive-definite matrix
        A_np = np.array([[4., 2.], [2., 5.]])
        A = lk.Matrix.from_numpy(A_np)
        L = A.cholesky()
        
        # Verify A = L @ L.T
        reconstructed = L.to_numpy() @ L.to_numpy().T
        np.testing.assert_allclose(reconstructed, A_np, atol=1e-10)
        
        # L should be lower triangular
        L_np = L.to_numpy()
        assert np.allclose(np.tril(L_np), L_np)
    
    def test_svd(self):
        """Test Singular Value Decomposition."""
        A = lk.Matrix.from_numpy([[1., 2., 3.],
                                   [4., 5., 6.]])
        U, S, Vt = A.svd(full_matrices=False)  # Use reduced SVD for easier reconstruction
        
        # Verify A = U @ diag(S) @ Vt
        reconstructed = U.to_numpy() @ np.diag(S) @ Vt.to_numpy()
        np.testing.assert_allclose(reconstructed, A.to_numpy(), atol=1e-10)
        
        # U should have orthonormal columns
        U_np = U.to_numpy()
        np.testing.assert_allclose(U_np.T @ U_np, np.eye(U_np.shape[1]), atol=1e-10)
        
        # Singular values should be non-negative and sorted descending
        assert all(S >= 0)
        assert all(S[:-1] >= S[1:])


class TestEigenvalues:
    """Test eigenvalue methods."""
    
    def test_eig(self):
        """Test eigenvalue decomposition."""
        A = lk.Matrix.from_numpy([[4., 2.],
                                   [1., 3.]])
        eigenvalues, eigenvectors = A.eig()
        
        # Verify A @ v = lambda * v for each eigenvalue/eigenvector pair
        A_np = A.to_numpy()
        V = eigenvectors.to_numpy()
        for i, lam in enumerate(eigenvalues):
            v = V[:, i]
            np.testing.assert_allclose(A_np @ v, lam * v, atol=1e-10)
    
    def test_eigvals(self):
        """Test eigenvalues only."""
        A = lk.Matrix.from_numpy([[4., 2.],
                                   [1., 3.]])
        eigenvalues = A.eigvals()
        
        # Should match numpy
        expected = np.linalg.eigvals(A.to_numpy())
        np.testing.assert_allclose(sorted(eigenvalues), sorted(expected), atol=1e-10)
    
    def test_eigh_symmetric(self):
        """Test eigenvalue decomposition for symmetric matrices."""
        # Symmetric positive-definite matrix
        A = lk.Matrix.from_numpy([[5., 2.],
                                   [2., 3.]])
        eigenvalues, eigenvectors = A.eigh()
        
        # Eigenvalues should be real (which they are, as numpy array of floats)
        assert eigenvalues.dtype in [np.float64, np.float32]
        
        # Eigenvectors should be orthonormal
        V = eigenvectors.to_numpy()
        np.testing.assert_allclose(V.T @ V, np.eye(2), atol=1e-10)


class TestSolvers:
    """Test linear system solvers."""
    
    def test_solve(self):
        """Test solving Ax = b."""
        A = lk.Matrix.from_numpy([[3., 1.],
                                   [1., 2.]])
        b = np.array([[9.], [8.]])
        
        x = A.solve(b)
        
        # Verify A @ x = b
        result = A.to_numpy() @ x.to_numpy()
        np.testing.assert_allclose(result, b, atol=1e-10)
    
    def test_inv(self):
        """Test matrix inverse."""
        A = lk.Matrix.from_numpy([[1., 2.],
                                   [3., 4.]])
        A_inv = A.inv()
        
        # Verify A @ A_inv = I
        product = A.to_numpy() @ A_inv.to_numpy()
        np.testing.assert_allclose(product, np.eye(2), atol=1e-10)
    
    def test_pinv(self):
        """Test Moore-Penrose pseudoinverse."""
        # Non-square matrix
        A = lk.Matrix.from_numpy([[1., 2.],
                                   [3., 4.],
                                   [5., 6.]])
        A_pinv = A.pinv()
        
        # Verify A @ A_pinv @ A = A
        A_np = A.to_numpy()
        result = A_np @ A_pinv.to_numpy() @ A_np
        np.testing.assert_allclose(result, A_np, atol=1e-10)
    
    def test_lstsq(self):
        """Test least-squares solution."""
        # Overdetermined system
        A = lk.Matrix.from_numpy([[1., 1.],
                                   [1., 2.],
                                   [1., 3.]])
        b = np.array([[1.], [2.], [2.]])
        
        x, residuals, rank, s = A.lstsq(b)
        
        # Rank should be 2 (full rank for 3x2 matrix)
        assert rank == 2


class TestNormsAndConditions:
    """Test matrix norms and condition numbers."""
    
    def test_frobenius_norm(self):
        """Test Frobenius norm (default)."""
        A = lk.Matrix.from_numpy([[1., 2.],
                                   [3., 4.]])
        norm = A.norm()  # Default is Frobenius
        expected = np.sqrt(1 + 4 + 9 + 16)  # sqrt(30)
        np.testing.assert_allclose(norm, expected, atol=1e-10)
    
    def test_spectral_norm(self):
        """Test spectral norm (2-norm)."""
        A = lk.Matrix.from_numpy([[1., 0.],
                                   [0., 2.]])
        norm = A.norm(ord=2)
        assert np.isclose(norm, 2.0)  # Largest singular value
    
    def test_condition_number(self):
        """Test condition number."""
        A = lk.Matrix.from_numpy([[1., 0.],
                                   [0., 2.]])
        cond = A.cond()
        assert np.isclose(cond, 2.0)  # max(s) / min(s) = 2/1
        
        # Identity matrix should have condition number 1
        I = lk.Matrix.identity(3)
        assert np.isclose(I.cond(), 1.0)
    
    def test_rank(self):
        """Test matrix rank computation."""
        # Full rank
        A = lk.Matrix.from_numpy([[1., 2.],
                                   [3., 4.]])
        assert A.rank() == 2
        
        # Rank deficient
        B = lk.Matrix.from_numpy([[1., 2.],
                                   [2., 4.]])
        assert B.rank() == 1
        
        # Zero matrix
        Z = lk.Matrix.zeros(3, 3)
        assert Z.rank() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
