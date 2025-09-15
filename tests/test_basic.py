import numpy as np
import LinAlgKit as lk


def test_construct_and_numpy_roundtrip():
    A_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    A = lk.Matrix.from_numpy(A_np)
    B_np = A.to_numpy()
    assert B_np.shape == (2, 2)
    np.testing.assert_allclose(B_np, A_np)


def test_ops_add_mul_transpose():
    A = lk.Matrix.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]]))
    I = lk.Matrix.identity(2)
    C = A + I
    AT = A.transpose()
    # simple checks
    np.testing.assert_allclose(C.to_numpy(), np.array([[2.0, 2.0], [3.0, 5.0]]))
    np.testing.assert_allclose(AT.to_numpy(), np.array([[1.0, 3.0], [2.0, 4.0]]))


def test_determinant_trace():
    A = lk.Matrix.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert lk.trace(A.to_numpy()) == np.trace(A.to_numpy())
    det_val = A.determinant()
    assert np.isclose(det_val, np.linalg.det(A.to_numpy()))
