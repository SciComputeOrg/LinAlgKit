"""
Tests for deep learning mathematical functions.
"""
import numpy as np
import pytest
import LinAlgKit as lk


class TestActivationFunctions:
    """Test activation functions."""
    
    def test_sigmoid(self):
        x = np.array([-2, -1, 0, 1, 2], dtype=float)
        output = lk.sigmoid(x)
        assert output.shape == x.shape
        assert np.all(output > 0) and np.all(output < 1)
        assert np.isclose(output[2], 0.5)  # sigmoid(0) = 0.5
    
    def test_relu(self):
        x = np.array([-2, -1, 0, 1, 2], dtype=float)
        output = lk.relu(x)
        expected = np.array([0, 0, 0, 1, 2], dtype=float)
        np.testing.assert_array_equal(output, expected)
    
    def test_leaky_relu(self):
        x = np.array([-2, -1, 0, 1, 2], dtype=float)
        output = lk.leaky_relu(x, alpha=0.1)
        expected = np.array([-0.2, -0.1, 0, 1, 2], dtype=float)
        np.testing.assert_allclose(output, expected)
    
    def test_softmax_sums_to_one(self):
        x = np.array([[2.0, 1.0, 0.1]])
        output = lk.softmax(x)
        assert np.isclose(np.sum(output), 1.0)
    
    def test_gelu(self):
        x = np.array([0.0])
        output = lk.gelu(x)
        assert np.isclose(output[0], 0.0)
    
    def test_tanh_range(self):
        x = np.array([-10, 0, 10], dtype=float)
        output = lk.tanh(x)
        assert np.all(output >= -1) and np.all(output <= 1)


class TestLossFunctions:
    """Test loss functions."""
    
    def test_mse_loss(self):
        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([1.0, 2.0, 3.0])
        loss = lk.mse_loss(pred, target)
        assert loss == 0.0
    
    def test_mse_loss_nonzero(self):
        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([2.0, 3.0, 4.0])
        loss = lk.mse_loss(pred, target)
        assert loss == 1.0  # (1^2 + 1^2 + 1^2) / 3 = 1
    
    def test_mae_loss(self):
        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([2.0, 3.0, 4.0])
        loss = lk.mae_loss(pred, target)
        assert loss == 1.0
    
    def test_binary_cross_entropy(self):
        pred = np.array([0.9, 0.1])
        target = np.array([1.0, 0.0])
        loss = lk.binary_cross_entropy(pred, target)
        assert loss < 0.2  # Should be low for correct predictions


class TestNormalization:
    """Test normalization functions."""
    
    def test_batch_norm(self):
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        output = lk.batch_norm(x)
        # After batch norm, mean along axis 0 should be ~0
        mean = np.mean(output, axis=0)
        np.testing.assert_allclose(mean, np.zeros(3), atol=1e-10)
    
    def test_layer_norm(self):
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        output = lk.layer_norm(x)
        # After layer norm, mean along axis -1 should be ~0
        mean = np.mean(output, axis=-1)
        np.testing.assert_allclose(mean, np.zeros(2), atol=1e-10)


class TestConvolutions:
    """Test convolution operations."""
    
    def test_max_pool2d(self):
        x = np.array([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12],
                      [13, 14, 15, 16]], dtype=float)
        output = lk.max_pool2d(x, kernel_size=2)
        expected = np.array([[6, 8], [14, 16]], dtype=float)
        np.testing.assert_array_equal(output, expected)
    
    def test_avg_pool2d(self):
        x = np.array([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12],
                      [13, 14, 15, 16]], dtype=float)
        output = lk.avg_pool2d(x, kernel_size=2)
        expected = np.array([[3.5, 5.5], [11.5, 13.5]], dtype=float)
        np.testing.assert_array_equal(output, expected)


class TestUtilities:
    """Test utility functions."""
    
    def test_one_hot(self):
        indices = np.array([0, 2, 1])
        output = lk.one_hot(indices, num_classes=3)
        expected = np.array([[1, 0, 0],
                             [0, 0, 1],
                             [0, 1, 0]], dtype=float)
        np.testing.assert_array_equal(output, expected)
    
    def test_clip(self):
        x = np.array([-5, 0, 5], dtype=float)
        output = lk.clip(x, -1, 1)
        expected = np.array([-1, 0, 1], dtype=float)
        np.testing.assert_array_equal(output, expected)
    
    def test_dropout_training(self):
        np.random.seed(42)
        x = np.ones((100, 100))
        output = lk.dropout(x, p=0.5, training=True)
        # Some values should be zero
        assert np.sum(output == 0) > 0
    
    def test_dropout_inference(self):
        x = np.ones((10, 10))
        output = lk.dropout(x, p=0.5, training=False)
        np.testing.assert_array_equal(output, x)


class TestInitialization:
    """Test weight initialization functions."""
    
    def test_xavier_uniform_shape(self):
        shape = (784, 256)
        weights = lk.xavier_uniform(shape)
        assert weights.shape == shape
    
    def test_he_normal_shape(self):
        shape = (256, 128)
        weights = lk.he_normal(shape)
        assert weights.shape == shape


class TestAdvancedMath:
    """Test advanced math functions."""
    
    def test_normalize(self):
        x = np.array([[3, 4], [6, 8]], dtype=float)
        output = lk.normalize(x, axis=1)
        norms = np.linalg.norm(output, axis=1)
        np.testing.assert_allclose(norms, [1, 1], atol=1e-10)
    
    def test_cosine_similarity(self):
        a = np.array([1, 0, 0], dtype=float)
        b = np.array([1, 0, 0], dtype=float)
        sim = lk.cosine_similarity(a, b)
        assert np.isclose(sim, 1.0)
        
        c = np.array([0, 1, 0], dtype=float)
        sim_orth = lk.cosine_similarity(a, c)
        assert np.isclose(sim_orth, 0.0)
    
    def test_euclidean_distance(self):
        a = np.array([0, 0, 0], dtype=float)
        b = np.array([3, 4, 0], dtype=float)
        dist = lk.euclidean_distance(a, b)
        assert np.isclose(dist, 5.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
