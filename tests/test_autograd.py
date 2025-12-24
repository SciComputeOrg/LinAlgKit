"""
Tests for LinAlgKit Autograd Engine

Tests automatic differentiation, Tensor operations, nn.Module, and optimizers.
"""
import pytest
import numpy as np
import sys
sys.path.insert(0, 'python_pkg')

import LinAlgKit as lk
from LinAlgKit import Tensor, nn, optim


class TestTensorBasics:
    """Test basic Tensor operations."""
    
    def test_tensor_creation(self):
        x = Tensor([1.0, 2.0, 3.0])
        assert x.shape == (3,)
        assert x.dtype == np.float32
    
    def test_tensor_requires_grad(self):
        x = Tensor([1.0, 2.0], requires_grad=True)
        assert x.requires_grad is True
        assert x.grad is None
    
    def test_tensor_add(self):
        x = Tensor([1.0, 2.0])
        y = Tensor([3.0, 4.0])
        z = x + y
        np.testing.assert_array_almost_equal(z.data, [4.0, 6.0])
    
    def test_tensor_mul(self):
        x = Tensor([1.0, 2.0])
        y = Tensor([3.0, 4.0])
        z = x * y
        np.testing.assert_array_almost_equal(z.data, [3.0, 8.0])
    
    def test_tensor_matmul(self):
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])
        y = Tensor([[5.0], [6.0]])
        z = x @ y
        np.testing.assert_array_almost_equal(z.data, [[17.0], [39.0]])


class TestAutograd:
    """Test automatic differentiation."""
    
    def test_simple_backward(self):
        """Test y = x * 2, dy/dx = 2"""
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x * 2
        z = y.sum()
        z.backward()
        
        np.testing.assert_array_almost_equal(x.grad, [2.0, 2.0, 2.0])
    
    def test_chain_backward(self):
        """Test y = (x + 1) * 2"""
        x = Tensor([1.0, 2.0], requires_grad=True)
        y = (x + 1) * 2
        z = y.sum()
        z.backward()
        
        # dy/dx = 2
        np.testing.assert_array_almost_equal(x.grad, [2.0, 2.0])
    
    def test_matmul_backward(self):
        """Test gradient for matrix multiplication."""
        x = Tensor([[1.0, 2.0]], requires_grad=True)  # (1, 2)
        w = Tensor([[3.0], [4.0]], requires_grad=True)  # (2, 1)
        y = x @ w  # (1, 1)
        y.backward()
        
        # dy/dx = w.T, dy/dw = x.T
        np.testing.assert_array_almost_equal(x.grad, [[3.0, 4.0]])
        np.testing.assert_array_almost_equal(w.grad, [[1.0], [2.0]])
    
    def test_relu_backward(self):
        """Test ReLU gradient."""
        x = Tensor([-1.0, 0.0, 1.0, 2.0], requires_grad=True)
        y = x.relu()
        z = y.sum()
        z.backward()
        
        np.testing.assert_array_almost_equal(x.grad, [0.0, 0.0, 1.0, 1.0])
    
    def test_sigmoid_backward(self):
        """Test sigmoid gradient."""
        x = Tensor([0.0], requires_grad=True)
        y = x.sigmoid()
        y.backward()
        
        # sigmoid'(0) = sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5 = 0.25
        np.testing.assert_array_almost_equal(x.grad, [0.25], decimal=5)
    
    def test_mse_loss_backward(self):
        """Test MSE loss gradient."""
        pred = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        target = Tensor([1.0, 2.0, 3.0])
        
        loss = lk.tensor_mse_loss(pred, target)
        loss.backward()
        
        # Perfect prediction, gradient should be 0
        np.testing.assert_array_almost_equal(pred.grad, [0.0, 0.0, 0.0])


class TestNumericalGradient:
    """Compare autograd with numerical gradients."""
    
    def test_numerical_vs_autograd(self):
        """Compare autograd gradients with numerical gradients."""
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        
        # Forward
        y = (x * x).sum()  # y = sum(x^2)
        y.backward()
        
        autograd_grad = x.grad
        
        # Numerical gradient: dx^2/dx = 2x
        expected_grad = 2 * x.data
        
        # Autograd should give exact results for simple polynomials
        np.testing.assert_array_almost_equal(autograd_grad, expected_grad, decimal=4)


class TestNNModule:
    """Test neural network modules."""
    
    def test_linear_layer(self):
        """Test Linear layer forward pass."""
        layer = nn.Linear(10, 5)
        x = Tensor(np.random.randn(32, 10).astype(np.float32))
        y = layer(x)
        
        assert y.shape == (32, 5)
    
    def test_linear_backward(self):
        """Test Linear layer backward pass."""
        layer = nn.Linear(3, 2)
        x = Tensor(np.ones((1, 3), dtype=np.float32), requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        
        assert layer.weight.grad is not None
        assert layer.bias.grad is not None
    
    def test_sequential(self):
        """Test Sequential container."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        
        x = Tensor(np.random.randn(32, 10).astype(np.float32))
        y = model(x)
        
        assert y.shape == (32, 5)
    
    def test_parameters(self):
        """Test parameter iteration."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 5)
        )
        
        params = list(model.parameters())
        assert len(params) == 4  # 2 weights + 2 biases
    
    def test_state_dict(self):
        """Test saving and loading state dict."""
        model = nn.Linear(10, 5)
        state = model.state_dict()
        
        # Create new model and load state
        model2 = nn.Linear(10, 5)
        model2.load_state_dict(state)
        
        np.testing.assert_array_equal(model.weight.data, model2.weight.data)
        np.testing.assert_array_equal(model.bias.data, model2.bias.data)


class TestOptimizers:
    """Test optimizers."""
    
    def test_sgd_step(self):
        """Test SGD optimizer step with proper backward."""
        layer = nn.Linear(3, 1)
        x = Tensor(np.ones((1, 3), dtype=np.float32), requires_grad=True)
        
        optimizer = optim.SGD(layer.parameters(), lr=0.1)
        
        # Forward and backward
        y = layer(x)
        y.backward()
        
        # Save old weight
        old_weight = layer.weight.data.copy()
        
        # Step
        optimizer.step()
        
        # Weight should have changed
        assert not np.array_equal(old_weight, layer.weight.data)
    
    def test_adam_step(self):
        """Test Adam optimizer step with proper backward."""
        layer = nn.Linear(3, 1)
        x = Tensor(np.ones((1, 3), dtype=np.float32), requires_grad=True)
        
        optimizer = optim.Adam(layer.parameters(), lr=0.001)
        
        # Forward and backward
        y = layer(x)
        y.backward()
        
        # Save old weight
        old_weight = layer.weight.data.copy()
        
        # Step
        optimizer.step()
        
        # Weight should have changed
        assert not np.array_equal(old_weight, layer.weight.data)
    
    def test_zero_grad(self):
        """Test zero_grad clears gradients."""
        layer = nn.Linear(3, 2)
        x = Tensor(np.ones((1, 3), dtype=np.float32), requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        
        assert layer.weight.grad is not None
        
        optimizer = optim.SGD(layer.parameters(), lr=0.01)
        optimizer.zero_grad()
        
        for param in layer.parameters():
            assert param.grad is None


class TestTrainingLoop:
    """Test a simple training loop."""
    
    def test_xor_problem(self):
        """Train a network to solve XOR."""
        np.random.seed(42)
        
        # XOR data
        X = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        y = Tensor([[0], [1], [1], [0]], dtype=np.float32)
        
        # Simple network
        model = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
        optimizer = optim.Adam(model.parameters(), lr=0.1)
        
        # Train
        for _ in range(100):
            optimizer.zero_grad()
            
            pred = model(X)
            loss = ((pred - y) ** 2).mean()
            loss.backward()
            
            optimizer.step()
        
        # Should have learned XOR (loss should be low)
        final_pred = model(X)
        final_loss = ((final_pred - y) ** 2).mean()
        
        assert final_loss.item() < 0.1  # Loss should be under 0.1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
