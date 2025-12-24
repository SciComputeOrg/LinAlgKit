#!/usr/bin/env python3
"""Benchmark standard vs fast implementations."""
import numpy as np
import time
import sys
sys.path.insert(0, 'python_pkg')

import LinAlgKit as lk
from LinAlgKit import fast

print('=' * 60)
print('LinAlgKit Performance Benchmark')
print('=' * 60)
print(f'Numba available: {lk.HAS_NUMBA}')
print()

# Test data
x = np.random.randn(5000, 1000)

def bench(name, std_func, fast_func, runs=5):
    """Benchmark function."""
    # Warmup
    std_func(x)
    fast_func(x)
    
    # Standard
    start = time.perf_counter()
    for _ in range(runs):
        std_func(x)
    std_time = (time.perf_counter() - start) / runs * 1000
    
    # Fast
    start = time.perf_counter()
    for _ in range(runs):
        fast_func(x)
    fast_time = (time.perf_counter() - start) / runs * 1000
    
    speedup = std_time / fast_time if fast_time > 0 else 0
    print(f'{name:15} | Standard: {std_time:8.2f}ms | Fast: {fast_time:8.2f}ms | Speedup: {speedup:.1f}x')

print('Activation Functions (5M elements):')
print('-' * 60)
bench('sigmoid', lk.sigmoid, fast.fast_sigmoid)
bench('relu', lk.relu, fast.fast_relu)
bench('gelu', lk.gelu, fast.fast_gelu)
bench('tanh', lk.tanh, fast.fast_tanh)
bench('softplus', lk.softplus, fast.fast_softplus)
bench('leaky_relu', lambda x: lk.leaky_relu(x, 0.01), lambda x: fast.fast_leaky_relu(x, 0.01))
bench('elu', lambda x: lk.elu(x, 1.0), lambda x: fast.fast_elu(x, 1.0))
bench('swish', lambda x: lk.swish(x, 1.0), lambda x: fast.fast_swish(x, 1.0))

print()
print('Loss Functions (5M elements):')
print('-' * 60)
y = np.random.randn(5000, 1000)
bench('mse_loss', lambda _: lk.mse_loss(x, y), lambda _: fast.fast_mse_loss(x, y))
bench('mae_loss', lambda _: lk.mae_loss(x, y), lambda _: fast.fast_mae_loss(x, y))

print()
print('Matrix In-place Operations (1000x1000):')
print('-' * 60)
A_np = np.random.randn(1000, 1000)
B_np = np.random.randn(1000, 1000)

# Standard - creates new matrix
def std_add():
    A = lk.Matrix.from_numpy(A_np)
    B = lk.Matrix.from_numpy(B_np)
    return A + B

# In-place - no allocation
def fast_add():
    A = lk.Matrix.from_numpy(A_np)
    B = lk.Matrix.from_numpy(B_np)
    return A.add_(B)

bench('matrix_add', std_add, fast_add, runs=20)

print()
print('=' * 60)
print('Benchmark complete!')
