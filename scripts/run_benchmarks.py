#!/usr/bin/env python3
"""
Python benchmarks for LinAlgKit.
Compares LinAlgKit performance against NumPy and SciPy.
"""

import argparse
import csv
import time
import sys
from pathlib import Path

import numpy as np

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python_pkg"))
import LinAlgKit as lk

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import scipy.linalg
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def benchmark(func, name, iterations=10, warmup=2):
    """Run a benchmark and return average time in milliseconds."""
    # Warmup
    for _ in range(warmup):
        func()
    
    # Actual timing
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    return {"name": name, "avg_ms": avg_time, "std_ms": std_time, "iterations": iterations}


def run_all_benchmarks():
    """Run all benchmarks and return results."""
    results = []
    
    # Matrix sizes to test
    sizes = [100, 500, 1000]
    
    for size in sizes:
        print(f"\n--- Benchmarking size {size}x{size} ---")
        
        # Create test matrices
        A_np = np.random.randn(size, size)
        B_np = np.random.randn(size, size)
        A_lk = lk.Matrix.from_numpy(A_np)
        B_lk = lk.Matrix.from_numpy(B_np)
        
        # Matrix multiplication
        print(f"  Matrix multiplication...")
        result = benchmark(lambda: A_lk * B_lk, f"matmul_{size}x{size}_LinAlgKit")
        results.append(result)
        
        result = benchmark(lambda: A_np @ B_np, f"matmul_{size}x{size}_NumPy")
        results.append(result)
        
        # Matrix addition
        print(f"  Matrix addition...")
        result = benchmark(lambda: A_lk + B_lk, f"add_{size}x{size}_LinAlgKit")
        results.append(result)
        
        result = benchmark(lambda: A_np + B_np, f"add_{size}x{size}_NumPy")
        results.append(result)
        
        # Transpose
        print(f"  Transpose...")
        result = benchmark(lambda: A_lk.transpose(), f"transpose_{size}x{size}_LinAlgKit")
        results.append(result)
        
        result = benchmark(lambda: A_np.T.copy(), f"transpose_{size}x{size}_NumPy")
        results.append(result)
        
        # Determinant (only for smaller sizes)
        if size <= 500:
            print(f"  Determinant...")
            result = benchmark(lambda: A_lk.determinant(), f"det_{size}x{size}_LinAlgKit")
            results.append(result)
            
            result = benchmark(lambda: np.linalg.det(A_np), f"det_{size}x{size}_NumPy")
            results.append(result)
        
        # SVD (only for smaller sizes)
        if size <= 500:
            print(f"  SVD...")
            result = benchmark(lambda: A_lk.svd(), f"svd_{size}x{size}_LinAlgKit", iterations=5)
            results.append(result)
            
            result = benchmark(lambda: np.linalg.svd(A_np), f"svd_{size}x{size}_NumPy", iterations=5)
            results.append(result)
        
        # QR decomposition
        if size <= 500:
            print(f"  QR decomposition...")
            result = benchmark(lambda: A_lk.qr(), f"qr_{size}x{size}_LinAlgKit", iterations=5)
            results.append(result)
            
            result = benchmark(lambda: np.linalg.qr(A_np), f"qr_{size}x{size}_NumPy", iterations=5)
            results.append(result)
        
        # Inverse (only for smaller sizes)
        if size <= 500:
            print(f"  Matrix inverse...")
            result = benchmark(lambda: A_lk.inv(), f"inv_{size}x{size}_LinAlgKit", iterations=5)
            results.append(result)
            
            result = benchmark(lambda: np.linalg.inv(A_np), f"inv_{size}x{size}_NumPy", iterations=5)
            results.append(result)
    
    # Activation functions benchmark
    print(f"\n--- Benchmarking activation functions ---")
    x = np.random.randn(10000, 1000)
    
    activations = [
        ("sigmoid", lk.sigmoid),
        ("relu", lk.relu),
        ("softmax", lk.softmax),
        ("gelu", lk.gelu),
        ("tanh", lk.tanh),
    ]
    
    for name, func in activations:
        print(f"  {name}...")
        result = benchmark(lambda f=func: f(x), f"{name}_10Mx1")
        results.append(result)
    
    return results


def save_csv(results, output_path):
    """Save benchmark results to CSV."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'avg_ms', 'std_ms', 'iterations'])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved results to: {output_path}")


def plot_results(results, output_path):
    """Plot benchmark results."""
    if plt is None:
        print("matplotlib not available; skipping plots")
        return
    
    # Group by operation
    operations = {}
    for r in results:
        parts = r['name'].rsplit('_', 1)
        if len(parts) == 2:
            op = parts[0]
            lib = parts[1]
        else:
            op = r['name']
            lib = 'LinAlgKit'
        
        if op not in operations:
            operations[op] = {}
        operations[op][lib] = r['avg_ms']
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    ops = list(operations.keys())
    x = np.arange(len(ops))
    width = 0.35
    
    linalgkit_times = [operations[op].get('LinAlgKit', 0) for op in ops]
    numpy_times = [operations[op].get('NumPy', 0) for op in ops]
    
    bars1 = ax.barh(x - width/2, linalgkit_times, width, label='LinAlgKit', color='#2ecc71')
    bars2 = ax.barh(x + width/2, numpy_times, width, label='NumPy', color='#3498db')
    
    ax.set_xlabel('Time (ms)')
    ax.set_title('LinAlgKit vs NumPy Benchmarks')
    ax.set_yticks(x)
    ax.set_yticklabels(ops, fontsize=8)
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Run LinAlgKit benchmarks')
    parser.add_argument('--csv', type=Path, default=Path('benchmark_results.csv'),
                        help='Output CSV file')
    parser.add_argument('--plot', type=Path, default=None,
                        help='Output PNG plot (optional)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("LinAlgKit Benchmark Suite")
    print("=" * 60)
    print(f"LinAlgKit version: {lk.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Backend: {lk.BACKEND}")
    
    results = run_all_benchmarks()
    
    save_csv(results, args.csv)
    
    if args.plot:
        plot_results(results, args.plot)
    
    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"{'Operation':<40} {'Avg (ms)':<12} {'Std (ms)':<12}")
    print("-" * 64)
    for r in results:
        print(f"{r['name']:<40} {r['avg_ms']:<12.3f} {r['std_ms']:<12.3f}")


if __name__ == "__main__":
    main()
