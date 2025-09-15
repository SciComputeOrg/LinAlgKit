# Performance Guide

This page explains the performance characteristics of LinAlgKit and how to reproduce benchmark results.

## Determinant algorithms

- `determinant()`: Bareiss fraction-free LU with partial pivoting; O(n^3), stable for integer matrices.
- `determinant_naive()`: Laplace expansion; O(n!) — only for tiny matrices/testing.

## Running benchmarks

Build with benchmarks enabled and run the harness:

```bash
mkdir -p ~/matrixlib_build && cd ~/matrixlib_build
cmake -G "Unix Makefiles" -DBUILD_BENCHMARKS=ON /path/to/repo
cmake --build . -j
python3 /path/to/repo/scripts/run_benchmarks.py --build-dir . --csv results.csv --plot results.png
```

The harness saves a CSV from Google Benchmark and optionally a simple bar plot (if matplotlib is installed).

## Interpreting benchmark output

- `real_time` is the per-iteration runtime (ns), averaged over repetitions.
- Determinant (optimized) should scale roughly with O(n^3).
- Determinant (naive) grows factorially; use only up to n≈8.

## Tips

- Build `Release` for meaningful numbers.
- Avoid running on a heavily loaded system.
- For multi-run studies, pin CPU frequency/governor and isolate cores if possible.
