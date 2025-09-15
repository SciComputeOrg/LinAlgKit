#include <benchmark/benchmark.h>
#include "matrixlib.h"
#include <random>
#include <vector>

using namespace matrixlib;

// Helper function to generate a random matrix
Matrixd generate_random_matrix(size_t size) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    Matrixd m(size, size);
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            m[i][j] = dist(gen);
        }
    }
    return m;
}

// Benchmark for matrix addition
static void BM_MatrixAddition(benchmark::State& state) {
    size_t size = state.range(0);
    auto a = generate_random_matrix(size);
    auto b = generate_random_matrix(size);
    
    for (auto _ : state) {
        auto c = a + b;
        benchmark::DoNotOptimize(c);
    }
    state.SetComplexityN(size);
}

// Benchmark for matrix multiplication
static void BM_MatrixMultiplication(benchmark::State& state) {
    size_t size = state.range(0);
    auto a = generate_random_matrix(size);
    auto b = generate_random_matrix(size);
    
    for (auto _ : state) {
        auto c = a * b;
        benchmark::DoNotOptimize(c);
    }
    state.SetComplexityN(size);
}

// Benchmark for matrix transpose
static void BM_MatrixTranspose(benchmark::State& state) {
    size_t size = state.range(0);
    auto a = generate_random_matrix(size);
    
    for (auto _ : state) {
        auto b = a.transpose();
        benchmark::DoNotOptimize(b);
    }
    state.SetComplexityN(size);
}

// Benchmark for matrix determinant (optimized LU/Bareiss)
static void BM_MatrixDeterminant_Optimized(benchmark::State& state) {
    size_t size = state.range(0);
    auto a = generate_random_matrix(size);
    for (auto _ : state) {
        auto det = a.determinant();
        benchmark::DoNotOptimize(det);
    }
    state.SetComplexityN(size);
}

// Benchmark for matrix determinant (naive recursive) — only very small sizes
static void BM_MatrixDeterminant_Naive(benchmark::State& state) {
    size_t size = state.range(0);
    auto a = generate_random_matrix(size);
    for (auto _ : state) {
        auto det = a.determinant_naive();
        benchmark::DoNotOptimize(det);
    }
    state.SetComplexityN(size);
}

// Register benchmarks with different matrix sizes
BENCHMARK(BM_MatrixAddition)
    ->RangeMultiplier(2)
    ->Range(8, 512)
    ->Unit(benchmark::kMillisecond)
    ->Complexity(benchmark::oN2);

BENCHMARK(BM_MatrixMultiplication)
    ->RangeMultiplier(2)
    ->Range(8, 256)
    ->Unit(benchmark::kMillisecond)
    ->Complexity(benchmark::oN3);

BENCHMARK(BM_MatrixTranspose)
    ->RangeMultiplier(2)
    ->Range(8, 2048)
    ->Unit(benchmark::kMillisecond)
    ->Complexity(benchmark::oN2);

// Only benchmark small matrices for determinant due to O(n!) complexity
BENCHMARK(BM_MatrixDeterminant_Optimized)
    ->RangeMultiplier(2)
    ->Range(8, 512)
    ->Unit(benchmark::kMillisecond);

// Only tiny sizes for naive due to O(n!) complexity
BENCHMARK(BM_MatrixDeterminant_Naive)
    ->DenseRange(2, 8, 1)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
