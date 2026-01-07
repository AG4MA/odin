"""
Baseline Benchmark: Pure Python MatMul
======================================
Reference per misurare speedup dei kernel WASM ottimizzati.
"""

import time
import random
from typing import List, Dict, Any

def matmul_naive(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Matrix multiplication O(n³) - baseline puro Python"""
    n = len(A)
    m = len(B[0])
    k = len(B)
    
    C = [[0.0] * m for _ in range(n)]
    
    for i in range(n):
        for j in range(m):
            for p in range(k):
                C[i][j] += A[i][p] * B[p][j]
    
    return C


def benchmark_matmul(size: int, iterations: int = 5) -> Dict[str, Any]:
    """Run benchmark for given matrix size"""
    
    # Generate random matrices
    A = [[random.random() for _ in range(size)] for _ in range(size)]
    B = [[random.random() for _ in range(size)] for _ in range(size)]
    
    times: List[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        _result = matmul_naive(A, B)  # Result unused, we're measuring time
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    flops = 2 * size ** 3  # multiply-add per element
    gflops = flops / avg_time / 1e9
    
    return {
        "size": size,
        "avg_time_ms": avg_time * 1000,
        "gflops": gflops
    }


if __name__ == "__main__":
    print("=" * 50)
    print("BASELINE BENCHMARK: Pure Python MatMul")
    print("=" * 50)
    
    sizes = [64, 128, 256]
    
    for size in sizes:
        result = benchmark_matmul(size, iterations=3)
        print(f"\nSize {size}x{size}:")
        print(f"  Time: {result['avg_time_ms']:.2f} ms")
        print(f"  GFLOPS: {result['gflops']:.4f}")
    
    print("\n" + "=" * 50)
    print("Target: WASM kernel should be 100x+ faster")
    print("=" * 50)
