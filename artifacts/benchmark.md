# Benchmark record

This record corresponds to [`benchmark.json`](benchmark.json).

## Environment

- Date: 2026-07-15
- CPU: Intel Core Ultra 7 258V, 8 cores / 8 logical processors
- OS: Microsoft Windows 11 Home
- Compiler: GCC 16.1.0, MinGW-w64 POSIX/UCRT
- Build: CMake Release
- Model: float32 `[64,128] → [64,256] → [64,128] → [64,64]`
- Warmups: 30 per mode
- Measured iterations: 500 per mode
- Multi-thread configuration: 4 persistent workers

## Results

| Mode | Minimum | Mean | Median | p95 | Peak activations |
|---|---:|---:|---:|---:|---:|
| Serial interpreter | 2.034 ms | 2.833 ms | 2.682 ms | 3.676 ms | 344,064 B |
| Compiled, 1 thread | 1.969 ms | 2.256 ms | 2.172 ms | 2.742 ms | 98,304 B |
| Compiled, 4 threads | 0.588 ms | 0.841 ms | 0.779 ms | 1.225 ms | 98,304 B |

- Operations: 9 → 3
- Fused `MatMul + BiasAdd + ReLU` groups: 3
- Serial-to-compiled-four-thread median speedup: 3.4417×
- Activation memory reduction: 71.4286%
- Maximum absolute output error: 0

Timing uses `std::chrono::steady_clock`; the validated interpreter, compiled sessions, and worker pools persist across iterations. Compilation, graph validation, and correctness checking are outside the timed region. Both runtimes reference the same immutable inputs and weights without per-run copies. Each timed invocation includes construction of the returned output tensor.
