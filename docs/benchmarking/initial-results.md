# Initial benchmark result artifact (issue #22)

This is the initial Criterion result artifact for the host/Rayon baseline. It
records the exact environment, the exact commands executed, the benchmark
results, the total duration, and the test results.

> **These numbers are specific to the machine below and must not be compared
> directly with numbers from any other machine.** They are a baseline, not a
> performance threshold or a supported-hardware claim.

## Exact environment identity

Recorded automatically by `support::Environment::probe` and printed to stderr at
the start of the run:

```
rustc: rustc 1.96.0 (ac68faa20 2026-05-25)
cargo: cargo 1.96.0 (30a34c682 2026-05-25)
os: linux
arch: x86_64
cpu_model: Intel(R) Xeon(R) Platinum 8275CL CPU @ 3.00GHz
logical_cores: 2
physical_cores: 2
feature_set: no-default-features (host-only, Rayon)
```

## Exact commands executed

```text
cargo fmt --all -- --check
cargo test --no-default-features
cargo bench --no-run --no-default-features
cargo bench --no-default-features
```

## Benchmark compilation result

```text
cargo bench --no-run --no-default-features
  Finished `bench` profile [optimized] target(s)
  Executable benches/host_benchmarks.rs (target/release/deps/host_benchmarks-...)
```

Compilation succeeds in a clean checkout with `--no-default-features`.

## Results

Criterion reports each benchmark as `[lower_bound mean upper_bound]`. The
machine-readable source of truth for every benchmark is
`target/criterion/<group>/<bench>/new/estimates.json` (not committed to the
repository); the values below are taken from that run.

| Benchmark                          | Time [lower  mean  upper]        |
|------------------------------------|----------------------------------|
| construct_convert/f32/16x16        | [88.469 ns  89.498 ns  90.417 ns]    |
| construct_convert/f32/256x256      | [7.3174 µs 7.4304 µs 7.5423 µs]      |
| construct_convert/f32/1024x1024    | [344.94 µs  345.59 µs  346.43 µs]    |
| unary_exp/f32/16x16                | [10.921 µs  12.143 µs  13.931 µs]    |
| unary_exp/f32/256x256              | [162.21 µs  165.19 µs  168.11 µs]    |
| unary_exp/f32/1024x1024            | [2.2983 ms  2.3161 ms  2.3318 ms]    |
| binary_add/f32/16x16               | [26.883 µs  27.760 µs  28.377 µs]    |
| binary_add/f32/256x256             | [78.554 µs  79.549 µs  80.895 µs]    |
| binary_add/f32/1024x1024           | [849.29 µs  857.92 µs  872.34 µs]    |
| binary_add_broadcast/f32/16x16     | [41.802 µs  44.219 µs  45.955 µs]    |
| binary_add_broadcast/f32/256x256   | [651.33 µs  661.02 µs  671.35 µs]    |
| binary_add_broadcast/f32/1024x1024 | [9.7063 ms  9.8600 ms  10.049 ms]    |
| reduce_sum_all/f32/16x16           | [11.767 µs  12.015 µs  12.203 µs]    |
| reduce_sum_all/f32/256x256         | [52.757 µs  53.330 µs  53.970 µs]    |
| reduce_sum_all/f32/1024x1024       | [616.56 µs  623.98 µs  631.21 µs]    |
| matmul/f32/16x16                   | [47.459 µs  48.198 µs  49.180 µs]    |
| matmul/f32/96x96                   | [1.5172 ms  1.5369 ms  1.5575 ms]    |
| matmul/f32/256x256                 | [14.218 ms  14.574 ms  14.998 ms]    |
| transpose/f32/16x16                | [24.586 µs  24.774 µs  24.931 µs]    |
| transpose/f32/256x256              | [562.52 µs  568.67 µs  574.14 µs]    |
| transpose/f32/1024x1024            | [12.287 ms  12.435 ms  12.686 ms]    |

Observations (not thresholds):

- The small (`16x16`) cases are latency-dominated by Rayon thread-pool dispatch
  (tens of µs), which is the intended latency-oriented characterization.
- The large elementwise/transform cases are throughput/memory-bound; `matmul`
  and `transpose` at `1024x1024`/`256x256` are the most compute-intensive.
- `binary_add_broadcast` is slower than same-shape `binary_add` because the
  broadcast operand is materialized (gathered with broadcast strides) within the
  measured region before the add.

## Total benchmark duration

```text
cargo bench --no-default-features  # 21 benchmarks, sample_size=10,
                                   # warm_up_time=1s, measurement_time=3s
Total wall time: 103 s
```

## Test results

```text
cargo test --no-default-features
```

All host-only tests pass, including the 5 new `bench_determinism` tests that
verify the fixture is deterministic, in range, and seed-distinct:

- `arithmetic` 2 passed
- `bench_determinism` 5 passed
- `compare` 7 passed
- `cond` 1 passed
- `construct` 3 passed
- `linalg` 2 passed
- `reduce` 3 passed
- `transform` 9 passed

## Known limitations and unavailable validation

- **No OpenCL / `all`-feature validation.** This baseline is host-only
  (`--no-default-features`). OpenCL/GPU/`all`-feature benchmarking is an
  explicit non-goal and is left to the CubeCL/OpenCL groups in #34-#50.
- **Axis-reduction is not benchmarked.** `NDArrayReduce::sum` (reduce over an
  axis) requires an owned `'static` accessor via the public `reduce_axes` path;
  `NDArrayReduceAll::sum_all` is used as the representative reduction instead.
- **Batched/broadcasted `matmul`** is not in the baseline shape matrix; only the
  square case is measured.
- **Results are machine-specific.** The numbers above are tied to the
  environment block at the top of this file and must not be compared with
  results from a different machine.
- No performance threshold, supported-hardware claim, or backend-retirement
  decision is introduced by this artifact.
