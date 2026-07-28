# Benchmarking ha-ndarray

This directory documents the reproducible Criterion benchmark baseline for
`ha-ndarray`, established on `main` before the CubeCL HAL migration changes
execution behavior (issue #22).

The baseline is **host-only / Rayon** and is built against the current public
API with `--no-default-features`. It is intentionally not tied to OpenCL,
GPU, or any accelerator.

## Layout

| Path                                  | Purpose                                                      |
|---------------------------------------|--------------------------------------------------------------|
| `benches/host_benchmarks.rs`          | Criterion targets for the six benchmark categories.          |
| `tests/benchmark_support/mod.rs`      | Shared deterministic fixture + environment-recording utility.|
| `tests/bench_determinism.rs`          | Tests that the fixture is deterministic and in range.        |
| `docs/benchmarking/initial-results.md`| The initial result artifact + exact environment identity.    |

## Benchmark inventory and operation mapping

`ha-ndarray` operations are **lazy**: building an op tree (e.g. `input.exp()`)
is essentially free, and computation only happens when a result is materialized
via `NDArrayRead::buffer` (or `into_read`). Operation-only benchmarks therefore
build deterministic input data once, outside the measured region, and measure
op construction + materialization. `criterion::black_box` is used to prevent the
compiler from eliminating work.

| # | Category              | Public API used                                  | Benchmark group          | End-to-end? |
|---|-----------------------|--------------------------------------------------|--------------------------|-------------|
| 1 | construction          | `ArrayBuf::convert(&data, shape)` (alloc + copy) | `construct_convert`      | yes         |
| 2 | unary elementwise     | `NDArrayUnary::exp`                              | `unary_exp`              | no          |
| 3 | binary elementwise    | `NDArrayMath::add` (same-shape)                  | `binary_add`             | no          |
| 3 | binary (broadcast)    | `NDArrayTransform::broadcast` + `NDArrayMath::add`| `binary_add_broadcast` | no          |
| 4 | reduction             | `NDArrayReduceAll::sum_all`                      | `reduce_sum_all`         | no          |
| 5 | matrix multiplication | `MatrixDual::matmul`                             | `matmul`                 | no          |
| 6 | non-contiguous view   | `NDArrayTransform::transpose` (materialized)     | `transpose`              | no          |

### Chosen operations and substitutions

- `exp` is the unary op: it exercises a transcendental elementwise `f32` kernel.
- `add` is the binary op; the broadcast variant broadcasts a `[1, N]` operand up
  to `[M, N]` before the add, exercising `Transform::broadcast` in the measured
  region.
- `sum_all` is the reduction. It is available on **borrowed** (non-`'static`)
  inputs via the public `NDArrayReduceAll` trait. Axis-reduction
  (`NDArrayReduce::sum`) requires an owned `'static` accessor (the
  `Accessor::from(A)` indirection used by `reduce_axes` has a `B: 'static`
  bound), so it is documented as a future addition rather than substituted with
  extra allocation in this baseline.
- `transpose` is the non-contiguous-view transform: it swaps strides to produce
  a non-contiguous view, then materializes it into a contiguous buffer.

No product functionality was added to make a benchmark possible.

## Shape matrix

Benchmark identifiers encode `dtype/shape` (e.g. `matmul/f32/256x256`) so that
results are machine-readable and comparable across runs on the same machine.

Elementwise / reduction / transform benchmarks:

| Case   | Shape          | Elements    | Orientation     |
|--------|----------------|-------------|-----------------|
| small  | `[16, 16]`     | 256         | latency         |
| medium | `[256, 256]`   | 65 536      | representative  |
| large  | `[1024, 1024]` | 1 048 576   | throughput      |

`matmul` uses square operands sized for a bounded budget:

| Case   | Operands                       | ~MACs     |
|--------|--------------------------------|-----------|
| small  | `[16, 16] x [16, 16]`          | 4 K       |
| medium | `[96, 96] x [96, 96]`          | 885 K     |
| large  | `[256, 256] x [256, 256]`      | 16.8 M    |

The large cases are bounded to remain safe in the documented Tembo environment
(see `initial-results.md` for the measured total duration).

## Measurement rules

- **Deterministic inputs**: all inputs are generated from a fixed seed
  (`FIXTURE_SEED`) by `tests/benchmark_support/mod.rs`, producing byte-identical
  data on every run and machine. Determinism is verified by
  `tests/bench_determinism.rs`.
- **Setup is not measured**: input data is constructed once outside the
  benchmark; per-iteration input views are built in the untimed `iter_batched`
  setup region. The only intentionally end-to-end group is `construct_convert`.
- **`black_box`**: every benchmark feeds its result through
  `criterion::black_box` to prevent optimization from eliminating work.
- **Bounded budget**: the Criterion config uses `sample_size = 10`,
  `warm_up_time = 1 s`, `measurement_time = 3 s` (see
  `tests/benchmark_support/mod.rs`). With 21 benchmarks this gives a documented,
  finite total run time recorded in `initial-results.md`.
- **Inputs are reused** across iterations (kept in cache). This is standard for
  in-process micro-benchmarks and does not change operation semantics.

## How to run locally

Compile only (safe to verify the suite builds without running timings):

```
cargo bench --no-run --no-default-features
```

Run the full host-only suite:

```
cargo bench --no-default-features
```

Run a single group or benchmark (Criterion filter):

```
cargo bench --no-default-features -- matmul
cargo bench --no-default-features -- "matmul/f32/256x256"
```

Criterion writes machine-readable output under `target/criterion/` (each
benchmark has `new/estimates.json` with mean/median/slope/throughput, plus
`benchmark.json`). The benchmark binary also prints the recorded environment to
stderr at start-up.

## Recording the environment

The environment is recorded automatically at the start of every benchmark run
via `support::Environment::probe` and printed to stderr. It captures:

- `rustc` / `cargo` versions
- OS and architecture
- CPU model (parsed from `/proc/cpuinfo` on Linux)
- logical and physical core counts (`num_cpus`)
- the active feature set

When publishing a result, copy this block verbatim into the result record (as
`initial-results.md` does) so the result is tied to an exact machine identity.

## Comparing results

- **Do not compare numbers from different machines as though they are directly
  equivalent.** CPU model, core count, frequency, and memory bandwidth differ.
- Compare a new run against a previous run **on the same machine** using
  Criterion's own comparison (`cargo bench` reuses `target/criterion/` baselines
  and reports regressions/improvements), or by diffing `estimates.json`.
- A faster or slower number is not, by itself, evidence of correctness.

## How future CubeCL/OpenCL groups should reuse this baseline (issues #34-#50)

The host baseline is intentionally stable and must not change when accelerator
groups are added:

1. **Do not modify the host benchmark groups** (`construct_convert`,
   `unary_exp`, `binary_add`, `binary_add_broadcast`, `reduce_sum_all`,
   `matmul`, `transpose`) or the shared fixture. Add new groups alongside them.
2. **Reuse the shared fixture** (`tests/benchmark_support/mod.rs`) so that
   CubeCL/OpenCL groups consume byte-identical inputs to the host groups. This is
   what makes an accelerator-vs-host comparison meaningful on the same machine.
3. **Mirror the shape matrix and naming** so that a future `matmul/cubecl/...`
   group can be compared against `matmul/f32/...` on the same machine.
4. **Gate accelerator groups behind their feature** (for example
   `#[cfg(feature = "opencl")]`), and document any all-feature/OpenCL validation
   that was not possible in this baseline.
5. **Record the same environment block** so accelerator results carry exact
   machine identity.

This baseline does not introduce any performance threshold, supported-hardware
claim, or backend-retirement decision.

## Known limitations

- No OpenCL, GPU, or `all`-feature benchmarking is performed or validated here;
  that is an explicit non-goal of this baseline and is left to the CubeCL/OpenCL
  groups in #34-#50.
- Axis-reduction (`NDArrayReduce::sum`) is not benchmarked because it requires
  an owned `'static` accessor through the public `reduce_axes` path; `sum_all`
  is used as the representative reduction instead.
- `matmul` does not exercise batched or broadcasted matrix multiply at the large
  size; only the square case is in the baseline shape matrix.
- Results are machine-specific and must not be compared across machines.
