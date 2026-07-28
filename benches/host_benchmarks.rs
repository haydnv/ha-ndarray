//! Host/Rayon Criterion benchmark baseline for `ha-ndarray`.
//!
//! This benchmark target is deliberately built against the **current public
//! API** of `ha-ndarray` with `--no-default-features` (host-only, Rayon). It
//! establishes a reproducible baseline on `main` before the CubeCL HAL
//! migration changes execution behavior.
//!
//! # Operation inventory
//!
//! | Category            | ha-ndarray public API                                | Benchmark group          |
//! |---------------------|------------------------------------------------------|--------------------------|
//! | 1. construction     | `ArrayBuf::convert(&data, shape)` (alloc + copy)     | `construct_convert`      |
//! | 2. unary elementwise| `NDArrayUnary::exp`                                   | `unary_exp`              |
//! | 3. binary elementwise| `NDArrayMath::add` (same-shape)                     | `binary_add`             |
//! | 3. binary broadcast | `NDArrayTransform::broadcast` + `NDArrayMath::add`   | `binary_add_broadcast`   |
//! | 4. reduction        | `NDArrayReduceAll::sum_all`                           | `reduce_sum_all`         |
//! | 5. matrix multiply  | `MatrixDual::matmul`                                  | `matmul`                 |
//! | 6. non-contig. view | `NDArrayTransform::transpose` (materialized view)    | `transpose`              |
//!
//! `exp` is chosen as the unary op because it exercises a transcendental
//! elementwise kernel on `f32`. `add` is chosen as the binary op, with the
//! broadcast variant broadcasting a `[1, N]` operand up to `[M, N]` before the
//! add. `sum_all` is chosen as the reduction because it is available on
//! borrowed (non-`'static`) inputs via the public `NDArrayReduceAll` trait;
//! axis-reduction (`NDArrayReduce::sum`) requires an owned `'static` accessor
//! and is documented as a future addition. `transpose` materializes a
//! non-contiguous view (swapped strides) into a contiguous buffer.
//!
//! # Shape matrix
//!
//! Elementwise / reduction / transform benchmarks use a `dtype x shape` matrix:
//!   * small  (latency):   `[16, 16]`    =    256 elements
//!   * medium:             `[256, 256]`  = 65 536 elements
//!   * large  (throughput):`[1024, 1024]`= 1 048 576 elements
//!
//! `matmul` uses square matrices sized for a bounded budget:
//!   * small: `[16, 16] x [16, 16]`     (~4 K MACs)
//!   * medium:`[96, 96] x [96, 96]`    (~885 K MACs)
//!   * large: `[256, 256] x [256, 256]`(~16.8 M MACs)
//!
//! # Measurement model
//!
//! `ha-ndarray` operations are lazy: building an op tree (e.g. `input.exp()`)
//! is essentially free, and computation only happens when the result is
//! materialized via `NDArrayRead::buffer` (or `into_read`). Operation-only
//! benchmarks therefore construct deterministic input data **once, outside the
//! measured region**, build a cheap borrowed view of that data per iteration in
//! the (untimed) `iter_batched` setup, and measure op construction +
//! materialization. The `construct_convert` group is explicitly end-to-end and
//! includes allocation + copy.
//!
//! Inputs are reused across iterations (kept in cache), which is standard for
//! in-process micro-benchmarks and does not change operation semantics.

#[path = "../tests/benchmark_support/mod.rs"]
mod support;

use criterion::{criterion_group, BatchSize, Criterion};
use ha_ndarray::*;

/// Elementwise / reduction / transform shape matrix: `(rows, cols)`.
const ELEM_SIZES: &[(usize, usize)] = &[(16, 16), (256, 256), (1024, 1024)];

/// `matmul` shape matrix: square `(k, k)` operands.
const MATMUL_SIZES: &[(usize, usize)] = &[(16, 16), (96, 96), (256, 256)];

/// Build a borrowed, buffer-backed input view over `data` with the given
/// `shape`. Construction is cheap (a slice reference + a cloned `SmallVec`
/// shape) and is used in the untimed `iter_batched` setup region.
fn input_view(data: &[f32], shape: Shape) -> ArrayBuf<f32, &[f32]> {
    ArrayBuf::new(data, shape).expect("construct input view")
}

/// Category 1: array construction / allocation (end-to-end).
///
/// Measures `ArrayBuf::convert(&data, shape)`, which allocates a host buffer
/// and copies the deterministic source slice into it. This is the only group
/// that intentionally includes allocation cost.
fn bench_construct(c: &mut Criterion) {
    let mut group = c.benchmark_group("construct_convert");
    for &(m, n) in ELEM_SIZES {
        let data = support::uniform_f32(m * n, support::FIXTURE_SEED);
        let shape = shape![m, n];
        let id = format!("f32/{}", support::shape_id(&[m, n]));
        group.bench_function(id, |b| {
            b.iter(|| {
                let arr: ArrayBuf<f32, Buffer<f32>> =
                    ArrayBuf::convert(&data[..], shape.clone()).expect("convert");
                criterion::black_box(arr);
            });
        });
    }
    group.finish();
}

/// Category 2: unary elementwise operation (`exp`), operation-only.
fn bench_unary(c: &mut Criterion) {
    let mut group = c.benchmark_group("unary_exp");
    for &(m, n) in ELEM_SIZES {
        let data = support::uniform_f32(m * n, support::FIXTURE_SEED);
        let shape = shape![m, n];
        let id = format!("f32/{}", support::shape_id(&[m, n]));
        group.bench_function(id, |b| {
            b.iter_batched(
                || input_view(&data, shape.clone()),
                |input| {
                    let out = input.exp().expect("exp");
                    criterion::black_box(out.buffer().expect("materialize"));
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// Category 3a: binary elementwise operation (`add`), same-shape, operation-only.
fn bench_binary(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary_add");
    for &(m, n) in ELEM_SIZES {
        let left = support::uniform_f32(m * n, support::FIXTURE_SEED);
        let right = support::uniform_f32(m * n, support::FIXTURE_SEED.wrapping_add(1));
        let shape = shape![m, n];
        let id = format!("f32/{}", support::shape_id(&[m, n]));
        group.bench_function(id, |b| {
            b.iter_batched(
                || {
                    (
                        input_view(&left, shape.clone()),
                        input_view(&right, shape.clone()),
                    )
                },
                |(l, r)| {
                    let out = l.add(r).expect("add");
                    criterion::black_box(out.buffer().expect("materialize"));
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// Category 3b: binary elementwise operation (`add`) with a broadcast operand.
///
/// Broadcasts a `[1, N]` operand up to `[M, N]` then adds, exercising the
/// `Transform::broadcast` view within the measured region.
fn bench_binary_broadcast(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary_add_broadcast");
    for &(m, n) in ELEM_SIZES {
        let left = support::uniform_f32(m * n, support::FIXTURE_SEED);
        let right = support::uniform_f32(n, support::FIXTURE_SEED.wrapping_add(2));
        let l_shape = shape![m, n];
        let r_shape = shape![1, n];
        let b_shape = shape![m, n];
        let id = format!("f32/{}", support::shape_id(&[m, n]));
        group.bench_function(id, |b| {
            b.iter_batched(
                || {
                    (
                        input_view(&left, l_shape.clone()),
                        input_view(&right, r_shape.clone()),
                    )
                },
                |(l, r)| {
                    let r_b = r.broadcast(b_shape.clone()).expect("broadcast");
                    let out = l.add(r_b).expect("add");
                    criterion::black_box(out.buffer().expect("materialize"));
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// Category 4: reduction (`sum_all`), operation-only.
///
/// Reduces the entire array to a single `f32` scalar via the public
/// `NDArrayReduceAll::sum_all` trait method, which works on borrowed inputs.
fn bench_reduce(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduce_sum_all");
    for &(m, n) in ELEM_SIZES {
        let data = support::uniform_f32(m * n, support::FIXTURE_SEED);
        let shape = shape![m, n];
        let id = format!("f32/{}", support::shape_id(&[m, n]));
        group.bench_function(id, |b| {
            b.iter_batched(
                || input_view(&data, shape.clone()),
                |input| {
                    let sum = input.sum_all().expect("sum_all");
                    criterion::black_box(sum);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// Category 5: matrix multiplication, operation-only.
fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");
    for &(k, _) in MATMUL_SIZES {
        let left = support::uniform_f32(k * k, support::FIXTURE_SEED);
        let right = support::uniform_f32(k * k, support::FIXTURE_SEED.wrapping_add(3));
        let l_shape = shape![k, k];
        let r_shape = shape![k, k];
        let id = format!("f32/{}", support::shape_id(&[k, k]));
        group.bench_function(id, |b| {
            b.iter_batched(
                || {
                    (
                        input_view(&left, l_shape.clone()),
                        input_view(&right, r_shape.clone()),
                    )
                },
                |(l, r)| {
                    let out = l.matmul(r).expect("matmul");
                    criterion::black_box(out.buffer().expect("materialize"));
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// Category 6: non-contiguous view / transform (`transpose`), operation-only.
///
/// Transposes a 2-D array (swapping strides to produce a non-contiguous view)
/// and materializes it into a contiguous buffer.
fn bench_transpose(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpose");
    for &(m, n) in ELEM_SIZES {
        let data = support::uniform_f32(m * n, support::FIXTURE_SEED);
        let shape = shape![m, n];
        let id = format!("f32/{}", support::shape_id(&[m, n]));
        group.bench_function(id, |b| {
            b.iter_batched(
                || input_view(&data, shape.clone()),
                |input| {
                    let out = input.transpose(None).expect("transpose");
                    criterion::black_box(out.buffer().expect("materialize"));
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(support::SAMPLE_SIZE)
        .warm_up_time(support::WARMUP_TIME)
        .measurement_time(support::MEASUREMENT_TIME);
    targets = bench_construct, bench_unary, bench_binary, bench_binary_broadcast, bench_reduce, bench_matmul, bench_transpose,
}

fn main() {
    let env = support::Environment::probe("no-default-features (host-only, Rayon)");
    eprintln!("[ha-ndarray benchmark environment]\n{}", env.report());
    benches();
    criterion::Criterion::default()
        .configure_from_args()
        .final_summary();
}
