//! Shared deterministic fixture and environment-recording utilities for the
//! ha-ndarray Criterion benchmark suite.
//!
//! This module is deliberately decoupled from the `ha-ndarray` public API: it
//! only produces raw `Vec<f32>` data and records host environment metadata. The
//! benchmark targets and the determinism test construct `ha-ndarray` arrays from
//! this data so that input construction is excluded from operation-only
//! measurements.
//!
//! It is intended to be included verbatim from two compilation contexts via
//! `#[path]`:
//!
//! * `benches/host_benchmarks.rs` - `#[path = "../tests/benchmark_support/mod.rs"] mod support;`
//! * `tests/bench_determinism.rs` - `#[path = "benchmark_support/mod.rs"] mod support;`
//!
//! Both contexts have access to the `rand` and `num_cpus` crate dependencies of
//! `ha-ndarray`, so this module depends on nothing else.

use std::time::Duration;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// The fixed seed used for all benchmark input generation.
///
/// Every benchmark input is derived from this seed so that a given shape always
/// produces byte-identical data on every run and every machine.
pub const FIXTURE_SEED: u64 = 0x6e64_6172_6179_0001;

/// The inclusive range from which deterministic fixture values are drawn.
///
/// The bounds are chosen so that every benchmarked operation stays in a safe
/// numeric range for `f32` (for example `exp(5.0) ~= 148.4` is well within the
/// `f32` range, and there are no zeros to cause division-by-zero surprises in
/// downstream comparisons).
pub const FIXTURE_LO: f32 = -5.0;
pub const FIXTURE_HI: f32 = 5.0;

/// Default Criterion measurement budget used by the benchmark targets.
///
/// These values are intentionally small so that a complete local run of the
/// host-only suite has a documented, finite wall-clock budget. See
/// `docs/benchmarking/README.md` for the expected total duration.
pub const SAMPLE_SIZE: usize = 10;
pub const WARMUP_TIME: Duration = Duration::from_secs(1);
pub const MEASUREMENT_TIME: Duration = Duration::from_secs(3);

/// Generate a deterministic `Vec<f32>` of `size` elements, drawn uniformly from
/// `[FIXTURE_LO, FIXTURE_HI]`.
///
/// The same `(size, seed)` pair always returns the same data, independent of
/// host, thread count, or run order. A different `seed` produces an independent
/// stream, which is useful for generating distinct operands (for example the two
/// sides of a binary operation).
pub fn uniform_f32(size: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..size)
        .map(|_| rng.gen_range(FIXTURE_LO..=FIXTURE_HI))
        .collect()
}

/// Generate a deterministic `Vec<f32>` of `size` elements with no zero values,
/// drawn from `[FIXTURE_LO, FIXTURE_HI]` but with any value whose absolute value
/// is below `eps` shifted away from zero.
///
/// Useful for operands that must avoid exact zeros (for example the denominator
/// of a division, or the base of a logarithm) while remaining deterministic.
/// Not consumed by the current host benchmarks, but provided for future groups
/// (for example a CubeCL division or logarithm benchmark) and validated by the
/// `bench_determinism` integration test.
#[allow(dead_code)]
pub fn uniform_f32_nonzero(size: usize, seed: u64, eps: f32) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..size)
        .map(|_| {
            let mut v = rng.gen_range(FIXTURE_LO..=FIXTURE_HI);
            if v.abs() < eps {
                v = if v >= 0.0 { eps } else { -eps };
            }
            v
        })
        .collect()
}

/// Render a shape slice as a compact, machine-readable `d1xd2x...` string used
/// to construct stable benchmark identifiers.
pub fn shape_id(shape: &[usize]) -> String {
    shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join("x")
}

/// Recorded identity of the host that produced a benchmark result.
///
/// All fields are strings so that the record can be serialized trivially and
/// compared across runs. None of these fields imply a performance threshold or a
/// supported-hardware claim.
#[derive(Debug, Clone)]
pub struct Environment {
    pub rustc: String,
    pub cargo: String,
    pub os: String,
    pub arch: String,
    pub cpu_model: String,
    pub logical_cores: usize,
    pub physical_cores: usize,
    pub feature_set: String,
}

impl Environment {
    /// Probe the current host and return its identity.
    ///
    /// Falling back to the string `"unknown"` for any field that cannot be
    /// determined, so that recording never panics on an unsupported platform.
    pub fn probe(feature_set: &str) -> Self {
        Self {
            rustc: command_output(["rustc", "--version"]),
            cargo: command_output(["cargo", "--version"]),
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            cpu_model: cpu_model_name().unwrap_or_else(|| "unknown".to_string()),
            logical_cores: num_cpus::get(),
            physical_cores: num_cpus::get_physical(),
            feature_set: feature_set.to_string(),
        }
    }

    /// Render the environment as a stable, human-readable block.
    pub fn report(&self) -> String {
        format!(
            "rustc: {}\n\
             cargo: {}\n\
             os: {}\n\
             arch: {}\n\
             cpu_model: {}\n\
             logical_cores: {}\n\
             physical_cores: {}\n\
             feature_set: {}\n",
            self.rustc,
            self.cargo,
            self.os,
            self.arch,
            self.cpu_model,
            self.logical_cores,
            self.physical_cores,
            self.feature_set,
        )
    }
}

fn command_output(cmd: [&str; 2]) -> String {
    std::process::Command::new(cmd[0])
        .arg(cmd[1])
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Read the CPU model name on Linux by parsing `/proc/cpuinfo`.
///
/// Returns `None` on non-Linux platforms or if the field is absent.
fn cpu_model_name() -> Option<String> {
    if std::env::consts::OS != "linux" {
        return None;
    }

    let contents = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    for line in contents.lines() {
        if let Some(rest) = line.strip_prefix("model name") {
            if let Some((_, value)) = rest.split_once(':') {
                return Some(value.trim().to_string());
            }
        }
    }

    None
}
