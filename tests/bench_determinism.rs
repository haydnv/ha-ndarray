//! Verifies that the shared benchmark fixture is deterministic and within the
//! documented numeric range. This guards the "deterministic inputs" acceptance
//! criterion of issue #22 independently of running the timing suite.

#[path = "benchmark_support/mod.rs"]
mod support;

#[test]
fn test_fixture_is_deterministic() {
    for &size in &[256, 65_536, 1_048_576] {
        let first = support::uniform_f32(size, support::FIXTURE_SEED);
        let second = support::uniform_f32(size, support::FIXTURE_SEED);
        assert_eq!(first, second, "fixture is not reproducible for size {size}");
    }
}

#[test]
fn test_fixture_distinct_seeds_differ() {
    let a = support::uniform_f32(1024, support::FIXTURE_SEED);
    let b = support::uniform_f32(1024, support::FIXTURE_SEED.wrapping_add(1));
    assert_ne!(a, b, "distinct seeds produced identical data");
}

#[test]
fn test_fixture_respects_range() {
    let data = support::uniform_f32(4096, support::FIXTURE_SEED);
    for v in data {
        assert!(
            v >= support::FIXTURE_LO && v <= support::FIXTURE_HI,
            "fixture value {v} outside documented range"
        );
    }
}

#[test]
fn test_fixture_nonzero_variant() {
    let eps = 0.25;
    let data = support::uniform_f32_nonzero(4096, support::FIXTURE_SEED, eps);
    assert!(
        data.iter().copied().all(|v| v.abs() >= eps),
        "nonzero fixture produced a value within the exclusion band"
    );
}

#[test]
fn test_environment_probe_succeeds() {
    let env = support::Environment::probe("test");
    assert!(!env.os.is_empty());
    assert!(env.logical_cores > 0);
    assert_eq!(env.feature_set, "test");
}
