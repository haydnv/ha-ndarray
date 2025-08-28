# Repository Guidelines

## Project Structure & Module Organization
- Source: `src/` with host and GPU backends: `src/host/`, `src/opencl/`, and core types in `src/array.rs`, `src/buffer.rs`, `src/ops/`.
- Library entry: `src/lib.rs` (crate type `rlib`/`cdylib`).
- OpenCL kernels: `src/opencl/programs/`.
- Tests: integration tests in `tests/*.rs` (e.g., `tests/construct.rs`).
- Packaging/build metadata: `Cargo.toml`, `Cargo.lock`; container setup in `Dockerfile`.

## Build, Test, and Development Commands
- Build (CPU/host only): `cargo build`.
- Enable features (e.g., OpenCL): `cargo build --features opencl` or all: `cargo build --features all`.
- Run tests: `cargo test` (host) or `cargo test --features opencl`.
- Docs: `cargo doc --no-deps` (add `--open` locally to view).
- Format and lint: `cargo fmt --all` and `cargo clippy --all-targets --all-features -D warnings`.

## Coding Style & Naming Conventions
- Rust 2021 edition; 4-space indentation; keep lines reasonably short.
- Modules/files: `snake_case` (e.g., `array.rs`); types/traits: `PascalCase` (e.g., `ArrayBuf`); functions/fields: `snake_case`.
- Prefer explicit types and small, focused modules; keep unsafe blocks minimal and well-justified.
- Run `cargo fmt` and `cargo clippy` before pushing.

## Testing Guidelines
- Framework: Rust `#[test]` with `cargo test`; integration tests live in `tests/*.rs`.
- Name tests descriptively (e.g., `#[test] fn transpose_concat_validates_dims()`), assert both values and shapes.
- For GPU-specific logic, guard with feature flags and provide host fallbacks when possible.
- Keep tests deterministic and fast; seed randomness when used.

## Commit & Pull Request Guidelines
- Commits: short, imperative subject (e.g., "implement Array::concat"), reference issues when relevant (e.g., `(#28)`).
- PRs: include a clear description, motivation, feature flags used (`opencl`, `complex`, etc.), and test coverage notes; add benchmarks only if necessary.
- Required: passing `cargo test`, `cargo fmt`, and `cargo clippy` with no new warnings.

## Security & Configuration Tips
- OpenCL support is optional (`--features opencl`) and requires drivers/ICD on the host; verify with `clinfo`.
- The provided `Dockerfile` can build with GPU support; run with `docker run --gpus=all` when testing OpenCL.
