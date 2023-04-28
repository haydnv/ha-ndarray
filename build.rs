fn main() {
    #[cfg(feature = "opencl")]
    pkg_config::Config::new()
        .atleast_version("2.0")
        .probe("OpenCL")
        .unwrap();
}
