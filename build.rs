fn main() {
    #[cfg(feature = "cuda")]
    pkg_config::Config::new()
        .atleast_version("11.0")
        .probe("cuda")
        .unwrap();

    #[cfg(feature = "opencl")]
    pkg_config::Config::new()
        .atleast_version("3.0")
        .probe("OpenCL")
        .unwrap();
}
