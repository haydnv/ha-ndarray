fn main() {
    #[cfg(feature = "opencl")]
    pkg_config::Config::new()
        .atleast_version("3.0")
        .probe("OpenCL")
        .unwrap();

    #[cfg(feature = "cuda")]
    {
        // CUDA does not provide a .pc file for use with pkg-config, so there's no check to run here
    }
}
