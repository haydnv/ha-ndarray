fn main() {
    pkg_config::Config::new()
        .atleast_version("2.0")
        .probe("OpenCL")
        .unwrap();
}
