#[cfg(test)]
mod tests {
    #[cfg(feature = "opencl")]
    #[test]
    fn test_create_buffer() {
        use ocl::builders::{BufferBuilder, ContextBuilder};

        let context = ContextBuilder::new().build().expect("context");
        let _buffer = BufferBuilder::<f32>::new().context(&context).len(8).build();
    }
}
