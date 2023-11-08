use crate::{BufferInstance, CType};

#[cfg(feature = "opencl")]
impl<T: CType> BufferInstance<T> for ocl::Buffer<T> {
    fn size(&self) -> usize {
        self.len()
    }
}

#[cfg(feature = "opencl")]
impl<'a, T: CType> BufferInstance<T> for &'a ocl::Buffer<T> {
    fn size(&self) -> usize {
        self.len()
    }
}
