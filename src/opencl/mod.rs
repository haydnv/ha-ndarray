use ocl::Buffer;

use crate::{BufferInstance, CType};

pub use platform::OpenCL;

mod kernels;
mod ops;
mod platform;

impl<T: CType> BufferInstance for Buffer<T> {
    fn size(&self) -> usize {
        self.len()
    }
}
