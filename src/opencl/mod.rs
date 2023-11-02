use lazy_static::lazy_static;
use ocl::Buffer;

use crate::{BufferInstance, CType, Enqueue, NDArrayRead, Op, PlatformInstance};

pub use platform::OpenCL;

mod ops;
mod platform;

impl<T: CType> BufferInstance for Buffer<T> {
    fn size(&self) -> usize {
        self.len()
    }
}
