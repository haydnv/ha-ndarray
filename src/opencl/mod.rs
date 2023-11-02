use lazy_static::lazy_static;
use ocl::Buffer;

use crate::{BufferInstance, CType};

pub use platform::OpenCL;

mod kernels;
mod ops;
mod platform;

#[cfg(feature = "opencl")]
lazy_static! {
    pub static ref CL_PLATFORM: OpenCL = OpenCL::default().expect("OpenCL platform");
}

impl<T: CType> BufferInstance for Buffer<T> {
    fn size(&self) -> usize {
        self.len()
    }
}
