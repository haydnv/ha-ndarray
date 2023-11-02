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

const GPU_MIN_SIZE: usize = 1024; // 1 KiB

const ACC_MIN_SIZE: usize = 2_147_483_648; // 1 GiB

impl<T: CType> BufferInstance for Buffer<T> {
    fn size(&self) -> usize {
        self.len()
    }
}

#[cfg(test)]
mod tests {
    use smallvec::smallvec;

    use super::*;

    use crate::array::ArrayBase;
    use crate::Error;

    #[test]
    fn test_add() -> Result<(), Error> {
        let buffer = CL_PLATFORM.create_buffer::<u64>(6)?;
        let left: ArrayBase<_, OpenCL> = ArrayBase::new(buffer, smallvec![1, 2, 3])?;

        let buffer = CL_PLATFORM.create_buffer::<u64>(6)?;
        let right: ArrayBase<_, OpenCL> = ArrayBase::new(buffer, smallvec![1, 2, 3])?;

        Ok(())
    }
}
