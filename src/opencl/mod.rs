use lazy_static::lazy_static;

use crate::host::VEC_MIN_SIZE;

use platform::CLPlatform;
pub use platform::OpenCL;

mod kernels;
mod ops;
mod platform;

#[cfg(feature = "opencl")]
lazy_static! {
    pub static ref CL_PLATFORM: CLPlatform = {
        assert!(VEC_MIN_SIZE < GPU_MIN_SIZE);
        assert!(GPU_MIN_SIZE < ACC_MIN_SIZE);

        CLPlatform::default().expect("OpenCL platform")
    };
}

pub const GPU_MIN_SIZE: usize = 1024; // 1 KiB

pub const ACC_MIN_SIZE: usize = 2_147_483_648; // 1 GiB

#[cfg(test)]
mod tests {
    use smallvec::smallvec;

    use super::*;

    use crate::{Array, Error};

    #[test]
    fn test_add() -> Result<(), Error> {
        let buffer = OpenCL::create_buffer::<u64>(6)?;
        let left = Array::new(buffer, smallvec![1, 2, 3])?;

        let buffer = OpenCL::create_buffer::<u64>(6)?;
        let right = Array::new(buffer, smallvec![3, 2, 1])?;

        let expected = OpenCL::copy_into_buffer(&[4, 4, 4]);

        Ok(())
    }
}
