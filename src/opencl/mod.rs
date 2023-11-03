use lazy_static::lazy_static;

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
        let right: ArrayBase<_, OpenCL> = ArrayBase::new(buffer, smallvec![3, 2, 1])?;

        let expected = CL_PLATFORM.copy_into_buffer(&[4, 4, 4]);

        Ok(())
    }
}
