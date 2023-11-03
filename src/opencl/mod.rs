use lazy_static::lazy_static;

use crate::host::VEC_MIN_SIZE;

use crate::access::AccessOp;
use crate::ops::ElementwiseDual;
use crate::{CType, Error, ReadBuf};

use platform::CLPlatform;
pub use platform::{OpenCL, ACC_MIN_SIZE, GPU_MIN_SIZE};

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

impl<T, L, R> ElementwiseDual<L, R, T> for OpenCL
where
    T: CType,
    L: ReadBuf<T>,
    R: ReadBuf<T>,
{
    type Output = ops::Dual<L, R, T>;

    fn add(self, left: L, right: R) -> Result<AccessOp<Self::Output, Self>, Error> {
        ops::Dual::add(self, left, right).map(AccessOp::from)
    }
}

#[cfg(test)]
mod tests {
    use smallvec::smallvec;

    use super::*;

    use crate::{Array, Error, Op, StackVec};

    #[test]
    fn test_add() -> Result<(), Error> {
        let buffer = OpenCL::create_buffer::<u64>(6)?;
        let left: Array<_, _, OpenCL> = Array::new(buffer, smallvec![1, 2, 3])?;

        let buffer = OpenCL::create_buffer::<u64>(6)?;
        let right: Array<_, _, OpenCL> = Array::new(buffer, smallvec![1, 2, 3])?;

        let expected = OpenCL::create_buffer::<u64>(6)?;

        let actual = left.add(right)?;

        Ok(())
    }
}
