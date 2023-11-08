use lazy_static::lazy_static;

use crate::host::VEC_MIN_SIZE;

use crate::access::AccessOp;
use crate::ops::{ElementwiseCompare, ElementwiseDual};
use crate::{CType, Error, ReadBuf};

pub use buffer::*;
pub use platform::{OpenCL, ACC_MIN_SIZE, GPU_MIN_SIZE};

mod buffer;
mod kernels;
pub mod ops;
mod platform;

lazy_static! {
    pub static ref CL_PLATFORM: platform::CLPlatform = {
        assert!(VEC_MIN_SIZE < GPU_MIN_SIZE);
        assert!(GPU_MIN_SIZE < ACC_MIN_SIZE);

        platform::CLPlatform::default().expect("OpenCL platform")
    };
}

impl<'a, T, L, R> ElementwiseCompare<L, R, T> for OpenCL
where
    T: CType,
    L: ReadBuf<'a, T>,
    R: ReadBuf<'a, T>,
{
    type Output = ops::Compare<L, R, T>;

    fn eq(self, left: L, right: R) -> Result<AccessOp<Self::Output, Self>, Error> {
        ops::Compare::eq(self, left, right).map(AccessOp::from)
    }
}

impl<'a, T, L, R> ElementwiseDual<L, R, T> for OpenCL
where
    T: CType,
    L: ReadBuf<'a, T>,
    R: ReadBuf<'a, T>,
{
    type Output = ops::Dual<L, R, T>;

    fn add(self, left: L, right: R) -> Result<AccessOp<Self::Output, Self>, Error> {
        ops::Dual::add(self, left, right).map(AccessOp::from)
    }

    fn sub(self, left: L, right: R) -> Result<AccessOp<Self::Output, Self>, Error> {
        ops::Dual::sub(self, left, right).map(AccessOp::from)
    }
}

#[cfg(test)]
mod tests {
    use smallvec::smallvec;

    use super::*;

    use crate::{Array, Error};

    #[test]
    fn test_add() -> Result<(), Error> {
        let shape = smallvec![1, 2, 3];

        let buffer = OpenCL::create_buffer::<u64>(6)?;
        let left: Array<_, _, OpenCL> = Array::new(buffer, shape.clone())?;

        let buffer = OpenCL::create_buffer::<u64>(6)?;
        let right: Array<_, _, OpenCL> = Array::new(buffer, shape.clone())?;

        let buffer = OpenCL::create_buffer::<u64>(6)?;
        let expected = Array::new(buffer, shape.clone())?;

        let actual = left.add(right)?;
        let eq = Array::eq(actual, expected)?;

        assert!(eq.all()?);

        Ok(())
    }

    #[test]
    fn test_sub() -> Result<(), Error> {
        let shape = smallvec![1, 2, 3];

        let buffer = OpenCL::copy_into_buffer(&[0, 1, 2, 3, 4, 5])?;
        let array: Array<_, _, OpenCL> = Array::new(buffer, shape.clone())?;

        let actual = array.as_ref().sub(array.as_ref())?;

        assert!(!actual.any()?);

        Ok(())
    }
}
