use lazy_static::lazy_static;

use crate::access::AccessBuffer;
use crate::host::VEC_MIN_SIZE;

pub use buffer::*;
pub use platform::{OpenCL, ACC_MIN_SIZE, GPU_MIN_SIZE};

mod buffer;
pub mod ops;
mod platform;
mod programs;

lazy_static! {
    pub static ref CL_PLATFORM: platform::CLPlatform = {
        assert!(VEC_MIN_SIZE < GPU_MIN_SIZE);
        assert!(GPU_MIN_SIZE < ACC_MIN_SIZE);

        platform::CLPlatform::default().expect("OpenCL platform")
    };
}

pub type Array<T> = crate::array::Array<T, AccessBuffer<ocl::Buffer<T>>, OpenCL>;

#[cfg(test)]
mod tests {
    use crate::{shape, slice, AxisRange, Error};

    use super::*;

    #[test]
    fn test_add() -> Result<(), Error> {
        let shape = shape![1, 2, 3];

        let buffer = OpenCL::create_buffer::<u64>(6)?;
        let left = Array::new(buffer, shape.clone())?;

        let buffer = OpenCL::create_buffer::<u64>(6)?;
        let right = Array::new(buffer, shape.clone())?;

        let buffer = OpenCL::create_buffer::<u64>(6)?;
        let expected = Array::new(buffer, shape.clone())?;

        let actual = left.add(right)?;
        let eq = actual.eq(expected)?;

        assert!(eq.all()?);

        Ok(())
    }

    #[test]
    fn test_sub() -> Result<(), Error> {
        let shape = shape![1, 2, 3];

        let buffer = OpenCL::copy_into_buffer(&[0, 1, 2, 3, 4, 5])?;
        let array = Array::new(buffer, shape.clone())?;

        let actual = array.as_ref().sub(array.as_ref())?;

        assert!(!actual.any()?);

        Ok(())
    }

    #[test]
    fn test_slice() -> Result<(), Error> {
        let buf = OpenCL::copy_into_buffer(&[0; 6])?;
        let array = Array::new(buf, shape![2, 3])?;
        let mut slice = array.slice(slice![AxisRange::In(0, 2, 1), AxisRange::At(1)])?;

        let buf = OpenCL::copy_into_buffer(&[0, 0])?;
        let zeros = Array::new(buf, shape![2])?;

        let buf = OpenCL::copy_into_buffer(&[0, 0])?;
        let ones = Array::new(buf, shape![2])?;

        assert!(slice.as_ref().eq(zeros)?.all()?);

        slice.write(&ones)?;

        Ok(())
    }
}

#[inline]
fn div_ceil(num: usize, denom: usize) -> usize {
    if num % denom == 0 {
        num / denom
    } else {
        (num / denom) + 1
    }
}
