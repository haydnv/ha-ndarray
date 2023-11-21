use crate::access::AccessBuffer;

pub use buffer::*;
pub use platform::*;

mod buffer;
pub mod ops;
mod platform;

pub type Array<T> = crate::array::Array<T, AccessBuffer<Buffer<T>>, Host>;

#[cfg(test)]
mod tests {
    use crate::{shape, slice, AxisRange, Error, NDArrayTransform};

    use super::*;

    #[test]
    fn test_slice() -> Result<(), Error> {
        let array = Array::new(vec![0; 6].into(), shape![2, 3])?;
        let mut slice = array.slice(slice![AxisRange::In(0, 2, 1), AxisRange::At(1)])?;

        let zeros = Array::new(vec![0, 0].into(), shape![2])?;
        let ones = Array::new(vec![1, 1].into(), shape![2])?;

        assert!(slice.as_ref().eq(zeros)?.all()?);

        slice.write(&ones)?;

        Ok(())
    }
}
