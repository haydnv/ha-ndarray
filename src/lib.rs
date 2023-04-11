use std::fmt;

pub use ocl::{Buffer, Device, DeviceType, OclPrm, Platform, Queue};

pub use array::*;
use ops::*;

mod array;
mod ops;

pub enum Error {
    Bounds(String),
    Platform(ocl::Error),
}

impl From<ocl::Error> for Error {
    fn from(cause: ocl::Error) -> Self {
        Self::Platform(cause)
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Bounds(cause) => f.write_str(cause),
            Self::Platform(cause) => cause.fmt(f),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Bounds(cause) => f.write_str(cause),
            Self::Platform(cause) => cause.fmt(f),
        }
    }
}

impl std::error::Error for Error {}

pub type Shape = Vec<usize>;

pub trait CDatatype: OclPrm {
    const TYPE_STR: &'static str;
}

impl CDatatype for f32 {
    const TYPE_STR: &'static str = "float";
}

pub trait NDArray: Sized {
    fn ndim(&self) -> usize {
        self.shape().len()
    }

    fn size(&self) -> usize {
        self.shape().iter().product()
    }

    fn shape(&self) -> &[usize];
}

pub trait NDArrayRead<T: OclPrm>: NDArray {
    fn read(self, queue: Queue, output: Option<Buffer<T>>) -> Result<Buffer<T>, Error>;
}

pub trait NDArrayWrite<O>: NDArray {
    fn write(&self, other: &O) -> Result<(), Error>;
}

pub trait NDArrayCast<O>: NDArray {
    fn cast(&self) -> Result<ArrayOp<ArrayCast<Self, O>>, Error>;
}

pub trait NDArrayCompare<O: NDArray>: NDArray {
    fn eq(self, other: O) -> Result<ArrayOp<ArrayEq<Self, O>>, Error> {
        let shape = broadcast_shape(self.shape(), other.shape())?;
        Ok(ArrayOp::new(ArrayEq::new(self, other), shape))
    }

    // fn gt(&self, other: &O) -> Result<ArrayOp<ArrayGT<Self, O>>, Error>;
    //
    // fn gte(&self, other: &O) -> Result<ArrayOp<ArrayGTE<Self, O>>, Error>;
    //
    // fn lt(&self, other: &O) -> Result<ArrayOp<ArrayLT<Self, O>>, Error>;
    //
    // fn lte(&self, other: &O) -> Result<ArrayOp<ArrayLTE<Self, O>>, Error>;
    //
    // fn ne(&self, other: &O) -> Result<ArrayOp<ArrayNE<Self, O>>, Error>;
}

pub trait NDArrayMath<T, O>: NDArray {
    fn matmul(&self, other: &O) -> Result<ArrayOp<MatMul<Self, O>>, Error>;
}

pub trait NDArrayReduce<T>: NDArray {
    fn all(&self) -> Result<bool, Error>;

    fn all_axis(&self, axis: usize) -> Result<ArrayOp<ArrayAll<Self>>, Error>;

    fn any(&self) -> Result<bool, Error>;

    fn any_axis(&self, axis: usize) -> Result<ArrayOp<ArrayAny<Self>>, Error>;

    fn max(&self) -> Result<T, Error>;

    fn max_axis(&self, axis: usize) -> Result<ArrayOp<ArrayMax<Self>>, Error>;

    fn min(&self) -> Result<T, Error>;

    fn min_axis(&self, axis: usize) -> Result<ArrayOp<ArrayMin<Self>>, Error>;

    fn product(&self) -> Result<T, Error>;

    fn product_axis(&self, axis: usize) -> Result<ArrayOp<ArrayProduct<Self>>, Error>;

    fn sum(&self) -> Result<T, Error>;

    fn sum_axis(&self, axis: usize) -> Result<ArrayOp<ArraySum<Self>>, Error>;
}

pub trait NDArrayTransform<T>: NDArray {
    fn broadcast(shape: Shape) -> Result<ArrayView<Self>, Error>;

    fn transpose(axes: Option<Vec<usize>>) -> Result<ArrayView<Self>, Error>;

    fn reshape(shape: Shape) -> Result<ArrayView<Self>, Error>;

    fn slice(bounds: Vec<AxisBound>) -> Result<ArraySlice<Self>, Error>;
}

pub enum AxisBound {
    At(u64),
    In(u64, u64, u64),
    Of(Vec<u64>),
}

#[inline]
fn broadcast_shape(left: &[usize], right: &[usize]) -> Result<Shape, Error> {
    if left.len() < right.len() {
        return broadcast_shape(right, left);
    }

    let mut shape = Vec::with_capacity(left.len());
    let offset = left.len() - right.len();

    for x in 0..offset {
        shape[x] = left[x];
    }

    for x in 0..right.len() {
        if right[x] == 1 || right[x] == left[x + offset] {
            shape[x + offset] = left[x + offset];
        } else if left[x + offset] == 1 {
            shape[x + offset] = right[x];
        } else {
            return Err(Error::Bounds(format!(
                "cannot broadcast dimensions {} and {}",
                left[x + offset],
                right[x]
            )));
        }
    }

    Ok(shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_array() -> Result<(), Error> {
        let array = ArrayBase::constant(0., vec![2, 3])?;
        assert!(!array.any()?);

        let array = ArrayBase::constant(1., vec![2, 3])?;
        assert!(array.all()?);

        Ok(())
    }
}
