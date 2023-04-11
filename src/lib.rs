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

pub trait CDatatype {
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

pub trait ArrayRead<T: OclPrm>: NDArray {
    fn read(&self, queue: Queue, output: Option<Buffer<T>>) -> Result<Buffer<T>, Error>;
}

pub trait ArrayWrite<O>: NDArray {
    fn write(&self, other: &O) -> Result<(), Error>;
}

pub trait ArrayCast<O>: NDArray {
    fn cast(&self) -> Result<ArrayOp<ops::ArrayCast<Self, O>>, Error>;
}

pub trait ArrayCompare<T, O>: NDArray {
    fn eq(&self, other: &O) -> Result<ArrayOp<ArrayEq<Self, O>>, Error>;

    fn gt(&self, other: &O) -> Result<ArrayOp<ArrayGT<Self, O>>, Error>;

    fn gte(&self, other: &O) -> Result<ArrayOp<ArrayGTE<Self, O>>, Error>;

    fn lt(&self, other: &O) -> Result<ArrayOp<ArrayLT<Self, O>>, Error>;

    fn lte(&self, other: &O) -> Result<ArrayOp<ArrayLTE<Self, O>>, Error>;

    fn ne(&self, other: &O) -> Result<ArrayOp<ArrayNE<Self, O>>, Error>;
}

pub trait ArrayMath<T, O>: NDArray {
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
