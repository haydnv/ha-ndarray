extern crate ocl;

use std::fmt;
use std::iter::Sum;
use std::ops::{Add, AddAssign};

use ocl::{Buffer, Context, Device, OclPrm, Platform, Queue};

pub use array::*;
use ops::*;

mod array;
mod kernels;
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

pub trait CDatatype: OclPrm + Add<Output = Self> + AddAssign + Sum {
    const TYPE_STR: &'static str;

    fn zero() -> Self;
}

impl CDatatype for f32 {
    const TYPE_STR: &'static str = "float";

    fn zero() -> Self {
        0.
    }
}

impl CDatatype for u8 {
    const TYPE_STR: &'static str = "char";

    fn zero() -> Self {
        0
    }
}

impl CDatatype for u32 {
    const TYPE_STR: &'static str = "uint";

    fn zero() -> Self {
        0
    }
}

impl CDatatype for i32 {
    const TYPE_STR: &'static str = "int";

    fn zero() -> Self {
        0
    }
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

pub trait NDArrayRead<T: CDatatype>: NDArray {
    fn copy(self) -> Result<ArrayBase<T>, Error> {
        let shape = self.shape().to_vec();

        let queue = autoqueue(None)?;
        let mut data = vec![T::zero(); self.size()];
        let buffer = self.read(queue, None)?;
        buffer.read(&mut data).enq()?;

        ArrayBase::from_vec(shape, data)
    }

    fn read(self, queue: Queue, output: Option<Buffer<T>>) -> Result<Buffer<T>, Error>;
}

pub trait NDArrayWrite<O>: NDArray {
    fn write(&self, other: &O) -> Result<(), Error>;
}

pub trait NDArrayCast<O>: NDArray {
    fn cast(&self) -> Result<ArrayOp<ArrayCast<Self, O>>, Error>;
}

pub trait NDArrayCompare<O: NDArray>: NDArray {
    fn eq(&self, other: O) -> Result<ArrayOp<ArrayCompare<&Self, O>>, Error> {
        let shape = check_shape(self.shape(), other.shape())?;
        Ok(ArrayOp::new(ArrayCompare::eq(self, other), shape))
    }

    fn gt(&self, other: O) -> Result<ArrayOp<ArrayCompare<&Self, O>>, Error> {
        let shape = check_shape(self.shape(), other.shape())?;
        Ok(ArrayOp::new(ArrayCompare::gt(self, other), shape))
    }

    fn gte(&self, other: O) -> Result<ArrayOp<ArrayCompare<&Self, O>>, Error> {
        let shape = check_shape(self.shape(), other.shape())?;
        Ok(ArrayOp::new(ArrayCompare::gte(self, other), shape))
    }

    fn lt(&self, other: O) -> Result<ArrayOp<ArrayCompare<&Self, O>>, Error> {
        let shape = check_shape(self.shape(), other.shape())?;
        Ok(ArrayOp::new(ArrayCompare::lt(self, other), shape))
    }

    fn lte(&self, other: O) -> Result<ArrayOp<ArrayCompare<&Self, O>>, Error> {
        let shape = check_shape(self.shape(), other.shape())?;
        Ok(ArrayOp::new(ArrayCompare::lte(self, other), shape))
    }

    fn ne(&self, other: O) -> Result<ArrayOp<ArrayCompare<&Self, O>>, Error> {
        let shape = check_shape(self.shape(), other.shape())?;
        Ok(ArrayOp::new(ArrayCompare::ne(self, other), shape))
    }
}

pub trait NDArrayCompareScalar<T>: NDArray {
    fn eq(&self, other: T) -> Result<ArrayOp<ArrayCompareScalar<&Self, T>>, Error> {
        let shape = self.shape().to_vec();
        Ok(ArrayOp::new(ArrayCompareScalar::eq(self, other), shape))
    }

    fn gt(&self, other: T) -> Result<ArrayOp<ArrayCompareScalar<&Self, T>>, Error> {
        let shape = self.shape().to_vec();
        Ok(ArrayOp::new(ArrayCompareScalar::gt(self, other), shape))
    }

    fn gte(&self, other: T) -> Result<ArrayOp<ArrayCompareScalar<&Self, T>>, Error> {
        let shape = self.shape().to_vec();
        Ok(ArrayOp::new(ArrayCompareScalar::gte(self, other), shape))
    }

    fn lt(&self, other: T) -> Result<ArrayOp<ArrayCompareScalar<&Self, T>>, Error> {
        let shape = self.shape().to_vec();
        Ok(ArrayOp::new(ArrayCompareScalar::lt(self, other), shape))
    }

    fn lte(&self, other: T) -> Result<ArrayOp<ArrayCompareScalar<&Self, T>>, Error> {
        let shape = self.shape().to_vec();
        Ok(ArrayOp::new(ArrayCompareScalar::lte(self, other), shape))
    }

    fn ne(&self, other: T) -> Result<ArrayOp<ArrayCompareScalar<&Self, T>>, Error> {
        let shape = self.shape().to_vec();
        Ok(ArrayOp::new(ArrayCompareScalar::ne(self, other), shape))
    }
}

pub trait NDArrayMath<T, O>: NDArray {
    fn matmul(&self, other: &O) -> Result<ArrayOp<MatMul<Self, O>>, Error>;
}

pub trait NDArrayReduce<T>: NDArray {
    fn all(&self) -> Result<bool, Error>;

    fn all_axis(&self, axis: usize) -> Result<ArrayOp<ArrayAll<&Self>>, Error>;

    fn any(&self) -> Result<bool, Error>;

    fn any_axis(&self, axis: usize) -> Result<ArrayOp<ArrayAny<&Self>>, Error>;

    fn max(&self) -> Result<T, Error>;

    fn max_axis(&self, axis: usize) -> Result<ArrayOp<ArrayMax<&Self>>, Error>;

    fn min(&self) -> Result<T, Error>;

    fn min_axis(&self, axis: usize) -> Result<ArrayOp<ArrayMin<&Self>>, Error>;

    fn product(&self) -> Result<T, Error>;

    fn product_axis(&self, axis: usize) -> Result<ArrayOp<ArrayProduct<&Self>>, Error>;

    fn sum(&self) -> Result<T, Error>;

    fn sum_axis(&self, axis: usize) -> Result<ArrayOp<ArraySum<&Self>>, Error>;
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

pub fn autoqueue(context: Option<Context>) -> Result<Queue, ocl::Error> {
    // TODO: select CPUs for small data, GPUs for medium data, accelerators for large data
    // TODO: rotate the device selection

    let (context, device) = if let Some(context) = context {
        let device = if let Some(platform) = context.platform()? {
            Device::first(platform)?
        } else {
            context.devices()[0]
        };

        (context, device)
    } else {
        let platform = Platform::default();
        let device = Device::first(platform)?;
        let context = Context::builder()
            .platform(platform)
            .devices(device)
            .build()?;

        (context, device)
    };

    Queue::new(&context, device, None)
}

#[inline]
fn broadcast_shape(left: &[usize], right: &[usize]) -> Result<Shape, Error> {
    if left.is_empty() || right.is_empty() {
        return Err(Error::Bounds("cannot broadcast empty shape".to_string()));
    } else if left.len() < right.len() {
        return broadcast_shape(right, left);
    }

    let mut shape = vec![0; left.len()];
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

    debug_assert!(!shape.iter().any(|dim| *dim == 0));

    Ok(shape)
}

#[inline]
fn check_shape(left: &[usize], right: &[usize]) -> Result<Shape, Error> {
    if left == right {
        Ok(left.to_vec())
    } else {
        const MSG: &str = "this operation expects arrays of the same shape";
        const HINT: &str = "consider calling broadcast() explicitly";
        Err(Error::Bounds(format!(
            "{} but found {:?} and {:?} ({})",
            MSG, left, right, HINT
        )))
    }
}
