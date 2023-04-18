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

    fn one() -> Self;

    fn zero() -> Self;
}

macro_rules! c_type {
    ($t:ty, $ct:expr, $zero:expr, $one:expr) => {
        impl CDatatype for $t {
            const TYPE_STR: &'static str = $ct;

            fn one() -> Self {
                $one
            }

            fn zero() -> Self {
                $zero
            }
        }
    };
}

c_type!(f32, "float", 0., 1.);
c_type!(f64, "double", 0., 1.);
c_type!(u8, "uchar", 0, 1);
c_type!(u16, "ushort", 0, 1);
c_type!(u32, "uint", 0, 1);
c_type!(u64, "ulong", 0, 1);
c_type!(i8, "char", 0, 1);
c_type!(i16, "short", 0, 1);
c_type!(i32, "int", 0, 1);
c_type!(i64, "long", 0, 1);

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
    fn copy(&self) -> Result<ArrayBase<T>, Error> {
        let shape = self.shape().to_vec();

        let queue = autoqueue(None)?;
        let mut data = vec![T::zero(); self.size()];
        let buffer = self.read(queue)?;
        buffer.read(&mut data).enq()?;

        ArrayBase::from_vec(shape, data)
    }

    fn read(&self, queue: Queue) -> Result<Buffer<T>, Error>;
}

pub trait NDArrayWrite<O>: NDArray {
    fn write(&self, other: &O) -> Result<(), Error>;
}

pub trait NDArrayCast<O>: NDArray {
    fn cast(&self) -> Result<ArrayOp<ArrayCast<Self, O>>, Error>;
}

pub trait NDArrayCompare<T: CDatatype, O: NDArray>: NDArray {
    fn eq<'a>(&'a self, other: &'a O) -> Result<ArrayOp<ArrayCompare<'a, Self, O>>, Error> {
        let shape = check_shape(self.shape(), other.shape())?;
        Ok(ArrayOp::new(ArrayCompare::eq(self, other), shape))
    }

    fn gt<'a>(&'a self, other: &'a O) -> Result<ArrayOp<ArrayCompare<'a, Self, O>>, Error> {
        let shape = check_shape(self.shape(), other.shape())?;
        Ok(ArrayOp::new(ArrayCompare::gt(self, other), shape))
    }

    fn gte<'a>(&'a self, other: &'a O) -> Result<ArrayOp<ArrayCompare<'a, Self, O>>, Error> {
        let shape = check_shape(self.shape(), other.shape())?;
        Ok(ArrayOp::new(ArrayCompare::gte(self, other), shape))
    }

    fn lt<'a>(&'a self, other: &'a O) -> Result<ArrayOp<ArrayCompare<'a, Self, O>>, Error> {
        let shape = check_shape(self.shape(), other.shape())?;
        Ok(ArrayOp::new(ArrayCompare::lt(self, other), shape))
    }

    fn lte<'a>(&'a self, other: &'a O) -> Result<ArrayOp<ArrayCompare<'a, Self, O>>, Error> {
        let shape = check_shape(self.shape(), other.shape())?;
        Ok(ArrayOp::new(ArrayCompare::lte(self, other), shape))
    }

    fn ne<'a>(&'a self, other: &'a O) -> Result<ArrayOp<ArrayCompare<'a, Self, O>>, Error> {
        let shape = check_shape(self.shape(), other.shape())?;
        Ok(ArrayOp::new(ArrayCompare::ne(self, other), shape))
    }
}

pub trait NDArrayCompareScalar<T>: NDArray {
    fn eq(&self, other: T) -> Result<ArrayOp<ArrayCompareScalar<Self, T>>, Error> {
        let shape = self.shape().to_vec();
        Ok(ArrayOp::new(ArrayCompareScalar::eq(self, other), shape))
    }

    fn gt(&self, other: T) -> Result<ArrayOp<ArrayCompareScalar<Self, T>>, Error> {
        let shape = self.shape().to_vec();
        Ok(ArrayOp::new(ArrayCompareScalar::gt(self, other), shape))
    }

    fn gte(&self, other: T) -> Result<ArrayOp<ArrayCompareScalar<Self, T>>, Error> {
        let shape = self.shape().to_vec();
        Ok(ArrayOp::new(ArrayCompareScalar::gte(self, other), shape))
    }

    fn lt(&self, other: T) -> Result<ArrayOp<ArrayCompareScalar<Self, T>>, Error> {
        let shape = self.shape().to_vec();
        Ok(ArrayOp::new(ArrayCompareScalar::lt(self, other), shape))
    }

    fn lte(&self, other: T) -> Result<ArrayOp<ArrayCompareScalar<Self, T>>, Error> {
        let shape = self.shape().to_vec();
        Ok(ArrayOp::new(ArrayCompareScalar::lte(self, other), shape))
    }

    fn ne(&self, other: T) -> Result<ArrayOp<ArrayCompareScalar<Self, T>>, Error> {
        let shape = self.shape().to_vec();
        Ok(ArrayOp::new(ArrayCompareScalar::ne(self, other), shape))
    }
}

pub trait MatrixMath<T: CDatatype, O: NDArrayRead<T>>: NDArray {
    fn matmul<'a>(&'a self, other: &'a O) -> Result<ArrayOp<MatMul<'a, Self, O>>, Error> {
        let ndim = self.ndim();
        let prefix = &self.shape()[..ndim - 2];

        if other.ndim() != ndim {
            return Err(Error::Bounds(format!(
                "matrix multiply expects at least two dimensions but found shapes {:?} and {:?}",
                self.shape(),
                other.shape()
            )));
        } else if &other.shape()[..ndim - 2] != prefix {
            return Err(Error::Bounds(format!(
                "matrix multiply requires the same batch shape, not {:?} and {:?}",
                prefix,
                &other.shape()[..ndim - 2]
            )));
        }

        let a = self.shape()[ndim - 2];
        let b = self.shape()[ndim - 1];
        let c = other.shape()[ndim - 1];

        if other.shape()[ndim - 2] != b {
            return Err(Error::Bounds(format!(
                "invalid dimensions for matrix multiply: {:?} and {:?}",
                self.shape(),
                other.shape()
            )));
        }

        let mut shape = Vec::with_capacity(ndim);
        shape.extend_from_slice(prefix);
        shape.push(a);
        shape.push(c);

        let op = MatMul::new(self, other);
        Ok(ArrayOp::new(op, shape))
    }
}

pub trait NDArrayReduce<T: CDatatype>: NDArray + NDArrayRead<T> + fmt::Debug {
    fn all(&self) -> Result<bool, Error> {
        let queue = autoqueue(None)?;
        let input = self.read(queue.clone())?;
        kernels::reduce_all(queue, input).map_err(Error::from)
    }

    fn any(&self) -> Result<bool, Error> {
        let queue = autoqueue(None)?;
        let input = self.read(queue.clone())?;
        kernels::reduce_any(queue, input).map_err(Error::from)
    }

    fn max(&self) -> Result<T, Error> {
        todo!()
    }

    fn max_axis(&self, axis: usize) -> Result<ArrayOp<ArrayReduce<Self>>, Error> {
        todo!()
    }

    fn min(&self) -> Result<T, Error> {
        todo!()
    }

    fn min_axis(&self, axis: usize) -> Result<ArrayOp<ArrayReduce<Self>>, Error> {
        todo!()
    }

    fn product(&self) -> Result<T, Error> {
        todo!()
    }

    fn product_axis(&self, axis: usize) -> Result<ArrayOp<ArrayReduce<Self>>, Error> {
        todo!()
    }

    fn sum(&self) -> Result<T, Error> {
        let queue = autoqueue(None)?;
        let input = self.read(queue.clone())?;
        kernels::reduce(T::zero(), "+", queue, input, std::iter::Sum::sum).map_err(Error::from)
    }

    fn sum_axis(&self, axis: usize) -> Result<ArrayOp<ArrayReduce<Self>>, Error> {
        if axis >= self.ndim() {
            return Err(Error::Bounds(format!(
                "axis {} is out of bounds for {:?}",
                axis, self
            )));
        }

        let shape = if self.ndim() == 1 {
            vec![1]
        } else {
            let mut shape = vec![0; self.ndim() - 1];
            shape[..axis].copy_from_slice(&self.shape()[..axis]);
            shape[axis..].copy_from_slice(&self.shape()[(axis + 1)..]);
            shape
        };

        let op = ArrayReduce::sum(self, axis);

        Ok(ArrayOp::new(op, shape))
    }
}

pub trait NDArrayTransform: NDArray {
    type Slice: NDArray;
    type View: NDArray;

    fn broadcast<'a>(&'a self, shape: Shape) -> Result<Self::View, Error>;

    fn expand_dims(&self, mut axes: Vec<usize>) -> Result<Self, Error> {
        axes.sort();

        let mut shape = Vec::with_capacity(self.ndim() + axes.len());
        shape.extend_from_slice(self.shape());

        for x in axes.into_iter().rev() {
            shape.insert(x, 1);
        }

        self.reshape(shape)
    }

    fn transpose(&self, axes: Option<Vec<usize>>) -> Result<Self::View, Error>;

    fn reshape(&self, shape: Shape) -> Result<Self, Error>;

    fn slice<Bounds: IntoIterator<Item = AxisBound>>(
        &self,
        bounds: Bounds,
    ) -> Result<Self::Slice, Error>;
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
