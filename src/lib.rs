extern crate ocl;

use std::fmt;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Range};

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

pub trait CDatatype: OclPrm + Add<Output = Self> + AddAssign + PartialOrd + Sum {
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

pub trait NDArrayRead: NDArray {
    type Out: CDatatype;

    fn copy(&self) -> Result<ArrayBase<Self::Out>, Error> {
        let shape = self.shape().to_vec();

        let queue = autoqueue(None)?;
        let mut data = vec![Self::Out::zero(); self.size()];
        let buffer = self.read(queue)?;
        buffer.read(&mut data).enq()?;

        ArrayBase::from_vec(shape, data)
    }

    fn read(&self, queue: Queue) -> Result<Buffer<Self::Out>, Error>;
}

pub trait NDArrayWrite<O: NDArrayRead>: NDArray {
    fn write(&self, other: &O) -> Result<(), Error>;
}

pub trait NDArrayWriteScalar: NDArrayRead {
    fn write(&self, scalar: Self::Out) -> Result<(), Error>;
}

pub trait NDArrayBoolean<O>: NDArrayRead
where
    O: NDArrayRead<Out = Self::Out>,
{
    fn and<'a>(&'a self, other: &'a O) -> Result<ArrayOp<ArrayBoolean<'a, Self, O>>, Error> {
        let shape = check_shape(self.shape(), other.shape())?;
        let op = ArrayBoolean::and(self, other);
        Ok(ArrayOp::new(op, shape))
    }

    fn or<'a>(&'a self, other: &'a O) -> Result<ArrayOp<ArrayBoolean<'a, Self, O>>, Error> {
        let shape = check_shape(self.shape(), other.shape())?;
        let op = ArrayBoolean::or(self, other);
        Ok(ArrayOp::new(op, shape))
    }

    fn xor<'a>(&'a self, other: &'a O) -> Result<ArrayOp<ArrayBoolean<'a, Self, O>>, Error> {
        let shape = check_shape(self.shape(), other.shape())?;
        let op = ArrayBoolean::xor(self, other);
        Ok(ArrayOp::new(op, shape))
    }
}

pub trait NDArrayAbs: NDArray + Clone {
    fn abs(&self) -> ArrayOp<ArrayUnary<Self>> {
        let op = ArrayUnary::abs(self.clone());
        ArrayOp::new(op, self.shape().to_vec())
    }
}

pub trait NDArrayExp: NDArray + Clone {
    fn exp(&self) -> ArrayOp<ArrayUnary<Self>> {
        let op = ArrayUnary::exp(self.clone());
        ArrayOp::new(op, self.shape().to_vec())
    }
}

pub trait NDArrayMath<O: NDArrayRead<Out = f64>>: NDArrayRead {
    fn log<'a>(&'a self, base: &'a O) -> Result<ArrayOp<ArrayDualFloat<&'a Self, &'a O>>, Error> {
        let shape = check_shape(self.shape(), base.shape())?;
        let op = ArrayDualFloat::log(self, base);
        Ok(ArrayOp::new(op, shape))
    }

    fn pow<'a>(&'a self, exp: &'a O) -> Result<ArrayOp<ArrayDualFloat<&'a Self, &'a O>>, Error> {
        let shape = check_shape(self.shape(), exp.shape())?;
        let op = ArrayDualFloat::pow(self, exp);
        Ok(ArrayOp::new(op, shape))
    }
}

pub trait NDArrayMathScalar: NDArrayRead + Clone {
    fn log(&self, base: f64) -> ArrayOp<ArrayScalar<f64, Self>> {
        let shape = self.shape().to_vec();
        let op = ArrayScalar::log(self.clone(), base);
        ArrayOp::new(op, shape)
    }

    fn pow(&self, exp: f64) -> ArrayOp<ArrayScalar<f64, Self>> {
        let shape = self.shape().to_vec();
        let op = ArrayScalar::pow(self.clone(), exp);
        ArrayOp::new(op, shape)
    }
}

pub trait NDArrayNumeric: NDArray + Clone {
    fn is_inf(&self) -> ArrayOp<ArrayUnary<Self>> {
        let shape = self.shape().to_vec();
        let op = ArrayUnary::inf(self.clone());
        ArrayOp::new(op, shape)
    }

    fn is_nan(&self) -> ArrayOp<ArrayUnary<Self>> {
        let shape = self.shape().to_vec();
        let op = ArrayUnary::nan(self.clone());
        ArrayOp::new(op, shape)
    }
}

pub trait NDArrayTrig: NDArray {
    fn asin(&self) -> ArrayOp<ArrayUnary<Self>>;

    fn sin(&self) -> ArrayOp<ArrayUnary<Self>>;

    fn sinh(&self) -> ArrayOp<ArrayUnary<Self>>;

    fn acos(&self) -> ArrayOp<ArrayUnary<Self>>;

    fn cos(&self) -> ArrayOp<ArrayUnary<Self>>;

    fn cosh(&self) -> ArrayOp<ArrayUnary<Self>>;

    fn atan(&self) -> ArrayOp<ArrayUnary<Self>>;

    fn tan(&self) -> ArrayOp<ArrayUnary<Self>>;

    fn tanh(&self) -> ArrayOp<ArrayUnary<Self>>;
}

pub trait NDArrayCast<O>: NDArray {
    fn cast(&self) -> ArrayOp<ArrayCast<Self, O>> {
        let shape = self.shape().to_vec();
        let op = ArrayCast::new(self);
        ArrayOp::new(op, shape)
    }
}

pub trait NDArrayCompare<O: NDArray>: NDArray {
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
    fn eq_scalar(&self, other: T) -> ArrayOp<ArrayCompareScalar<Self, T>> {
        let shape = self.shape().to_vec();
        ArrayOp::new(ArrayCompareScalar::eq(self, other), shape)
    }

    fn gt_scalar(&self, other: T) -> ArrayOp<ArrayCompareScalar<Self, T>> {
        let shape = self.shape().to_vec();
        ArrayOp::new(ArrayCompareScalar::gt(self, other), shape)
    }

    fn gte_scalar(&self, other: T) -> ArrayOp<ArrayCompareScalar<Self, T>> {
        let shape = self.shape().to_vec();
        ArrayOp::new(ArrayCompareScalar::gte(self, other), shape)
    }

    fn lt_scalar(&self, other: T) -> ArrayOp<ArrayCompareScalar<Self, T>> {
        let shape = self.shape().to_vec();
        ArrayOp::new(ArrayCompareScalar::lt(self, other), shape)
    }

    fn lte_scalar(&self, other: T) -> ArrayOp<ArrayCompareScalar<Self, T>> {
        let shape = self.shape().to_vec();
        ArrayOp::new(ArrayCompareScalar::lte(self, other), shape)
    }

    fn ne_scalar(&self, other: T) -> ArrayOp<ArrayCompareScalar<Self, T>> {
        let shape = self.shape().to_vec();
        ArrayOp::new(ArrayCompareScalar::ne(self, other), shape)
    }
}

pub trait MatrixMath<O: NDArrayRead>: NDArrayRead {
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

pub trait NDArrayReduce: NDArrayRead + Clone + fmt::Debug {
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

    fn max(&self) -> Result<Self::Out, Error> {
        let queue = autoqueue(None)?;
        let input = self.read(queue.clone())?;
        kernels::reduce(Self::Out::zero(), "max", queue, input, |l, r| {
            if r > l {
                r
            } else {
                l
            }
        })
        .map_err(Error::from)
    }

    fn max_axis(&self, axis: usize) -> Result<ArrayOp<ArrayReduce<Self>>, Error> {
        todo!()
    }

    fn min(&self) -> Result<Self::Out, Error> {
        todo!()
    }

    fn min_axis(&self, axis: usize) -> Result<ArrayOp<ArrayReduce<Self>>, Error> {
        todo!()
    }

    fn product(&self) -> Result<Self::Out, Error> {
        todo!()
    }

    fn product_axis(&self, axis: usize) -> Result<ArrayOp<ArrayReduce<Self>>, Error> {
        todo!()
    }

    fn sum(&self) -> Result<Self::Out, Error> {
        let queue = autoqueue(None)?;
        let input = self.read(queue.clone())?;
        kernels::reduce(Self::Out::zero(), "add", queue, input, Add::add).map_err(Error::from)
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

pub trait NDArrayTransform: NDArray + fmt::Debug {
    type Broadcast: NDArray;
    type Expand: NDArray;
    type Reshape: NDArray;
    type Slice: NDArray;
    type Transpose: NDArray;

    fn broadcast(&self, shape: Shape) -> Result<Self::Broadcast, Error>;

    fn expand_dim(&self, axis: usize) -> Result<Self::Expand, Error>;

    fn expand_dims(&self, axes: Vec<usize>) -> Result<Self::Expand, Error>;

    fn reshape(&self, shape: Shape) -> Result<Self::Reshape, Error>;

    fn slice(&self, bounds: Vec<AxisBound>) -> Result<Self::Slice, Error>;

    fn transpose(&self, axes: Option<Vec<usize>>) -> Result<Self::Transpose, Error>;
}

#[derive(Clone)]
pub enum AxisBound {
    At(usize),
    In(usize, usize, usize),
    Of(Vec<usize>),
}

impl AxisBound {
    pub fn size(&self) -> usize {
        match self {
            Self::At(_) => 0,
            Self::In(start, stop, step) => (stop - start) / step,
            Self::Of(indices) => indices.len(),
        }
    }
}

impl From<usize> for AxisBound {
    fn from(i: usize) -> Self {
        Self::At(i)
    }
}

impl From<Range<usize>> for AxisBound {
    fn from(range: Range<usize>) -> Self {
        Self::In(range.start, range.end, 1)
    }
}

impl From<Vec<usize>> for AxisBound {
    fn from(indices: Vec<usize>) -> Self {
        Self::Of(indices)
    }
}

impl fmt::Debug for AxisBound {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::At(i) => write!(f, "{}", i),
            Self::In(start, stop, 1) => write!(f, "{}:{}", start, stop),
            Self::In(start, stop, step) => write!(f, "{}:{}:{}", start, stop, step),
            Self::Of(indices) => write!(f, "{:?}", indices),
        }
    }
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
