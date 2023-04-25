extern crate ocl;

use std::fmt;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Range};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

pub use array::*;
use ops::*;

mod array;
mod kernels;
mod ops;

// TODO: set this to a reasonable value after implementing CPU operations
const GPU_MIN_DEFAULT: usize = 0;

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

pub trait CDatatype: ocl::OclPrm + Add<Output = Self> + AddAssign + PartialOrd + Sum {
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

#[derive(Clone, Default)]
struct DeviceList {
    devices: Vec<ocl::Device>,
    next: Arc<AtomicUsize>,
}

impl DeviceList {
    fn is_empty(&self) -> bool {
        self.devices.is_empty()
    }

    fn next(&self) -> Option<ocl::Device> {
        if self.devices.is_empty() {
            None
        } else {
            let idx = self.next.fetch_add(1, Ordering::Relaxed);
            self.devices.get(idx % self.devices.len()).copied()
        }
    }
}

impl From<Vec<ocl::Device>> for DeviceList {
    fn from(devices: Vec<ocl::Device>) -> Self {
        Self {
            devices,
            next: Arc::new(AtomicUsize::default()),
        }
    }
}

#[derive(Clone)]
pub enum Buffer<T: CDatatype> {
    CL(ocl::Buffer<T>),
    Host(Vec<T>),
}

impl<T: CDatatype> From<Vec<T>> for Buffer<T> {
    fn from(buffer: Vec<T>) -> Self {
        Self::Host(buffer)
    }
}

impl<T: CDatatype> From<ocl::Buffer<T>> for Buffer<T> {
    fn from(buffer: ocl::Buffer<T>) -> Self {
        Self::CL(buffer)
    }
}

#[derive(Clone, Default)]
pub struct Platform {
    cl_cpus: DeviceList,
    cl_gpus: DeviceList,
    cl_accs: DeviceList,
}

impl Platform {
    fn has_gpu(&self) -> bool {
        !self.cl_gpus.is_empty()
    }

    fn next_cpu(&self) -> Option<ocl::Device> {
        self.cl_cpus.next()
    }

    fn next_gpu(&self) -> Option<ocl::Device> {
        self.cl_gpus.next()
    }

    fn next_acc(&self) -> Option<ocl::Device> {
        self.cl_accs.next()
    }
}

impl TryFrom<ocl::Platform> for Platform {
    type Error = ocl::Error;

    fn try_from(cl_platform: ocl::Platform) -> Result<Self, Self::Error> {
        let cl_cpus = ocl::Device::list(cl_platform, Some(ocl::DeviceType::CPU))?;
        let cl_gpus = ocl::Device::list(cl_platform, Some(ocl::DeviceType::GPU))?;
        let cl_accs = ocl::Device::list(cl_platform, Some(ocl::DeviceType::ACCELERATOR))?;

        Ok(Self {
            cl_cpus: cl_cpus.into(),
            cl_gpus: cl_gpus.into(),
            cl_accs: cl_accs.into(),
        })
    }
}

#[derive(Clone)]
pub struct Context {
    platform: Platform,
    cl_context: ocl::Context,
    gpu_min: usize,
    acc_min: usize,
}

impl Context {
    pub fn default() -> Result<Self, Error> {
        let cl_platform = ocl::Platform::first()?;
        let cl_context = ocl::Context::builder().platform(cl_platform).build()?;
        let platform = Platform::try_from(cl_platform)?;
        let acc_min = if platform.has_gpu() {
            usize::MAX
        } else {
            GPU_MIN_DEFAULT
        };

        Ok(Self {
            platform,
            cl_context,
            gpu_min: GPU_MIN_DEFAULT,
            acc_min,
        })
    }

    fn cl_context(&self) -> &ocl::Context {
        &self.cl_context
    }

    pub fn queue(&self, size_hint: usize) -> Result<Queue, Error> {
        let device = if size_hint < self.gpu_min {
            self.platform.next_cpu()
        } else if size_hint < self.acc_min {
            self.platform
                .next_gpu()
                .or_else(|| self.platform.next_cpu())
        } else {
            self.platform
                .next_acc()
                .or_else(|| self.platform.next_gpu())
                .or_else(|| self.platform.next_cpu())
        };

        if let Some(device) = device {
            Queue::with_device(self.clone(), device)
        } else {
            todo!("CPU queue")
        }
    }
}

enum DeviceQueue {
    CPU,
    CL(ocl::Queue),
}

pub struct Queue {
    context: Context,
    device_queue: DeviceQueue,
}

impl Queue {
    fn with_device(context: Context, device: ocl::Device) -> Result<Self, Error> {
        let cl_queue = ocl::Queue::new(context.cl_context(), device, None)?;

        Ok(Self {
            context,
            device_queue: DeviceQueue::CL(cl_queue),
        })
    }

    fn device_queue(&self) -> &DeviceQueue {
        &self.device_queue
    }

    fn context(&self) -> &Context {
        &self.context
    }
}

pub trait NDArray: Sized {
    type DType: CDatatype;

    fn ndim(&self) -> usize {
        self.shape().len()
    }

    fn size(&self) -> usize {
        self.shape().iter().product()
    }

    fn shape(&self) -> &[usize];
}

pub trait NDArrayRead: NDArray {
    fn read(&self, queue: Queue) -> Result<Buffer<Self::DType>, Error>;
}

pub trait NDArrayWrite<O: NDArray<DType = Self::DType>>: NDArray {
    fn write(&self, other: &O) -> Result<(), Error>;
}

pub trait NDArrayWriteScalar: NDArray {
    fn write(&self, scalar: Self::DType) -> Result<(), Error>;
}

pub trait NDArrayBoolean<O>: NDArray
where
    O: NDArray<DType = Self::DType>,
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

pub trait NDArrayMath<O: NDArray<DType = f64>>: NDArrayRead {
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

pub trait MatrixMath<O: NDArray<DType = Self::DType>>: NDArray {
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
        let context = Context::default()?;
        let queue = context.queue(self.size())?;
        let input = self.read(queue)?;

        match input {
            Buffer::CL(buffer) => {
                let cl_queue = buffer.default_queue().expect("queue").clone();
                kernels::reduce_all(cl_queue, buffer).map_err(Error::from)
            }
            Buffer::Host(buffer) => {
                let zero = Self::DType::zero();
                Ok(buffer.into_iter().all(|n| n != zero))
            }
        }
    }

    fn any(&self) -> Result<bool, Error> {
        let context = Context::default()?;
        let queue = context.queue(self.size())?;
        let input = self.read(queue)?;

        match input {
            Buffer::CL(buffer) => {
                let cl_queue = buffer.default_queue().expect("queue").clone();
                kernels::reduce_any(cl_queue, buffer).map_err(Error::from)
            }
            Buffer::Host(buffer) => {
                let zero = Self::DType::zero();
                Ok(buffer.into_iter().any(|n| n != zero))
            }
        }
    }

    fn max(&self) -> Result<Self::DType, Error> {
        let context = Context::default()?;
        let queue = context.queue(self.size())?;
        let input = self.read(queue)?;
        let collector = |l, r| {
            if r > l {
                r
            } else {
                l
            }
        };

        match input {
            Buffer::CL(buffer) => {
                let cl_queue = buffer.default_queue().expect("queue").clone();
                kernels::reduce(Self::DType::zero(), "max", cl_queue, buffer, collector)
                    .map_err(Error::from)
            }
            Buffer::Host(buffer) => {
                let zero = Self::DType::zero();
                Ok(buffer.into_iter().fold(zero, collector))
            }
        }
    }

    fn max_axis(&self, axis: usize) -> Result<ArrayOp<ArrayReduce<Self>>, Error> {
        todo!()
    }

    fn min(&self) -> Result<Self::DType, Error> {
        todo!()
    }

    fn min_axis(&self, axis: usize) -> Result<ArrayOp<ArrayReduce<Self>>, Error> {
        todo!()
    }

    fn product(&self) -> Result<Self::DType, Error> {
        todo!()
    }

    fn product_axis(&self, axis: usize) -> Result<ArrayOp<ArrayReduce<Self>>, Error> {
        todo!()
    }

    fn sum(&self) -> Result<Self::DType, Error> {
        let context = Context::default()?;
        let queue = context.queue(self.size())?;
        let input = self.read(queue)?;

        match input {
            Buffer::CL(buffer) => {
                let cl_queue = buffer.default_queue().expect("queue").clone();
                kernels::reduce(Self::DType::zero(), "add", cl_queue, buffer, Add::add)
                    .map_err(Error::from)
            }
            Buffer::Host(buffer) => {
                let zero = Self::DType::zero();
                Ok(buffer.into_iter().fold(zero, Add::add))
            }
        }
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
