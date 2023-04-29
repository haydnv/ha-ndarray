#[cfg(feature = "opencl")]
extern crate ocl;

use std::convert::identity;
use std::fmt;
use std::iter::Sum;
use std::ops::{Add, Div, Mul, Range, Rem, Sub};
use std::sync::Arc;

use rayon::prelude::*;
#[allow(unused_imports)]
use safecast::{as_type, AsType};

pub use array::*;
use ops::*;

mod array;
#[cfg(feature = "opencl")]
mod cl_programs;
mod ops;

pub mod construct {
    pub use super::ops::{RandomNormal, RandomUniform};
}

const GPU_MIN_DEFAULT: usize = 1024;

pub enum Error {
    Bounds(String),
    Interface(String),
    #[cfg(feature = "opencl")]
    OCL(ocl::Error),
}

#[cfg(feature = "opencl")]
impl From<ocl::Error> for Error {
    fn from(cause: ocl::Error) -> Self {
        Self::OCL(cause)
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Bounds(cause) => f.write_str(cause),
            Self::Interface(cause) => f.write_str(cause),
            #[cfg(feature = "opencl")]
            Self::OCL(cause) => cause.fmt(f),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Bounds(cause) => f.write_str(cause),
            Self::Interface(cause) => f.write_str(cause),
            #[cfg(feature = "opencl")]
            Self::OCL(cause) => cause.fmt(f),
        }
    }
}

impl std::error::Error for Error {}

pub type Shape = Vec<usize>;

// TODO: is there a better way to implement the OclPrm trait bound?
#[cfg(feature = "opencl")]
pub trait CDatatype:
    ocl::OclPrm
    + Copy
    + Add<Output = Self>
    + Div<Output = Self>
    + Mul<Output = Self>
    + Rem<Output = Self>
    + Sub<Output = Self>
    + PartialEq
    + PartialOrd
    + Sum
    + Send
    + Sync
{
    const TYPE_STR: &'static str;

    type Float: Float;
    type Neg: CDatatype;

    fn one() -> Self;

    fn zero() -> Self;

    fn from_f64(float: f64) -> Self;

    fn abs(self) -> Self;

    fn exp(self) -> Self;

    fn log(self, base: f64) -> Self {
        Self::from_f64(self.to_f64().log(base))
    }

    fn neg(self) -> Self::Neg;

    fn not(self) -> u8 {
        if self == Self::zero() {
            1
        } else {
            0
        }
    }

    fn pow(self, exp: f64) -> Self {
        Self::from_f64(self.to_f64().powf(exp))
    }

    fn to_f64(self) -> f64;
}

#[cfg(not(feature = "opencl"))]
pub trait CDatatype:
    Copy
    + Add<Output = Self>
    + Div<Output = Self>
    + Mul<Output = Self>
    + Rem<Output = Self>
    + Sub<Output = Self>
    + PartialEq
    + PartialOrd
    + Sum
    + Send
    + Sync
{
    const TYPE_STR: &'static str;

    type Float: Float;
    type Neg: CDatatype;

    fn one() -> Self;

    fn zero() -> Self;

    fn from_f64(float: f64) -> Self;

    fn abs(self) -> Self;

    fn exp(self) -> Self;

    fn log(self, base: f64) -> Self {
        Self::from_f64(self.to_f64().log(base))
    }

    fn neg(self) -> Self::Neg;

    fn not(self) -> u8 {
        if self == Self::zero() {
            1
        } else {
            0
        }
    }

    fn pow(self, exp: f64) -> Self {
        Self::from_f64(self.to_f64().powf(exp))
    }

    fn to_f64(self) -> f64;
}

macro_rules! c_type {
    ($t:ty, $ct:expr, $zero:expr, $one:expr, $abs:expr, $float:ty, $neg:ty) => {
        impl CDatatype for $t {
            const TYPE_STR: &'static str = $ct;

            type Float = $float;
            type Neg = $neg;

            fn one() -> Self {
                $one
            }

            fn zero() -> Self {
                $zero
            }

            fn from_f64(float: f64) -> Self {
                float as $t
            }

            fn abs(self) -> Self {
                $abs(self)
            }

            fn exp(self) -> Self {
                Self::from_f64(std::f64::consts::E.pow(self.to_f64()))
            }

            fn neg(self) -> Self::Neg {
                if self >= Self::zero() {
                    self as $neg
                } else {
                    -(self as $neg)
                }
            }

            fn to_f64(self) -> f64 {
                self as f64
            }
        }
    };
}

c_type!(f32, "float", 0., 1., f32::abs, f32, f32);
c_type!(f64, "double", 0., 1., f64::abs, f64, f64);
c_type!(u8, "uchar", 0, 1, identity, f32, i8);
c_type!(u16, "ushort", 0, 1, identity, f32, i16);
c_type!(u32, "uint", 0, 1, identity, f32, i32);
c_type!(u64, "ulong", 0, 1, identity, f64, i64);
c_type!(i8, "char", 0, 1, i8::abs, f32, i8);
c_type!(i16, "short", 0, 1, i16::abs, f32, i16);
c_type!(i32, "int", 0, 1, i32::abs, f32, i32);
c_type!(i64, "long", 0, 1, i64::abs, f64, i64);

pub trait Float: CDatatype {
    fn is_inf(self) -> u8;

    fn is_nan(self) -> u8;
}

impl Float for f32 {
    fn is_inf(self) -> u8 {
        if f32::is_infinite(self) {
            1
        } else {
            0
        }
    }

    fn is_nan(self) -> u8 {
        if f32::is_nan(self) {
            1
        } else {
            0
        }
    }
}

impl Float for f64 {
    fn is_inf(self) -> u8 {
        if f64::is_infinite(self) {
            1
        } else {
            0
        }
    }

    fn is_nan(self) -> u8 {
        if f64::is_nan(self) {
            1
        } else {
            0
        }
    }
}

#[cfg(feature = "opencl")]
#[derive(Clone, Default)]
struct DeviceList {
    devices: Vec<ocl::Device>,
    next: Arc<std::sync::atomic::AtomicUsize>,
}

#[cfg(feature = "opencl")]
impl DeviceList {
    fn is_empty(&self) -> bool {
        self.devices.is_empty()
    }

    fn iter(&self) -> std::slice::Iter<ocl::Device> {
        self.devices.iter()
    }

    fn next(&self) -> Option<ocl::Device> {
        if self.devices.is_empty() {
            None
        } else {
            let idx = self.next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.devices.get(idx % self.devices.len()).copied()
        }
    }
}

#[cfg(feature = "opencl")]
impl From<Vec<ocl::Device>> for DeviceList {
    fn from(devices: Vec<ocl::Device>) -> Self {
        Self {
            devices,
            next: Arc::new(Default::default()),
        }
    }
}

#[cfg(feature = "opencl")]
impl FromIterator<ocl::Device> for DeviceList {
    fn from_iter<T: IntoIterator<Item = ocl::Device>>(iter: T) -> Self {
        Self::from(iter.into_iter().collect::<Vec<ocl::Device>>())
    }
}

#[cfg(feature = "opencl")]
#[derive(Clone)]
pub enum Buffer<T: CDatatype> {
    Host(Arc<Vec<T>>),
    CL(ocl::Buffer<T>),
}

#[cfg(not(feature = "opencl"))]
#[derive(Clone)]
pub enum Buffer<T: CDatatype> {
    Host(Arc<Vec<T>>),
}

#[cfg(feature = "opencl")]
impl<T: CDatatype> From<ocl::Buffer<T>> for Buffer<T> {
    fn from(buffer: ocl::Buffer<T>) -> Self {
        Self::CL(buffer)
    }
}

impl<T: CDatatype> From<Vec<T>> for Buffer<T> {
    fn from(buffer: Vec<T>) -> Self {
        Self::Host(Arc::new(buffer))
    }
}

#[derive(Clone, Default)]
pub struct Platform {
    #[cfg(feature = "opencl")]
    cl_cpus: DeviceList,
    #[cfg(feature = "opencl")]
    cl_gpus: DeviceList,
    #[cfg(feature = "opencl")]
    cl_accs: DeviceList,
    #[cfg(feature = "opencl")]
    cl_platform: ocl::Platform,
}

impl Platform {
    #[cfg(feature = "opencl")]
    fn devices(&self) -> DeviceList {
        self.cl_cpus
            .iter()
            .copied()
            .chain(self.cl_gpus.iter().copied())
            .chain(self.cl_accs.iter().copied())
            .collect()
    }

    #[cfg(feature = "opencl")]
    fn has_gpu(&self) -> bool {
        !self.cl_gpus.is_empty()
    }

    #[cfg(feature = "opencl")]
    fn next_cpu(&self) -> Option<ocl::Device> {
        self.cl_cpus.next()
    }

    #[cfg(feature = "opencl")]
    fn next_gpu(&self) -> Option<ocl::Device> {
        self.cl_gpus.next()
    }

    #[cfg(feature = "opencl")]
    fn next_acc(&self) -> Option<ocl::Device> {
        self.cl_accs.next()
    }
}

#[cfg(feature = "opencl")]
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
            cl_platform,
        })
    }
}

#[derive(Clone)]
pub struct Context {
    platform: Platform,
    gpu_min: usize,
    acc_min: usize,
    #[cfg(feature = "opencl")]
    cl_context: ocl::Context,
}

impl Context {
    #[cfg(feature = "opencl")]
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
            gpu_min: GPU_MIN_DEFAULT,
            acc_min,
            cl_context,
        })
    }

    #[cfg(not(feature = "opencl"))]
    pub fn default() -> Result<Self, Error> {
        Ok(Self {
            platform: Platform {},
            gpu_min: GPU_MIN_DEFAULT,
            acc_min: GPU_MIN_DEFAULT,
        })
    }

    #[cfg(feature = "opencl")]
    pub fn new(gpu_min: usize, acc_min: usize, platform: Option<Platform>) -> Result<Self, Error> {
        let platform = if let Some(platform) = platform {
            platform
        } else {
            let cl_platform = ocl::Platform::first()?;
            Platform::try_from(cl_platform.clone())?
        };

        let cl_context = ocl::Context::builder()
            .platform(platform.cl_platform.clone())
            .build()?;

        Ok(Self {
            platform,
            gpu_min,
            acc_min,
            cl_context,
        })
    }

    #[cfg(not(feature = "opencl"))]
    pub fn new(gpu_min: usize, acc_min: usize, platform: Option<Platform>) -> Result<Self, Error> {
        let platform = platform.unwrap_or_default();

        Ok(Self {
            platform,
            gpu_min,
            acc_min,
        })
    }

    #[cfg(feature = "opencl")]
    fn cl_context(&self) -> &ocl::Context {
        &self.cl_context
    }

    pub fn queue(&self, size_hint: usize) -> Result<Queue, Error> {
        #[cfg(feature = "opencl")]
        if let Some(device) = if size_hint < self.gpu_min {
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
        } {
            return Queue::with_device(self.clone(), device);
        }

        Ok(Queue::default(self.clone()))
    }

    #[cfg(feature = "opencl")]
    fn cl_queue(&self, device_queue: &DeviceQueue, size_hint: usize) -> Result<ocl::Queue, Error> {
        if let DeviceQueue::CL(cl_queue) = device_queue {
            Ok(cl_queue.clone())
        } else if let Some(device) = if size_hint < self.gpu_min {
            self.platform
                .next_cpu()
                .or_else(|| self.platform.next_gpu())
                .or_else(|| self.platform.next_acc())
        } else if size_hint < self.acc_min {
            self.platform
                .next_gpu()
                .or_else(|| self.platform.next_cpu())
                .or_else(|| self.platform.next_acc())
        } else {
            self.platform
                .next_acc()
                .or_else(|| self.platform.next_gpu())
                .or_else(|| self.platform.next_cpu())
        } {
            ocl::Queue::new(&self.cl_context, device, None).map_err(Error::from)
        } else {
            Err(Error::Interface("no OpenCL devices available".to_string()))
        }
    }
}

enum DeviceQueue {
    Host,
    #[cfg(feature = "opencl")]
    CL(ocl::Queue),
}

#[cfg(feature = "opencl")]
as_type!(DeviceQueue, CL, ocl::Queue);

pub struct Queue {
    context: Context,
    device_queue: DeviceQueue,
}

impl Queue {
    fn default(context: Context) -> Self {
        Self {
            context,
            device_queue: DeviceQueue::Host,
        }
    }

    #[cfg(feature = "opencl")]
    fn with_device(context: Context, device: ocl::Device) -> Result<Self, Error> {
        let cl_queue = ocl::Queue::new(&context.cl_context, device, None)?;

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

pub trait NDArray: Send + Sync + Sized {
    type DType: CDatatype;

    fn context(&self) -> &Context;

    fn ndim(&self) -> usize {
        self.shape().len()
    }

    fn size(&self) -> usize {
        self.shape().iter().product()
    }

    fn shape(&self) -> &[usize];
}

pub trait NDArrayRead: NDArray + fmt::Debug {
    fn read(&self, queue: &Queue) -> Result<Buffer<Self::DType>, Error>;

    fn to_vec(&self, queue: &Queue) -> Result<Arc<Vec<Self::DType>>, Error> {
        match self.read(queue)? {
            Buffer::Host(buffer) => {
                debug_assert_eq!(buffer.len(), self.size());
                Ok(buffer)
            }
            #[cfg(feature = "opencl")]
            Buffer::CL(cl_buffer) => {
                debug_assert_eq!(cl_buffer.len(), self.size());

                let mut buffer = vec![Self::DType::zero(); cl_buffer.len()];
                cl_buffer.read(&mut buffer[..]).enq()?;
                Ok(Arc::new(buffer))
            }
        }
    }

    #[cfg(feature = "opencl")]
    fn to_cl_buffer(&self, queue: &Queue) -> Result<ocl::Buffer<Self::DType>, Error> {
        match self.read(queue)? {
            Buffer::CL(buffer) => {
                debug_assert_eq!(buffer.len(), self.size());
                Ok(buffer)
            }
            Buffer::Host(host_buffer) => {
                debug_assert_eq!(host_buffer.len(), self.size());

                let cl_queue = queue
                    .context()
                    .cl_queue(queue.device_queue(), self.size())?;

                ocl::Buffer::builder()
                    .queue(cl_queue)
                    .len(host_buffer.len())
                    .copy_host_slice(&host_buffer[..])
                    .build()
                    .map_err(Error::from)
            }
        }
    }
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
    fn and<'a>(
        &'a self,
        other: &'a O,
    ) -> Result<ArrayOp<ArrayBoolean<'a, Self::DType, Self, O>>, Error> {
        let shape = check_shape(self.shape(), other.shape())?;
        let op = ArrayBoolean::and(self, other)?;
        Ok(ArrayOp::new(shape, op))
    }

    fn or<'a>(
        &'a self,
        other: &'a O,
    ) -> Result<ArrayOp<ArrayBoolean<'a, Self::DType, Self, O>>, Error> {
        let shape = check_shape(self.shape(), other.shape())?;
        let op = ArrayBoolean::or(self, other)?;
        Ok(ArrayOp::new(shape, op))
    }

    fn xor<'a>(
        &'a self,
        other: &'a O,
    ) -> Result<ArrayOp<ArrayBoolean<'a, Self::DType, Self, O>>, Error> {
        let shape = check_shape(self.shape(), other.shape())?;
        let op = ArrayBoolean::xor(self, other)?;
        Ok(ArrayOp::new(shape, op))
    }
}

impl<T: CDatatype, A: NDArray<DType = T>, O: NDArray<DType = T>> NDArrayBoolean<O> for A {}

pub trait NDArrayAbs: NDArray + Clone {
    fn abs(&self) -> Result<ArrayOp<ArrayUnary<Self::DType, Self::DType, Self>>, Error> {
        let op = ArrayUnary::abs(self.clone())?;
        Ok(ArrayOp::new(self.shape().to_vec(), op))
    }
}

impl<A: NDArray + Clone> NDArrayAbs for A {}

pub trait NDArrayExp: NDArray + Clone {
    fn exp(&self) -> Result<ArrayOp<ArrayUnary<Self::DType, Self::DType, Self>>, Error> {
        let op = ArrayUnary::exp(self.clone())?;
        Ok(ArrayOp::new(self.shape().to_vec(), op))
    }
}

impl<A: NDArray + Clone> NDArrayExp for A {}

pub trait NDArrayMath<O: NDArray + Clone>: NDArray + Clone {
    fn add<'a>(&'a self, rhs: &'a O) -> Result<ArrayOp<ArrayDual<Self::DType, Self, O>>, Error>
    where
        O: NDArray<DType = Self::DType>,
    {
        let shape = check_shape(self.shape(), rhs.shape())?;
        let op = ArrayDual::add(self.clone(), rhs.clone())?;
        Ok(ArrayOp::new(shape, op))
    }

    fn div<'a>(&'a self, rhs: &'a O) -> Result<ArrayOp<ArrayDual<Self::DType, Self, O>>, Error>
    where
        O: NDArray<DType = Self::DType>,
    {
        let shape = check_shape(self.shape(), rhs.shape())?;
        let op = ArrayDual::div(self.clone(), rhs.clone())?;
        Ok(ArrayOp::new(shape, op))
    }

    fn mul<'a>(&'a self, rhs: &'a O) -> Result<ArrayOp<ArrayDual<Self::DType, Self, O>>, Error>
    where
        O: NDArray<DType = Self::DType>,
    {
        let shape = check_shape(self.shape(), rhs.shape())?;
        let op = ArrayDual::mul(self.clone(), rhs.clone())?;
        Ok(ArrayOp::new(shape, op))
    }

    fn rem<'a>(&'a self, rhs: &'a O) -> Result<ArrayOp<ArrayDual<Self::DType, Self, O>>, Error>
    where
        O: NDArray<DType = Self::DType>,
    {
        let shape = check_shape(self.shape(), rhs.shape())?;
        let op = ArrayDual::rem(self.clone(), rhs.clone())?;
        Ok(ArrayOp::new(shape, op))
    }

    fn sub<'a>(&'a self, rhs: &'a O) -> Result<ArrayOp<ArrayDual<Self::DType, Self, O>>, Error>
    where
        O: NDArray<DType = Self::DType>,
    {
        let shape = check_shape(self.shape(), rhs.shape())?;
        let op = ArrayDual::sub(self.clone(), rhs.clone())?;
        Ok(ArrayOp::new(shape, op))
    }

    fn log<'a>(&'a self, base: &'a O) -> Result<ArrayOp<ArrayDualFloat<Self, O>>, Error>
    where
        O: NDArray<DType = f64>,
    {
        let shape = check_shape(self.shape(), base.shape())?;
        let op = ArrayDualFloat::log(self.clone(), base.clone());
        Ok(ArrayOp::new(shape, op))
    }

    fn pow<'a>(&'a self, exp: &'a O) -> Result<ArrayOp<ArrayDualFloat<Self, O>>, Error>
    where
        O: NDArray<DType = f64>,
    {
        let shape = check_shape(self.shape(), exp.shape())?;
        let op = ArrayDualFloat::pow(self.clone(), exp.clone());
        Ok(ArrayOp::new(shape, op))
    }
}

impl<A: NDArray + Clone, O: NDArray + Clone> NDArrayMath<O> for A {}

pub trait NDArrayMathScalar: NDArray + Clone {
    fn add_scalar(
        &self,
        rhs: Self::DType,
    ) -> Result<ArrayOp<ArrayScalar<Self::DType, Self>>, Error> {
        let shape = self.shape().to_vec();
        let op = ArrayScalar::add(self.clone(), rhs)?;
        Ok(ArrayOp::new(shape, op))
    }

    fn div_scalar(
        &self,
        rhs: Self::DType,
    ) -> Result<ArrayOp<ArrayScalar<Self::DType, Self>>, Error> {
        let shape = self.shape().to_vec();
        let op = ArrayScalar::div(self.clone(), rhs)?;
        Ok(ArrayOp::new(shape, op))
    }

    fn mul_scalar(
        &self,
        rhs: Self::DType,
    ) -> Result<ArrayOp<ArrayScalar<Self::DType, Self>>, Error> {
        let shape = self.shape().to_vec();
        let op = ArrayScalar::mul(self.clone(), rhs)?;
        Ok(ArrayOp::new(shape, op))
    }

    fn rem_scalar(
        &self,
        rhs: Self::DType,
    ) -> Result<ArrayOp<ArrayScalar<Self::DType, Self>>, Error> {
        let shape = self.shape().to_vec();
        let op = ArrayScalar::rem(self.clone(), rhs)?;
        Ok(ArrayOp::new(shape, op))
    }

    fn sub_scalar(
        &self,
        rhs: Self::DType,
    ) -> Result<ArrayOp<ArrayScalar<Self::DType, Self>>, Error> {
        let shape = self.shape().to_vec();
        let op = ArrayScalar::sub(self.clone(), rhs)?;
        Ok(ArrayOp::new(shape, op))
    }

    fn log_scalar(&self, base: f64) -> Result<ArrayOp<ArrayScalarFloat<Self::DType, Self>>, Error> {
        let shape = self.shape().to_vec();
        let op = ArrayScalarFloat::log(self.clone(), base)?;
        Ok(ArrayOp::new(shape, op))
    }

    fn pow_scalar(&self, exp: f64) -> Result<ArrayOp<ArrayScalarFloat<Self::DType, Self>>, Error> {
        let shape = self.shape().to_vec();
        let op = ArrayScalarFloat::pow(self.clone(), exp)?;
        Ok(ArrayOp::new(shape, op))
    }
}

impl<A: NDArray + Clone> NDArrayMathScalar for A {}

pub trait NDArrayNumeric: NDArray + Clone
where
    Self::DType: Float,
{
    fn is_inf(&self) -> Result<ArrayOp<ArrayUnary<Self::DType, u8, Self>>, Error> {
        let shape = self.shape().to_vec();
        let op = ArrayUnary::inf(self.clone())?;
        Ok(ArrayOp::new(shape, op))
    }

    fn is_nan(&self) -> Result<ArrayOp<ArrayUnary<Self::DType, u8, Self>>, Error> {
        let shape = self.shape().to_vec();
        let op = ArrayUnary::nan(self.clone()).expect("op");
        Ok(ArrayOp::new(shape, op))
    }
}

impl<A: NDArray + Clone> NDArrayNumeric for A where A::DType: Float {}

pub trait NDArrayTrig: NDArray {
    fn asin(&self) -> ArrayOp<ArrayUnary<Self::DType, <Self::DType as CDatatype>::Float, Self>>;

    fn sin(&self) -> ArrayOp<ArrayUnary<Self::DType, <Self::DType as CDatatype>::Float, Self>>;

    fn sinh(&self) -> ArrayOp<ArrayUnary<Self::DType, <Self::DType as CDatatype>::Float, Self>>;

    fn acos(&self) -> ArrayOp<ArrayUnary<Self::DType, <Self::DType as CDatatype>::Float, Self>>;

    fn cos(&self) -> ArrayOp<ArrayUnary<Self::DType, <Self::DType as CDatatype>::Float, Self>>;

    fn cosh(&self) -> ArrayOp<ArrayUnary<Self::DType, <Self::DType as CDatatype>::Float, Self>>;

    fn atan(&self) -> ArrayOp<ArrayUnary<Self::DType, <Self::DType as CDatatype>::Float, Self>>;

    fn tan(&self) -> ArrayOp<ArrayUnary<Self::DType, <Self::DType as CDatatype>::Float, Self>>;

    fn tanh(&self) -> ArrayOp<ArrayUnary<Self::DType, <Self::DType as CDatatype>::Float, Self>>;
}

// TODO: implement trigonometry methods

pub trait NDArrayCast<O: CDatatype>: NDArray {
    fn cast(&self) -> Result<ArrayOp<ArrayCast<Self, O>>, Error> {
        let shape = self.shape().to_vec();
        let op = ArrayCast::new(self)?;
        Ok(ArrayOp::new(shape, op))
    }
}

impl<A: NDArray, O: CDatatype> NDArrayCast<O> for A {}

pub trait NDArrayCompare<O: NDArray<DType = Self::DType>>: NDArray {
    fn eq<'a>(
        &'a self,
        other: &'a O,
    ) -> Result<ArrayOp<ArrayCompare<'a, Self::DType, Self, O>>, Error> {
        let shape = check_shape(self.shape(), other.shape())?;
        let op = ArrayCompare::eq(self, other)?;
        Ok(ArrayOp::new(shape, op))
    }

    fn gt<'a>(
        &'a self,
        other: &'a O,
    ) -> Result<ArrayOp<ArrayCompare<'a, Self::DType, Self, O>>, Error> {
        let shape = check_shape(self.shape(), other.shape())?;
        let op = ArrayCompare::gt(self, other)?;
        Ok(ArrayOp::new(shape, op))
    }

    fn ge<'a>(
        &'a self,
        other: &'a O,
    ) -> Result<ArrayOp<ArrayCompare<'a, Self::DType, Self, O>>, Error> {
        let shape = check_shape(self.shape(), other.shape())?;
        let op = ArrayCompare::ge(self, other)?;
        Ok(ArrayOp::new(shape, op))
    }

    fn lt<'a>(
        &'a self,
        other: &'a O,
    ) -> Result<ArrayOp<ArrayCompare<'a, Self::DType, Self, O>>, Error> {
        let shape = check_shape(self.shape(), other.shape())?;
        let op = ArrayCompare::lt(self, other)?;
        Ok(ArrayOp::new(shape, op))
    }

    fn le<'a>(
        &'a self,
        other: &'a O,
    ) -> Result<ArrayOp<ArrayCompare<'a, Self::DType, Self, O>>, Error> {
        let shape = check_shape(self.shape(), other.shape())?;
        let op = ArrayCompare::le(self, other)?;
        Ok(ArrayOp::new(shape, op))
    }

    fn ne<'a>(
        &'a self,
        other: &'a O,
    ) -> Result<ArrayOp<ArrayCompare<'a, Self::DType, Self, O>>, Error> {
        let shape = check_shape(self.shape(), other.shape())?;
        let op = ArrayCompare::ne(self, other)?;
        Ok(ArrayOp::new(shape, op))
    }
}

impl<A: NDArray, O: NDArray> NDArrayCompare<O> for A where O: NDArray<DType = A::DType> {}

pub trait NDArrayCompareScalar: NDArray {
    fn eq_scalar(
        &self,
        other: Self::DType,
    ) -> Result<ArrayOp<ArrayCompareScalar<Self::DType, Self>>, Error> {
        let shape = self.shape().to_vec();
        let op = ArrayCompareScalar::eq(self, other)?;
        Ok(ArrayOp::new(shape, op))
    }

    fn gt_scalar(
        &self,
        other: Self::DType,
    ) -> Result<ArrayOp<ArrayCompareScalar<Self::DType, Self>>, Error> {
        let shape = self.shape().to_vec();
        let op = ArrayCompareScalar::gt(self, other)?;
        Ok(ArrayOp::new(shape, op))
    }

    fn ge_scalar(
        &self,
        other: Self::DType,
    ) -> Result<ArrayOp<ArrayCompareScalar<Self::DType, Self>>, Error> {
        let shape = self.shape().to_vec();
        let op = ArrayCompareScalar::ge(self, other)?;
        Ok(ArrayOp::new(shape, op))
    }

    fn lt_scalar(
        &self,
        other: Self::DType,
    ) -> Result<ArrayOp<ArrayCompareScalar<Self::DType, Self>>, Error> {
        let shape = self.shape().to_vec();
        let op = ArrayCompareScalar::lt(self, other)?;
        Ok(ArrayOp::new(shape, op))
    }

    fn le_scalar(
        &self,
        other: Self::DType,
    ) -> Result<ArrayOp<ArrayCompareScalar<Self::DType, Self>>, Error> {
        let shape = self.shape().to_vec();
        let op = ArrayCompareScalar::le(self, other)?;
        Ok(ArrayOp::new(shape, op))
    }

    fn ne_scalar(
        &self,
        other: Self::DType,
    ) -> Result<ArrayOp<ArrayCompareScalar<Self::DType, Self>>, Error> {
        let shape = self.shape().to_vec();
        let op = ArrayCompareScalar::ne(self, other)?;
        Ok(ArrayOp::new(shape, op))
    }
}

impl<A: NDArray> NDArrayCompareScalar for A {}

pub trait MatrixMath<O: NDArray<DType = Self::DType>>: NDArray {
    fn matmul<'a>(
        &'a self,
        other: &'a O,
    ) -> Result<ArrayOp<MatMul<'a, Self::DType, Self, O>>, Error> {
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

        let op = MatMul::new(self, other)?;
        Ok(ArrayOp::new(shape, op))
    }
}

impl<A: NDArray, O: NDArray> MatrixMath<O> for A where O: NDArray<DType = A::DType> {}

pub trait NDArrayReduce: NDArrayRead + Clone + fmt::Debug {
    fn all(&self) -> Result<bool, Error> {
        let queue = self.context().queue(self.size())?;
        match queue.device_queue() {
            DeviceQueue::Host => {
                let input = self.to_vec(&queue)?;
                let zero = Self::DType::zero();
                Ok(input.par_iter().copied().all(|n| n != zero))
            }
            #[cfg(feature = "opencl")]
            DeviceQueue::CL(cl_queue) => {
                let input = self.to_cl_buffer(&queue)?;
                cl_programs::reduce_all(cl_queue.clone(), input).map_err(Error::from)
            }
        }
    }

    fn any(&self) -> Result<bool, Error> {
        let queue = self.context().queue(self.size())?;
        match queue.device_queue() {
            DeviceQueue::Host => {
                let input = self.to_vec(&queue)?;
                let zero = Self::DType::zero();
                Ok(input.par_iter().copied().any(|n| n != zero))
            }
            #[cfg(feature = "opencl")]
            DeviceQueue::CL(cl_queue) => {
                let input = self.to_cl_buffer(&queue)?;
                cl_programs::reduce_any(cl_queue.clone(), input).map_err(Error::from)
            }
        }
    }

    fn max(&self) -> Result<Self::DType, Error> {
        let zero = Self::DType::zero();
        let collector = |l, r| {
            if r > l {
                r
            } else {
                l
            }
        };

        let queue = self.context().queue(self.size())?;
        match queue.device_queue() {
            DeviceQueue::Host => {
                let input = self.to_vec(&queue)?;
                Ok(input.par_iter().copied().reduce(|| zero, collector))
            }
            #[cfg(feature = "opencl")]
            DeviceQueue::CL(cl_queue) => {
                let input = self.to_cl_buffer(&queue)?;
                cl_programs::reduce(zero, "max", cl_queue.clone(), input, collector)
                    .map_err(Error::from)
            }
        }
    }

    fn max_axis(&self, axis: usize) -> Result<ArrayOp<ArrayReduceAxis<Self::DType, Self>>, Error> {
        todo!()
    }

    fn min(&self) -> Result<Self::DType, Error> {
        todo!()
    }

    fn min_axis(&self, axis: usize) -> Result<ArrayOp<ArrayReduceAxis<Self::DType, Self>>, Error> {
        todo!()
    }

    fn product(&self) -> Result<Self::DType, Error> {
        todo!()
    }

    fn product_axis(
        &self,
        axis: usize,
    ) -> Result<ArrayOp<ArrayReduceAxis<Self::DType, Self>>, Error> {
        todo!()
    }

    fn sum(&self) -> Result<Self::DType, Error> {
        let zero = Self::DType::zero();
        let queue = self.context().queue(self.size())?;

        match queue.device_queue() {
            DeviceQueue::Host => {
                let input = self.to_vec(&queue)?;
                Ok(input
                    .as_slice()
                    .par_iter()
                    .copied()
                    .reduce(|| zero, Add::add))
            }
            #[cfg(feature = "opencl")]
            DeviceQueue::CL(cl_queue) => {
                let input = self.to_cl_buffer(&queue)?;
                cl_programs::reduce(zero, "add", cl_queue.clone(), input, Add::add)
                    .map_err(Error::from)
            }
        }
    }

    fn sum_axis(&self, axis: usize) -> Result<ArrayOp<ArrayReduceAxis<Self::DType, Self>>, Error> {
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

        let op = ArrayReduceAxis::sum(self, axis);

        Ok(ArrayOp::new(shape, op))
    }
}

impl<A: NDArrayRead + Clone + fmt::Debug> NDArrayReduce for A {}

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
pub fn broadcast_shape(left: &[usize], right: &[usize]) -> Result<Shape, Error> {
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

#[cfg(feature = "opencl")]
#[inline]
fn div_ceil(num: usize, denom: usize) -> usize {
    if num % denom == 0 {
        num / denom
    } else {
        (num / denom) + 1
    }
}
