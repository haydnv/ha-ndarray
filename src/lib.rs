/// An n-dimensional array with automatic parallelization using [`rayon`]
/// and optional support for hardware acceleration using the "opencl" feature flag.

#[cfg(feature = "opencl")]
extern crate ocl;

use std::convert::identity;
use std::fmt;
use std::iter::Sum;
use std::ops::{Add, Div, Mul, Range, Rem, Sub};

pub use array::*;
pub use buffer::*;
use ops::*;

mod array;
mod buffer;
#[cfg(feature = "opencl")]
mod cl_programs;
pub mod ops;

/// N-dimensional array constructor op definitions
pub mod construct {
    pub use super::ops::{RandomNormal, RandomUniform, Range};
}

const GPU_MIN_DEFAULT: usize = 1024;

/// An array math error
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

/// The shape of an [`NDArray`]
pub type Shape = Vec<usize>;

// TODO: is there a better way to implement the OclPrm trait bound?
// TODO: rename to CType
/// A type which supports hardware-accelerated arithmetic operations
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
    + 'static
{
    const TYPE_STR: &'static str;

    type Float: Float;
    type Neg: CDatatype;

    fn max() -> Self;

    fn min() -> Self;

    fn one() -> Self;

    fn zero() -> Self;

    fn from_float(f: Self::Float) -> Self;

    fn from_f64(f: f64) -> Self;

    fn abs(self) -> Self;

    fn neg(self) -> Self::Neg;

    fn not(self) -> u8 {
        if self == Self::zero() {
            1
        } else {
            0
        }
    }

    fn round(self) -> Self;

    fn to_float(self) -> Self::Float;

    fn to_f64(self) -> f64;
}

#[cfg(not(feature = "opencl"))]
/// A type which supports hardware-accelerated arithmetic operations
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
    + 'static
{
    const TYPE_STR: &'static str;

    type Float: Float;
    type Neg: CDatatype;

    fn max() -> Self;

    fn min() -> Self;

    fn one() -> Self;

    fn zero() -> Self;

    fn from_float(float: Self::Float) -> Self;

    fn from_f64(f: f64) -> Self;

    fn abs(self) -> Self;

    fn neg(self) -> Self::Neg;

    fn not(self) -> u8 {
        if self == Self::zero() {
            1
        } else {
            0
        }
    }

    fn round(self) -> Self;

    fn to_float(self) -> Self::Float;

    fn to_f64(self) -> f64;
}

macro_rules! c_type {
    ($t:ty, $ct:expr, $max:expr, $min: expr, $one:expr, $zero:expr, $abs:expr, $float:ty, $neg:ty, $round:expr) => {
        impl CDatatype for $t {
            const TYPE_STR: &'static str = $ct;

            type Float = $float;
            type Neg = $neg;

            fn max() -> Self {
                $max
            }

            fn min() -> Self {
                $min
            }

            fn one() -> Self {
                $one
            }

            fn zero() -> Self {
                $zero
            }

            fn from_float(float: Self::Float) -> Self {
                float as $t
            }

            fn from_f64(f: f64) -> Self {
                f as $t
            }

            fn abs(self) -> Self {
                $abs(self)
            }

            fn neg(self) -> Self::Neg {
                if self >= Self::zero() {
                    self as $neg
                } else {
                    -(self as $neg)
                }
            }

            fn round(self) -> Self {
                $round(self)
            }

            fn to_float(self) -> Self::Float {
                self as $float
            }

            fn to_f64(self) -> f64 {
                self as f64
            }
        }
    };
}

c_type!(
    f32,
    "float",
    f32::MAX,
    f32::MIN,
    1.,
    0.,
    f32::abs,
    f32,
    f32,
    f32::round
);

c_type!(
    f64,
    "double",
    f64::MAX,
    f64::MIN,
    1.,
    0.,
    f64::abs,
    f64,
    f64,
    f64::round
);

c_type!(
    u8,
    "uchar",
    u8::MAX,
    u8::MIN,
    1,
    0,
    identity,
    f32,
    i8,
    identity
);

c_type!(
    u16,
    "ushort",
    u16::MAX,
    u16::MIN,
    1,
    0,
    identity,
    f32,
    i16,
    identity
);

c_type!(
    u32,
    "uint",
    u32::MAX,
    u32::MIN,
    1,
    0,
    identity,
    f32,
    i32,
    identity
);

c_type!(
    u64,
    "ulong",
    u64::MAX,
    u64::MIN,
    1,
    0,
    identity,
    f64,
    i64,
    identity
);

c_type!(
    i8,
    "char",
    i8::MAX,
    i8::MIN,
    1,
    0,
    i8::abs,
    f32,
    i8,
    identity
);

c_type!(
    i16,
    "short",
    i16::MAX,
    i16::MIN,
    1,
    0,
    i16::abs,
    f32,
    i16,
    identity
);

c_type!(
    i32,
    "int",
    i32::MAX,
    i32::MIN,
    1,
    0,
    i32::abs,
    f32,
    i32,
    identity
);

c_type!(
    i64,
    "long",
    i64::MAX,
    i64::MIN,
    1,
    0,
    i64::abs,
    f64,
    i64,
    identity
);

/// Logarithm-related operations on a scalar value
pub trait Log {
    /// Compute the natural log of this value.
    fn ln(self) -> Self;

    /// Compute the logarithm of this value with respect to the given `base`.
    fn log(self, base: Self) -> Self;

    /// Compute `e^self`.
    fn exp(self) -> Self;

    /// Compute the exponent of this value to the given power.
    fn pow(self, exp: Self) -> Self;
}

impl Log for f32 {
    fn ln(self) -> Self {
        f32::ln(self)
    }

    fn log(self, base: f32) -> Self {
        f32::log(self, base)
    }

    fn exp(self) -> Self {
        f32::exp(self)
    }

    fn pow(self, n: f32) -> Self {
        f32::powf(self, n)
    }
}

impl Log for f64 {
    fn ln(self) -> Self {
        f64::ln(self)
    }

    fn log(self, base: f64) -> Self {
        f64::log(self, base)
    }

    fn exp(self) -> Self {
        f64::exp(self)
    }

    fn pow(self, n: f64) -> Self {
        f64::powf(self, n)
    }
}

/// Trigonometric operations on a scalar value
pub trait Trig {
    /// Compute the sine of this value.
    fn sin(self) -> Self;

    /// Compute the arcsine of this value.
    fn asin(self) -> Self;

    /// Compute the hyperbolic sine of this value.
    fn sinh(self) -> Self;

    /// Compute the cosine of this value.
    fn cos(self) -> Self;

    /// Compute the arccosine of this value.
    fn acos(self) -> Self;

    /// Compute the hyperbolic cosine of this value.
    fn cosh(self) -> Self;

    /// Compute the tangent of this value.
    fn tan(self) -> Self;

    /// Compute the arctangent of this value.
    fn atan(self) -> Self;

    /// Compute the hyperbolic tangent of this value.
    fn tanh(self) -> Self;
}

impl Trig for f32 {
    fn sin(self) -> Self {
        f32::sin(self)
    }

    fn asin(self) -> Self {
        f32::asin(self)
    }

    fn sinh(self) -> Self {
        f32::sinh(self)
    }

    fn cos(self) -> Self {
        f32::cos(self)
    }

    fn acos(self) -> Self {
        f32::acos(self)
    }

    fn cosh(self) -> Self {
        f32::cosh(self)
    }

    fn tan(self) -> Self {
        f32::tan(self)
    }

    fn atan(self) -> Self {
        f32::atan(self)
    }

    fn tanh(self) -> Self {
        f32::tanh(self)
    }
}

impl Trig for f64 {
    fn sin(self) -> Self {
        f64::sin(self)
    }

    fn asin(self) -> Self {
        f64::asin(self)
    }

    fn sinh(self) -> Self {
        f64::sinh(self)
    }

    fn cos(self) -> Self {
        f64::cos(self)
    }

    fn acos(self) -> Self {
        f64::acos(self)
    }

    fn cosh(self) -> Self {
        f64::cosh(self)
    }

    fn tan(self) -> Self {
        f64::tan(self)
    }

    fn atan(self) -> Self {
        f64::atan(self)
    }

    fn tanh(self) -> Self {
        f64::tanh(self)
    }
}

/// Float-specific operations on a scalar floating point value
pub trait Float: CDatatype + Log + Trig {
    /// Return `1` if this value is infinite, otherwise `0`.
    fn is_inf(self) -> u8;

    /// Return `1` if this value is not a number, otherwise `0`.
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
    next: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

#[cfg(feature = "opencl")]
impl DeviceList {
    fn is_empty(&self) -> bool {
        self.devices.is_empty()
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
            next: std::sync::Arc::new(Default::default()),
        }
    }
}

#[cfg(feature = "opencl")]
impl FromIterator<ocl::Device> for DeviceList {
    fn from_iter<T: IntoIterator<Item = ocl::Device>>(iter: T) -> Self {
        Self::from(iter.into_iter().collect::<Vec<ocl::Device>>())
    }
}

#[derive(Clone, Default)]
/// An OpenCL platform
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
#[allow(unused)]
/// An execution context
pub struct Context {
    platform: Platform,
    gpu_min: usize,
    acc_min: usize,
    #[cfg(feature = "opencl")]
    cl_context: ocl::Context,
}

impl Context {
    #[cfg(feature = "opencl")]
    /// Construct a default [`Context`] with all available devices.
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
    /// Construct a default [`Context`] with the host device.
    pub fn default() -> Result<Self, Error> {
        Ok(Self {
            platform: Platform {},
            gpu_min: GPU_MIN_DEFAULT,
            acc_min: GPU_MIN_DEFAULT,
        })
    }

    #[cfg(feature = "opencl")]
    /// Construct a default [`Context`] with the given configuration.
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
    /// Construct a host [`Context`].
    /// The given configuration will be ignored since the "opencl" feature flag is disabled.
    pub fn new(gpu_min: usize, acc_min: usize, platform: Option<Platform>) -> Result<Self, Error> {
        let platform = if let Some(platform) = platform {
            platform
        } else {
            Platform::default()
        };

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

    #[cfg(feature = "opencl")]
    fn select_device(&self, size_hint: usize) -> Option<ocl::Device> {
        if size_hint < self.gpu_min {
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
        }
    }
}

/// A queue of array operations
#[derive(Clone)]
pub struct Queue {
    context: Context,
    #[cfg(feature = "opencl")]
    cl_queue: Option<ocl::Queue>,
}

impl Queue {
    fn default(context: Context) -> Self {
        Self {
            context,
            #[cfg(feature = "opencl")]
            cl_queue: None,
        }
    }

    #[cfg(feature = "opencl")]
    /// Construct a new [`Queue`] with the given [`Context`].
    /// If `size_hint` is large, a new OpenCL context may be initialized.
    pub fn new(context: Context, size_hint: usize) -> Result<Self, Error> {
        if let Some(device) = context.select_device(size_hint) {
            let cl_queue = ocl::Queue::new(context.cl_context(), device, None)?;

            Ok(Self {
                context,
                cl_queue: Some(cl_queue),
            })
        } else {
            Ok(Self::default(context))
        }
    }

    #[cfg(not(feature = "opencl"))]
    /// Construct a new host [`Queue`] with the given [`Context`].
    /// `size_hint` will be ignored since the "opencl" feature flag is disabled.
    pub fn new(context: Context, _size_hint: usize) -> Result<Self, Error> {
        Ok(Self::default(context))
    }

    #[cfg(feature = "opencl")]
    fn cl_queue(&self, default: Option<&ocl::Queue>) -> ocl::Queue {
        self.cl_queue
            .as_ref()
            .or_else(|| default)
            .cloned()
            .expect("OpenCL queue")
    }

    #[allow(unused)]
    fn context(&self) -> &Context {
        &self.context
    }

    #[allow(unused)]
    fn split(&self, size_hint: usize) -> Result<Self, Error> {
        #[cfg(feature = "opencl")]
        let cl_queue = if let Some(left_queue) = &self.cl_queue {
            if let Some(device) = self.context.select_device(size_hint) {
                ocl::Queue::new(&left_queue.context(), device, None).map(Some)?
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            context: self.context.clone(),
            #[cfg(feature = "opencl")]
            cl_queue,
        })
    }
}

/// An n-dimensional array
pub trait NDArray: Send + Sync {
    /// The data type of the elements in this array
    type DType: CDatatype;

    /// Borrow the execution [`Context`] of this array.
    fn context(&self) -> &Context;

    /// Return the number of dimensions in this array.
    fn ndim(&self) -> usize {
        self.shape().len()
    }

    /// Return the number of elements in this array.
    fn size(&self) -> usize {
        self.shape().iter().product()
    }

    /// Borrow the shape of this array.
    fn shape(&self) -> &[usize];
}

impl<A: NDArray + ?Sized> NDArray for Box<A> {
    type DType = A::DType;

    fn context(&self) -> &Context {
        (**self).context()
    }

    fn shape(&self) -> &[usize] {
        (**self).shape()
    }
}

/// Array buffer access methods
pub trait AsBuffer: NDArray {
    /// Dereference this [`NDArray`] as a [`BufferConverter`].
    fn as_buffer(&self) -> BufferConverter<Self::DType>;

    /// Dereference this [`NDArray`] as a [`BufferConverterMut`].
    fn as_buffer_mut(&mut self) -> BufferConverterMut<Self::DType>;
}

/// Access methods for an [`NDArray`]
pub trait NDArrayRead: NDArray + fmt::Debug + Sized {
    /// Read the value of this [`NDArray`].
    fn read(&self, queue: &Queue) -> Result<BufferConverter<Self::DType>, Error>;

    /// Read the value at one `coord` in this [`NDArray`].
    fn read_value(&self, coord: &[usize]) -> Result<Self::DType, Error>;

    /// Read the value of this [`NDArray`] as a [`SliceConverter`] in main memory
    fn to_host(&self, queue: &Queue) -> Result<SliceConverter<Self::DType>, Error> {
        let converter = self.read(queue)?;
        converter.to_slice()
    }

    #[cfg(feature = "opencl")]
    /// Read the value of this [`NDArray`] as a [`CLConverter`] in OpenCL memory
    fn to_cl_buffer(&self, queue: &Queue) -> Result<CLConverter<Self::DType>, Error> {
        let converter = self.read(queue)?;
        converter.to_cl(queue)
    }
}

/// Access methods for a mutable [`NDArray`]
pub trait NDArrayWrite: NDArray + fmt::Debug + Sized {
    /// Overwrite this [`NDArray`] with the value of the `other` array.
    fn write<O: NDArrayRead<DType = Self::DType>>(&mut self, other: &O) -> Result<(), Error>;

    /// Overwrite this [`NDArray`] with a constant scalar `value`.
    fn write_value(&mut self, value: Self::DType) -> Result<(), Error>;

    /// Write the given `value` at the given `coord` of this [`NDArray`].
    fn write_value_at(&mut self, coord: &[usize], value: Self::DType) -> Result<(), Error>;
}

/// Boolean array operations
pub trait NDArrayBoolean<O>: NDArray + Sized
where
    O: NDArray<DType = Self::DType>,
{
    /// Construct a boolean and comparison with the `other` array.
    fn and(self, other: O) -> Result<ArrayOp<ArrayBoolean<Self::DType, Self, O>>, Error> {
        let shape = check_shape(self.shape(), other.shape())?;
        let op = ArrayBoolean::and(self, other)?;
        Ok(ArrayOp::new(shape, op))
    }

    /// Construct a boolean or comparison with the `other` array.
    fn or(self, other: O) -> Result<ArrayOp<ArrayBoolean<Self::DType, Self, O>>, Error> {
        let shape = check_shape(self.shape(), other.shape())?;
        let op = ArrayBoolean::or(self, other)?;
        Ok(ArrayOp::new(shape, op))
    }

    /// Construct a boolean xor comparison with the `other` array.
    fn xor(self, other: O) -> Result<ArrayOp<ArrayBoolean<Self::DType, Self, O>>, Error> {
        let shape = check_shape(self.shape(), other.shape())?;
        let op = ArrayBoolean::xor(self, other)?;
        Ok(ArrayOp::new(shape, op))
    }
}

impl<T: CDatatype, A: NDArray<DType = T>, O: NDArray<DType = T>> NDArrayBoolean<O> for A {}

/// Boolean array operations with a scalar argument
pub trait NDArrayBooleanScalar: NDArray + Sized {
    /// Construct a boolean and operation with the `other` value.
    fn and_scalar(
        self,
        other: Self::DType,
    ) -> Result<ArrayOp<ArrayBooleanScalar<Self, Self::DType>>, Error> {
        let shape = self.shape().to_vec();
        let op = ArrayBooleanScalar::and(self, other)?;
        Ok(ArrayOp::new(shape, op))
    }

    /// Construct a boolean or operation with the `other` value.
    fn or_scalar(
        self,
        other: Self::DType,
    ) -> Result<ArrayOp<ArrayBooleanScalar<Self, Self::DType>>, Error> {
        let shape = self.shape().to_vec();
        let op = ArrayBooleanScalar::or(self, other)?;
        Ok(ArrayOp::new(shape, op))
    }

    /// Construct a boolean xor operation with the `other` value.
    fn xor_scalar(
        self,
        other: Self::DType,
    ) -> Result<ArrayOp<ArrayBooleanScalar<Self, Self::DType>>, Error> {
        let shape = self.shape().to_vec();
        let op = ArrayBooleanScalar::xor(self, other)?;
        Ok(ArrayOp::new(shape, op))
    }
}

impl<T: CDatatype, A: NDArray<DType = T>> NDArrayBooleanScalar for A {}

/// Unary array operations
pub trait NDArrayUnary: NDArray + Sized {
    /// Construct an absolute value operation.
    fn abs(self) -> Result<ArrayOp<ArrayUnary<Self::DType, Self::DType, Self>>, Error> {
        let shape = self.shape().to_vec();
        let op = ArrayUnary::abs(self)?;
        Ok(ArrayOp::new(shape, op))
    }

    /// Construct an exponentiation operation.
    fn exp(self) -> Result<ArrayOp<ArrayUnary<Self::DType, Self::DType, Self>>, Error> {
        let shape = self.shape().to_vec();
        let op = ArrayUnary::exp(self)?;
        Ok(ArrayOp::new(shape, op))
    }

    /// Construct a natural logarithm operation.
    fn ln(self) -> Result<ArrayOp<ArrayUnary<Self::DType, Self::DType, Self>>, Error> {
        let shape = self.shape().to_vec();
        let op = ArrayUnary::ln(self)?;
        Ok(ArrayOp::new(shape, op))
    }

    /// Construct a boolean not operation.
    fn not(self) -> Result<ArrayOp<ArrayUnary<Self::DType, u8, Self>>, Error> {
        let shape = self.shape().to_vec();
        let op = ArrayUnary::not(self)?;
        Ok(ArrayOp::new(shape, op))
    }

    /// Construct an integer rounding operation.
    fn round(self) -> Result<ArrayOp<ArrayUnary<Self::DType, Self::DType, Self>>, Error> {
        let shape = self.shape().to_vec();
        let op = ArrayUnary::round(self)?;
        Ok(ArrayOp::new(shape, op))
    }
}

impl<A: NDArray> NDArrayUnary for A {}

/// Array arithmetic operations
pub trait NDArrayMath: NDArray + Sized {
    /// Construct an addition operation with the given `rhs`.
    fn add<O>(self, rhs: O) -> Result<ArrayOp<ArrayDual<Self::DType, Self, O>>, Error>
    where
        O: NDArray<DType = Self::DType> + Sized,
    {
        let shape = check_shape(self.shape(), rhs.shape())?;
        let op = ArrayDual::add(self, rhs)?;
        Ok(ArrayOp::new(shape, op))
    }

    /// Construct a division operation with the given `rhs`
    /// which will return an error if `rhs` contains zeros.
    fn checked_div<O>(self, rhs: O) -> Result<ArrayOp<ArrayDual<Self::DType, Self, O>>, Error>
    where
        O: NDArray<DType = Self::DType> + Sized,
    {
        let shape = check_shape(self.shape(), rhs.shape())?;
        let op = ArrayDual::checked_div(self, rhs)?;
        Ok(ArrayOp::new(shape, op))
    }

    /// Construct a division operation with the given `rhs`
    /// which will enter undefined behavior if `rhs` contains zeros.
    fn div<O>(self, rhs: O) -> Result<ArrayOp<ArrayDual<Self::DType, Self, O>>, Error>
    where
        O: NDArray<DType = Self::DType> + Sized,
    {
        let shape = check_shape(self.shape(), rhs.shape())?;
        let op = ArrayDual::div(self, rhs)?;
        Ok(ArrayOp::new(shape, op))
    }

    /// Construct an array multiplication operation with the given `rhs`.
    fn mul<O>(self, rhs: O) -> Result<ArrayOp<ArrayDual<Self::DType, Self, O>>, Error>
    where
        O: NDArray<DType = Self::DType> + Sized,
    {
        let shape = check_shape(self.shape(), rhs.shape())?;
        let op = ArrayDual::mul(self, rhs)?;
        Ok(ArrayOp::new(shape, op))
    }

    /// Construct an array modulo operation with the given `rhs`.
    fn rem<O>(self, rhs: O) -> Result<ArrayOp<ArrayDual<Self::DType, Self, O>>, Error>
    where
        O: NDArray<DType = Self::DType> + Sized,
    {
        let shape = check_shape(self.shape(), rhs.shape())?;
        let op = ArrayDual::rem(self, rhs)?;
        Ok(ArrayOp::new(shape, op))
    }

    /// Construct an array subtraction operation with the given `rhs`.
    fn sub<O>(self, rhs: O) -> Result<ArrayOp<ArrayDual<Self::DType, Self, O>>, Error>
    where
        O: NDArray<DType = Self::DType> + Sized,
    {
        let shape = check_shape(self.shape(), rhs.shape())?;
        let op = ArrayDual::sub(self, rhs)?;
        Ok(ArrayOp::new(shape, op))
    }

    /// Construct an array logarithm operation with the given `base`.
    fn log<O>(self, base: O) -> Result<ArrayOp<ArrayDualFloat<Self::DType, Self, O>>, Error>
    where
        O: NDArray<DType = <Self::DType as CDatatype>::Float> + Sized,
    {
        let shape = check_shape(self.shape(), base.shape())?;
        let op = ArrayDualFloat::log(self, base)?;
        Ok(ArrayOp::new(shape, op))
    }

    /// Construct an array exponentiation operation with the given power.
    fn pow<O>(self, exp: O) -> Result<ArrayOp<ArrayDualFloat<Self::DType, Self, O>>, Error>
    where
        O: NDArray<DType = <Self::DType as CDatatype>::Float> + Sized,
    {
        let shape = check_shape(self.shape(), exp.shape())?;
        let op = ArrayDualFloat::pow(self, exp)?;
        Ok(ArrayOp::new(shape, op))
    }
}

impl<A: NDArray> NDArrayMath for A {}

/// Array arithmetic operations with a scalar argument
pub trait NDArrayMathScalar: NDArray + Sized {
    /// Construct a scalar addition operation.
    fn add_scalar(
        self,
        rhs: Self::DType,
    ) -> Result<ArrayOp<ArrayScalar<Self::DType, Self>>, Error> {
        let shape = self.shape().to_vec();
        let op = ArrayScalar::add(self, rhs)?;
        Ok(ArrayOp::new(shape, op))
    }

    /// Construct a scalar division operation.
    fn div_scalar(
        self,
        rhs: Self::DType,
    ) -> Result<ArrayOp<ArrayScalar<Self::DType, Self>>, Error> {
        if rhs == Self::DType::zero() {
            Err(Error::Bounds("divide by zero".to_string()))
        } else {
            let shape = self.shape().to_vec();
            let op = ArrayScalar::div(self, rhs)?;
            Ok(ArrayOp::new(shape, op))
        }
    }

    /// Construct a scalar multiplication operation.
    fn mul_scalar(
        self,
        rhs: Self::DType,
    ) -> Result<ArrayOp<ArrayScalar<Self::DType, Self>>, Error> {
        let shape = self.shape().to_vec();
        let op = ArrayScalar::mul(self, rhs)?;
        Ok(ArrayOp::new(shape, op))
    }

    /// Construct a scalar modulo operation.
    fn rem_scalar(
        self,
        rhs: Self::DType,
    ) -> Result<ArrayOp<ArrayScalar<Self::DType, Self>>, Error> {
        let shape = self.shape().to_vec();
        let op = ArrayScalar::rem(self, rhs)?;
        Ok(ArrayOp::new(shape, op))
    }

    /// Construct a scalar subtraction operation.
    fn sub_scalar(
        self,
        rhs: Self::DType,
    ) -> Result<ArrayOp<ArrayScalar<Self::DType, Self>>, Error> {
        let shape = self.shape().to_vec();
        let op = ArrayScalar::sub(self, rhs)?;
        Ok(ArrayOp::new(shape, op))
    }

    /// Construct a scalar logarithm operation.
    fn log_scalar(
        self,
        base: <Self::DType as CDatatype>::Float,
    ) -> Result<ArrayOp<ArrayScalarFloat<Self::DType, Self>>, Error> {
        let shape = self.shape().to_vec();
        let op = ArrayScalarFloat::log(self, base)?;
        Ok(ArrayOp::new(shape, op))
    }

    /// Construct a scalar exponentiation operation.
    fn pow_scalar(
        self,
        exp: <Self::DType as CDatatype>::Float,
    ) -> Result<ArrayOp<ArrayScalarFloat<Self::DType, Self>>, Error> {
        let shape = self.shape().to_vec();
        let op = ArrayScalarFloat::pow(self, exp)?;
        Ok(ArrayOp::new(shape, op))
    }
}

impl<A: NDArray> NDArrayMathScalar for A {}

/// Float-specific array methods
pub trait NDArrayNumeric: NDArray + Sized
where
    Self::DType: Float,
{
    /// Test which elements of this array are infinite.
    fn is_inf(self) -> Result<ArrayOp<ArrayUnary<Self::DType, u8, Self>>, Error> {
        let shape = self.shape().to_vec();
        let op = ArrayUnary::inf(self)?;
        Ok(ArrayOp::new(shape, op))
    }

    /// Test which elements of this array are not-a-number.
    fn is_nan(self) -> Result<ArrayOp<ArrayUnary<Self::DType, u8, Self>>, Error> {
        let shape = self.shape().to_vec();
        let op = ArrayUnary::nan(self).expect("op");
        Ok(ArrayOp::new(shape, op))
    }
}

impl<A: NDArray> NDArrayNumeric for A where A::DType: Float {}

/// Array trigonometry methods
pub trait NDArrayTrig: NDArray + Sized {
    /// Construct a new arcsine operation.
    fn asin(
        self,
    ) -> Result<ArrayOp<ArrayUnary<Self::DType, <Self::DType as CDatatype>::Float, Self>>, Error>
    {
        let shape = self.shape().to_vec();
        let op = ArrayUnary::asin(self)?;
        Ok(ArrayOp::new(shape, op))
    }

    /// Construct a new sine operation.
    fn sin(
        self,
    ) -> Result<ArrayOp<ArrayUnary<Self::DType, <Self::DType as CDatatype>::Float, Self>>, Error>
    {
        let shape = self.shape().to_vec();
        let op = ArrayUnary::sin(self)?;
        Ok(ArrayOp::new(shape, op))
    }

    /// Construct a new hyperbolic sine operation.
    fn sinh(
        self,
    ) -> Result<ArrayOp<ArrayUnary<Self::DType, <Self::DType as CDatatype>::Float, Self>>, Error>
    {
        let shape = self.shape().to_vec();
        let op = ArrayUnary::sinh(self)?;
        Ok(ArrayOp::new(shape, op))
    }

    /// Construct a new arccosine operation.
    fn acos(
        self,
    ) -> Result<ArrayOp<ArrayUnary<Self::DType, <Self::DType as CDatatype>::Float, Self>>, Error>
    {
        let shape = self.shape().to_vec();
        let op = ArrayUnary::acos(self)?;
        Ok(ArrayOp::new(shape, op))
    }

    /// Construct a new cosine operation.
    fn cos(
        self,
    ) -> Result<ArrayOp<ArrayUnary<Self::DType, <Self::DType as CDatatype>::Float, Self>>, Error>
    {
        let shape = self.shape().to_vec();
        let op = ArrayUnary::cos(self)?;
        Ok(ArrayOp::new(shape, op))
    }

    /// Construct a new hyperbolic cosine operation.
    fn cosh(
        self,
    ) -> Result<ArrayOp<ArrayUnary<Self::DType, <Self::DType as CDatatype>::Float, Self>>, Error>
    {
        let shape = self.shape().to_vec();
        let op = ArrayUnary::cosh(self)?;
        Ok(ArrayOp::new(shape, op))
    }

    /// Construct a new arctangent operation.
    fn atan(
        self,
    ) -> Result<ArrayOp<ArrayUnary<Self::DType, <Self::DType as CDatatype>::Float, Self>>, Error>
    {
        let shape = self.shape().to_vec();
        let op = ArrayUnary::atan(self)?;
        Ok(ArrayOp::new(shape, op))
    }

    /// Construct a new tangent operation.
    fn tan(
        self,
    ) -> Result<ArrayOp<ArrayUnary<Self::DType, <Self::DType as CDatatype>::Float, Self>>, Error>
    {
        let shape = self.shape().to_vec();
        let op = ArrayUnary::tan(self)?;
        Ok(ArrayOp::new(shape, op))
    }

    /// Construct a new hyperbolic tangent operation.
    fn tanh(
        self,
    ) -> Result<ArrayOp<ArrayUnary<Self::DType, <Self::DType as CDatatype>::Float, Self>>, Error>
    {
        let shape = self.shape().to_vec();
        let op = ArrayUnary::tanh(self)?;
        Ok(ArrayOp::new(shape, op))
    }
}

impl<A: NDArray> NDArrayTrig for A {}

/// Array cast operations
pub trait NDArrayCast: NDArray + Sized {
    /// Construct a new array cast operation.
    fn cast<O: CDatatype>(self) -> Result<ArrayOp<ArrayCast<Self, O>>, Error> {
        let shape = self.shape().to_vec();
        let op = ArrayCast::new(self)?;
        Ok(ArrayOp::new(shape, op))
    }
}

impl<A: NDArray> NDArrayCast for A {}

/// Array comparison operations
pub trait NDArrayCompare<O: NDArray<DType = Self::DType>>: NDArray + Sized {
    /// Construct an equality comparison with the `other` array.
    fn eq(self, other: O) -> Result<ArrayOp<ArrayCompare<Self::DType, Self, O>>, Error> {
        let shape = check_shape(self.shape(), other.shape())?;
        let op = ArrayCompare::eq(self, other)?;
        Ok(ArrayOp::new(shape, op))
    }

    /// Construct a greater-than comparison with the `other` array.
    fn gt(self, other: O) -> Result<ArrayOp<ArrayCompare<Self::DType, Self, O>>, Error> {
        let shape = check_shape(self.shape(), other.shape())?;
        let op = ArrayCompare::gt(self, other)?;
        Ok(ArrayOp::new(shape, op))
    }

    /// Construct an equal-or-greater-than comparison with the `other` array.
    fn ge(self, other: O) -> Result<ArrayOp<ArrayCompare<Self::DType, Self, O>>, Error> {
        let shape = check_shape(self.shape(), other.shape())?;
        let op = ArrayCompare::ge(self, other)?;
        Ok(ArrayOp::new(shape, op))
    }

    /// Construct an equal-or-less-than comparison with the `other` array.
    fn lt(self, other: O) -> Result<ArrayOp<ArrayCompare<Self::DType, Self, O>>, Error> {
        let shape = check_shape(self.shape(), other.shape())?;
        let op = ArrayCompare::lt(self, other)?;
        Ok(ArrayOp::new(shape, op))
    }

    /// Construct an equal-or-less-than comparison with the `other` array.
    fn le(self, other: O) -> Result<ArrayOp<ArrayCompare<Self::DType, Self, O>>, Error> {
        let shape = check_shape(self.shape(), other.shape())?;
        let op = ArrayCompare::le(self, other)?;
        Ok(ArrayOp::new(shape, op))
    }

    /// Construct an not-equal comparison with the `other` array.
    fn ne(self, other: O) -> Result<ArrayOp<ArrayCompare<Self::DType, Self, O>>, Error> {
        let shape = check_shape(self.shape(), other.shape())?;
        let op = ArrayCompare::ne(self, other)?;
        Ok(ArrayOp::new(shape, op))
    }
}

impl<A: NDArray, O: NDArray> NDArrayCompare<O> for A where O: NDArray<DType = A::DType> {}

/// Array comparison operations with a scalar argument
pub trait NDArrayCompareScalar: NDArray + Sized {
    /// Construct an equality comparison with the `other` value.
    fn eq_scalar(
        self,
        other: Self::DType,
    ) -> Result<ArrayOp<ArrayCompareScalar<Self::DType, Self>>, Error> {
        let shape = self.shape().to_vec();
        let op = ArrayCompareScalar::eq(self, other)?;
        Ok(ArrayOp::new(shape, op))
    }

    /// Construct a greater-than comparison with the `other` value.
    fn gt_scalar(
        self,
        other: Self::DType,
    ) -> Result<ArrayOp<ArrayCompareScalar<Self::DType, Self>>, Error> {
        let shape = self.shape().to_vec();
        let op = ArrayCompareScalar::gt(self, other)?;
        Ok(ArrayOp::new(shape, op))
    }

    /// Construct an equal-or-greater-than comparison with the `other` value.
    fn ge_scalar(
        self,
        other: Self::DType,
    ) -> Result<ArrayOp<ArrayCompareScalar<Self::DType, Self>>, Error> {
        let shape = self.shape().to_vec();
        let op = ArrayCompareScalar::ge(self, other)?;
        Ok(ArrayOp::new(shape, op))
    }

    /// Construct a less-than comparison with the `other` value.
    fn lt_scalar(
        self,
        other: Self::DType,
    ) -> Result<ArrayOp<ArrayCompareScalar<Self::DType, Self>>, Error> {
        let shape = self.shape().to_vec();
        let op = ArrayCompareScalar::lt(self, other)?;
        Ok(ArrayOp::new(shape, op))
    }

    /// Construct an equal-or-less-than comparison with the `other` value.
    fn le_scalar(
        self,
        other: Self::DType,
    ) -> Result<ArrayOp<ArrayCompareScalar<Self::DType, Self>>, Error> {
        let shape = self.shape().to_vec();
        let op = ArrayCompareScalar::le(self, other)?;
        Ok(ArrayOp::new(shape, op))
    }

    /// Construct an not-equal comparison with the `other` value.
    fn ne_scalar(
        self,
        other: Self::DType,
    ) -> Result<ArrayOp<ArrayCompareScalar<Self::DType, Self>>, Error>
    where
        Self: Sized,
    {
        let shape = self.shape().to_vec();
        let op = ArrayCompareScalar::ne(self, other)?;
        Ok(ArrayOp::new(shape, op))
    }
}

impl<A: NDArray> NDArrayCompareScalar for A {}

/// Matrix operations
pub trait MatrixMath: NDArray + fmt::Debug {
    /// Construct an operation to read the diagonal of this matrix or batch of matrices.
    fn diagonal(self) -> Result<ArrayOp<MatDiag<Self>>, Error>
    where
        Self: Sized,
    {
        if self.ndim() >= 2 && self.shape()[self.ndim() - 1] == self.shape()[self.ndim() - 2] {
            let shape = self.shape().iter().take(self.ndim() - 1).copied().collect();
            let op = MatDiag::new(self)?;
            Ok(ArrayOp::new(shape, op))
        } else {
            Err(Error::Bounds(format!(
                "diagonal requires a square matrix, not {:?}",
                self
            )))
        }
    }

    /// Construct an operation to multiply this matrix or batch of matrices with the `other`.
    fn matmul<O>(self, other: O) -> Result<ArrayOp<MatMul<Self::DType, Self, O>>, Error>
    where
        O: NDArray<DType = Self::DType> + fmt::Debug,
        Self: Sized,
    {
        if self.ndim() < 2 || other.ndim() < 2 {
            return Err(Error::Bounds(format!(
                "invalid matrices for matmul: {:?} and {:?}",
                self, other
            )));
        }

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

impl<A: NDArray + fmt::Debug> MatrixMath for A {}

/// Boolean array reduce operations
pub trait NDArrayReduceBoolean: NDArrayRead {
    /// Return `true` if this array contains only non-zero elements.
    fn all(&self) -> Result<bool, Error> {
        let queue = Queue::new(self.context().clone(), self.size())?;
        let buffer = self.read(&queue)?;
        buffer.all(&queue)
    }

    /// Return `true` if this array contains any non-zero elements.
    fn any(&self) -> Result<bool, Error> {
        let queue = Queue::new(self.context().clone(), self.size())?;
        let buffer = self.read(&queue)?;
        buffer.any(&queue)
    }
}

impl<A: NDArrayRead> NDArrayReduceBoolean for A {}

/// Array reduce operations
pub trait NDArrayReduceAll: NDArrayRead {
    /// Return the maximum element in this array.
    fn max_all(&self) -> Result<Self::DType, Error> {
        let queue = Queue::new(self.context().clone(), self.size())?;
        let buffer = self.read(&queue)?;
        buffer.max(&queue)
    }

    /// Return the minimum element in this array.
    fn min_all(&self) -> Result<Self::DType, Error> {
        let queue = Queue::new(self.context().clone(), self.size())?;
        let buffer = self.read(&queue)?;
        buffer.min(&queue)
    }

    /// Return the product of all elements in this array.
    fn product_all(&self) -> Result<Self::DType, Error> {
        let queue = Queue::new(self.context().clone(), self.size())?;
        let buffer = self.read(&queue)?;
        buffer.product(&queue)
    }

    /// Return the sum of all elements in this array.
    fn sum_all(&self) -> Result<Self::DType, Error> {
        let queue = Queue::new(self.context().clone(), self.size())?;
        let buffer = self.read(&queue)?;
        buffer.sum(&queue)
    }
}

impl<A: NDArrayRead> NDArrayReduceAll for A {}

/// Axis-wise array reduce operations
pub trait NDArrayReduce: NDArrayRead + NDArrayTransform + fmt::Debug
where
    Array<Self::DType>: From<Self> + From<Self::Transpose>,
{
    /// Construct a max-reduce operation over the given `axes`.
    fn max(
        self,
        mut axes: Vec<usize>,
        keepdims: bool,
    ) -> Result<ArrayOp<ArrayReduceAxes<Self::DType, Array<Self::DType>>>, Error> {
        axes.sort();
        axes.dedup();

        let shape = reduce_axes(self.shape(), &axes, keepdims)?;
        let stride = axes.iter().copied().map(|x| self.shape()[x]).product();
        let this = permute_for_reduce(self, axes)?;
        let op = ArrayReduceAxes::max(this, stride);
        Ok(ArrayOp::new(shape, op))
    }

    /// Construct a min-reduce operation over the given `axes`.
    fn min(
        self,
        mut axes: Vec<usize>,
        keepdims: bool,
    ) -> Result<ArrayOp<ArrayReduceAxes<Self::DType, Array<Self::DType>>>, Error> {
        axes.sort();
        axes.dedup();

        let shape = reduce_axes(self.shape(), &axes, keepdims)?;
        let stride = axes.iter().copied().map(|x| self.shape()[x]).product();
        let this = permute_for_reduce(self, axes)?;
        let op = ArrayReduceAxes::min(this, stride);
        Ok(ArrayOp::new(shape, op))
    }

    /// Construct a product-reduce operation over the given `axes`.
    fn product(
        self,
        mut axes: Vec<usize>,
        keepdims: bool,
    ) -> Result<ArrayOp<ArrayReduceAxes<Self::DType, Array<Self::DType>>>, Error> {
        axes.sort();
        axes.dedup();

        let shape = reduce_axes(self.shape(), &axes, keepdims)?;
        let stride = axes.iter().copied().map(|x| self.shape()[x]).product();
        let this = permute_for_reduce(self, axes)?;
        let op = ArrayReduceAxes::product(this, stride);
        Ok(ArrayOp::new(shape, op))
    }

    /// Construct a sum-reduce operation over the given `axes`.
    fn sum(
        self,
        mut axes: Vec<usize>,
        keepdims: bool,
    ) -> Result<ArrayOp<ArrayReduceAxes<Self::DType, Array<Self::DType>>>, Error> {
        axes.sort();
        axes.dedup();

        let shape = reduce_axes(self.shape(), &axes, keepdims)?;
        let stride = axes.iter().copied().map(|x| self.shape()[x]).product();
        let this = permute_for_reduce(self, axes)?;
        let op = ArrayReduceAxes::sum(this, stride);
        Ok(ArrayOp::new(shape, op))
    }
}

impl<A> NDArrayReduce for A
where
    Array<A::DType>: From<A> + From<A::Transpose>,
    A: NDArrayRead + NDArrayTransform + fmt::Debug,
{
}

/// Conditional selection (boolean logic) methods
pub trait NDArrayWhere: NDArray<DType = u8> + fmt::Debug {
    /// Construct a boolean selection operation.
    /// The resulting array will return values from `then` where `self` is `true`
    /// and from `or_else` where `self` is `false`.
    fn cond<T, L, R>(self, then: L, or_else: R) -> Result<ArrayOp<GatherCond<Self, T, L, R>>, Error>
    where
        T: CDatatype,
        L: NDArray<DType = T> + fmt::Debug,
        R: NDArray<DType = T> + fmt::Debug,
        Self: Sized,
    {
        if self.shape() == then.shape() && self.shape() == or_else.shape() {
            let shape = self.shape().to_vec();
            let op = GatherCond::new(self, then, or_else)?;
            Ok(ArrayOp::new(shape, op))
        } else {
            Err(Error::Bounds(format!(
                "cannot gather from {:?} and {:?} conditionally based on {:?} (wrong shape)",
                then, or_else, self
            )))
        }
    }
}

impl<A: NDArray<DType = u8> + fmt::Debug> NDArrayWhere for A {}

/// Array transform operations
pub trait NDArrayTransform: NDArray + fmt::Debug {
    /// The type returned by `broadcast`
    type Broadcast: NDArray<DType = Self::DType> + NDArrayRead + NDArrayTransform;

    /// The type returned by `expand_dims`
    type Expand: NDArray<DType = Self::DType> + NDArrayRead + NDArrayTransform;

    /// The type returned by `reshape`
    type Reshape: NDArray<DType = Self::DType> + NDArrayRead + NDArrayTransform;

    /// The type returned by `slice`
    type Slice: NDArray<DType = Self::DType> + NDArrayRead + NDArrayTransform;

    /// The type returned by `transpose`
    type Transpose: NDArray<DType = Self::DType> + NDArrayRead + NDArrayTransform;

    /// Broadcast this array into the given `shape`.
    fn broadcast(self, shape: Shape) -> Result<Self::Broadcast, Error>;

    /// Expand the given `axes` of this array.
    fn expand_dims(self, axes: Vec<usize>) -> Result<Self::Expand, Error>;

    /// Reshape this array into the given `shape`.
    fn reshape(self, shape: Shape) -> Result<Self::Reshape, Error>;

    /// Construct a slice of this array.
    fn slice(self, bounds: Vec<AxisBound>) -> Result<Self::Slice, Error>;

    /// Transpose this array according to the given `permutation`.
    /// If no permutation is given, the array axes will be reversed.
    fn transpose(self, permutatin: Option<Vec<usize>>) -> Result<Self::Transpose, Error>;
}

/// Bounds on an individual array axis
#[derive(Clone)]
pub enum AxisBound {
    At(usize),
    In(usize, usize, usize),
    Of(Vec<usize>),
}

impl AxisBound {
    /// Return `true` if this is an index bound (i.e. not a slice)
    pub fn is_index(&self) -> bool {
        match self {
            Self::At(_) => true,
            _ => false,
        }
    }

    /// Return the number of elements contained within this bound.
    /// Returns `0` for an index bound.
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
/// Compute the shape which results from broadcasting the `left` and `right` shapes, if possible.
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

#[inline]
fn offset_of(coord: &[usize], shape: &[usize]) -> usize {
    let strides = shape.iter().enumerate().map(|(x, dim)| {
        if *dim == 1 {
            0
        } else {
            shape.iter().rev().take(shape.len() - 1 - x).product()
        }
    });

    coord
        .iter()
        .copied()
        .zip(strides)
        .map(|(i, stride)| i * stride)
        .sum()
}

#[inline]
fn permute_for_reduce<A: NDArrayTransform>(
    array: A,
    axes: Vec<usize>,
) -> Result<Array<A::DType>, Error>
where
    Array<A::DType>: From<A> + From<A::Transpose>,
{
    let mut permutation = Vec::with_capacity(array.ndim());
    permutation.extend((0..array.ndim()).into_iter().filter(|x| !axes.contains(x)));
    permutation.extend(axes);

    if permutation.iter().copied().enumerate().all(|(i, x)| i == x) {
        Ok(array.into())
    } else {
        array.transpose(Some(permutation)).map(Array::from)
    }
}

#[inline]
fn reduce_axes(shape: &[usize], axes: &[usize], keepdims: bool) -> Result<Shape, Error> {
    let mut shape = shape.to_vec();

    for x in axes.iter().copied().rev() {
        if x >= shape.len() {
            return Err(Error::Bounds(format!(
                "axis {x} is out of bounds for {shape:?}"
            )));
        } else if keepdims {
            shape[x] = 1;
        } else {
            shape.remove(x);
        }
    }

    if shape.is_empty() {
        Ok(vec![1])
    } else {
        Ok(shape)
    }
}

#[inline]
/// Compute the strides of the given shape, with a result of length `ndim`.
pub fn strides_for(shape: &[usize], ndim: usize) -> Vec<usize> {
    debug_assert!(ndim >= shape.len());

    let zeros = std::iter::repeat(0).take(ndim - shape.len());

    let strides = shape.iter().copied().enumerate().map(|(x, dim)| {
        if dim == 1 {
            0
        } else {
            shape.iter().rev().take(shape.len() - 1 - x).product()
        }
    });

    zeros.chain(strides).collect()
}
