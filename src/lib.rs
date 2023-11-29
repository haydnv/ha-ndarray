use std::cmp::Ordering;
use std::fmt;
use std::iter::{Product, Sum};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign};

pub use smallvec::smallvec as axes;
pub use smallvec::smallvec as range;
pub use smallvec::smallvec as slice;
pub use smallvec::smallvec as shape;
pub use smallvec::smallvec as stackvec;
use smallvec::SmallVec;

pub use access::*;
pub use array::{
    MatrixMath, NDArray, NDArrayCast, NDArrayCompare, NDArrayCompareScalar, NDArrayMath,
    NDArrayMathScalar, NDArrayNumeric, NDArrayRead, NDArrayReduce, NDArrayReduceAll,
    NDArrayReduceBoolean, NDArrayTransform, NDArrayUnary, NDArrayWhere, NDArrayWrite,
};
pub use buffer::{Buffer, BufferConverter, BufferInstance};
use ops::*;
pub use platform::*;

mod access;
mod array;
mod buffer;
pub mod host;
#[cfg(feature = "opencl")]
pub mod opencl;
mod ops;
mod platform;

#[cfg(feature = "opencl")]
pub trait CType:
    ocl::OclPrm
    + Add<Output = Self>
    + AddAssign
    + Sum
    + Mul<Output = Self>
    + MulAssign
    + Product
    + Div<Output = Self>
    + DivAssign
    + Sub<Output = Self>
    + SubAssign
    + Rem<Output = Self>
    + RemAssign
    + PartialEq
    + PartialOrd
    + Copy
    + Send
    + Sync
    + fmt::Display
    + fmt::Debug
    + 'static
{
    const TYPE: &'static str;

    const MAX: Self;

    const MIN: Self;

    const ZERO: Self;

    const ONE: Self;

    type Float: Float;

    fn is_float() -> bool;

    fn from_f64(float: f64) -> Self;

    fn from_float(float: Self::Float) -> Self;

    fn abs(self) -> Self;

    fn min(l: Self, r: Self) -> Self;

    fn max(l: Self, r: Self) -> Self;

    fn pow(self, exp: Self) -> Self;

    fn round(self) -> Self;

    fn to_f64(self) -> f64;

    fn to_float(self) -> Self::Float;
}

#[cfg(not(feature = "opencl"))]
pub trait CType:
    Add<Output = Self>
    + AddAssign
    + Sum
    + Mul<Output = Self>
    + MulAssign
    + Product
    + Div<Output = Self>
    + DivAssign
    + Sub<Output = Self>
    + SubAssign
    + Rem<Output = Self>
    + RemAssign
    + PartialEq
    + PartialOrd
    + Copy
    + Send
    + Sync
    + fmt::Display
    + fmt::Debug
    + 'static
{
    const TYPE: &'static str;

    const MAX: Self;

    const MIN: Self;

    const ZERO: Self;

    const ONE: Self;

    type Float: Float;

    fn is_float() -> bool;

    fn from_f64(float: f64) -> Self;

    fn from_float(float: Self::Float) -> Self;

    fn abs(self) -> Self;

    fn min(l: Self, r: Self) -> Self;

    fn max(l: Self, r: Self) -> Self;

    fn pow(self, exp: Self) -> Self;

    fn round(self) -> Self;

    fn to_f64(self) -> f64;

    fn to_float(self) -> Self::Float;
}

macro_rules! c_type {
    ($t:ty, $str:expr, $is_float:expr, $one:expr, $zero:expr, $float:ty, $abs:expr, $round:expr, $pow:expr, $cmp_max:expr, $cmp_min:expr) => {
        impl CType for $t {
            const TYPE: &'static str = $str;

            const MAX: Self = <$t>::MAX;

            const MIN: Self = <$t>::MIN;

            const ZERO: Self = $zero;

            const ONE: Self = $one;

            type Float = $float;

            fn is_float() -> bool {
                $is_float
            }

            fn from_f64(float: f64) -> Self {
                float as $t
            }

            fn from_float(float: $float) -> Self {
                float as $t
            }

            fn abs(self) -> Self {
                $abs(self)
            }

            fn min(l: Self, r: Self) -> Self {
                $cmp_min(l, r)
            }

            fn max(l: Self, r: Self) -> Self {
                $cmp_max(l, r)
            }

            fn pow(self, exp: Self) -> Self {
                ($pow)(self, exp)
            }

            fn round(self) -> Self {
                $round(self)
            }

            fn to_f64(self) -> f64 {
                self as f64
            }

            fn to_float(self) -> $float {
                self as $float
            }
        }
    };
}

c_type!(
    f32,
    "float",
    true,
    1.0,
    0.0,
    Self,
    f32::abs,
    f32::round,
    max_f32,
    f32::powf,
    min_f32
);

c_type!(
    f64,
    "double",
    true,
    1.0,
    0.0,
    Self,
    f64::abs,
    f64::round,
    f64::powf,
    max_f64,
    min_f64
);

c_type!(
    i8,
    "char",
    false,
    1,
    0,
    f32,
    i8::abs,
    id,
    |a, e| f32::powi(a as f32, e as i32) as i8,
    Ord::max,
    Ord::min
);

c_type!(
    i16,
    "short",
    false,
    1,
    0,
    f32,
    i16::abs,
    id,
    |a, e| f32::powi(a as f32, e as i32) as i16,
    Ord::max,
    Ord::min
);

c_type!(
    i32,
    "int",
    false,
    1,
    0,
    f32,
    i32::abs,
    id,
    |a, e| f32::powi(a as f32, e) as i32,
    Ord::max,
    Ord::min
);

c_type!(
    i64,
    "long",
    false,
    1,
    0,
    f64,
    i64::abs,
    id,
    |a, e| f64::powi(
        a as f64,
        i32::try_from(e).unwrap_or_else(|_| if e >= 0 { i32::MAX } else { i32::MIN })
    ) as i64,
    Ord::max,
    Ord::min
);

c_type!(
    u8,
    "uchar",
    false,
    1,
    0,
    f32,
    id,
    id,
    |a, e| u8::pow(a, e as u32),
    Ord::max,
    Ord::min
);

c_type!(
    u16,
    "ushort",
    false,
    1,
    0,
    f32,
    id,
    id,
    |a, e| u16::pow(a, e as u32),
    Ord::max,
    Ord::min
);

c_type!(
    u32,
    "uint",
    false,
    1,
    0,
    f32,
    id,
    id,
    |a, e| u32::pow(a, e),
    Ord::max,
    Ord::min
);

c_type!(
    u64,
    "ulong",
    false,
    1,
    0,
    f64,
    id,
    id,
    |a, e| u64::pow(a, u32::try_from(e).unwrap_or(u32::MAX)),
    Ord::max,
    Ord::min
);

fn id<T>(this: T) -> T {
    this
}

fn max_f32(l: f32, r: f32) -> f32 {
    match l.total_cmp(&r) {
        Ordering::Less => r,
        Ordering::Equal => l,
        Ordering::Greater => l,
    }
}

fn min_f32(l: f32, r: f32) -> f32 {
    match l.total_cmp(&r) {
        Ordering::Less => l,
        Ordering::Equal => l,
        Ordering::Greater => r,
    }
}

fn max_f64(l: f64, r: f64) -> f64 {
    match l.total_cmp(&r) {
        Ordering::Less => r,
        Ordering::Equal => l,
        Ordering::Greater => l,
    }
}

fn min_f64(l: f64, r: f64) -> f64 {
    match l.total_cmp(&r) {
        Ordering::Less => l,
        Ordering::Equal => l,
        Ordering::Greater => r,
    }
}

pub trait Float: CType {
    fn exp(self) -> Self;

    fn is_inf(self) -> bool;

    fn is_nan(self) -> bool;

    fn ln(self) -> Self;

    fn log(self, base: Self) -> Self;

    fn to_f64(self) -> f64;
}

impl Float for f32 {
    fn exp(self) -> Self {
        f32::exp(self)
    }

    fn is_inf(self) -> bool {
        self.is_infinite()
    }

    fn is_nan(self) -> bool {
        f32::is_nan(self)
    }

    fn ln(self) -> Self {
        f32::ln(self)
    }

    fn log(self, base: Self) -> Self {
        f32::log(self, base)
    }

    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl Float for f64 {
    fn exp(self) -> Self {
        f64::exp(self)
    }

    fn is_inf(self) -> bool {
        self.is_infinite()
    }

    fn is_nan(self) -> bool {
        f64::is_nan(self)
    }

    fn ln(self) -> Self {
        f64::ln(self)
    }

    fn log(self, base: Self) -> Self {
        f64::log(self, base)
    }

    fn to_f64(self) -> f64 {
        self
    }
}

/// An array math error
pub enum Error {
    Bounds(String),
    Interface(String),
    Unsupported(String),
    #[cfg(feature = "opencl")]
    OCL(std::sync::Arc<ocl::Error>),
}

// Clone is required to support memorizing OpenCL programs
// since constructing an [`ocl::Program`] can return an error
impl Clone for Error {
    fn clone(&self) -> Self {
        match self {
            Self::Bounds(msg) => Self::Bounds(msg.clone()),
            Self::Interface(msg) => Self::Interface(msg.clone()),
            Self::Unsupported(msg) => Self::Unsupported(msg.clone()),
            #[cfg(feature = "opencl")]
            Self::OCL(cause) => Self::OCL(cause.clone()),
        }
    }
}

#[cfg(feature = "opencl")]
impl From<ocl::Error> for Error {
    fn from(cause: ocl::Error) -> Self {
        #[cfg(debug_assertions)]
        panic!("OpenCL error: {:?}", cause);

        #[cfg(not(debug_assertions))]
        Self::OCL(std::sync::Arc::new(cause))
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Bounds(cause) => f.write_str(cause),
            Self::Interface(cause) => f.write_str(cause),
            Self::Unsupported(cause) => f.write_str(cause),
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
            Self::Unsupported(cause) => f.write_str(cause),
            #[cfg(feature = "opencl")]
            Self::OCL(cause) => cause.fmt(f),
        }
    }
}

impl std::error::Error for Error {}

pub type Axes = SmallVec<[usize; 8]>;

pub type Range = SmallVec<[AxisRange; 8]>;

pub type Shape = SmallVec<[usize; 8]>;

pub type Strides = SmallVec<[usize; 8]>;

pub type Array<T, A> = array::Array<T, A, Platform>;

pub type ArrayBuf<T, B> = array::Array<T, AccessBuffer<B>, Platform>;

pub type ArrayOp<T, O> = array::Array<T, AccessOp<O, Platform>, Platform>;

/// Bounds on an individual array axis
#[derive(Clone, Eq, PartialEq, Hash)]
pub enum AxisRange {
    At(usize),
    In(usize, usize, usize),
    Of(SmallVec<[usize; 8]>),
}

impl AxisRange {
    /// Return `true` if this is an index bound (i.e. not a slice)
    pub fn is_index(&self) -> bool {
        match self {
            Self::At(_) => true,
            _ => false,
        }
    }

    /// Return the number of elements contained within this bound.
    /// Returns `None` for an index bound.
    pub fn size(&self) -> Option<usize> {
        match self {
            Self::At(_) => None,
            Self::In(start, stop, step) => Some((stop - start) / step),
            Self::Of(indices) => Some(indices.len()),
        }
    }
}

impl From<usize> for AxisRange {
    fn from(i: usize) -> Self {
        Self::At(i)
    }
}

impl From<std::ops::Range<usize>> for AxisRange {
    fn from(range: std::ops::Range<usize>) -> Self {
        Self::In(range.start, range.end, 1)
    }
}

impl From<SmallVec<[usize; 8]>> for AxisRange {
    fn from(indices: SmallVec<[usize; 8]>) -> Self {
        Self::Of(indices)
    }
}

impl fmt::Debug for AxisRange {
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

    let offset = left.len() - right.len();

    let mut shape = Shape::with_capacity(left.len());
    shape.extend_from_slice(&left[..offset]);

    for (l, r) in left.into_iter().copied().zip(right.into_iter().copied()) {
        if r == 1 || r == l {
            shape.push(l);
        } else if l == 1 {
            shape.push(r);
        } else {
            return Err(Error::Bounds(format!(
                "cannot broadcast dimensions {l} and {r}"
            )));
        }
    }

    debug_assert!(!shape.iter().any(|dim| *dim == 0));

    Ok(shape)
}

#[inline]
fn range_shape(source_shape: &[usize], range: &[AxisRange]) -> Shape {
    debug_assert_eq!(source_shape.len(), range.len());
    range.iter().filter_map(|ar| ar.size()).collect()
}

#[inline]
fn strides_for<'a>(shape: &'a [usize], ndim: usize) -> impl Iterator<Item = usize> + 'a {
    debug_assert!(ndim >= shape.len());

    let zeros = std::iter::repeat(0).take(ndim - shape.len());

    let strides = shape.iter().copied().enumerate().map(|(x, dim)| {
        if dim == 1 {
            0
        } else {
            shape.iter().rev().take(shape.len() - 1 - x).product()
        }
    });

    zeros.chain(strides)
}
