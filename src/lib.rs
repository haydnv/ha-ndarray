use std::fmt;
use std::iter::Sum;
use std::ops::{Add, Sub};

pub use smallvec::smallvec as slice;
pub use smallvec::smallvec as shape;
use smallvec::SmallVec;

pub use access::*;
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
    + Sub<Output = Self>
    + Sum
    + PartialEq
    + Copy
    + Send
    + Sync
    + fmt::Display
    + fmt::Debug
    + 'static
{
    const TYPE: &'static str;

    const ZERO: Self;

    const ONE: Self;

    type Float: Float;

    fn from_float(float: Self::Float) -> Self;

    fn to_float(self) -> Self::Float;
}

#[cfg(not(feature = "opencl"))]
pub trait CType:
    Add<Output = Self>
    + Sub<Output = Self>
    + Sum
    + PartialEq
    + Copy
    + Send
    + Sync
    + fmt::Display
    + fmt::Debug
    + 'static
{
    const TYPE: &'static str;

    const ZERO: Self;

    const ONE: Self;

    type Float: Float;

    fn from_float(float: Self::Float) -> Self;

    fn to_float(self) -> Self::Float;
}

macro_rules! c_type {
    ($t:ty, $str:expr, $one:expr, $zero:expr, $float:ty) => {
        impl CType for $t {
            const TYPE: &'static str = $str;

            const ZERO: Self = $zero;

            const ONE: Self = $one;

            type Float = $float;

            fn from_float(float: $float) -> Self {
                float as $t
            }

            fn to_float(self) -> $float {
                self as $float
            }
        }
    };
}

c_type!(f32, "float", 1.0, 0.0, Self);
c_type!(f64, "double", 1.0, 0.0, Self);
c_type!(i8, "char", 1, 0, f32);
c_type!(i16, "short", 1, 0, f32);
c_type!(i32, "int", 1, 0, f32);
c_type!(i64, "long", 1, 0, f64);
c_type!(u8, "uchar", 1, 0, f32);
c_type!(u16, "ushort", 1, 0, f32);
c_type!(u32, "uint", 1, 0, f32);
c_type!(u64, "ulong", 1, 0, f64);

pub trait Float: CType {
    fn ln(self) -> Self;
}

impl Float for f32 {
    fn ln(self) -> Self {
        f32::ln(self)
    }
}

impl Float for f64 {
    fn ln(self) -> Self {
        f64::ln(self)
    }
}

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

pub type Range = SmallVec<[AxisRange; 8]>;

pub type Shape = SmallVec<[usize; 8]>;

pub type Strides = SmallVec<[usize; 8]>;

pub type Array<T> = array::Array<T, AccessBuffer<Buffer<T>>, Platform>;

pub type ArrayBuf<T, B> = array::Array<T, AccessBuffer<B>, Platform>;

/// Bounds on an individual array axis
#[derive(Clone)]
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
fn strides_for(shape: &[usize], ndim: usize) -> Strides {
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
