use std::fmt;
use std::iter::Sum;
use std::ops::{Add, Sub};

pub use smallvec::smallvec as shape;
use smallvec::SmallVec;

pub use access::*;
pub use buffer::{Buffer, BufferConverter, BufferInstance};
pub use host::{Host, StackVec};
use ops::*;
pub use platform::*;

mod access;
mod array;
mod buffer;
mod host;
#[cfg(feature = "opencl")]
mod opencl;
mod ops;
mod platform;

#[cfg(feature = "opencl")]
pub trait CType:
    ocl::OclPrm
    + Add<Output = Self>
    + Sub<Output = Self>
    + Sum
    + Eq
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
}

#[cfg(not(feature = "opencl"))]
pub trait CType:
    Add<Output = Self>
    + Sub<Output = Self>
    + Sum
    + Eq
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
}

impl CType for u8 {
    const TYPE: &'static str = "uchar";

    const ZERO: Self = 0;

    const ONE: Self = 1;
}

impl CType for i32 {
    const TYPE: &'static str = "int";
    const ZERO: Self = 0;
    const ONE: Self = 1;
}

impl CType for u32 {
    const TYPE: &'static str = "uint";
    const ZERO: Self = 0;
    const ONE: Self = 1;
}

impl CType for u64 {
    const TYPE: &'static str = "ulong";
    const ZERO: Self = 0;
    const ONE: Self = 1;
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

pub type Shape = SmallVec<[usize; 8]>;

pub type Strides = SmallVec<[usize; 8]>;

pub type Array<T> = array::Array<T, AccessBuffer<Buffer<T>>, Platform>;

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
