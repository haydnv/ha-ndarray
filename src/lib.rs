use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Div, Mul, Rem, Sub};

pub use smallvec::smallvec as axes;
pub use smallvec::smallvec as range;
pub use smallvec::smallvec as slice;
pub use smallvec::smallvec as shape;
pub use smallvec::smallvec as stackvec;
use smallvec::SmallVec;

pub use access::*;
pub use array::{
    MatrixDual, MatrixUnary, NDArray, NDArrayBoolean, NDArrayBooleanScalar, NDArrayCast,
    NDArrayCompare, NDArrayCompareScalar, NDArrayMath, NDArrayMathScalar, NDArrayNumeric,
    NDArrayRead, NDArrayReduce, NDArrayReduceAll, NDArrayReduceBoolean, NDArrayTransform,
    NDArrayTrig, NDArrayUnary, NDArrayUnaryBoolean, NDArrayWhere, NDArrayWrite,
};
pub use buffer::{Buffer, BufferConverter, BufferInstance, BufferMut};
pub use host::StackVec;
pub use platform::*;

mod access;
mod array;
mod buffer;
pub mod host;
#[cfg(feature = "opencl")]
pub mod opencl;
pub mod ops;
mod platform;

/// A numeric type supported by ha-ndarray.
#[cfg(feature = "opencl")]
pub trait CType:
    ocl::OclPrm + PartialEq + PartialOrd + Copy + Send + Sync + fmt::Display + fmt::Debug + 'static
{
    // type information

    /// The C-language type name of this data type.
    const TYPE: &'static str;

    /// The maximum value of this data type.
    const MAX: Self;

    /// The minimum value of this data type.
    const MIN: Self;

    /// The zero value of this data type.
    const ZERO: Self;

    /// The one value of this data type.
    const ONE: Self;

    /// Whether this is a floating-point data type.
    const IS_FLOAT: bool;

    /// The floating-point type used to represent this type in floating-point-only operations.
    type Float: Float;

    // constructors

    /// Construct an instance of this type from a [`f64`].
    fn from_f64(float: f64) -> Self;

    /// Construct an instance of this type from an instance of its floating-point type.
    fn from_float(float: Self::Float) -> Self;

    // arithmetic

    /// Construct an instance of this type from a [`f64`].
    fn abs(self) -> Self;

    /// Add two instances of this type.
    fn add(self, other: Self) -> Self;

    /// Divide two instances of this type.
    fn div(self, other: Self) -> Self;

    /// Multiply two instances of this type.
    fn mul(self, other: Self) -> Self;

    /// Subtract two instances of this type.
    fn sub(self, other: Self) -> Self;

    /// Compute the remainder of `self.div(other)`.
    fn rem(self, other: Self) -> Self;

    // comparisons

    /// Return the minimum of two values of this type.
    fn min(l: Self, r: Self) -> Self;

    /// Return the maximum of two values of this type.
    fn max(l: Self, r: Self) -> Self;

    // logarithms

    /// Raise this value to the power of the given `exp`onent.
    fn pow(self, exp: Self) -> Self;

    // conversions

    /// Round this value to the nearest integer.
    fn round(self) -> Self;

    /// Return the minimum of two values of this type.
    fn to_f64(self) -> f64;

    /// Convert this value to a floating-point value.
    fn to_float(self) -> Self::Float;
}

/// A numeric type supported by ha-ndarray.
#[cfg(not(feature = "opencl"))]
pub trait CType:
    PartialEq + PartialOrd + Copy + Send + Sync + fmt::Display + fmt::Debug + 'static
{
    // type information

    /// The C-language type name of this data type.
    const TYPE: &'static str;

    /// The maximum value of this data type.
    const MAX: Self;

    /// The minimum value of this data type.
    const MIN: Self;

    /// The zero value of this data type.
    const ZERO: Self;

    /// The one value of this data type.
    const ONE: Self;

    /// Whether this is a floating-point data type.
    const IS_FLOAT: bool;

    /// The floating-point type used to represent this type in floating-point-only operations.
    type Float: Float;

    // constructors

    /// Construct an instance of this type from a [`f64`].
    fn from_f64(float: f64) -> Self;

    /// Construct an instance of this type from an instance of its floating-point type.
    fn from_float(float: Self::Float) -> Self;

    // arithmetic

    /// Construct an instance of this type from a [`f64`].
    fn abs(self) -> Self;

    /// Add two instances of this type.
    fn add(self, other: Self) -> Self;

    /// Divide two instances of this type.
    fn div(self, other: Self) -> Self;

    /// Multiply two instances of this type.
    fn mul(self, other: Self) -> Self;

    /// Subtract two instances of this type.
    fn sub(self, other: Self) -> Self;

    /// Compute the remainder of `self.div(other)`.
    fn rem(self, other: Self) -> Self;

    // comparisons

    /// Return the minimum of two values of this type.
    fn min(l: Self, r: Self) -> Self;

    /// Return the maximum of two values of this type.
    fn max(l: Self, r: Self) -> Self;

    // logarithms

    /// Raise this value to the power of the given `exp`onent.
    fn pow(self, exp: Self) -> Self;

    // conversions

    /// Round this value to the nearest integer.
    fn round(self) -> Self;

    /// Return the minimum of two values of this type.
    fn to_f64(self) -> f64;

    /// Convert this value to a floating-point value.
    fn to_float(self) -> Self::Float;
}

macro_rules! c_type {
    ($t:ty, $str:expr, $is_float:expr, $one:expr, $zero:expr, $float:ty, $abs:expr, $add:expr, $div:expr, $mul:expr, $sub:expr, $rem:expr, $round:expr, $pow:expr, $cmp_max:expr, $cmp_min:expr) => {
        impl CType for $t {
            const TYPE: &'static str = $str;

            const MAX: Self = <$t>::MAX;

            const MIN: Self = <$t>::MIN;

            const ZERO: Self = $zero;

            const ONE: Self = $one;

            const IS_FLOAT: bool = $is_float;

            type Float = $float;

            fn from_f64(float: f64) -> Self {
                float as $t
            }

            fn from_float(float: $float) -> Self {
                float as $t
            }

            fn abs(self) -> Self {
                $abs(self)
            }

            fn add(self, other: Self) -> Self {
                $add(self, other)
            }

            fn div(self, other: Self) -> Self {
                $div(self, other)
            }

            fn mul(self, other: Self) -> Self {
                $mul(self, other)
            }

            fn sub(self, other: Self) -> Self {
                $sub(self, other)
            }

            fn rem(self, other: Self) -> Self {
                $rem(self, other)
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
    1.,
    0.,
    Self,
    f32::abs,
    Add::add,
    Div::div,
    Mul::mul,
    Sub::sub,
    Rem::rem,
    f32::round,
    f32::powf,
    max_f32,
    min_f32
);

c_type!(
    f64,
    "double",
    true,
    1.,
    0.,
    Self,
    f64::abs,
    Add::add,
    Div::div,
    Mul::mul,
    Sub::sub,
    Rem::rem,
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
    Self::wrapping_abs,
    Self::wrapping_add,
    |l, r| if r == 0 { 0 } else { Self::wrapping_div(l, r) },
    Self::wrapping_mul,
    Self::wrapping_sub,
    Self::wrapping_rem,
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
    Self::wrapping_abs,
    Self::wrapping_add,
    |l, r| if r == 0 { 0 } else { Self::wrapping_div(l, r) },
    Self::wrapping_mul,
    Self::wrapping_sub,
    Self::wrapping_rem,
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
    Self::wrapping_abs,
    Self::wrapping_add,
    |l, r| if r == 0 { 0 } else { Self::wrapping_div(l, r) },
    Self::wrapping_mul,
    Self::wrapping_sub,
    Self::wrapping_rem,
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
    Self::wrapping_abs,
    Self::wrapping_add,
    |l, r| if r == 0 { 0 } else { Self::wrapping_div(l, r) },
    Self::wrapping_mul,
    Self::wrapping_sub,
    Self::wrapping_rem,
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
    Self::wrapping_add,
    |l, r| if r == 0 { 0 } else { Self::wrapping_div(l, r) },
    Self::wrapping_mul,
    Self::wrapping_sub,
    Self::wrapping_rem,
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
    Self::wrapping_add,
    |l, r| if r == 0 { 0 } else { Self::wrapping_div(l, r) },
    Self::wrapping_mul,
    Self::wrapping_sub,
    Self::wrapping_rem,
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
    Self::wrapping_add,
    |l, r| if r == 0 { 0 } else { Self::wrapping_div(l, r) },
    Self::wrapping_mul,
    Self::wrapping_sub,
    Self::wrapping_rem,
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
    Self::wrapping_add,
    |l, r| if r == 0 { 0 } else { Self::wrapping_div(l, r) },
    Self::wrapping_mul,
    Self::wrapping_sub,
    Self::wrapping_rem,
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

/// A floating-point [`CType`].
pub trait Float: CType<Float = Self> {
    // numeric methods
    /// Return `true` if this [`Float`] is infinite (positive or negative infinity).
    fn is_inf(self) -> bool;

    /// Return `true` if this [`Float`] is not a number (e.g. a float representation of `1.0 / 0.0`).
    fn is_nan(self) -> bool;

    // logarithms
    /// Exponentiate this number (equivalent to `consts::E.pow(self)`).
    fn exp(self) -> Self;

    /// Return the natural logarithm of this [`Float`].
    fn ln(self) -> Self;

    /// Calculate the logarithm of this [`Float`] w/r/t the given `base`.
    fn log(self, base: Self) -> Self;

    // trigonometry
    /// Return the sine of this [`Float`] (in radians).
    fn sin(self) -> Self;

    /// Return the arcsine of this [`Float`] (in radians).
    fn asin(self) -> Self;

    /// Return the hyperbolic sine of this [`Float`] (in radians).
    fn sinh(self) -> Self;

    /// Return the cosine of this [`Float`] (in radians).
    fn cos(self) -> Self;

    /// Return the arcsine of this [`Float`] (in radians).
    fn acos(self) -> Self;

    /// Return the hyperbolic cosine of this [`Float`] (in radians).
    fn cosh(self) -> Self;

    /// Return the tangent of this [`Float`] (in radians).
    fn tan(self) -> Self;

    /// Return the arctangent of this [`Float`] (in radians).
    fn atan(self) -> Self;

    /// Return the hyperbolic tangent of this [`Float`] (in radians).
    fn tanh(self) -> Self;

    // utility
    /// Cast this [`Float`] to an [`f64`].
    fn to_f64(self) -> f64;
}

macro_rules! float_type {
    ($t:ty) => {
        impl Float for $t {
            fn is_inf(self) -> bool {
                <$t>::is_infinite(self)
            }

            fn is_nan(self) -> bool {
                <$t>::is_nan(self)
            }

            fn exp(self) -> Self {
                <$t>::exp(self)
            }

            fn ln(self) -> Self {
                <$t>::ln(self)
            }

            fn log(self, base: Self) -> Self {
                <$t>::log(self, base)
            }

            fn sin(self) -> Self {
                <$t>::sin(self)
            }

            fn asin(self) -> Self {
                <$t>::asin(self)
            }

            fn sinh(self) -> Self {
                <$t>::sinh(self)
            }

            fn cos(self) -> Self {
                <$t>::cos(self)
            }

            fn acos(self) -> Self {
                <$t>::acos(self)
            }

            fn cosh(self) -> Self {
                <$t>::cosh(self)
            }

            fn tan(self) -> Self {
                <$t>::tan(self)
            }

            fn atan(self) -> Self {
                <$t>::atan(self)
            }

            fn tanh(self) -> Self {
                <$t>::tanh(self)
            }

            fn to_f64(self) -> f64 {
                self as f64
            }
        }
    };
}

float_type!(f32);
float_type!(f64);

/// An array math error
pub enum Error {
    Bounds(String),
    Interface(String),
    Unsupported(String),
    #[cfg(feature = "opencl")]
    OCL(std::sync::Arc<ocl::Error>),
}

// Clone is required to support memoizing OpenCL programs
// since constructing an [`ocl::Program`] may return an error
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

/// A list of n-dimensional array axes
pub type Axes = SmallVec<[usize; 8]>;

/// An n-dimensional selection range, used to slice an array
pub type Range = SmallVec<[AxisRange; 8]>;

/// The shape of an n-dimensional array
pub type Shape = SmallVec<[usize; 8]>;

/// The strides used to access an n-dimensional array
pub type Strides = SmallVec<[usize; 8]>;

/// An n-dimensional array on the top-level [`Platform`]
pub type Array<T, A> = array::Array<T, A, Platform>;

/// An n-dimensional array backed by a buffer on the top-level [`Platform`]
pub type ArrayBuf<T, B> = array::Array<T, AccessBuf<B>, Platform>;

/// The result of an n-dimensional array operation
pub type ArrayOp<T, Op> = array::Array<T, AccessOp<Op>, Platform>;

/// A general type of n-dimensional array used to elide recursive types
pub type ArrayAccess<T> = array::Array<T, Accessor<T>, Platform>;

/// An accessor for the result of an n-dimensional array operation on the top-level [`Platform`]
pub type AccessOp<Op> = access::AccessOp<Op, Platform>;

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

/// Compute the shape which results from broadcasting the `left` and `right` shapes, if possible.
#[inline]
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

/// Construct an iterator over the strides for the given shape and number of dimensions.
#[inline]
pub fn strides_for<'a>(shape: &'a [usize], ndim: usize) -> impl Iterator<Item = usize> + 'a {
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
