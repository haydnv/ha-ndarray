use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Div, Mul, Rem, Sub};

#[cfg(feature = "complex")]
use num_complex::{Complex32, Complex64};
use number_general as ng;
use safecast::CastFrom;

pub use smallvec::smallvec as axes;
pub use smallvec::smallvec as coord;
pub use smallvec::smallvec as range;
pub use smallvec::smallvec as slice;
pub use smallvec::smallvec as shape;
pub use smallvec::smallvec as stackvec;
use smallvec::SmallVec;

pub use access::*;
pub use array::{
    MatrixDual, MatrixUnary, NDArray, NDArrayAbs, NDArrayBoolean, NDArrayBooleanScalar,
    NDArrayCast, NDArrayCompare, NDArrayCompareScalar, NDArrayMath, NDArrayMathScalar,
    NDArrayNumeric, NDArrayRead, NDArrayReduce, NDArrayReduceAll, NDArrayReduceBoolean,
    NDArrayTransform, NDArrayTrig, NDArrayUnary, NDArrayUnaryBoolean, NDArrayWhere, NDArrayWrite,
};
#[cfg(feature = "complex")]
pub use array::{NDArrayComplex, NDArrayFourier};
pub use buffer::{Buffer, BufferConverter, BufferInstance, BufferMut};
pub use host::StackVec;
pub use platform::*;

mod access;
mod array;
mod buffer;
#[cfg(feature = "complex")]
pub mod fft;
pub mod host;
#[cfg(feature = "opencl")]
pub mod opencl;
pub mod ops;
mod platform;

fn id<T>(this: T) -> T {
    this
}

pub trait CLType: PartialEq + Copy + Send + Sync + fmt::Display + fmt::Debug + 'static {
    #[cfg(feature = "opencl")]
    type CType: ocl::OclPrm;

    const TYPE: &'static str;
}

macro_rules! cl_type {
    ($t:ty, $ct:ty, $str:expr) => {
        impl CLType for $t {
            #[cfg(feature = "opencl")]
            type CType = $ct;
            const TYPE: &'static str = $str;
        }
    };
}

#[cfg(feature = "complex")]
impl CLType for Complex32 {
    #[cfg(feature = "opencl")]
    type CType = Self;

    const TYPE: &'static str = "float2";
}

#[cfg(feature = "complex")]
impl CLType for Complex64 {
    #[cfg(feature = "opencl")]
    type CType = Self;

    const TYPE: &'static str = "double2";
}

cl_type!(f32, Self, "float");
cl_type!(f64, Self, "double");
cl_type!(i8, Self, "char");
cl_type!(i16, Self, "short");
cl_type!(i32, Self, "int");
cl_type!(i64, Self, "long");
cl_type!(u8, Self, "uchar");
cl_type!(u16, Self, "ushort");
cl_type!(u32, Self, "uint");
cl_type!(u64, Self, "ulong");

/// A numeric type supported by ha-ndarray
pub trait Number: CLType + Into<ng::Number> + CastFrom<ng::Number> + Default {
    /// The zero value of this data type.
    const ZERO: Self;

    /// The one value of this data type.
    const ONE: Self;

    /// Whether this is a floating-point data type.
    const IS_FLOAT: bool;

    /// Whether this is a read-valued data type.
    const IS_REAL: bool;

    /// The absolute value type of this [`Number`].
    type Abs: Number;

    /// The floating-point type used to represent this type in floating-point-only operations.
    type Float: Float;

    // constructors

    /// Construct an instance of this type from its OpenCL type
    #[cfg(feature = "opencl")]
    fn from_cl(value: Self::CType) -> Self;

    /// Construct an instance of this type from an instance of its floating-point type.
    fn from_float(float: Self::Float) -> Self;

    // arithmetic

    /// Construct an instance of this type from a [`f64`].
    fn abs(self) -> Self::Abs;

    /// Add two instances of this type.
    fn add(self, other: Self) -> Self;

    /// Divide two instances of this type.
    fn div(self, other: Self) -> Self;

    /// Multiply two instances of this type.
    fn mul(self, other: Self) -> Self;

    /// Subtract two instances of this type.
    fn sub(self, other: Self) -> Self;

    /// Raise this value to the power of the given `exp`onent.
    fn pow(self, exp: Self) -> Self;

    // conversions

    /// Return this value as an instance of its OpenCL type.
    #[cfg(feature = "opencl")]
    fn to_cl(&self) -> Self::CType;

    /// Convert this value to a floating-point value.
    fn to_float(self) -> Self::Float;
}

macro_rules! number {
    ($t:ty, $is_float:expr, $is_real:expr, $abs_t:ty, $one:expr, $zero:expr, $float:ty, $abs:expr, $add:expr, $div:expr, $mul:expr, $sub:expr, $pow:expr) => {
        impl Number for $t {
            const ONE: Self = $one;

            const ZERO: Self = $zero;

            const IS_FLOAT: bool = $is_float;

            const IS_REAL: bool = $is_float;

            type Abs = $abs_t;

            type Float = $float;

            #[cfg(feature = "opencl")]
            fn from_cl(value: Self) -> Self {
                value
            }

            fn from_float(float: $float) -> Self {
                float as $t
            }

            fn abs(self) -> Self::Abs {
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

            fn pow(self, exp: Self) -> Self {
                ($pow)(self, exp)
            }

            #[cfg(feature = "opencl")]
            fn to_cl(&self) -> Self::CType {
                *self
            }

            fn to_float(self) -> $float {
                self as $float
            }
        }
    };
}

#[cfg(feature = "complex")]
number!(
    Complex32,
    true,
    false,
    f32,
    Complex32::new(1., 0.),
    Complex32::new(0., 0.),
    Self,
    Complex32::norm,
    Add::add,
    Div::div,
    Mul::mul,
    Sub::sub,
    Complex32::powc
);

#[cfg(feature = "complex")]
number!(
    Complex64,
    true,
    false,
    f64,
    Complex64::new(1., 0.),
    Complex64::new(0., 0.),
    Self,
    Complex64::norm,
    Add::add,
    Div::div,
    Mul::mul,
    Sub::sub,
    Complex64::powc
);

number!(
    f32,
    true,
    true,
    Self,
    1.,
    0.,
    Self,
    f32::abs,
    Add::add,
    Div::div,
    Mul::mul,
    Sub::sub,
    f32::powf
);

number!(
    f64,
    true,
    true,
    Self,
    1.,
    0.,
    Self,
    f64::abs,
    Add::add,
    Div::div,
    Mul::mul,
    Sub::sub,
    f64::powf
);

number!(
    i8,
    false,
    true,
    Self,
    1,
    0,
    f32,
    Self::wrapping_abs,
    Self::wrapping_add,
    |l, r| if r == 0 { 0 } else { Self::wrapping_div(l, r) },
    Self::wrapping_mul,
    Self::wrapping_sub,
    |a, e| f32::powi(a as f32, e as i32) as i8
);

number!(
    i16,
    false,
    true,
    Self,
    1,
    0,
    f32,
    Self::wrapping_abs,
    Self::wrapping_add,
    |l, r| if r == 0 { 0 } else { Self::wrapping_div(l, r) },
    Self::wrapping_mul,
    Self::wrapping_sub,
    |a, e| f32::powi(a as f32, e as i32) as i16
);

number!(
    i32,
    false,
    true,
    Self,
    1,
    0,
    f32,
    Self::wrapping_abs,
    Self::wrapping_add,
    |l, r| if r == 0 { 0 } else { Self::wrapping_div(l, r) },
    Self::wrapping_mul,
    Self::wrapping_sub,
    |a, e| f32::powi(a as f32, e) as i32
);

number!(
    i64,
    false,
    true,
    Self,
    1,
    0,
    f64,
    Self::wrapping_abs,
    Self::wrapping_add,
    |l, r| if r == 0 { 0 } else { Self::wrapping_div(l, r) },
    Self::wrapping_mul,
    Self::wrapping_sub,
    |a, e| f64::powi(
        a as f64,
        i32::try_from(e).unwrap_or_else(|_| if e >= 0 { i32::MAX } else { i32::MIN })
    ) as i64
);

number!(
    u8,
    false,
    true,
    Self,
    1,
    0,
    f32,
    id,
    Self::wrapping_add,
    |l, r| if r == 0 { 0 } else { Self::wrapping_div(l, r) },
    Self::wrapping_mul,
    Self::wrapping_sub,
    |a, e| u8::pow(a, e as u32)
);

number!(
    u16,
    false,
    true,
    Self,
    1,
    0,
    f32,
    id,
    Self::wrapping_add,
    |l, r| if r == 0 { 0 } else { Self::wrapping_div(l, r) },
    Self::wrapping_mul,
    Self::wrapping_sub,
    |a, e| u16::pow(a, e as u32)
);

number!(
    u32,
    false,
    true,
    Self,
    1,
    0,
    f32,
    id,
    Self::wrapping_add,
    |l, r| if r == 0 { 0 } else { Self::wrapping_div(l, r) },
    Self::wrapping_mul,
    Self::wrapping_sub,
    |a, e| u32::pow(a, e)
);

number!(
    u64,
    false,
    true,
    Self,
    1,
    0,
    f64,
    id,
    Self::wrapping_add,
    |l, r| if r == 0 { 0 } else { Self::wrapping_div(l, r) },
    Self::wrapping_mul,
    Self::wrapping_sub,
    |a, e| u64::pow(a, u32::try_from(e).unwrap_or(u32::MAX))
);

/// A real-valued [`Number`]
pub trait Real: Number {
    /// The maximum value of this data type.
    const MAX: Self;

    /// The minimum value of this data type.
    const MIN: Self;

    /// Return the maximum of the given values.
    fn max(l: Self, r: Self) -> Self;

    /// Return the maximum of the given values.
    fn min(l: Self, r: Self) -> Self;

    /// Compute the remainder of `self.div(other)`.
    fn rem(self, other: Self) -> Self;

    /// Round this value to the nearest integer.
    fn round(self) -> Self;
}

macro_rules! real {
    ($t:ty, $rem:expr, $ord:expr, $round:expr) => {
        impl Real for $t {
            const MAX: Self = <$t>::MAX;

            const MIN: Self = <$t>::MIN;

            fn max(l: Self, r: Self) -> $t {
                match $ord(&l, &r) {
                    Ordering::Greater | Ordering::Equal => l,
                    Ordering::Less => r,
                }
            }

            fn min(l: Self, r: Self) -> $t {
                match $ord(&l, &r) {
                    Ordering::Less | Ordering::Equal => l,
                    Ordering::Greater => r,
                }
            }

            fn rem(self, other: Self) -> Self {
                $rem(self, other)
            }

            fn round(self) -> Self {
                $round(self)
            }
        }
    };
}

real!(f32, Rem::rem, f32::total_cmp, f32::round);
real!(f64, Rem::rem, f64::total_cmp, f64::round);
real!(i8, Self::wrapping_rem, Ord::cmp, id);
real!(i16, Self::wrapping_rem, Ord::cmp, id);
real!(i32, Self::wrapping_rem, Ord::cmp, id);
real!(i64, Self::wrapping_rem, Ord::cmp, id);
real!(u8, Self::wrapping_rem, Ord::cmp, id);
real!(u16, Self::wrapping_rem, Ord::cmp, id);
real!(u32, Self::wrapping_rem, Ord::cmp, id);
real!(u64, Self::wrapping_rem, Ord::cmp, id);

/// A floating-point [`Number`]
pub trait Float: Number<Float = Self> {
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
}

macro_rules! float_type {
    ($t:ty, $inf:expr, $nan:expr) => {
        impl Float for $t {
            fn is_inf(self) -> bool {
                $inf(self)
            }

            fn is_nan(self) -> bool {
                $nan(self)
            }

            fn exp(self) -> Self {
                <$t>::exp(self)
            }

            fn ln(self) -> Self {
                <$t>::ln(self)
            }

            fn log(self, base: Self) -> Self {
                self.ln() / base.ln()
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
        }
    };
}

#[cfg(feature = "complex")]
float_type!(Complex32, |_| false, |_| false);
#[cfg(feature = "complex")]
float_type!(Complex64, |_| false, |_| false);
float_type!(f32, f32::is_infinite, f32::is_nan);
float_type!(f64, f64::is_infinite, f64::is_nan);

#[cfg(feature = "complex")]
/// A complex [`Number`]
pub trait Complex: Float<Abs = Self::Real> {
    type Real: Float + Real;

    fn angle(self) -> Self::Real;

    fn conj(self) -> Self;

    fn im(self) -> Self::Real;

    fn re(self) -> Self::Real;
}

#[cfg(feature = "complex")]
macro_rules! complex_type {
    ($t:ty, $r:ty) => {
        impl Complex for $t {
            type Real = $r;

            fn angle(self) -> $r {
                Self::arg(self)
            }

            fn conj(self) -> Self {
                num_complex::Complex::<$r>::conj(&self)
            }

            fn im(self) -> $r {
                self.im
            }

            fn re(self) -> $r {
                self.re
            }
        }
    };
}

#[cfg(feature = "complex")]
complex_type!(Complex32, f32);
#[cfg(feature = "complex")]
complex_type!(Complex64, f64);

/// An array math error
pub enum Error {
    Bounds(String),
    Unsupported(String),
    #[cfg(feature = "opencl")]
    OCL(std::sync::Arc<ocl::Error>),
}

impl Error {
    fn bounds(msg: String) -> Self {
        #[cfg(feature = "debug_crash")]
        panic!("{}", msg);

        #[cfg(not(feature = "debug_crash"))]
        Self::Bounds(msg)
    }

    fn unsupported(msg: String) -> Self {
        #[cfg(feature = "debug_crash")]
        panic!("{}", msg);

        #[cfg(not(feature = "debug_crash"))]
        Self::Unsupported(msg)
    }
}

// Clone is required to support memoizing OpenCL programs
// since constructing an [`ocl::Program`] may return an error
impl Clone for Error {
    fn clone(&self) -> Self {
        match self {
            Self::Bounds(msg) => Self::Bounds(msg.clone()),
            Self::Unsupported(msg) => Self::Unsupported(msg.clone()),
            #[cfg(feature = "opencl")]
            Self::OCL(cause) => Self::OCL(cause.clone()),
        }
    }
}

#[cfg(feature = "opencl")]
impl From<ocl::Error> for Error {
    fn from(cause: ocl::Error) -> Self {
        #[cfg(feature = "debug_crash")]
        panic!("OpenCL error: {:?}", cause);

        #[cfg(not(feature = "debug_crash"))]
        Self::OCL(std::sync::Arc::new(cause))
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Bounds(cause) => f.write_str(cause),
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
    let ndim = usize::max(left.len(), right.len());
    let mut shape = Shape::with_capacity(ndim);

    let mut left = left.into_iter().rev().copied();
    let mut right = right.into_iter().rev().copied();

    loop {
        if let Some(dim) = broadcast_dim(left.next(), right.next())? {
            shape.push(dim)
        } else {
            break;
        }
    }

    shape.reverse();

    Ok(shape)
}

/// Compute the shapes needed to multiply the `left` and `right` matrices, if possible.
#[inline]
pub fn broadcast_matmul_shape(left: &[usize], right: &[usize]) -> Result<(Shape, Shape), Error> {
    let (left_ndim, right_ndim) = (left.len(), right.len());
    let ndim = usize::max(left_ndim, right_ndim);

    let mut left = left.into_iter().rev().copied();
    let mut right = right.into_iter().rev().copied();

    let k = right.next().unwrap_or(1);
    let j = match (left.next(), right.next()) {
        (Some(jl), Some(jr)) => match (jl, jr) {
            (jl, jr) if jl == jr => Ok(jl),
            (jl, jr) if jl == 1 => Ok(jr),
            (jl, jr) if jr == 1 => Ok(jl),
            _ => Err(Error::bounds(format!(
                "cannot matrix-multiply shapes {left:?} and {right:?}"
            ))),
        },
        (Some(jl), None) => Ok(jl),
        (None, Some(jr)) => Ok(jr),
        (None, None) => Ok(1),
    }?;
    let i = left.next().unwrap_or(1);

    let mut left = left.rev();
    let mut right = right.rev();
    let mut broadcast_shape = Shape::with_capacity(ndim);
    loop {
        if let Some(dim) = broadcast_dim(left.next(), right.next())? {
            broadcast_shape.push(dim);
        } else {
            break;
        }
    }

    let left = broadcast_shape.iter().copied().chain([i, j]).collect();
    let right = broadcast_shape.into_iter().chain([j, k]).collect();
    Ok((left, right))
}

#[inline]
fn broadcast_dim(left: Option<usize>, right: Option<usize>) -> Result<Option<usize>, Error> {
    match (left, right) {
        (Some(l), Some(r)) if l == r => Ok(Some(l)),
        (Some(1), Some(r)) => Ok(Some(r)),
        (Some(l), Some(1)) => Ok(Some(l)),
        (None, Some(r)) => Ok(Some(r)),
        (Some(l), None) => Ok(Some(l)),
        (None, None) => Ok(None),
        (l, r) => Err(Error::bounds(format!(
            "cannot broadcast dimensions {l:?} and {r:?}"
        ))),
    }
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
