use std::borrow::{Borrow, BorrowMut};
use std::fmt;
use std::marker::PhantomData;

use crate::access::*;
use crate::buffer::BufferInstance;
use crate::ops::*;
use crate::platform::PlatformInstance;
use crate::{
    shape, strides_for, AxisRange, BufferConverter, CType, Convert, Error, Platform, Range, Shape,
};

pub struct Array<T, A, P> {
    shape: Shape,
    access: A,
    platform: P,
    dtype: PhantomData<T>,
}

impl<T, A, P> Array<T, A, P> {
    fn apply<O, OT, Op>(self, op: Op) -> Result<Array<OT, AccessOp<O, P>, P>, Error>
    where
        P: Copy,
        Op: Fn(P, A) -> Result<AccessOp<O, P>, Error>,
    {
        let access = (op)(self.platform, self.access)?;

        Ok(Array {
            shape: self.shape,
            access,
            platform: self.platform,
            dtype: PhantomData,
        })
    }

    pub fn into_inner(self) -> A {
        self.access
    }
}

// constructors
impl<T: CType> Array<T, Accessor<T>, Platform> {
    pub fn from<A, P>(array: Array<T, A, P>) -> Self
    where
        Accessor<T>: From<A>,
        Platform: From<P>,
    {
        Self {
            shape: array.shape,
            access: array.access.into(),
            platform: array.platform.into(),
            dtype: array.dtype,
        }
    }
}

impl<T, B, P> Array<T, AccessBuffer<B>, P>
where
    T: CType,
    B: BufferInstance<T>,
    P: PlatformInstance,
{
    pub fn new(buffer: B, shape: Shape) -> Result<Self, Error> {
        if shape.iter().product::<usize>() == buffer.size() {
            let platform = P::select(buffer.size());
            let access = buffer.into();

            Ok(Self {
                shape,
                access,
                platform,
                dtype: PhantomData,
            })
        } else {
            Err(Error::Bounds(format!(
                "cannot construct an array with shape {shape:?} from a buffer of size {}",
                buffer.size()
            )))
        }
    }
}

// op constructors
impl<T: CType, P: PlatformInstance> Array<T, AccessOp<P::Range, P>, P>
where
    P: Construct<T>,
{
    pub fn range(start: T, stop: T, size: usize) -> Result<Self, Error> {
        let platform = P::select(size);
        let shape = shape![size];

        platform.range(start, stop, size).map(|access| Self {
            shape,
            access,
            platform,
            dtype: PhantomData,
        })
    }
}

impl<P: PlatformInstance> Array<f32, AccessOp<P::Normal, P>, P>
where
    P: Random,
{
    pub fn random_normal(size: usize) -> Result<Self, Error> {
        let platform = P::select(size);
        let shape = shape![size];

        platform.random_normal(size).map(|access| Self {
            shape,
            access,
            platform,
            dtype: PhantomData,
        })
    }
}

impl<P: PlatformInstance> Array<f32, AccessOp<P::Uniform, P>, P>
where
    P: Random,
{
    pub fn random_uniform(size: usize) -> Result<Self, Error> {
        let platform = P::select(size);
        let shape = shape![size];

        platform.random_uniform(size).map(|access| Self {
            shape,
            access,
            platform,
            dtype: PhantomData,
        })
    }
}

// references
impl<T, B, P> Array<T, AccessBuffer<B>, P>
where
    T: CType,
    B: BufferInstance<T>,
    P: PlatformInstance,
{
    pub fn as_mut<RB: ?Sized>(&mut self) -> Array<T, AccessBuffer<&mut RB>, P>
    where
        B: BorrowMut<RB>,
    {
        Array {
            shape: Shape::from_slice(&self.shape),
            access: self.access.as_mut(),
            platform: self.platform,
            dtype: PhantomData,
        }
    }

    pub fn as_ref<RB: ?Sized>(&self) -> Array<T, AccessBuffer<&RB>, P>
    where
        B: Borrow<RB>,
    {
        Array {
            shape: Shape::from_slice(&self.shape),
            access: self.access.as_ref(),
            platform: self.platform,
            dtype: PhantomData,
        }
    }
}

// references
impl<T, O, P> Array<T, AccessOp<O, P>, P>
where
    T: CType,
    O: Enqueue<P, T>,
    P: PlatformInstance,
{
    pub fn as_mut<'a>(&'a mut self) -> Array<T, &'a mut AccessOp<O, P>, P>
    where
        O: Write<'a, P, T>,
    {
        Array {
            shape: Shape::from_slice(&self.shape),
            access: &mut self.access,
            platform: self.platform,
            dtype: PhantomData,
        }
    }

    pub fn as_ref(&self) -> Array<T, &AccessOp<O, P>, P> {
        Array {
            shape: Shape::from_slice(&self.shape),
            access: &self.access,
            platform: self.platform,
            dtype: PhantomData,
        }
    }
}

// traits

/// An n-dimensional array
pub trait NDArray: Send + Sync {
    /// The data type of the elements in this array
    type DType: CType;

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

impl<T, A, P> NDArray for Array<T, A, P>
where
    T: CType,
    A: Send + Sync,
    P: Send + Sync,
{
    type DType = T;

    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

/// Access methods for an [`NDArray`]
pub trait NDArrayRead: NDArray + fmt::Debug + Sized {
    /// Read the value of this [`NDArray`].
    fn read(&self) -> Result<BufferConverter<Self::DType>, Error>;

    /// Read the value at one `coord` in this [`NDArray`].
    fn read_value(&self, coord: &[usize]) -> Result<Self::DType, Error>;
}

impl<T, A, P> NDArrayRead for Array<T, A, P>
where
    T: CType,
    A: Access<T>,
    P: PlatformInstance,
{
    fn read(&self) -> Result<BufferConverter<Self::DType>, Error> {
        self.access.read()
    }

    fn read_value(&self, coord: &[usize]) -> Result<Self::DType, Error> {
        valid_coord(coord, self.shape())?;

        let strides = strides_for(self.shape(), self.ndim());

        let offset = coord
            .iter()
            .zip(strides)
            .map(|(i, stride)| i * stride)
            .sum();

        self.access.read_value(offset)
    }
}

/// Access methods for a mutable [`NDArray`]
pub trait NDArrayWrite<'a>: NDArray + fmt::Debug + Sized {
    /// Overwrite this [`NDArray`] with the value of the `other` array.
    fn write<O: NDArrayRead<DType = Self::DType>>(&'a mut self, other: &'a O) -> Result<(), Error>;

    /// Overwrite this [`NDArray`] with a constant scalar `value`.
    fn write_value(&'a mut self, value: Self::DType) -> Result<(), Error>;

    /// Write the given `value` at the given `coord` of this [`NDArray`].
    fn write_value_at(&'a mut self, coord: &[usize], value: Self::DType) -> Result<(), Error>;
}

// write ops
impl<'a, T, L, P> NDArrayWrite<'a> for Array<T, L, P>
where
    T: CType,
    L: AccessMut<'a, T>,
    P: Convert<'a, T, Buffer = L::Data>,
{
    fn write<O>(&'a mut self, other: &'a O) -> Result<(), Error>
    where
        O: NDArrayRead<DType = Self::DType>,
    {
        same_shape("write", self.shape(), other.shape())?;

        let data = other.read().and_then(|buf| self.platform.convert(buf))?;

        self.access.write(data)
    }

    fn write_value(&'a mut self, value: Self::DType) -> Result<(), Error> {
        self.access.write_value(value)
    }

    fn write_value_at(&'a mut self, coord: &[usize], value: Self::DType) -> Result<(), Error> {
        valid_coord(coord, self.shape())?;

        let offset = coord
            .iter()
            .zip(strides_for(self.shape(), self.ndim()))
            .map(|(i, stride)| i * stride)
            .sum();

        self.access.write_value_at(offset, value)
    }
}

// op traits

/// Array transform operations
pub trait NDArrayTransform: NDArray + fmt::Debug {
    /// The type returned by `broadcast`
    type Broadcast: NDArray<DType = Self::DType>;

    /// The type returned by `slice`
    type Slice: NDArray<DType = Self::DType>;

    /// Broadcast this array into the given `shape`.
    fn broadcast(self, shape: Shape) -> Result<Self::Broadcast, Error>;

    /// Construct a slice of this array.
    fn slice(self, range: Range) -> Result<Self::Slice, Error>;
}

impl<T, A, P> NDArrayTransform for Array<T, A, P>
where
    T: CType,
    A: Access<T>,
    P: Transform<A, T>,
{
    type Broadcast = Array<T, AccessOp<P::Broadcast, P>, P>;
    type Slice = Array<T, AccessOp<P::Slice, P>, P>;

    fn broadcast(self, shape: Shape) -> Result<Self::Broadcast, Error> {
        if !can_broadcast(self.shape(), &shape) {
            return Err(Error::Bounds(format!(
                "cannot broadcast {self:?} into {shape:?}"
            )));
        }

        let platform = P::select(shape.iter().product());
        let broadcast = Shape::from_slice(&shape);
        let access = platform.broadcast(self.access, self.shape, broadcast)?;

        Ok(Array {
            shape,
            access,
            platform,
            dtype: PhantomData,
        })
    }

    fn slice(self, range: Range) -> Result<Array<T, AccessOp<P::Slice, P>, P>, Error>
    where
        P: Transform<A, T>,
    {
        for (dim, range) in self.shape.iter().zip(&range) {
            match range {
                AxisRange::At(i) if i < dim => Ok(()),
                AxisRange::In(start, stop, _step) if start < dim && stop <= dim => Ok(()),
                AxisRange::Of(indices) if indices.iter().all(|i| i < dim) => Ok(()),
                range => Err(Error::Bounds(format!(
                    "invalid range {range:?} for dimension {dim}"
                ))),
            }?;
        }

        let shape = range.iter().filter_map(|ar| ar.size()).collect::<Shape>();
        let platform = P::select(shape.iter().product());
        let access = platform.slice(self.access, &self.shape, range)?;

        Ok(Array {
            shape,
            access,
            platform,
            dtype: PhantomData,
        })
    }
}

/// Unary array operations
pub trait NDArrayUnary: NDArray + Sized {
    /// The return type of an `ln` operation.
    type Log: NDArray<DType = Self::DType>;

    /// Construct a natural logarithm operation.
    fn ln(self) -> Result<Self::Log, Error>;
}

impl<T, A, P> NDArrayUnary for Array<T, A, P>
where
    T: CType,
    A: Access<T>,
    P: ElementwiseUnary<A, T>,
{
    type Log = Array<T, AccessOp<P::Op, P>, P>;

    fn ln(self) -> Result<Self::Log, Error>
    where
        P: ElementwiseUnary<A, T>,
    {
        self.platform.ln(self.access).map(|access| Array {
            access,
            shape: self.shape,
            platform: self.platform,
            dtype: self.dtype,
        })
    }
}

/// Array comparison operations
pub trait NDArrayCompare<O: NDArray<DType = Self::DType>>: NDArray + Sized {
    type Output: NDArray<DType = u8>;

    fn eq(self, other: O) -> Result<Self::Output, Error>;
}

impl<T, L, R, P> NDArrayCompare<Array<T, R, P>> for Array<T, L, P>
where
    T: CType,
    L: Access<T>,
    R: Access<T>,
    P: ElementwiseCompare<L, R, T>,
{
    type Output = Array<u8, AccessOp<P::Op, P>, P>;

    fn eq(self, other: Array<T, R, P>) -> Result<Self::Output, Error> {
        same_shape("compare", self.shape(), other.shape())?;

        Ok(Array {
            shape: self.shape,
            access: self.platform.eq(self.access, other.access)?,
            platform: self.platform,
            dtype: PhantomData,
        })
    }
}

pub trait NDArrayCompareScalar: NDArray + Sized {
    type Output: NDArray<DType = u8>;

    /// Construct an equality comparison with the `other` value.
    fn eq_scalar(self, other: Self::DType) -> Result<Self::Output, Error>;

    /// Construct a greater-than comparison with the `other` value.
    fn gt_scalar(self, other: Self::DType) -> Result<Self::Output, Error>;

    /// Construct an equal-or-greater-than comparison with the `other` value.
    fn ge_scalar(self, other: Self::DType) -> Result<Self::Output, Error>;

    /// Construct a less-than comparison with the `other` value.
    fn lt_scalar(self, other: Self::DType) -> Result<Self::Output, Error>;

    /// Construct an equal-or-less-than comparison with the `other` value.
    fn le_scalar(self, other: Self::DType) -> Result<Self::Output, Error>;

    /// Construct an not-equal comparison with the `other` value.
    fn ne_scalar(self, other: Self::DType) -> Result<Self::Output, Error>;
}

impl<T, A, P> NDArrayCompareScalar for Array<T, A, P>
where
    T: CType,
    A: Access<T>,
    P: ElementwiseScalarCompare<A, T>,
{
    type Output = Array<u8, AccessOp<P::Op, P>, P>;

    fn eq_scalar(self, other: Self::DType) -> Result<Self::Output, Error> {
        self.apply(|platform, access| platform.eq_scalar(access, other))
    }

    fn gt_scalar(self, other: Self::DType) -> Result<Self::Output, Error> {
        self.apply(|platform, access| platform.gt_scalar(access, other))
    }

    fn ge_scalar(self, other: Self::DType) -> Result<Self::Output, Error> {
        self.apply(|platform, access| platform.ge_scalar(access, other))
    }

    fn lt_scalar(self, other: Self::DType) -> Result<Self::Output, Error> {
        self.apply(|platform, access| platform.lt_scalar(access, other))
    }

    fn le_scalar(self, other: Self::DType) -> Result<Self::Output, Error> {
        self.apply(|platform, access| platform.le_scalar(access, other))
    }

    fn ne_scalar(self, other: Self::DType) -> Result<Self::Output, Error> {
        self.apply(|platform, access| platform.ne_scalar(access, other))
    }
}

/// Array arithmetic operations
pub trait NDArrayMath<O: NDArray<DType = Self::DType>>: NDArray + Sized {
    type Output: NDArray<DType = Self::DType>;

    /// Construct an addition operation with the given `rhs`.
    fn add(self, rhs: O) -> Result<Self::Output, Error>;

    /// Construct an array subtraction operation with the given `rhs`.
    fn sub(self, rhs: O) -> Result<Self::Output, Error>;
}

impl<T, L, R, P> NDArrayMath<Array<T, R, P>> for Array<T, L, P>
where
    T: CType,
    L: Access<T>,
    R: Access<T>,
    P: ElementwiseDual<L, R, T>,
{
    type Output = Array<T, AccessOp<P::Op, P>, P>;

    fn add(self, rhs: Array<T, R, P>) -> Result<Self::Output, Error> {
        same_shape("add", self.shape(), rhs.shape())?;

        Ok(Array {
            shape: self.shape,
            access: self.platform.add(self.access, rhs.access)?,
            platform: self.platform,
            dtype: PhantomData,
        })
    }

    fn sub(self, rhs: Array<T, R, P>) -> Result<Self::Output, Error> {
        same_shape("subtract", self.shape(), rhs.shape())?;

        Ok(Array {
            shape: self.shape,
            access: self.platform.sub(self.access, rhs.access)?,
            platform: self.platform,
            dtype: PhantomData,
        })
    }
}

/// Boolean array reduce operations
pub trait NDArrayReduceBoolean: NDArrayRead {
    /// Return `true` if this array contains only non-zero elements.
    fn all(self) -> Result<bool, Error>;

    /// Return `true` if this array contains any non-zero elements.
    fn any(self) -> Result<bool, Error>;
}

impl<T, A, P> NDArrayReduceBoolean for Array<T, A, P>
where
    T: CType,
    A: Access<T>,
    P: Reduce<A, T>,
{
    fn all(self) -> Result<bool, Error> {
        self.platform.all(self.access)
    }

    fn any(self) -> Result<bool, Error> {
        self.platform.any(self.access)
    }
}

/// Array reduce operations
pub trait NDArrayReduceAll: NDArrayRead {
    /// Return the sum of all elements in this array.
    fn sum_all(self) -> Result<Self::DType, Error>;
}

impl<'a, T, A, P> NDArrayReduceAll for Array<T, A, P>
where
    T: CType,
    A: Access<T>,
    P: Reduce<A, T>,
{
    fn sum_all(self) -> Result<T, Error> {
        self.platform.sum(self.access)
    }
}

impl<T, A, P> fmt::Debug for Array<T, A, P> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "a {} array of shape {:?}",
            std::any::type_name::<T>(),
            self.shape
        )
    }
}

#[inline]
fn can_broadcast(left: &[usize], right: &[usize]) -> bool {
    if left.len() < right.len() {
        return can_broadcast(right, left);
    }

    for (l, r) in left.iter().copied().rev().zip(right.iter().copied().rev()) {
        if l == r || l == 1 || r == 1 {
            // pass
        } else {
            return false;
        }
    }

    true
}

#[inline]
fn same_shape(op_name: &'static str, left: &[usize], right: &[usize]) -> Result<(), Error> {
    if left == right {
        Ok(())
    } else if can_broadcast(left, right) {
        Err(Error::Bounds(format!(
            "cannot {op_name} arrays with shapes {left:?} and {right:?} (consider broadcasting)"
        )))
    } else {
        Err(Error::Bounds(format!(
            "cannot {op_name} arrays with shapes {left:?} and {right:?}"
        )))
    }
}

#[inline]
fn valid_coord(coord: &[usize], shape: &[usize]) -> Result<(), Error> {
    if coord.len() == shape.len() {
        if coord.iter().zip(shape).all(|(i, dim)| i < dim) {
            return Ok(());
        }
    }

    Err(Error::Bounds(format!(
        "invalid coordinate {coord:?} for shape {shape:?}"
    )))
}
