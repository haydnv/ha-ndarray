//! Array operations

use crate::access::*;
use crate::buffer::Buffer;
#[cfg(feature = "opencl")]
use crate::opencl;
use crate::platform::{Platform, PlatformInstance};
use crate::{
    host, range_shape, strides_for, Axes, AxisRange, BufferConverter, CType, Error, Range, Shape,
    Strides,
};

macro_rules! op_dispatch {
    ($this:expr, $op:ident, $call:expr) => {
        match $this {
            #[cfg(feature = "opencl")]
            Self::CL($op) => $call,
            Self::Host($op) => $call,
        }
    };
}

macro_rules! op_enqueue {
    ($this:expr, $t:ty) => {
        match $this {
            #[cfg(feature = "opencl")]
            Self::CL(op) => Enqueue::<opencl::OpenCL, $t>::enqueue(op).map(Buffer::CL),
            Self::Host(op) => Enqueue::<host::Host, $t>::enqueue(op).map(Buffer::Host),
        }
    };
}

pub trait Op: Send + Sync {
    fn size(&self) -> usize;
}

pub trait Enqueue<P: PlatformInstance, T: CType>: Op {
    type Buffer: Into<BufferConverter<'static, T>>;

    fn enqueue(&self) -> Result<Self::Buffer, Error>;
}

pub trait ReadValue<P: PlatformInstance, T: CType>: Op {
    fn read_value(&self, offset: usize) -> Result<T, Error>;
}

pub trait ReadOp<P, T>: Enqueue<P, T> + ReadValue<P, T>
where
    P: PlatformInstance,
    T: CType,
{
}

impl<O, P, T> ReadOp<P, T> for O
where
    O: Enqueue<P, T> + ReadValue<P, T>,
    P: PlatformInstance,
    T: CType,
{
}

pub trait Write<P: PlatformInstance, T: CType>: Enqueue<P, T> {
    fn write<'a>(&mut self, data: BufferConverter<'a, T>) -> Result<(), Error>;

    fn write_value(&mut self, value: T) -> Result<(), Error>;

    fn write_value_at(&mut self, offset: usize, value: T) -> Result<(), Error>;
}

pub trait Construct<T: CType>: PlatformInstance {
    type Range: Enqueue<Self, T>;

    fn range(self, start: T, stop: T, size: usize) -> Result<AccessOp<Self::Range, Self>, Error>;
}

pub trait ElementwiseBoolean<L, R, T>: PlatformInstance {
    type Op: ReadOp<Self, u8>;

    fn and(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error>;

    fn or(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error>;

    fn xor(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error>;
}

pub trait ElementwiseBooleanScalar<A, T>: PlatformInstance {
    type Op: ReadOp<Self, u8>;

    fn and_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error>;

    fn or_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error>;

    fn xor_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error>;
}

pub trait ElementwiseCast<A, IT, OT>: PlatformInstance
where
    A: Access<IT>,
    IT: CType,
    OT: CType,
{
    type Op: ReadOp<Self, OT>;

    fn cast(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error>;
}

pub trait ElementwiseCompare<L, R, T>: PlatformInstance {
    type Op: ReadOp<Self, u8>;

    fn eq(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error>;

    fn ge(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error>;

    fn gt(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error>;

    fn le(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error>;

    fn lt(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error>;

    fn ne(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error>;
}

pub trait ElementwiseScalarCompare<A, T>: PlatformInstance {
    type Op: ReadOp<Self, u8>;

    fn eq_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error>;

    fn ge_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error>;

    fn gt_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error>;

    fn le_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error>;

    fn lt_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error>;

    fn ne_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error>;
}

pub trait ElementwiseDual<L, R, T>: PlatformInstance
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Op: ReadOp<Self, T>;

    fn add(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error>;

    fn div(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error>;

    fn log(self, arg: L, base: R) -> Result<AccessOp<Self::Op, Self>, Error>;

    fn mul(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error>;

    fn pow(self, arg: L, exp: R) -> Result<AccessOp<Self::Op, Self>, Error>;

    fn rem(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error>;

    fn sub(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error>;
}

pub trait ElementwiseScalar<A, T>: PlatformInstance
where
    A: Access<T>,
    T: CType,
{
    type Op: ReadOp<Self, T>;

    fn add_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error>;

    fn div_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error>;

    fn log_scalar(self, arg: A, base: T) -> Result<AccessOp<Self::Op, Self>, Error>;

    fn mul_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error>;

    fn pow_scalar(self, arg: A, exp: T) -> Result<AccessOp<Self::Op, Self>, Error>;

    fn rem_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error>;

    fn sub_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error>;
}

pub trait ElementwiseNumeric<A, T>: PlatformInstance
where
    A: Access<T>,
    T: CType,
{
    type Op: ReadOp<Self, u8>;

    fn is_inf(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error>;

    fn is_nan(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error>;
}

pub trait ElementwiseTrig<A, T>: PlatformInstance
where
    A: Access<T>,
    T: CType,
{
    type Op: ReadOp<Self, T::Float>;

    fn sin(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error>;

    fn asin(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error>;

    fn sinh(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error>;

    fn cos(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error>;

    fn acos(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error>;

    fn cosh(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error>;

    fn tan(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error>;

    fn atan(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error>;

    fn tanh(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error>;
}

pub trait ElementwiseUnary<A, T>: PlatformInstance
where
    A: Access<T>,
    T: CType,
{
    type Op: ReadOp<Self, T>;

    fn abs(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error>;

    fn exp(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error>;

    fn ln(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error>;

    fn round(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error>;
}

pub trait ElementwiseUnaryBoolean<A, T>: PlatformInstance
where
    A: Access<T>,
    T: CType,
{
    type Op: ReadOp<Self, u8>;

    fn not(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error>;
}

pub trait GatherCond<A, L, R, T>: PlatformInstance
where
    A: Access<u8>,
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Op: ReadOp<Self, T>;

    fn cond(self, cond: A, then: L, or_else: R) -> Result<AccessOp<Self::Op, Self>, Error>;
}

pub trait LinAlgDual<L, R, T>: PlatformInstance
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Op: ReadOp<Self, T>;

    fn matmul(self, left: L, right: R, dims: [usize; 4])
        -> Result<AccessOp<Self::Op, Self>, Error>;
}

pub trait LinAlgUnary<A, T>: PlatformInstance
where
    A: Access<T>,
    T: CType,
{
    type Op: ReadOp<Self, T>;

    fn diag(
        self,
        access: A,
        batch_size: usize,
        dim: usize,
    ) -> Result<AccessOp<Self::Op, Self>, Error>;
}

pub trait Random: PlatformInstance {
    type Normal: Enqueue<Self, f32>;
    type Uniform: Enqueue<Self, f32>;

    fn random_normal(self, size: usize) -> Result<AccessOp<Self::Normal, Self>, Error>;

    fn random_uniform(self, size: usize) -> Result<AccessOp<Self::Uniform, Self>, Error>;
}

pub trait ReduceAll<A, T>: PlatformInstance {
    fn all(self, access: A) -> Result<bool, Error>;

    fn any(self, access: A) -> Result<bool, Error>;

    fn max(self, access: A) -> Result<T, Error>;

    fn min(self, access: A) -> Result<T, Error>;

    fn product(self, access: A) -> Result<T, Error>;

    fn sum(self, access: A) -> Result<T, Error>;
}

pub trait ReduceAxes<A: Access<T>, T: CType>: PlatformInstance {
    type Op: ReadOp<Self, T>;

    fn max(self, access: A, stride: usize) -> Result<AccessOp<Self::Op, Self>, Error>;

    fn min(self, access: A, stride: usize) -> Result<AccessOp<Self::Op, Self>, Error>;

    fn product(self, access: A, stride: usize) -> Result<AccessOp<Self::Op, Self>, Error>;

    fn sum(self, access: A, stride: usize) -> Result<AccessOp<Self::Op, Self>, Error>;
}

pub trait Transform<A: Access<T>, T: CType>: PlatformInstance {
    type Broadcast: ReadOp<Self, T>;
    type Slice: ReadOp<Self, T>;
    type Transpose: ReadOp<Self, T>;

    fn broadcast(
        self,
        access: A,
        shape: Shape,
        broadcast: Shape,
    ) -> Result<AccessOp<Self::Broadcast, Self>, Error>;

    fn slice(
        self,
        access: A,
        shape: &[usize],
        range: Range,
    ) -> Result<AccessOp<Self::Slice, Self>, Error>;

    fn transpose(
        self,
        access: A,
        shape: Shape,
        permutation: Axes,
    ) -> Result<AccessOp<Self::Transpose, Self>, Error>;
}

pub enum Cast<A, IT, OT> {
    #[cfg(feature = "opencl")]
    CL(opencl::ops::Cast<A, IT, OT>),
    Host(host::ops::Cast<A, IT, OT>),
}

impl<A: Access<IT>, IT: CType, OT: CType> Op for Cast<A, IT, OT> {
    fn size(&self) -> usize {
        op_dispatch!(self, op, op.size())
    }
}

impl<A: Access<IT>, IT: CType, OT: CType> Enqueue<Platform, OT> for Cast<A, IT, OT> {
    type Buffer = Buffer<OT>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        op_enqueue!(self, OT)
    }
}

impl<A: Access<IT>, IT: CType, OT: CType> ReadValue<Platform, OT> for Cast<A, IT, OT> {
    fn read_value(&self, offset: usize) -> Result<OT, Error> {
        op_dispatch!(self, op, op.read_value(offset))
    }
}

impl<A, IT, OT> From<host::ops::Cast<A, IT, OT>> for Cast<A, IT, OT> {
    fn from(op: host::ops::Cast<A, IT, OT>) -> Cast<A, IT, OT> {
        Self::Host(op)
    }
}

#[cfg(feature = "opencl")]
impl<A, IT, OT> From<opencl::ops::Cast<A, IT, OT>> for Cast<A, IT, OT> {
    fn from(op: opencl::ops::Cast<A, IT, OT>) -> Cast<A, IT, OT> {
        Self::CL(op)
    }
}

pub enum Cond<A, L, R, T> {
    #[cfg(feature = "opencl")]
    CL(opencl::ops::Cond<A, L, R, T>),
    Host(host::ops::Cond<A, L, R, T>),
}

impl<A, L, R, T> Op for Cond<A, L, R, T>
where
    A: Access<u8>,
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    fn size(&self) -> usize {
        op_dispatch!(self, op, op.size())
    }
}

impl<A, L, R, T> Enqueue<Platform, T> for Cond<A, L, R, T>
where
    A: Access<u8>,
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Buffer = Buffer<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        op_enqueue!(self, T)
    }
}

impl<A, L, R, T> ReadValue<Platform, T> for Cond<A, L, R, T>
where
    A: Access<u8>,
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    fn read_value(&self, offset: usize) -> Result<T, Error> {
        op_dispatch!(self, op, op.read_value(offset))
    }
}

impl<A, L, R, T> From<host::ops::Cond<A, L, R, T>> for Cond<A, L, R, T> {
    fn from(op: host::ops::Cond<A, L, R, T>) -> Self {
        Self::Host(op)
    }
}

#[cfg(feature = "opencl")]
impl<A, L, R, T> From<opencl::ops::Cond<A, L, R, T>> for Cond<A, L, R, T> {
    fn from(op: opencl::ops::Cond<A, L, R, T>) -> Self {
        Self::CL(op)
    }
}

pub enum Dual<L, R, IT, OT> {
    #[cfg(feature = "opencl")]
    CL(opencl::ops::Dual<L, R, IT, OT>),
    Host(host::ops::Dual<L, R, IT, OT>),
}

impl<L, R, IT, OT> Op for Dual<L, R, IT, OT>
where
    L: Access<IT>,
    R: Access<IT>,
    IT: CType,
    OT: CType,
{
    fn size(&self) -> usize {
        op_dispatch!(self, op, op.size())
    }
}

impl<L, R, IT, OT> Enqueue<Platform, OT> for Dual<L, R, IT, OT>
where
    L: Access<IT>,
    R: Access<IT>,
    IT: CType,
    OT: CType,
{
    type Buffer = Buffer<OT>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        op_enqueue!(self, OT)
    }
}

impl<L, R, IT, OT> ReadValue<Platform, OT> for Dual<L, R, IT, OT>
where
    L: Access<IT>,
    R: Access<IT>,
    IT: CType,
    OT: CType,
{
    fn read_value(&self, offset: usize) -> Result<OT, Error> {
        op_dispatch!(self, op, op.read_value(offset))
    }
}

#[cfg(feature = "opencl")]
impl<L, R, IT, OT> From<opencl::ops::Dual<L, R, IT, OT>> for Dual<L, R, IT, OT> {
    fn from(op: opencl::ops::Dual<L, R, IT, OT>) -> Self {
        Self::CL(op)
    }
}

impl<L, R, IT, OT> From<host::ops::Dual<L, R, IT, OT>> for Dual<L, R, IT, OT> {
    fn from(op: host::ops::Dual<L, R, IT, OT>) -> Self {
        Self::Host(op)
    }
}

pub enum Linear<T> {
    #[cfg(feature = "opencl")]
    CL(opencl::ops::Linear<T>),
    Host(host::ops::Linear<T>),
}

#[cfg(feature = "opencl")]
impl<T> From<opencl::ops::Linear<T>> for Linear<T> {
    fn from(op: opencl::ops::Linear<T>) -> Self {
        Self::CL(op)
    }
}

impl<T> From<host::ops::Linear<T>> for Linear<T> {
    fn from(op: host::ops::Linear<T>) -> Self {
        Self::Host(op)
    }
}

impl<T: Send + Sync> Op for Linear<T> {
    fn size(&self) -> usize {
        op_dispatch!(self, op, op.size())
    }
}

impl<T: CType> Enqueue<Platform, T> for Linear<T> {
    type Buffer = Buffer<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        op_enqueue!(self, T)
    }
}

impl<T: CType> ReadValue<Platform, T> for Linear<T> {
    fn read_value(&self, offset: usize) -> Result<T, Error> {
        op_dispatch!(self, op, op.read_value(offset))
    }
}

pub enum MatDiag<A, T> {
    #[cfg(feature = "opencl")]
    CL(opencl::ops::MatDiag<A, T>),
    Host(host::ops::MatDiag<A, T>),
}

impl<A: Access<T>, T: CType> Op for MatDiag<A, T> {
    fn size(&self) -> usize {
        op_dispatch!(self, op, op.size())
    }
}

impl<A: Access<T>, T: CType> Enqueue<Platform, T> for MatDiag<A, T> {
    type Buffer = Buffer<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        op_enqueue!(self, T)
    }
}

impl<A: Access<T>, T: CType> ReadValue<Platform, T> for MatDiag<A, T> {
    fn read_value(&self, offset: usize) -> Result<T, Error> {
        op_dispatch!(self, op, op.read_value(offset))
    }
}

impl<A, T> From<host::ops::MatDiag<A, T>> for MatDiag<A, T> {
    fn from(op: host::ops::MatDiag<A, T>) -> Self {
        Self::Host(op)
    }
}

#[cfg(feature = "opencl")]
impl<A, T> From<opencl::ops::MatDiag<A, T>> for MatDiag<A, T> {
    fn from(op: opencl::ops::MatDiag<A, T>) -> Self {
        Self::CL(op)
    }
}

pub enum MatMul<L, R, T> {
    #[cfg(feature = "opencl")]
    CL(opencl::ops::MatMul<L, R, T>),
    Host(host::ops::MatMul<L, R, T>),
}

impl<L, R, T> Op for MatMul<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    fn size(&self) -> usize {
        op_dispatch!(self, op, op.size())
    }
}

impl<L, R, T> Enqueue<Platform, T> for MatMul<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Buffer = Buffer<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        op_enqueue!(self, T)
    }
}

impl<L, R, T> ReadValue<Platform, T> for MatMul<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    fn read_value(&self, offset: usize) -> Result<T, Error> {
        op_dispatch!(self, op, op.read_value(offset))
    }
}

#[cfg(feature = "opencl")]
impl<L, R, T> From<opencl::ops::MatMul<L, R, T>> for MatMul<L, R, T> {
    fn from(op: opencl::ops::MatMul<L, R, T>) -> Self {
        Self::CL(op)
    }
}

impl<L, R, T> From<host::ops::MatMul<L, R, T>> for MatMul<L, R, T> {
    fn from(op: host::ops::MatMul<L, R, T>) -> Self {
        Self::Host(op)
    }
}

pub enum RandomNormal {
    #[cfg(feature = "opencl")]
    CL(opencl::ops::RandomNormal),
    Host(host::ops::RandomNormal),
}

#[cfg(feature = "opencl")]
impl From<opencl::ops::RandomNormal> for RandomNormal {
    fn from(op: opencl::ops::RandomNormal) -> Self {
        Self::CL(op)
    }
}

impl From<host::ops::RandomNormal> for RandomNormal {
    fn from(op: host::ops::RandomNormal) -> Self {
        Self::Host(op)
    }
}

pub enum RandomUniform {
    #[cfg(feature = "opencl")]
    CL(opencl::ops::RandomUniform),
    Host(host::ops::RandomUniform),
}

#[cfg(feature = "opencl")]
impl From<opencl::ops::RandomUniform> for RandomUniform {
    fn from(op: opencl::ops::RandomUniform) -> Self {
        Self::CL(op)
    }
}

impl From<host::ops::RandomUniform> for RandomUniform {
    fn from(op: host::ops::RandomUniform) -> Self {
        Self::Host(op)
    }
}

macro_rules! impl_random {
    ($t:ty) => {
        impl Op for $t {
            fn size(&self) -> usize {
                op_dispatch!(self, op, op.size())
            }
        }

        impl Enqueue<Platform, f32> for $t {
            type Buffer = Buffer<f32>;

            fn enqueue(&self) -> Result<Self::Buffer, Error> {
                op_enqueue!(self, f32)
            }
        }

        impl ReadValue<Platform, f32> for $t {
            fn read_value(&self, offset: usize) -> Result<f32, Error> {
                op_dispatch!(self, op, op.read_value(offset))
            }
        }
    };
}

impl_random!(RandomNormal);
impl_random!(RandomUniform);

macro_rules! impl_unary {
    ($op:ty, $t:ty) => {
        impl<A: Access<T>, T: CType> Op for $op {
            fn size(&self) -> usize {
                op_dispatch!(self, op, op.size())
            }
        }

        impl<A: Access<T>, T: CType> Enqueue<Platform, $t> for $op {
            type Buffer = Buffer<$t>;

            fn enqueue(&self) -> Result<Self::Buffer, Error> {
                op_enqueue!(self, $t)
            }
        }

        impl<A: Access<T>, T: CType> ReadValue<Platform, $t> for $op {
            fn read_value(&self, offset: usize) -> Result<$t, Error> {
                op_dispatch!(self, op, op.read_value(offset))
            }
        }
    };
}

pub enum Reduce<A, T: CType> {
    #[cfg(feature = "opencl")]
    CL(opencl::ops::Reduce<A, T>),
    Host(host::ops::Reduce<A, T>),
}

impl_unary!(Reduce<A, T>, T);

impl<A, T: CType> From<host::ops::Reduce<A, T>> for Reduce<A, T> {
    fn from(op: host::ops::Reduce<A, T>) -> Self {
        Self::Host(op)
    }
}

#[cfg(feature = "opencl")]
impl<A, T: CType> From<opencl::ops::Reduce<A, T>> for Reduce<A, T> {
    fn from(op: opencl::ops::Reduce<A, T>) -> Self {
        Self::CL(op)
    }
}

#[derive(Clone, Eq, PartialEq, Hash)]
pub struct SliceSpec {
    pub range: Range,
    pub shape: Shape,
    pub strides: Strides,
    pub source_strides: Strides,
}

impl SliceSpec {
    pub fn new(source_shape: &[usize], range: Range) -> Self {
        debug_assert!(range.len() <= source_shape.len());

        let shape = range_shape(source_shape, &range);
        let strides = strides_for(&shape, shape.len()).collect();
        let source_strides = strides_for(source_shape, source_shape.len()).collect();

        Self {
            range,
            shape,
            strides,
            source_strides,
        }
    }

    pub fn source_offset(&self, offset: usize) -> usize {
        debug_assert!(!self.shape.is_empty());
        debug_assert_eq!(self.shape.len(), self.strides.len());

        let mut coord = self
            .strides
            .iter()
            .copied()
            .zip(&self.shape)
            .map(|(stride, dim)| {
                if stride == 0 {
                    0
                } else {
                    (offset / stride) % dim
                }
            });

        let mut offset = 0;
        for (stride, bound) in self.source_strides.iter().zip(self.range.iter()) {
            let i = match bound {
                AxisRange::At(i) => *i,
                AxisRange::In(start, stop, step) => {
                    let i = start + (coord.next().expect("i") * step);
                    debug_assert!(i < *stop);
                    i
                }
                AxisRange::Of(indices) => indices[coord.next().expect("i")],
            };

            offset += i * stride;
        }

        offset
    }

    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }
}

pub enum Scalar<A, IT, OT> {
    #[cfg(feature = "opencl")]
    CL(opencl::ops::Scalar<A, IT, OT>),
    Host(host::ops::Scalar<A, IT, OT>),
}

impl<A, IT, OT> Op for Scalar<A, IT, OT>
where
    A: Access<IT>,
    IT: CType,
    OT: CType,
{
    fn size(&self) -> usize {
        op_dispatch!(self, op, op.size())
    }
}

impl<A, IT, OT> Enqueue<Platform, OT> for Scalar<A, IT, OT>
where
    A: Access<IT>,
    IT: CType,
    OT: CType,
{
    type Buffer = Buffer<OT>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        op_enqueue!(self, OT)
    }
}

impl<A, IT, OT> ReadValue<Platform, OT> for Scalar<A, IT, OT>
where
    A: Access<IT>,
    IT: CType,
    OT: CType,
{
    fn read_value(&self, offset: usize) -> Result<OT, Error> {
        op_dispatch!(self, op, op.read_value(offset))
    }
}

#[cfg(feature = "opencl")]
impl<A, IT, OT> From<opencl::ops::Scalar<A, IT, OT>> for Scalar<A, IT, OT> {
    fn from(op: opencl::ops::Scalar<A, IT, OT>) -> Self {
        Self::CL(op)
    }
}

impl<A, IT, OT> From<host::ops::Scalar<A, IT, OT>> for Scalar<A, IT, OT> {
    fn from(op: host::ops::Scalar<A, IT, OT>) -> Self {
        Self::Host(op)
    }
}

pub enum Slice<A, T> {
    #[cfg(feature = "opencl")]
    CL(opencl::ops::Slice<A, T>),
    Host(host::ops::Slice<A, T>),
}

impl_unary!(Slice<A, T>, T);

#[cfg(feature = "opencl")]
impl<A, T> Write<Platform, T> for Slice<A, T>
where
    A: AccessMut<T> + std::fmt::Debug,
    T: CType,
{
    fn write<'a>(&mut self, data: BufferConverter<'a, T>) -> Result<(), Error> {
        match self {
            Self::CL(op) => Write::<opencl::OpenCL, T>::write(op, data),
            Self::Host(op) => Write::<host::Host, T>::write(op, data),
        }
    }

    fn write_value(&mut self, value: T) -> Result<(), Error> {
        match self {
            Self::CL(op) => Write::<opencl::OpenCL, T>::write_value(op, value),
            Self::Host(op) => Write::<host::Host, T>::write_value(op, value),
        }
    }

    fn write_value_at(&mut self, offset: usize, value: T) -> Result<(), Error> {
        match self {
            Self::CL(op) => Write::<opencl::OpenCL, T>::write_value_at(op, offset, value),
            Self::Host(op) => Write::<host::Host, T>::write_value_at(op, offset, value),
        }
    }
}

#[cfg(not(feature = "opencl"))]
impl<A, T> Write<Platform, T> for Slice<A, T>
where
    T: CType,
    A: AccessMut<T>,
{
    fn write<'a>(&mut self, data: BufferConverter<'a, T>) -> Result<(), Error> {
        match self {
            Self::Host(op) => Write::<host::Host, T>::write(op, data),
        }
    }

    fn write_value(&mut self, value: T) -> Result<(), Error> {
        match self {
            Self::Host(op) => Write::<host::Host, T>::write_value(op, value),
        }
    }

    fn write_value_at(&mut self, offset: usize, value: T) -> Result<(), Error> {
        match self {
            Self::Host(op) => Write::<host::Host, T>::write_value_at(op, offset, value),
        }
    }
}

#[cfg(feature = "opencl")]
impl<A, T> From<opencl::ops::Slice<A, T>> for Slice<A, T> {
    fn from(op: opencl::ops::Slice<A, T>) -> Self {
        Self::CL(op)
    }
}

impl<A, T> From<host::ops::Slice<A, T>> for Slice<A, T> {
    fn from(op: host::ops::Slice<A, T>) -> Self {
        Self::Host(op)
    }
}

pub enum Unary<A, IT, OT> {
    #[cfg(feature = "opencl")]
    CL(opencl::ops::Unary<A, IT, OT>),
    Host(host::ops::Unary<A, IT, OT>),
}

impl<A, IT, OT> Op for Unary<A, IT, OT>
where
    A: Access<IT>,
    IT: CType,
    OT: CType,
{
    fn size(&self) -> usize {
        op_dispatch!(self, op, op.size())
    }
}

impl<A, IT, OT> Enqueue<Platform, OT> for Unary<A, IT, OT>
where
    A: Access<IT>,
    IT: CType,
    OT: CType,
{
    type Buffer = Buffer<OT>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        op_enqueue!(self, OT)
    }
}

impl<A, IT, OT> ReadValue<Platform, OT> for Unary<A, IT, OT>
where
    A: Access<IT>,
    IT: CType,
    OT: CType,
{
    fn read_value(&self, offset: usize) -> Result<OT, Error> {
        op_dispatch!(self, op, op.read_value(offset))
    }
}

impl<A, IT, OT> From<host::ops::Unary<A, IT, OT>> for Unary<A, IT, OT> {
    fn from(op: host::ops::Unary<A, IT, OT>) -> Self {
        Self::Host(op)
    }
}

#[cfg(feature = "opencl")]
impl<A, IT, OT> From<opencl::ops::Unary<A, IT, OT>> for Unary<A, IT, OT> {
    fn from(op: opencl::ops::Unary<A, IT, OT>) -> Self {
        Self::CL(op)
    }
}

#[derive(Clone, Eq, PartialEq, Hash)]
pub struct ViewSpec {
    pub shape: Shape,
    pub strides: Strides,
    pub source_strides: Strides,
}

impl ViewSpec {
    pub fn new(shape: Shape, source_strides: Strides) -> Self {
        let strides = strides_for(&shape, shape.len()).collect();

        Self {
            shape,
            strides,
            source_strides,
        }
    }

    pub fn source_offset(&self, offset: usize) -> usize {
        debug_assert!(offset < self.size());

        let source_offset = self
            .strides
            .iter()
            .copied()
            .zip(self.shape.iter().copied())
            .rev()
            .take(self.source_strides.len())
            .map(|(stride, dim)| {
                if stride == 0 {
                    0
                } else {
                    (offset / stride) % dim
                }
            }) // coord
            .zip(self.source_strides.iter().rev().copied())
            .map(|(i, source_stride)| i * source_stride)
            .sum::<usize>();

        source_offset
    }

    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }
}

pub enum View<A, T> {
    #[cfg(feature = "opencl")]
    CL(opencl::ops::View<A, T>),
    Host(host::ops::View<A, T>),
}

impl_unary!(View<A, T>, T);

#[cfg(feature = "opencl")]
impl<A, T> From<opencl::ops::View<A, T>> for View<A, T> {
    fn from(op: opencl::ops::View<A, T>) -> Self {
        Self::CL(op)
    }
}

impl<A, T> From<host::ops::View<A, T>> for View<A, T> {
    fn from(op: host::ops::View<A, T>) -> Self {
        Self::Host(op)
    }
}
