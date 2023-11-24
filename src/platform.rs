use crate::access::{Access, AccessOp};
use crate::buffer::BufferConverter;
#[cfg(feature = "opencl")]
use crate::opencl;
use crate::ops::*;
use crate::{host, Axes, CType, Error, Range, Shape};

pub trait PlatformInstance: PartialEq + Eq + Clone + Copy + Send + Sync {
    fn select(size_hint: usize) -> Self;
}

pub trait Convert<'a, T: CType>: PlatformInstance {
    type Buffer;

    fn convert(&self, buffer: BufferConverter<'a, T>) -> Result<Self::Buffer, Error>;
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum Platform {
    #[cfg(feature = "opencl")]
    CL(opencl::OpenCL),
    Host(host::Host),
}

#[cfg(feature = "opencl")]
impl PlatformInstance for Platform {
    fn select(size_hint: usize) -> Self {
        if size_hint < opencl::GPU_MIN_SIZE {
            Self::Host(host::Host::select(size_hint))
        } else {
            Self::CL(opencl::OpenCL)
        }
    }
}

#[cfg(not(feature = "opencl"))]
impl PlatformInstance for Platform {
    fn select(size_hint: usize) -> Self {
        Self::Host(host::Host::select(size_hint))
    }
}

#[cfg(feature = "opencl")]
impl From<opencl::OpenCL> for Platform {
    fn from(opencl: opencl::OpenCL) -> Self {
        Self::CL(opencl)
    }
}

impl From<host::Host> for Platform {
    fn from(host: host::Host) -> Self {
        Self::Host(host)
    }
}

impl<'a, T: CType> Convert<'a, T> for Platform {
    type Buffer = BufferConverter<'a, T>;

    fn convert(&self, buffer: BufferConverter<'a, T>) -> Result<Self::Buffer, Error> {
        Ok(buffer)
    }
}

#[cfg(not(feature = "opencl"))]
impl<T: CType> Construct<T> for Platform {
    type Range = Linear<T>;

    fn range(self, start: T, stop: T, size: usize) -> Result<AccessOp<Self::Range, Self>, Error> {
        match self {
            Self::Host(host) => host.range(start, stop, size).map(AccessOp::wrap),
        }
    }
}

#[cfg(feature = "opencl")]
impl<T: CType> Construct<T> for Platform {
    type Range = Linear<T>;

    fn range(self, start: T, stop: T, size: usize) -> Result<AccessOp<Self::Range, Self>, Error> {
        match self {
            Self::CL(cl) => cl.range(start, stop, size).map(AccessOp::wrap),
            Self::Host(host) => host.range(start, stop, size).map(AccessOp::wrap),
        }
    }
}

#[cfg(not(feature = "opencl"))]
impl<L, R, T> ElementwiseCompare<L, R, T> for Platform
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Op = Compare<L, R, T>;

    fn eq(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            Self::Host(host) => host.eq(left, right).map(AccessOp::wrap),
        }
    }
}

#[cfg(feature = "opencl")]
impl<L, R, T> ElementwiseCompare<L, R, T> for Platform
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Op = Compare<L, R, T>;

    fn eq(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            Self::CL(cl) => cl.eq(left, right).map(AccessOp::wrap),
            Self::Host(host) => host.eq(left, right).map(AccessOp::wrap),
        }
    }
}

#[cfg(not(feature = "opencl"))]
impl<A: Access<T>, T: CType> ElementwiseScalarCompare<A, T> for Platform {
    type Op = CompareScalar<A, T>;

    fn eq_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            Self::Host(host) => host.eq_scalar(left, right).map(AccessOp::wrap),
        }
    }

    fn ge_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            Self::Host(host) => host.ge_scalar(left, right).map(AccessOp::wrap),
        }
    }

    fn gt_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            Self::Host(host) => host.gt_scalar(left, right).map(AccessOp::wrap),
        }
    }

    fn le_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            Self::Host(host) => host.le_scalar(left, right).map(AccessOp::wrap),
        }
    }

    fn lt_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            Self::Host(host) => host.lt_scalar(left, right).map(AccessOp::wrap),
        }
    }

    fn ne_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            Self::Host(host) => host.ne_scalar(left, right).map(AccessOp::wrap),
        }
    }
}

#[cfg(feature = "opencl")]
impl<A: Access<T>, T: CType> ElementwiseScalarCompare<A, T> for Platform {
    type Op = CompareScalar<A, T>;

    fn eq_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            Self::CL(cl) => cl.eq_scalar(left, right).map(AccessOp::wrap),
            Self::Host(host) => host.eq_scalar(left, right).map(AccessOp::wrap),
        }
    }

    fn ge_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            Self::CL(cl) => cl.ge_scalar(left, right).map(AccessOp::wrap),
            Self::Host(host) => host.ge_scalar(left, right).map(AccessOp::wrap),
        }
    }

    fn gt_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            Self::CL(cl) => cl.gt_scalar(left, right).map(AccessOp::wrap),
            Self::Host(host) => host.gt_scalar(left, right).map(AccessOp::wrap),
        }
    }

    fn le_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            Self::CL(cl) => cl.le_scalar(left, right).map(AccessOp::wrap),
            Self::Host(host) => host.le_scalar(left, right).map(AccessOp::wrap),
        }
    }

    fn lt_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            Self::CL(cl) => cl.lt_scalar(left, right).map(AccessOp::wrap),
            Self::Host(host) => host.lt_scalar(left, right).map(AccessOp::wrap),
        }
    }

    fn ne_scalar(self, left: A, right: T) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            Self::CL(cl) => cl.ne_scalar(left, right).map(AccessOp::wrap),
            Self::Host(host) => host.ne_scalar(left, right).map(AccessOp::wrap),
        }
    }
}

#[cfg(not(feature = "opencl"))]
impl<L, R, T> ElementwiseDual<L, R, T> for Platform
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Op = Dual<L, R, T>;

    fn add(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            Self::Host(host) => host.add(left, right).map(AccessOp::wrap),
        }
    }

    fn sub(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            Self::Host(host) => host.sub(left, right).map(AccessOp::wrap),
        }
    }
}

#[cfg(feature = "opencl")]
impl<L, R, T> ElementwiseDual<L, R, T> for Platform
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Op = Dual<L, R, T>;

    fn add(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            Self::CL(cl) => cl.add(left, right).map(AccessOp::wrap),
            Self::Host(host) => host.add(left, right).map(AccessOp::wrap),
        }
    }

    fn sub(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            Self::CL(cl) => cl.sub(left, right).map(AccessOp::wrap),
            Self::Host(host) => host.sub(left, right).map(AccessOp::wrap),
        }
    }
}

#[cfg(not(feature = "opencl"))]
impl<A: Access<T>, T: CType> ElementwiseUnary<A, T> for Platform {
    type Op = Unary<A, T, T>;

    fn ln(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            Self::Host(host) => host.ln(access).map(AccessOp::wrap),
        }
    }
}

#[cfg(feature = "opencl")]
impl<A: Access<T>, T: CType> ElementwiseUnary<A, T> for Platform {
    type Op = Unary<A, T, T>;

    fn ln(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            Self::CL(cl) => cl.ln(access).map(AccessOp::wrap),
            Self::Host(host) => host.ln(access).map(AccessOp::wrap),
        }
    }
}

#[cfg(not(feature = "opencl"))]
impl Random for Platform {
    type Normal = RandomNormal;
    type Uniform = RandomUniform;

    fn random_normal(self, size: usize) -> Result<AccessOp<Self::Normal, Self>, Error> {
        match self {
            Self::Host(host) => host.random_normal(size).map(AccessOp::wrap),
        }
    }

    fn random_uniform(self, size: usize) -> Result<AccessOp<Self::Uniform, Self>, Error> {
        match self {
            Self::Host(host) => host.random_uniform(size).map(AccessOp::wrap),
        }
    }
}

#[cfg(feature = "opencl")]
impl Random for Platform {
    type Normal = RandomNormal;
    type Uniform = RandomUniform;

    fn random_normal(self, size: usize) -> Result<AccessOp<Self::Normal, Self>, Error> {
        match self {
            Self::CL(cl) => cl.random_normal(size).map(AccessOp::wrap),
            Self::Host(host) => host.random_normal(size).map(AccessOp::wrap),
        }
    }

    fn random_uniform(self, size: usize) -> Result<AccessOp<Self::Uniform, Self>, Error> {
        match self {
            Self::CL(cl) => cl.random_uniform(size).map(AccessOp::wrap),
            Self::Host(host) => host.random_uniform(size).map(AccessOp::wrap),
        }
    }
}

#[cfg(not(feature = "opencl"))]
impl<A: Access<T>, T: CType> ReduceAll<A, T> for Platform {
    fn all(self, access: A) -> Result<bool, Error> {
        match self {
            Self::Host(host) => host.all(access),
        }
    }

    fn any(self, access: A) -> Result<bool, Error> {
        match self {
            Self::Host(host) => host.any(access),
        }
    }

    fn max(self, access: A) -> Result<T, Error> {
        match self {
            Self::Host(host) => ReduceAll::max(host, access),
        }
    }

    fn min(self, access: A) -> Result<T, Error> {
        match self {
            Self::Host(host) => ReduceAll::min(host, access),
        }
    }

    fn product(self, access: A) -> Result<T, Error> {
        match self {
            Self::Host(host) => ReduceAll::product(host, access),
        }
    }

    fn sum(self, access: A) -> Result<T, Error> {
        match self {
            Self::Host(host) => ReduceAll::sum(host, access),
        }
    }
}

#[cfg(feature = "opencl")]
impl<A, T> ReduceAll<A, T> for Platform
where
    A: Access<T>,
    T: CType,
{
    fn all(self, access: A) -> Result<bool, Error> {
        match self {
            Self::CL(cl) => cl.all(access),
            Self::Host(host) => host.all(access),
        }
    }

    fn any(self, access: A) -> Result<bool, Error> {
        match self {
            Self::CL(cl) => cl.all(access),
            Self::Host(host) => host.all(access),
        }
    }

    fn max(self, access: A) -> Result<T, Error> {
        match self {
            Self::CL(cl) => ReduceAll::max(cl, access),
            Self::Host(host) => ReduceAll::max(host, access),
        }
    }

    fn min(self, access: A) -> Result<T, Error> {
        match self {
            Self::CL(cl) => ReduceAll::min(cl, access),
            Self::Host(host) => ReduceAll::min(host, access),
        }
    }

    fn product(self, access: A) -> Result<T, Error> {
        match self {
            Self::CL(cl) => ReduceAll::product(cl, access),
            Self::Host(host) => ReduceAll::product(host, access),
        }
    }

    fn sum(self, access: A) -> Result<T, Error> {
        match self {
            Self::CL(cl) => ReduceAll::sum(cl, access),
            Self::Host(host) => ReduceAll::sum(host, access),
        }
    }
}

#[cfg(not(feature = "opencl"))]
impl<A: Access<T>, T: CType> ReduceAxis<A, T> for Platform {
    type Op = Reduce<A, T>;

    fn max(self, access: A, stride: usize) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            Self::Host(host) => ReduceAxis::max(host, access, stride).map(AccessOp::wrap),
        }
    }

    fn min(self, access: A, stride: usize) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            Self::Host(host) => ReduceAxis::min(host, access, stride).map(AccessOp::wrap),
        }
    }

    fn product(self, access: A, stride: usize) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            Self::Host(host) => ReduceAxis::product(host, access, stride).map(AccessOp::wrap),
        }
    }

    fn sum(self, access: A, stride: usize) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            Self::Host(host) => ReduceAxis::sum(host, access, stride).map(AccessOp::wrap),
        }
    }
}

#[cfg(feature = "opencl")]
impl<A: Access<T>, T: CType> ReduceAxis<A, T> for Platform {
    type Op = Reduce<A, T>;

    fn max(self, access: A, stride: usize) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            Self::CL(cl) => ReduceAxis::max(cl, access, stride).map(AccessOp::wrap),
            Self::Host(host) => ReduceAxis::max(host, access, stride).map(AccessOp::wrap),
        }
    }

    fn min(self, access: A, stride: usize) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            Self::CL(cl) => ReduceAxis::min(cl, access, stride).map(AccessOp::wrap),
            Self::Host(host) => ReduceAxis::min(host, access, stride).map(AccessOp::wrap),
        }
    }

    fn product(self, access: A, stride: usize) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            Self::CL(cl) => ReduceAxis::product(cl, access, stride).map(AccessOp::wrap),
            Self::Host(host) => ReduceAxis::product(host, access, stride).map(AccessOp::wrap),
        }
    }

    fn sum(self, access: A, stride: usize) -> Result<AccessOp<Self::Op, Self>, Error> {
        match self {
            Self::CL(cl) => ReduceAxis::sum(cl, access, stride).map(AccessOp::wrap),
            Self::Host(host) => ReduceAxis::sum(host, access, stride).map(AccessOp::wrap),
        }
    }
}

impl<A: Access<T>, T: CType> Transform<A, T> for Platform {
    type Broadcast = View<A, T>;
    type Slice = Slice<A, T>;
    type Transpose = View<A, T>;

    fn broadcast(
        self,
        access: A,
        shape: Shape,
        broadcast: Shape,
    ) -> Result<AccessOp<Self::Broadcast, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.broadcast(access, shape, broadcast).map(AccessOp::wrap),
            Self::Host(host) => host.broadcast(access, shape, broadcast).map(AccessOp::wrap),
        }
    }

    fn slice(
        self,
        access: A,
        shape: &[usize],
        range: Range,
    ) -> Result<AccessOp<Self::Slice, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.slice(access, shape, range).map(AccessOp::wrap),
            Self::Host(host) => host.slice(access, shape, range).map(AccessOp::wrap),
        }
    }

    fn transpose(
        self,
        access: A,
        shape: Shape,
        permutation: Axes,
    ) -> Result<AccessOp<Self::Transpose, Self>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.transpose(access, shape, permutation).map(AccessOp::wrap),
            Self::Host(host) => host
                .transpose(access, shape, permutation)
                .map(AccessOp::wrap),
        }
    }
}
