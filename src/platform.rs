use crate::access::{Access, AccessOp};
use crate::buffer::BufferConverter;
#[cfg(feature = "opencl")]
use crate::opencl;
use crate::ops::*;
use crate::{host, CType, Error, Range, Shape};

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
impl<A: Access<T>, T: CType> Reduce<A, T> for Platform {
    fn all(self, access: A) -> Result<bool, Error> {
        match Self::select(access.size()) {
            Self::Host(host) => host.all(access),
        }
    }

    fn any(self, access: A) -> Result<bool, Error> {
        match Self::select(access.size()) {
            Self::Host(host) => host.all(access),
        }
    }

    fn sum(self, access: A) -> Result<T, Error> {
        match Self::select(access.size()) {
            Self::Host(host) => host.sum(access),
        }
    }
}

#[cfg(feature = "opencl")]
impl<A, T> Reduce<A, T> for Platform
where
    A: Access<T>,
    T: CType,
{
    fn all(self, access: A) -> Result<bool, Error> {
        match Self::select(access.size()) {
            Self::CL(cl) => cl.all(access),
            Self::Host(host) => host.all(access),
        }
    }

    fn any(self, access: A) -> Result<bool, Error> {
        match Self::select(access.size()) {
            Self::CL(cl) => cl.all(access),
            Self::Host(host) => host.all(access),
        }
    }

    fn sum(self, access: A) -> Result<T, Error> {
        match Self::select(access.size()) {
            Self::CL(cl) => cl.sum(access),
            Self::Host(host) => host.sum(access),
        }
    }
}

impl<A, T> Transform<A, T> for Platform
where
    A: Access<T>,
    T: CType,
{
    type Broadcast = View<A, T>;
    type Slice = Slice<A, T>;

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
}
