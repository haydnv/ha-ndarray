use crate::access::{Access, AccessOp};
use crate::buffer::{Buffer, BufferConverter, BufferInstance};
#[cfg(feature = "opencl")]
use crate::opencl;
use crate::ops::*;
use crate::{host, CType, Error, Shape};

pub trait PlatformInstance: PartialEq + Eq + Clone + Copy + Send + Sync {
    fn select(size_hint: usize) -> Self;
}

pub trait Convert<T: CType>: PlatformInstance {
    type Buffer: BufferInstance<T>;

    fn convert<'a>(&self, buffer: BufferConverter<'a, T>) -> Result<Self::Buffer, Error>;
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

impl<T: CType> Convert<T> for Platform {
    type Buffer = Buffer<T>;

    fn convert<'a>(&self, buffer: BufferConverter<'a, T>) -> Result<Self::Buffer, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(cl) => cl.convert(buffer).map(Buffer::CL),
            Self::Host(host) => host.convert(buffer).map(Buffer::Host),
        }
    }
}

#[cfg(not(feature = "opencl"))]
impl<'a, L, R, T> ElementwiseCompare<L, R, T> for Platform
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Output = Compare<L, R, T>;

    fn eq(self, left: L, right: R) -> Result<AccessOp<Self::Output, Self>, Error> {
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
    type Output = Compare<L, R, T>;

    fn eq(self, left: L, right: R) -> Result<AccessOp<Self::Output, Self>, Error> {
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
    type Output = Dual<L, R, T>;

    fn add(self, left: L, right: R) -> Result<AccessOp<Self::Output, Self>, Error> {
        match self {
            Self::Host(host) => host.add(left, right).map(AccessOp::wrap),
        }
    }

    fn sub(self, left: L, right: R) -> Result<AccessOp<Self::Output, Self>, Error> {
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
    type Output = Dual<L, R, T>;

    fn add(self, left: L, right: R) -> Result<AccessOp<Self::Output, Self>, Error> {
        match self {
            Self::CL(cl) => cl.add(left, right).map(AccessOp::wrap),
            Self::Host(host) => host.add(left, right).map(AccessOp::wrap),
        }
    }

    fn sub(self, left: L, right: R) -> Result<AccessOp<Self::Output, Self>, Error> {
        match self {
            Self::CL(cl) => cl.sub(left, right).map(AccessOp::wrap),
            Self::Host(host) => host.sub(left, right).map(AccessOp::wrap),
        }
    }
}

#[cfg(not(feature = "opencl"))]
impl<'a, A, T> Reduce<A, T> for Platform
where
    A: Access<T>,
    T: CType,
{
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
}
