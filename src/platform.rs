use crate::access::{Access, AccessOp};
use crate::buffer::{BufferConverter, BufferInstance};
#[cfg(feature = "opencl")]
use crate::opencl;
use crate::ops::*;
use crate::{host, CType, Error, Host};

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
    fn from(host: Host) -> Self {
        Self::Host(host)
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
}
