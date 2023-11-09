use crate::buffer::{BufferConverter, BufferInstance};
#[cfg(feature = "opencl")]
use crate::opencl;
use crate::ops::*;
use crate::{host, CType, Error, ReadBuf};

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

#[cfg(not(feature = "opencl"))]
impl<'a, A, T> Reduce<A, T> for Platform
where
    A: ReadBuf<'a, T>,
    T: CType,
    host::Host: Reduce<A, T>,
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
impl<'a, A, T> Reduce<A, T> for Platform
where
    A: ReadBuf<'a, T>,
    T: CType,
    host::Host: Reduce<A, T>,
    opencl::OpenCL: Reduce<A, T>,
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
