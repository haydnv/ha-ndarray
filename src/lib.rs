use std::fmt;
use std::ops::{Add, Sub};

use smallvec::SmallVec;

use crate::access::AccessBuffer;
pub use buffer::{Buffer, BufferConverter, BufferInstance};
pub use host::{Host, StackVec};
use ops::*;

mod access;
mod array;
mod buffer;
mod host;
#[cfg(feature = "opencl")]
mod opencl;
mod ops;

#[cfg(feature = "opencl")]
pub trait CType:
    ocl::OclPrm
    + Add<Output = Self>
    + Sub<Output = Self>
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
    Host(Host),
}

#[cfg(feature = "opencl")]
impl PlatformInstance for Platform {
    fn select(size_hint: usize) -> Self {
        if size_hint < opencl::GPU_MIN_SIZE {
            Self::Host(Host::select(size_hint))
        } else {
            Self::CL(opencl::OpenCL)
        }
    }
}

#[cfg(not(feature = "opencl"))]
impl PlatformInstance for Platform {
    fn select(size_hint: usize) -> Self {
        Self::Host(Host::select(size_hint))
    }
}

#[cfg(not(feature = "opencl"))]
impl<'a, A, T> Reduce<A, T> for Platform
where
    A: ReadBuf<'a, T>,
    T: CType,
    Host: Reduce<A, T>,
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
    Host: Reduce<A, T>,
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

pub type Shape = SmallVec<[usize; 8]>;

pub type Array<T> = array::Array<T, AccessBuffer<Buffer<T>>, Platform>;

pub trait ReadBuf<'a, T: CType>: Send + Sync {
    fn read(self) -> Result<BufferConverter<'a, T>, Error>;

    fn size(&self) -> usize;
}
