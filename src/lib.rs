use std::fmt;

use smallvec::SmallVec;

mod array;
mod host;
#[cfg(feature = "opencl")]
mod opencl;

#[cfg(feature = "opencl")]
pub trait CType: ocl::OclPrm {
    const TYPE: &'static str;
}

#[cfg(not(feature = "opencl"))]
pub trait CType {
    const TYPE: &'static str;
}

impl CType for u32 {
    const TYPE: &'static str = "uint";
}

impl CType for u64 {
    const TYPE: &'static str = "uint";
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

pub trait PlatformInstance {
    fn select(size_hint: usize) -> Self;
}

enum Platform {
    #[cfg(feature = "opencl")]
    CL(opencl::OpenCL),
    Host(host::Host),
}

pub trait BufferInstance {
    fn size(&self) -> usize;
}

pub type Shape = SmallVec<[usize; 8]>;

pub trait NDArray {
    type Platform: PlatformInstance;

    fn platform(&self) -> &Self::Platform;

    fn shape(&self) -> &[usize];

    fn size(&self) -> usize {
        self.shape().iter().product()
    }
}

pub trait NDArrayRead<Buf: BufferInstance>: NDArray {
    fn read(&self) -> Result<Buf, Error>;
}

pub trait Op: Sized {
    type DType: CType;
}

pub trait Enqueue<P: PlatformInstance>: Op {
    type Buffer: BufferInstance;

    fn enqueue(&self, platform: &P) -> Result<Self::Buffer, Error>;
}
