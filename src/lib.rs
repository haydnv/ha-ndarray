use std::fmt;
use std::ops::Add;

use smallvec::SmallVec;

pub use host::StackBuf;

mod array;
mod host;
#[cfg(feature = "opencl")]
mod opencl;

#[cfg(feature = "opencl")]
pub trait CType: ocl::OclPrm + Add<Output = Self> + Copy + Send + Sync {
    const TYPE: &'static str;
}

#[cfg(not(feature = "opencl"))]
pub trait CType: Add<Output = Self> + Copy + Send + Sync {
    const TYPE: &'static str;
}

impl CType for i32 {
    const TYPE: &'static str = "int";
}

impl CType for u32 {
    const TYPE: &'static str = "uint";
}

impl CType for u64 {
    const TYPE: &'static str = "ulong";
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

pub trait PlatformInstance: Send + Sync {
    fn select(size_hint: usize) -> Self;
}

enum Platform {
    #[cfg(feature = "opencl")]
    CL(opencl::OpenCL),
    Host(host::Host),
}

pub trait BufferInstance: Send + Sync {
    fn size(&self) -> usize;
}

impl<'a, T: CType> BufferInstance for &'a [T] {
    fn size(&self) -> usize {
        self.len()
    }
}

impl<T: CType> BufferInstance for Vec<T> {
    fn size(&self) -> usize {
        self.len()
    }
}

impl<T: CType> BufferInstance for StackBuf<T> {
    fn size(&self) -> usize {
        self.len()
    }
}

#[cfg(feature = "opencl")]
impl<T: CType> BufferInstance for ocl::Buffer<T> {
    fn size(&self) -> usize {
        self.len()
    }
}

#[cfg(feature = "opencl")]
impl<'a, T: CType> BufferInstance for &'a ocl::Buffer<T> {
    fn size(&self) -> usize {
        self.len()
    }
}

pub type Shape = SmallVec<[usize; 8]>;

pub trait NDArray: Send + Sync {
    type Platform: PlatformInstance;

    fn platform(&self) -> &Self::Platform;

    fn shape(&self) -> &[usize];

    fn size(&self) -> usize {
        self.shape().iter().product()
    }
}

pub trait NDArrayMath<O: NDArray<Platform = Self::Platform>>: NDArray {
    type Op: Op + Enqueue<Self::Platform>;

    fn add(self, other: O) -> Result<array::ArrayOp<Self::Op, Self::Platform>, Error>;
}

pub trait ReadBuf<Buf: BufferInstance>: Send + Sync {
    fn read(self) -> Result<Buf, Error>;
}

pub trait Op: Send + Sync + Sized {
    type DType: CType;
}

pub trait Enqueue<P: PlatformInstance>: Op {
    type Buffer: BufferInstance;

    fn enqueue(self, platform: &P) -> Result<Self::Buffer, Error>;
}
