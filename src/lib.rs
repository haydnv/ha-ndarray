use std::fmt;
use std::ops::Add;

use smallvec::SmallVec;

use access::*;
pub use host::{Host, StackVec};

mod access;
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

pub trait PlatformInstance: PartialEq + Eq + Clone + Copy + Send + Sync {
    fn select(size_hint: usize) -> Self;
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum Platform {
    #[cfg(feature = "opencl")]
    CL(opencl::OpenCL),
    Host(host::Host),
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

pub trait BufferInstance: Sized {
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

impl<T: CType> BufferInstance for StackVec<T> {
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

pub trait ReadBuf {
    type Buffer: BufferInstance;

    fn read(self) -> Result<Self::Buffer, Error>;
}

pub trait Op: Sized {
    type DType: CType;
}

pub trait Enqueue<P: PlatformInstance>: Op {
    type Buffer: BufferInstance;

    fn enqueue(self, platform: P) -> Result<Self::Buffer, Error>;
}

pub struct Array<A> {
    shape: Shape,
    access: A,
}

impl<B: BufferInstance> Array<AccessBuffer<B>> {
    pub fn new(buffer: B, shape: Shape) -> Result<Self, Error> {
        if shape.iter().product::<usize>() == buffer.size() {
            Ok(Self {
                shape,
                access: AccessBuffer::from(buffer),
            })
        } else {
            Err(Error::Bounds(format!(
                "cannot construct an array with shape {shape:?} from a buffer of size {}",
                buffer.size()
            )))
        }
    }

    pub fn as_ref<RB>(&self) -> Array<AccessBuffer<&RB>>
    where
        B: AsRef<RB>,
    {
        Array {
            shape: Shape::from_slice(&self.shape),
            access: self.access.as_ref(),
        }
    }
}
