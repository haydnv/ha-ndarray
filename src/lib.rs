use std::fmt;
use std::marker::PhantomData;

use smallvec::SmallVec;

use host::{Heap, Stack};
#[cfg(feature = "opencl")]
use opencl::{OpenCL, CL_PLATFORM};

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

pub trait PlatformInstance {}

enum Platform {
    #[cfg(feature = "opencl")]
    CL(OpenCL),
    Host(host::Host),
}

pub trait BufferInstance {
    fn size(&self) -> usize;
}

pub type Shape = SmallVec<[usize; 8]>;

pub trait NDArray {
    type Platform: PlatformInstance;

    fn platform(&self) -> &Self::Platform;
}

pub trait NDArrayRead<Buf: BufferInstance>: NDArray {
    fn read(&self) -> Result<Buf, Error>;
}

pub struct ArrayBase<B, P> {
    buffer: B,
    platform: PhantomData<P>,
}

pub struct ArrayOp<O, P> {
    op: O,
    platform: P,
}

impl<O> NDArray for ArrayOp<O, Stack>
where
    O: Enqueue<Stack>,
{
    type Platform = Stack;

    fn platform(&self) -> &Self::Platform {
        &self.platform
    }
}

impl<O> NDArray for ArrayOp<O, Heap>
where
    O: Enqueue<Heap>,
{
    type Platform = Heap;

    fn platform(&self) -> &Self::Platform {
        &self.platform
    }
}

#[cfg(feature = "opencl")]
impl<O> NDArray for ArrayOp<O, OpenCL>
where
    O: Enqueue<OpenCL>,
{
    type Platform = OpenCL;

    fn platform(&self) -> &OpenCL {
        &self.platform
    }
}

impl<O: Op, P: PlatformInstance> NDArrayRead<O::Buffer> for ArrayOp<O, P>
where
    O: Enqueue<P>,
    Self: NDArray<Platform = P>,
{
    fn read(&self) -> Result<O::Buffer, Error> {
        todo!()
    }
}

pub struct ArraySlice<A, P> {
    source: A,
    platform: PhantomData<P>,
}

pub struct ArrayView<A, P> {
    source: A,
    platform: PhantomData<P>,
}

pub trait Op: Sized {
    type DType: CType;
}

pub trait Enqueue<P: PlatformInstance>: Op {
    type Buffer: BufferInstance;

    fn enqueue(&self, platform: &P) -> Result<Self::Buffer, Error>;
}
