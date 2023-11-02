use std::fmt;
use std::marker::PhantomData;

use smallvec::SmallVec;

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
    CL(opencl::OpenCL),
    Host(host::Host),
}

pub trait BufferInstance {
    fn size(&self) -> usize;
}

pub type Shape = SmallVec<[usize; 8]>;

pub trait NDArray {}

pub trait NDArrayRead: NDArray {
    type Buffer: BufferInstance;

    fn read(&self) -> Result<Self::Buffer, Error>;
}

pub struct ArrayBase<B, P> {
    buffer: B,
    platform: PhantomData<P>,
}

pub struct ArrayOp<O, P> {
    op: O,
    platform: PhantomData<P>,
}

impl<O, P> NDArray for ArrayOp<O, P> {}

impl<O: Op, P: PlatformInstance> NDArrayRead for ArrayOp<O, P>
where
    P: Enqueue<O>,
{
    type Buffer = <P as Enqueue<O>>::Buffer;

    fn read(&self) -> Result<Self::Buffer, Error> {
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

pub trait Enqueue<O: Op>: PlatformInstance {
    type Buffer: BufferInstance;

    fn enqueue(&self, op: O) -> Result<Self::Buffer, Error>;
}
