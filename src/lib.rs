use std::fmt;
use std::marker::PhantomData;
use std::ops::Add;

use smallvec::SmallVec;

use access::*;
pub use host::{Host, StackVec};
use ops::*;

mod access;
mod host;
#[cfg(feature = "opencl")]
mod opencl;
mod ops;

#[cfg(feature = "opencl")]
pub trait CType: ocl::OclPrm + Add<Output = Self> + Eq + Copy + Send + Sync {
    const TYPE: &'static str;

    const ZERO: Self;

    const ONE: Self;
}

#[cfg(not(feature = "opencl"))]
pub trait CType: Add<Output = Self> + Eq + Copy + Send + Sync {
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

pub trait BufferInstance<T: CType>: Sized {
    fn size(&self) -> usize;
}

impl<'a, T: CType> BufferInstance<T> for &'a [T] {
    fn size(&self) -> usize {
        self.len()
    }
}

impl<T: CType> BufferInstance<T> for Vec<T> {
    fn size(&self) -> usize {
        self.len()
    }
}

impl<T: CType> BufferInstance<T> for StackVec<T> {
    fn size(&self) -> usize {
        self.len()
    }
}

#[cfg(feature = "opencl")]
impl<T: CType> BufferInstance<T> for ocl::Buffer<T> {
    fn size(&self) -> usize {
        self.len()
    }
}

#[cfg(feature = "opencl")]
impl<'a, T: CType> BufferInstance<T> for &'a ocl::Buffer<T> {
    fn size(&self) -> usize {
        self.len()
    }
}

pub type Shape = SmallVec<[usize; 8]>;

pub trait ReadBuf<T: CType> {
    type Buffer: BufferInstance<T>;

    fn read(self) -> Result<Self::Buffer, Error>;
}

pub trait Op: Sized {
    type DType: CType;
}

pub trait Enqueue<P: PlatformInstance>: Op {
    type Buffer: BufferInstance<Self::DType>;

    fn enqueue(self) -> Result<Self::Buffer, Error>;
}

pub struct Array<T, A, P> {
    shape: Shape,
    access: A,
    platform: P,
    dtype: PhantomData<T>,
}

impl<T, A, P> Array<T, A, P> {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }
}

impl<T, B, P> Array<T, AccessBuffer<B>, P>
where
    T: CType,
    B: BufferInstance<T>,
    P: PlatformInstance,
{
    pub fn new(buffer: B, shape: Shape) -> Result<Self, Error> {
        if shape.iter().product::<usize>() == buffer.size() {
            let platform = P::select(buffer.size());
            // TODO: this should call a Platform::convert method
            let access = buffer.into();

            Ok(Self {
                shape,
                access,
                platform,
                dtype: PhantomData,
            })
        } else {
            Err(Error::Bounds(format!(
                "cannot construct an array with shape {shape:?} from a buffer of size {}",
                buffer.size()
            )))
        }
    }

    pub fn as_ref<RB>(&self) -> Array<T, AccessBuffer<&RB>, P>
    where
        B: AsRef<RB>,
    {
        Array {
            shape: Shape::from_slice(&self.shape),
            access: self.access.as_ref(),
            platform: self.platform,
            dtype: PhantomData,
        }
    }
}

impl<T, L, P> Array<T, L, P> {
    pub fn eq<R>(self, other: Array<T, R, P>) -> Result<Array<T, AccessOp<P::Output, P>, P>, Error>
    where
        P: ElementwiseCompare<L, R, T>,
    {
        if self.shape == other.shape {
            Ok(Array {
                shape: self.shape,
                access: self.platform.eq(self.access, other.access)?,
                platform: self.platform,
                dtype: PhantomData,
            })
        } else {
            todo!("broadcast")
        }
    }

    pub fn add<R>(self, other: Array<T, R, P>) -> Result<Array<T, AccessOp<P::Output, P>, P>, Error>
    where
        P: ElementwiseDual<L, R, T>,
    {
        if self.shape == other.shape {
            Ok(Array {
                shape: self.shape,
                access: self.platform.add(self.access, other.access)?,
                platform: self.platform,
                dtype: PhantomData,
            })
        } else {
            todo!("broadcast")
        }
    }
}
