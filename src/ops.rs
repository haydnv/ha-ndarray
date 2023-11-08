use crate::access::AccessOp;
use crate::buffer::{Buffer, BufferInstance};
#[cfg(feature = "opencl")]
use crate::opencl;
use crate::{host, CType, Error, Platform, PlatformInstance, ReadBuf};

pub trait Op: Send + Sync + Sized {
    type DType: CType;

    fn size(&self) -> usize;
}

pub trait Enqueue<P: PlatformInstance>: Op {
    type Buffer: BufferInstance<Self::DType>;

    fn enqueue(self) -> Result<Self::Buffer, Error>;
}

pub trait ElementwiseCompare<L, R, T>: PlatformInstance {
    type Output;

    fn eq(self, left: L, right: R) -> Result<AccessOp<Self::Output, Self>, Error>;
}

pub trait ElementwiseDual<L, R, T>: PlatformInstance {
    type Output;

    fn add(self, left: L, right: R) -> Result<AccessOp<Self::Output, Self>, Error>;

    fn sub(self, left: L, right: R) -> Result<AccessOp<Self::Output, Self>, Error>;
}

pub trait Reduce<A, T>: PlatformInstance {
    fn all(self, access: A) -> Result<bool, Error>;

    fn any(self, access: A) -> Result<bool, Error>;
}

pub enum Dual<L, R, T> {
    #[cfg(feature = "opencl")]
    CL(opencl::ops::Dual<L, R, T>),
    Host(host::ops::Dual<L, R, T>),
}

impl<'a, L, R, T> Op for Dual<L, R, T>
where
    L: ReadBuf<'a, T>,
    R: ReadBuf<'a, T>,
    T: CType,
{
    type DType = T;

    fn size(&self) -> usize {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(op) => op.size(),
            Self::Host(op) => op.size(),
        }
    }
}

impl<'a, L, R, T> Enqueue<Platform> for Dual<L, R, T>
where
    L: ReadBuf<'a, T>,
    R: ReadBuf<'a, T>,
    T: CType,
{
    type Buffer = Buffer<T>;

    fn enqueue(self) -> Result<Self::Buffer, Error> {
        todo!()
    }
}
