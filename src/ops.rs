use crate::access::*;
use crate::array::Array;
use crate::buffer::{Buffer, BufferInstance};
#[cfg(feature = "opencl")]
use crate::opencl;
use crate::platform::{Platform, PlatformInstance};
use crate::{host, CType, Error};

pub trait Op: Send + Sync + Sized {
    type DType: CType;

    fn size(&self) -> usize;
}

pub trait Enqueue<P: PlatformInstance>: Op {
    type Buffer: BufferInstance<Self::DType>;

    fn enqueue(&self) -> Result<Self::Buffer, Error>;
}

pub trait ElementwiseCompare<L, R, T>: PlatformInstance {
    type Output: Enqueue<Self>;

    fn eq(self, left: L, right: R) -> Result<AccessOp<Self::Output, Self>, Error>;
}

pub trait ElementwiseDual<L, R, T>: PlatformInstance {
    type Output: Enqueue<Self>;

    fn add(self, left: L, right: R) -> Result<AccessOp<Self::Output, Self>, Error>;

    fn sub(self, left: L, right: R) -> Result<AccessOp<Self::Output, Self>, Error>;
}

pub trait Reduce<A, T>: PlatformInstance {
    fn all(self, access: A) -> Result<bool, Error>;

    fn any(self, access: A) -> Result<bool, Error>;
}

pub trait Transform<A, T>: PlatformInstance {
    type Broadcast: Enqueue<Self>;

    fn broadcast(
        self,
        array: Array<T, A, Self>,
        shape: &[usize],
    ) -> Result<AccessOp<Self::Broadcast, Self>, Error>;
}

pub enum Compare<L, R, T> {
    #[cfg(feature = "opencl")]
    CL(opencl::ops::Compare<L, R, T>),
    Host(host::ops::Compare<L, R, T>),
}

#[cfg(feature = "opencl")]
impl<L, R, T> From<opencl::ops::Compare<L, R, T>> for Compare<L, R, T> {
    fn from(op: opencl::ops::Compare<L, R, T>) -> Self {
        Self::CL(op)
    }
}

impl<L, R, T> From<host::ops::Compare<L, R, T>> for Compare<L, R, T> {
    fn from(op: host::ops::Compare<L, R, T>) -> Self {
        Self::Host(op)
    }
}

pub enum Dual<L, R, T> {
    #[cfg(feature = "opencl")]
    CL(opencl::ops::Dual<L, R, T>),
    Host(host::ops::Dual<L, R, T>),
}

#[cfg(feature = "opencl")]
impl<L, R, T> From<opencl::ops::Dual<L, R, T>> for Dual<L, R, T> {
    fn from(op: opencl::ops::Dual<L, R, T>) -> Self {
        Self::CL(op)
    }
}

impl<L, R, T> From<host::ops::Dual<L, R, T>> for Dual<L, R, T> {
    fn from(op: host::ops::Dual<L, R, T>) -> Self {
        Self::Host(op)
    }
}

macro_rules! impl_dual {
    ($op:ty, $t:ty) => {
        impl<'a, L, R, T> Op for $op
        where
            L: Access<T>,
            R: Access<T>,
            T: CType,
        {
            type DType = $t;

            fn size(&self) -> usize {
                match self {
                    #[cfg(feature = "opencl")]
                    Self::CL(op) => op.size(),
                    Self::Host(op) => op.size(),
                }
            }
        }

        impl<'a, L, R, T> Enqueue<Platform> for $op
        where
            L: Access<T>,
            R: Access<T>,
            T: CType,
        {
            type Buffer = Buffer<$t>;

            fn enqueue(&self) -> Result<Self::Buffer, Error> {
                match self {
                    #[cfg(feature = "opencl")]
                    Self::CL(op) => Enqueue::<opencl::OpenCL>::enqueue(op).map(Buffer::CL),
                    Self::Host(op) => Enqueue::<host::Host>::enqueue(op).map(Buffer::Host),
                }
            }
        }
    };
}

impl_dual!(Compare<L, R, T>, u8);
impl_dual!(Dual<L, R, T>, T);

pub enum View<A, T> {
    #[cfg(feature = "opencl")]
    CL(opencl::ops::View<A, T>),
    Host(host::ops::View<A, T>),
}

impl<'a, A, T> Op for View<A, T>
where
    A: Access<T>,
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

impl<'a, A, T> Enqueue<Platform> for View<A, T>
where
    A: Access<T>,
    T: CType,
{
    type Buffer = Buffer<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(op) => Enqueue::<opencl::OpenCL>::enqueue(op).map(Buffer::CL),
            Self::Host(op) => Enqueue::<host::Host>::enqueue(op).map(Buffer::Host),
        }
    }
}
