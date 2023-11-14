use crate::access::*;
use crate::buffer::Buffer;
#[cfg(feature = "opencl")]
use crate::opencl;
use crate::platform::{Platform, PlatformInstance};
use crate::{host, BufferConverter, CType, Error, Range, Shape};

pub trait Op: Send + Sync {
    fn size(&self) -> usize;
}

pub trait Enqueue<P: PlatformInstance>: Op {
    type Buffer;

    fn enqueue(&self) -> Result<Self::Buffer, Error>;
}

pub trait Write<'a, P: PlatformInstance>: Enqueue<P> {
    type Data;

    fn write(&'a mut self, data: Self::Data) -> Result<(), Error>;
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

    fn sum(self, access: A) -> Result<T, Error>;
}

pub trait Transform<A, T>: PlatformInstance {
    type Broadcast: Enqueue<Self>;
    type Slice: Enqueue<Self>;

    fn broadcast(
        self,
        access: A,
        shape: Shape,
        broadcast: Shape,
    ) -> Result<AccessOp<Self::Broadcast, Self>, Error>;

    fn slice(
        self,
        access: A,
        shape: &[usize],
        range: Range,
    ) -> Result<AccessOp<Self::Slice, Self>, Error>;
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

macro_rules! impl_unary {
    ($op:ty, $t:ty) => {
        impl<'a, A, T> Op for $op
        where
            A: Access<T>,
            T: CType,
        {
            fn size(&self) -> usize {
                match self {
                    #[cfg(feature = "opencl")]
                    Self::CL(op) => op.size(),
                    Self::Host(op) => op.size(),
                }
            }
        }

        impl<'a, A, T> Enqueue<Platform> for $op
        where
            A: Access<T>,
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

macro_rules! impl_dual {
    ($op:ty, $t:ty) => {
        impl<'a, L, R, T> Op for $op
        where
            L: Access<T>,
            R: Access<T>,
            T: CType,
        {
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

pub enum Slice<A, T> {
    #[cfg(feature = "opencl")]
    CL(opencl::ops::Slice<A, T>),
    Host(host::ops::Slice<A, T>),
}

impl_unary!(Slice<A, T>, T);

#[cfg(feature = "opencl")]
impl<'a, A, T> Write<'a, Platform> for Slice<A, T>
where
    A: Access<T>,
    T: CType,
    host::ops::Slice<A, T>: Write<'a, host::Host, Data = host::SliceConverter<'a, T>>,
    opencl::ops::Slice<A, T>: Write<'a, opencl::OpenCL, Data = opencl::CLConverter<'a, T>>,
{
    type Data = BufferConverter<'a, T>;

    fn write(&'a mut self, data: Self::Data) -> Result<(), Error> {
        match self {
            Self::CL(op) => Write::write(op, data.to_cl()?),
            Self::Host(op) => Write::write(op, data.to_slice()?),
        }
    }
}

#[cfg(not(feature = "opencl"))]
impl<'a, A, T> Write<'a, Platform> for Slice<A, T>
where
    A: Access<T>,
    T: CType,
    host::ops::Slice<A, T>: Write<'a, host::Host, Data = host::SliceConverter<'a, T>>,
{
    type Data = BufferConverter<'a, T>;

    fn write(&'a mut self, data: Self::Data) -> Result<(), Error> {
        match self {
            Self::Host(op) => Write::write(op, data.to_slice()?),
        }
    }
}

#[cfg(feature = "opencl")]
impl<A, T> From<opencl::ops::Slice<A, T>> for Slice<A, T> {
    fn from(op: opencl::ops::Slice<A, T>) -> Self {
        Self::CL(op)
    }
}

impl<A, T> From<host::ops::Slice<A, T>> for Slice<A, T> {
    fn from(op: host::ops::Slice<A, T>) -> Self {
        Self::Host(op)
    }
}

pub enum View<A, T> {
    #[cfg(feature = "opencl")]
    CL(opencl::ops::View<A, T>),
    Host(host::ops::View<A, T>),
}

impl_unary!(View<A, T>, T);

#[cfg(feature = "opencl")]
impl<A, T> From<opencl::ops::View<A, T>> for View<A, T> {
    fn from(op: opencl::ops::View<A, T>) -> Self {
        Self::CL(op)
    }
}

impl<A, T> From<host::ops::View<A, T>> for View<A, T> {
    fn from(op: host::ops::View<A, T>) -> Self {
        Self::Host(op)
    }
}
