use std::fmt;
use std::marker::PhantomData;

use crate::host;
#[cfg(feature = "opencl")]
use crate::opencl;
use crate::{
    BufferInstance, CType, Enqueue, Error, NDArray, NDArrayMath, Op, PlatformInstance, ReadBuf,
    Shape,
};

pub struct ArrayBase<B, P> {
    shape: Shape,
    buffer: B,
    platform: P,
}

impl<B: BufferInstance, P: PlatformInstance> ArrayBase<B, P> {
    pub fn new(buffer: B, shape: Shape) -> Result<Self, Error> {
        if buffer.size() == shape.iter().product() {
            let platform = P::select(buffer.size());

            Ok(Self {
                shape,
                buffer,
                platform,
            })
        } else {
            Err(Error::Bounds(format!(
                "invalid buffer size {} for shape {shape:?}",
                buffer.size()
            )))
        }
    }
}

impl<B, P> NDArray for ArrayBase<B, P>
where
    B: BufferInstance,
    P: PlatformInstance,
{
    type Platform = P;

    fn platform(&self) -> &Self::Platform {
        &self.platform
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl<'a, B, P> ReadBuf<B> for ArrayBase<B, P>
where
    B: BufferInstance,
    P: PlatformInstance,
{
    fn read(self) -> Result<B, Error> {
        Ok(self.buffer)
    }
}

impl<T: CType> NDArrayMath<Self> for ArrayBase<Vec<T>, host::Heap> {
    type Op = host::Dual<Self, Self, T>;

    fn add(self, other: Self) -> Result<ArrayOp<Self::Op, Self::Platform>, Error> {
        if self.shape == other.shape {
            let shape = self.shape.clone();
            let op = host::Dual::add(self, other);
            Ok(ArrayOp::new(op, shape, host::Heap))
        } else {
            Err(Error::Bounds(format!("cannot add {self:?} and {other:?}")))
        }
    }
}

impl<B, P> fmt::Debug for ArrayBase<B, P> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "array buffer of shape {:?}", self.shape)
    }
}

pub struct ArrayOp<O, P> {
    shape: Shape,
    op: O,
    platform: P,
}

impl<O, P> ArrayOp<O, P> {
    fn new(op: O, shape: Shape, platform: P) -> Self {
        Self {
            shape,
            op,
            platform,
        }
    }
}

impl<O> NDArray for ArrayOp<O, host::Stack>
where
    O: Enqueue<host::Stack>,
{
    type Platform = host::Stack;

    fn platform(&self) -> &Self::Platform {
        &self.platform
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl<O> NDArray for ArrayOp<O, host::Heap>
where
    O: Enqueue<host::Heap>,
{
    type Platform = host::Heap;

    fn platform(&self) -> &Self::Platform {
        &self.platform
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

#[cfg(feature = "opencl")]
impl<O> NDArray for ArrayOp<O, opencl::OpenCL>
where
    O: Enqueue<opencl::OpenCL>,
{
    type Platform = opencl::OpenCL;

    fn platform(&self) -> &opencl::OpenCL {
        &self.platform
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl<O: Op, P: PlatformInstance> ReadBuf<O::Buffer> for ArrayOp<O, P>
where
    O: Enqueue<P>,
    Self: NDArray<Platform = P>,
{
    fn read(self) -> Result<O::Buffer, Error> {
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
