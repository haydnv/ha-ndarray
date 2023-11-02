use std::marker::PhantomData;

use crate::host::{Heap, Stack};
#[cfg(feature = "opencl")]
use crate::opencl::OpenCL;
use crate::{BufferInstance, Enqueue, Error, NDArray, NDArrayRead, Op, PlatformInstance, Shape};

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

pub struct ArrayOp<O, P> {
    shape: Shape,
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

    fn shape(&self) -> &[usize] {
        &self.shape
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

    fn shape(&self) -> &[usize] {
        &self.shape
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

    fn shape(&self) -> &[usize] {
        &self.shape
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
