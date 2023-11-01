use ocl::Buffer;
use std::marker::PhantomData;

use crate::{BufferInstance, CType, Enqueue, Error, NDArrayRead, Op, PlatformInstance};

impl<T: CType> BufferInstance for Buffer<T> {
    fn size(&self) -> usize {
        self.len()
    }
}

pub struct OpenCL;

impl PlatformInstance for OpenCL {}

struct Dual<L, R, T> {
    left: L,
    right: R,
    dtype: PhantomData<T>,
}

impl<L, R, T: CType> Op for Dual<L, R, T> {
    type DType = T;
}

impl<L, R, T> Enqueue<Dual<L, R, T>> for OpenCL
where
    L: NDArrayRead,
    R: NDArrayRead,
    T: CType,
{
    type Buffer = Buffer<T>;

    fn enqueue(&self, op: Dual<L, R, T>) -> Result<Self::Buffer, Error> {
        todo!()
    }
}
