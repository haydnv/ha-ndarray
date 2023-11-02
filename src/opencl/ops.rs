use std::marker::PhantomData;

use ocl::Buffer;

use crate::{CType, Enqueue, Error, NDArrayRead, Op};

use super::platform::OpenCL;

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
    L: NDArrayRead<Buffer = Buffer<T>>,
    R: NDArrayRead<Buffer = Buffer<T>>,
    T: CType,
{
    type Buffer = Buffer<T>;

    fn enqueue(&self, op: Dual<L, R, T>) -> Result<Self::Buffer, Error> {
        let left = op.left.read()?;
        let right = op.right.read()?;

        let context = self.context();

        todo!()
    }
}
