use std::marker::PhantomData;

use ocl::{Buffer, Kernel};

use crate::opencl::kernels;
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

impl<L, R, T> Dual<L, R, T> {
    pub fn add(left: L, right: R) -> Self {
        Self {
            left,
            right,
            dtype: PhantomData,
        }
    }
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

        debug_assert_eq!(left.len(), right.len());

        let queue = self.queue(left.default_queue(), left.len())?;

        let output = Buffer::builder()
            .queue(queue.clone())
            .len(left.len())
            .build()?;

        let program = kernels::elementwise::dual::<T, T>("add", self.context())?;

        let kernel = Kernel::builder()
            .name("dual")
            .program(&program)
            .queue(queue)
            .global_work_size(left.len())
            .arg(&left)
            .arg(&right)
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }
}
