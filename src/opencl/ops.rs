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

impl<L, R, T> Enqueue<OpenCL> for Dual<L, R, T>
where
    L: NDArrayRead<Buffer<T>>,
    R: NDArrayRead<Buffer<T>>,
    T: CType,
{
    type Buffer = Buffer<T>;

    fn enqueue(&self, platform: &OpenCL) -> Result<Self::Buffer, Error> {
        let left = self.left.read()?;
        let right = self.right.read()?;
        debug_assert_eq!(left.len(), right.len());

        let queue = platform.queue(left.len(), left.default_queue(), right.default_queue())?;

        let output = Buffer::builder()
            .queue(queue.clone())
            .len(left.len())
            .build()?;

        let program = kernels::elementwise::dual::<T, T>("add", platform.context())?;

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
