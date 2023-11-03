use std::borrow::Borrow;
use std::marker::PhantomData;

use ocl::{Buffer, Kernel};

use crate::{BufferInstance, CType, Enqueue, Error, Op, ReadBuf};

use super::kernels;
use super::platform::{CLBuf, OpenCL};

pub struct Dual<L, R, T> {
    left: L,
    right: R,
    dtype: PhantomData<T>,
}

impl<L, R, T> Op for Dual<L, R, T>
where
    L: Send + Sync,
    R: Send + Sync,
    T: CType,
{
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
    L: ReadBuf<T> + Send + Sync,
    R: ReadBuf<T> + Send + Sync,
    T: CType,
    L::Buffer: CLBuf<T>,
    R::Buffer: CLBuf<T>,
{
    type Buffer = Buffer<T>;

    fn enqueue(self, platform: OpenCL) -> Result<Self::Buffer, Error> {
        let left = self.left.read()?;
        let right = self.right.read()?;
        debug_assert_eq!(left.size(), right.size());

        let left = left.borrow();
        let right = right.borrow();

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
            .arg(left)
            .arg(right)
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }
}
