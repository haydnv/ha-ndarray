use std::borrow::Borrow;
use std::marker::PhantomData;

use ocl::{Buffer, Kernel, Program};

use crate::ops::Reduce;
use crate::{BufferInstance, CType, Enqueue, Error, Op, ReadBuf};

use super::kernels;
use super::platform::OpenCL;

pub struct Compare<L, R, T> {
    left: L,
    right: R,
    platform: OpenCL,
    program: Program,
    dtype: PhantomData<T>,
}

impl<L, R, T> Op for Compare<L, R, T>
where
    L: Send + Sync,
    R: Send + Sync,
    T: CType,
{
    type DType = u8;
}

impl<'a, L, R, T> Op for &'a Compare<L, R, T>
where
    L: Send + Sync,
    R: Send + Sync,
    T: CType,
{
    type DType = u8;
}

impl<L, R, T: CType> Compare<L, R, T> {
    pub fn eq(platform: OpenCL, left: L, right: R) -> Result<Self, Error> {
        let program = kernels::elementwise::compare::<T>("eq", platform.context())?;

        Ok(Self {
            left,
            right,
            platform,
            program,
            dtype: PhantomData,
        })
    }
}

impl<L, R, T> Enqueue<OpenCL> for Compare<L, R, T>
where
    L: ReadBuf<T> + Send + Sync,
    R: ReadBuf<T> + Send + Sync,
    T: CType,
    <L as ReadBuf<T>>::Buffer: Borrow<Buffer<T>>,
    <R as ReadBuf<T>>::Buffer: Borrow<Buffer<T>>,
{
    type Buffer = Buffer<u8>;

    fn enqueue(self) -> Result<Self::Buffer, Error> {
        let left = self.left.read()?;
        let right = self.right.read()?;
        debug_assert_eq!(left.size(), right.size());

        let left = left.borrow();
        let right = right.borrow();

        let queue = self
            .platform
            .queue(left.len(), left.default_queue(), right.default_queue())?;

        let output = Buffer::builder()
            .queue(queue.clone())
            .len(left.len())
            .build()?;

        let kernel = Kernel::builder()
            .name("compare")
            .program(&self.program)
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

impl<'a, L, R, T> Enqueue<OpenCL> for &'a Compare<L, R, T>
where
    L: Send + Sync,
    R: Send + Sync,
    &'a L: ReadBuf<T> + Send + Sync,
    &'a R: ReadBuf<T> + Send + Sync,
    T: CType,
    <&'a L as ReadBuf<T>>::Buffer: Borrow<Buffer<T>>,
    <&'a R as ReadBuf<T>>::Buffer: Borrow<Buffer<T>>,
{
    type Buffer = Buffer<u8>;

    fn enqueue(self) -> Result<Self::Buffer, Error> {
        let left = self.left.read()?;
        let right = self.right.read()?;
        debug_assert_eq!(left.size(), right.size());

        let left = left.borrow();
        let right = right.borrow();

        let queue = self
            .platform
            .queue(left.len(), left.default_queue(), right.default_queue())?;

        let output = Buffer::builder()
            .queue(queue.clone())
            .len(left.len())
            .build()?;

        let kernel = Kernel::builder()
            .name("compare")
            .program(&self.program)
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

pub struct Dual<L, R, T> {
    left: L,
    right: R,
    platform: OpenCL,
    program: Program,
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

impl<'a, L, R, T> Op for &'a Dual<L, R, T>
where
    L: Send + Sync,
    R: Send + Sync,
    T: CType,
{
    type DType = T;
}

impl<L, R, T: CType> Dual<L, R, T> {
    pub fn add(platform: OpenCL, left: L, right: R) -> Result<Self, Error> {
        let program = kernels::elementwise::dual::<T>("add", platform.context())?;

        Ok(Self {
            left,
            right,
            platform,
            program,
            dtype: PhantomData,
        })
    }
}

impl<L, R, T> Enqueue<OpenCL> for Dual<L, R, T>
where
    L: ReadBuf<T> + Send + Sync,
    R: ReadBuf<T> + Send + Sync,
    T: CType,
    <L as ReadBuf<T>>::Buffer: Borrow<Buffer<T>>,
    <R as ReadBuf<T>>::Buffer: Borrow<Buffer<T>>,
{
    type Buffer = Buffer<T>;

    fn enqueue(self) -> Result<Self::Buffer, Error> {
        let left = self.left.read()?;
        let right = self.right.read()?;
        debug_assert_eq!(left.size(), right.size());

        let left = left.borrow();
        let right = right.borrow();

        let queue = self
            .platform
            .queue(left.len(), left.default_queue(), right.default_queue())?;

        let output = Buffer::builder()
            .queue(queue.clone())
            .len(left.len())
            .build()?;

        let kernel = Kernel::builder()
            .name("dual")
            .program(&self.program)
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

impl<'a, L, R, T> Enqueue<OpenCL> for &'a Dual<L, R, T>
where
    L: Send + Sync,
    R: Send + Sync,
    &'a L: ReadBuf<T> + Send + Sync,
    &'a R: ReadBuf<T> + Send + Sync,
    T: CType,
    <&'a L as ReadBuf<T>>::Buffer: Borrow<Buffer<T>>,
    <&'a R as ReadBuf<T>>::Buffer: Borrow<Buffer<T>>,
{
    type Buffer = Buffer<T>;

    fn enqueue(self) -> Result<Self::Buffer, Error> {
        let left = self.left.read()?;
        let right = self.right.read()?;
        debug_assert_eq!(left.size(), right.size());

        let left = left.borrow();
        let right = right.borrow();

        let queue = self
            .platform
            .queue(left.len(), left.default_queue(), right.default_queue())?;

        let output = Buffer::builder()
            .queue(queue.clone())
            .len(left.len())
            .build()?;

        let kernel = Kernel::builder()
            .name("dual")
            .program(&self.program)
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

impl<A, T> Reduce<A, T> for OpenCL
where
    A: ReadBuf<T>,
    T: CType,
    <A as ReadBuf<T>>::Buffer: Borrow<Buffer<T>>,
{
    fn all(self, access: A) -> Result<bool, Error> {
        let buffer = access.read()?;
        let buffer = buffer.borrow();

        let result = [1];

        let program = kernels::reduce::all::<T>(self.context())?;

        let flag = unsafe {
            Buffer::builder()
                .context(self.context())
                .use_host_slice(&result)
                .len(1)
                .build()?
        };

        let queue = self.queue(buffer.len(), buffer.default_queue(), None)?;

        let kernel = Kernel::builder()
            .name("all")
            .program(&program)
            .queue(queue.clone())
            .global_work_size(buffer.len())
            .arg(&flag)
            .arg(buffer)
            .build()?;

        unsafe { kernel.enq()? }

        queue.finish()?;

        Ok(result == [1])
    }
}
