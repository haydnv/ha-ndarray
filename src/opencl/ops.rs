use std::marker::PhantomData;

use ocl::{Buffer, Kernel, Program};

use crate::ops::Reduce;
use crate::{CType, Enqueue, Error, Op, ReadBuf};

use super::kernels;
use super::platform::OpenCL;

pub struct Compare<L, R, T> {
    left: L,
    right: R,
    platform: OpenCL,
    program: Program,
    dtype: PhantomData<T>,
}

impl<'a, L, R, T> Op for Compare<L, R, T>
where
    L: ReadBuf<'a, T>,
    R: ReadBuf<'a, T>,
    T: CType,
{
    type DType = u8;

    fn size(&self) -> usize {
        self.left.size()
    }
}

impl<'a, 'b, L, R, T> Op for &'a Compare<L, R, T>
where
    L: Send + Sync,
    R: Send + Sync,
    &'a L: ReadBuf<'b, T>,
    &'a R: ReadBuf<'b, T>,
    T: CType,
{
    type DType = u8;

    fn size(&self) -> usize {
        ReadBuf::size(&&self.left)
    }
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

impl<'a, L, R, T> Enqueue<OpenCL> for Compare<L, R, T>
where
    L: ReadBuf<'a, T>,
    R: ReadBuf<'a, T>,
    T: CType,
{
    type Buffer = Buffer<u8>;

    fn enqueue(self) -> Result<Self::Buffer, Error> {
        let left = self.left.read()?.to_cl()?;
        let right = self.right.read()?.to_cl()?;
        debug_assert_eq!(left.size(), right.size());

        let left = left.as_ref();
        let right = right.as_ref();

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

impl<'a, 'b, L, R, T> Enqueue<OpenCL> for &'a Compare<L, R, T>
where
    L: Send + Sync,
    R: Send + Sync,
    &'a L: ReadBuf<'b, T>,
    &'a R: ReadBuf<'b, T>,
    T: CType,
{
    type Buffer = Buffer<u8>;

    fn enqueue(self) -> Result<Self::Buffer, Error> {
        let left = self.left.read()?.to_cl()?;
        let right = self.right.read()?.to_cl()?;
        debug_assert_eq!(left.size(), right.size());

        let left = left.as_ref();
        let right = right.as_ref();

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

impl<'a, L, R, T> Op for Dual<L, R, T>
where
    L: ReadBuf<'a, T>,
    R: ReadBuf<'a, T>,
    T: CType,
{
    type DType = T;

    fn size(&self) -> usize {
        self.left.size()
    }
}

impl<'a, 'b, L, R, T> Op for &'a Dual<L, R, T>
where
    L: Send + Sync,
    R: Send + Sync,
    &'a L: ReadBuf<'b, T>,
    &'a R: ReadBuf<'b, T>,
    T: CType,
{
    type DType = T;

    fn size(&self) -> usize {
        ReadBuf::size(&&self.left)
    }
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

    pub fn sub(platform: OpenCL, left: L, right: R) -> Result<Self, Error> {
        let program = kernels::elementwise::dual::<T>("sub", platform.context())?;

        Ok(Self {
            left,
            right,
            platform,
            program,
            dtype: PhantomData,
        })
    }
}

impl<'a, L, R, T> Enqueue<OpenCL> for Dual<L, R, T>
where
    L: ReadBuf<'a, T>,
    R: ReadBuf<'a, T>,
    T: CType,
{
    type Buffer = Buffer<T>;

    fn enqueue(self) -> Result<Self::Buffer, Error> {
        let left = self.left.read()?.to_cl()?;
        let right = self.right.read()?.to_cl()?;
        debug_assert_eq!(left.size(), right.size());

        let left = left.as_ref();
        let right = right.as_ref();

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

impl<'a, 'b, L, R, T> Enqueue<OpenCL> for &'a Dual<L, R, T>
where
    L: Send + Sync,
    R: Send + Sync,
    &'a L: ReadBuf<'b, T>,
    &'a R: ReadBuf<'b, T>,
    T: CType,
{
    type Buffer = Buffer<T>;

    fn enqueue(self) -> Result<Self::Buffer, Error> {
        let left = self.left.read()?.to_cl()?;
        let right = self.right.read()?.to_cl()?;
        debug_assert_eq!(left.size(), right.size());

        let left = left.as_ref();
        let right = right.as_ref();

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

impl<'a, A, T> Reduce<A, T> for OpenCL
where
    A: ReadBuf<'a, T>,
    T: CType,
{
    fn all(self, access: A) -> Result<bool, Error> {
        let buffer = access.read()?.to_cl()?;
        let buffer = buffer.as_ref();

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

    fn any(self, access: A) -> Result<bool, Error> {
        let buffer = access.read()?.to_cl()?;
        let buffer = buffer.as_ref();

        let result = [0];

        let program = kernels::reduce::any::<T>(self.context())?;

        let flag = unsafe {
            Buffer::builder()
                .context(self.context())
                .use_host_slice(&result)
                .len(1)
                .build()?
        };

        let queue = self.queue(buffer.len(), buffer.default_queue(), None)?;

        let kernel = Kernel::builder()
            .name("any")
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
