use std::marker::PhantomData;

use ocl::{Buffer, Kernel, Program};

use crate::access::Access;
use crate::array::Array;
use crate::{strides_for, CType, Enqueue, Error, Op};

use super::kernels;
use super::platform::OpenCL;

pub struct Compare<L, R, T> {
    left: L,
    right: R,
    platform: OpenCL,
    program: Program,
    dtype: PhantomData<T>,
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

impl<L, R, T> Op for Compare<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    fn size(&self) -> usize {
        self.left.size()
    }
}

impl<L, R, T> Enqueue<OpenCL> for Compare<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Buffer = Buffer<u8>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
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

impl<'a, L, R, T> Op for Dual<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    fn size(&self) -> usize {
        self.left.size()
    }
}

impl<L, R, T> Enqueue<OpenCL> for Dual<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Buffer = Buffer<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
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

pub struct View<A, T> {
    access: A,
    program: Program,
    size: usize,
    dtype: PhantomData<T>,
}

impl<A, T> View<A, T>
where
    T: CType,
{
    pub fn new<P>(
        source: Array<T, A, P>,
        shape: &[usize],
        strides: &[usize],
    ) -> Result<Self, Error> {
        let size = shape.iter().product();
        let source_strides = strides_for(source.shape(), source.ndim());

        let program = kernels::view::view::<T>(OpenCL.context(), shape, strides, &source_strides)?;

        Ok(Self {
            access: source.into_inner(),
            program,
            size,
            dtype: PhantomData,
        })
    }
}

impl<A, T> Op for View<A, T>
where
    A: Access<T>,
    T: CType,
{
    fn size(&self) -> usize {
        self.size
    }
}

impl<A, T> Enqueue<OpenCL> for View<A, T>
where
    A: Access<T>,
    T: CType,
{
    type Buffer = Buffer<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let source = self.access.read()?.to_cl()?;
        let source = source.as_ref();

        let queue = OpenCL.queue(self.size, source.default_queue(), None)?;

        let output = Buffer::builder()
            .queue(queue.clone())
            .len(self.size)
            .build()?;

        let kernel = Kernel::builder()
            .name("reorder")
            .program(&self.program)
            .queue(queue)
            .global_work_size(self.size)
            .arg(source)
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }
}
