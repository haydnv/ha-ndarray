use std::borrow::Borrow;
use std::marker::PhantomData;

use ocl::{Buffer, Kernel, Program};

use crate::access::Access;
use crate::ops::Write;
use crate::{strides_for, AccessBuffer, CType, Enqueue, Error, Op, Range, Shape, Strides};

use super::platform::OpenCL;
use super::{programs, CLConverter};

pub struct Compare<L, R, T> {
    left: L,
    right: R,
    program: Program,
    dtype: PhantomData<T>,
}

impl<L, R, T: CType> Compare<L, R, T> {
    pub fn eq(left: L, right: R) -> Result<Self, Error> {
        let program = programs::elementwise::compare::<T>("eq", OpenCL::context())?;

        Ok(Self {
            left,
            right,
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

        let queue = OpenCL::queue(left.len(), left.default_queue(), right.default_queue())?;

        let output = Buffer::builder()
            .queue(queue.clone())
            .len(left.len())
            .build()?;

        let kernel = Kernel::builder()
            .name("compare")
            .program(&self.program)
            .queue(queue)
            .global_work_size(left.len())
            .arg(&*left)
            .arg(&*right)
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }
}

pub struct Dual<L, R, T> {
    left: L,
    right: R,
    program: Program,
    dtype: PhantomData<T>,
}

impl<L, R, T: CType> Dual<L, R, T> {
    pub fn add(left: L, right: R) -> Result<Self, Error> {
        let program = programs::elementwise::dual::<T>("add", OpenCL::context())?;

        Ok(Self {
            left,
            right,
            program,
            dtype: PhantomData,
        })
    }

    pub fn sub(left: L, right: R) -> Result<Self, Error> {
        let program = programs::elementwise::dual::<T>("sub", OpenCL::context())?;

        Ok(Self {
            left,
            right,
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

        let queue = OpenCL::queue(left.len(), left.default_queue(), right.default_queue())?;

        let output = Buffer::builder()
            .queue(queue.clone())
            .len(left.len())
            .build()?;

        let kernel = Kernel::builder()
            .name("dual")
            .program(&self.program)
            .queue(queue)
            .global_work_size(left.len())
            .arg(&*left)
            .arg(&*right)
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }
}

pub struct Slice<A, T> {
    access: A,
    range: Range,
    shape: Shape,
    strides: Strides,
    source_strides: Strides,
    read: Program,
    write: Option<Program>,
    dtype: PhantomData<T>,
}

impl<A, T: CType> Slice<A, T> {
    pub fn new(access: A, shape: &[usize], range: Range) -> Result<Self, Error> {
        let source_strides = strides_for(shape, shape.len());
        let shape = range.iter().filter_map(|ar| ar.size()).collect::<Shape>();
        let strides = strides_for(&shape, shape.len());

        let read = programs::slice::read_slice::<T>(
            OpenCL::context(),
            &shape,
            &strides,
            &range,
            &source_strides,
        )?;

        Ok(Self {
            access,
            range,
            shape,
            strides,
            source_strides,
            read,
            write: None,
            dtype: PhantomData,
        })
    }
}

impl<A: Send + Sync, T: Send + Sync> Op for Slice<A, T> {
    fn size(&self) -> usize {
        self.shape.iter().product()
    }
}

impl<A, T> Enqueue<OpenCL> for Slice<A, T>
where
    A: Access<T>,
    T: CType,
{
    type Buffer = Buffer<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let source = self.access.read()?.to_cl()?;
        let queue = OpenCL::queue(self.size(), source.default_queue(), None)?;

        let output = Buffer::builder()
            .queue(queue.clone())
            .len(self.size())
            .build()?;

        let kernel = Kernel::builder()
            .name("read_slice")
            .program(&self.read)
            .queue(queue)
            .global_work_size(output.len())
            .arg(&*source)
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }
}

impl<'a, B, T> Write<'a, OpenCL> for Slice<AccessBuffer<B>, T>
where
    B: Borrow<Buffer<T>>,
    T: CType,
    AccessBuffer<B>: Access<T>,
{
    type Data = CLConverter<'a, T>;

    fn write(&'a mut self, data: Self::Data) -> Result<(), Error> {
        let size_hint = self.size();
        let source = self.access.as_ref().into_inner();
        let queue = OpenCL::queue(size_hint, source.default_queue(), None)?;

        if self.write.is_none() {
            let program = programs::slice::write_to_slice::<T>(
                OpenCL::context(),
                &self.shape,
                &self.strides,
                &self.range,
                &self.source_strides,
            )?;

            self.write = Some(program);
        }

        let kernel = Kernel::builder()
            .name("write_slice")
            .program(self.write.as_ref().expect("CL write op"))
            .queue(queue)
            .global_work_size(data.size())
            .arg(source)
            .arg(&*data)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(())
    }
}

pub struct Unary<A, IT, OT> {
    access: A,
    program: Program,
    dtype: PhantomData<(IT, OT)>,
}

impl<A, T> Unary<A, T, T>
where
    T: CType,
{
    pub fn ln(access: A) -> Result<Self, Error> {
        let program = programs::elementwise::unary::<T, T>("_log", OpenCL::context())?;

        Ok(Self {
            access,
            program,
            dtype: PhantomData,
        })
    }
}

impl<A, IT, OT> Op for Unary<A, IT, OT>
where
    A: Access<IT>,
    IT: CType,
    OT: CType,
{
    fn size(&self) -> usize {
        self.access.size()
    }
}

impl<A, IT, OT> Enqueue<OpenCL> for Unary<A, IT, OT>
where
    A: Access<IT>,
    IT: CType,
    OT: CType,
{
    type Buffer = Buffer<OT>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let input = self.access.read()?.to_cl()?;
        let queue = OpenCL::queue(input.len(), input.default_queue(), None)?;

        let output = Buffer::builder()
            .queue(queue.clone())
            .len(input.len())
            .build()?;

        let kernel = Kernel::builder()
            .name("unary")
            .program(&self.program)
            .queue(queue)
            .global_work_size(input.len())
            .arg(&*input)
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
    pub fn new(
        access: A,
        shape: &[usize],
        broadcast: &[usize],
        strides: &[usize],
    ) -> Result<Self, Error> {
        let size = broadcast.iter().product();
        let source_strides = strides_for(shape, shape.len());

        let program =
            programs::view::view::<T>(OpenCL::context(), shape, strides, &source_strides)?;

        Ok(Self {
            access,
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

        let queue = OpenCL::queue(self.size, source.default_queue(), None)?;

        let output = Buffer::builder()
            .queue(queue.clone())
            .len(self.size)
            .build()?;

        let kernel = Kernel::builder()
            .name("view")
            .program(&self.program)
            .queue(queue)
            .global_work_size(self.size)
            .arg(&*source)
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }
}
