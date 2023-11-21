use std::borrow::Borrow;
use std::marker::PhantomData;
use std::ops::{Add, Sub};

use ocl::{Buffer, Kernel, Program};
use rand::{random, Rng};

use crate::access::Access;
use crate::ops::{ReadValue, SliceSpec, ViewSpec, Write};
use crate::{strides_for, AccessBuffer, CType, Enqueue, Error, Float, Op, Range, Shape, Strides};

use super::platform::OpenCL;
use super::{programs, CLConverter, WG_SIZE};

pub struct Compare<L, R, T> {
    left: L,
    right: R,
    program: Program,
    op: fn(T, T) -> bool,
    dtype: PhantomData<T>,
}

impl<L, R, T: CType> Compare<L, R, T> {
    pub fn eq(left: L, right: R) -> Result<Self, Error> {
        let program = programs::elementwise::compare(T::TYPE, "eq")?;

        Ok(Self {
            left,
            right,
            program,
            op: |l, r| l == r,
            dtype: PhantomData,
        })
    }
}

impl<L: Access<T>, R: Access<T>, T: CType> Op for Compare<L, R, T> {
    fn size(&self) -> usize {
        self.left.size()
    }
}

impl<L, R, T> Enqueue<OpenCL, u8> for Compare<L, R, T>
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

impl<L, R, T> ReadValue<OpenCL, u8> for Compare<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    fn read_value(&self, offset: usize) -> Result<u8, Error> {
        let l = self.left.read_value(offset)?;
        let r = self.right.read_value(offset)?;

        if (self.op)(l, r) {
            Ok(1)
        } else {
            Ok(0)
        }
    }
}

pub struct Dual<L, R, T> {
    left: L,
    right: R,
    program: Program,
    op: fn(T, T) -> T,
    dtype: PhantomData<T>,
}

impl<L, R, T: CType> Dual<L, R, T> {
    pub fn add(left: L, right: R) -> Result<Self, Error> {
        let program = programs::elementwise::dual(T::TYPE, "add")?;

        Ok(Self {
            left,
            right,
            program,
            op: Add::add,
            dtype: PhantomData,
        })
    }

    pub fn sub(left: L, right: R) -> Result<Self, Error> {
        let program = programs::elementwise::dual(T::TYPE, "sub")?;

        Ok(Self {
            left,
            right,
            program,
            op: Sub::sub,
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

impl<L, R, T> Enqueue<OpenCL, T> for Dual<L, R, T>
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

impl<L, R, T> ReadValue<OpenCL, T> for Dual<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    fn read_value(&self, offset: usize) -> Result<T, Error> {
        let l = self.left.read_value(offset)?;
        let r = self.right.read_value(offset)?;
        Ok((self.op)(l, r))
    }
}

pub struct Linear<T> {
    start: T,
    step: f64,
    size: usize,
    program: Program,
}

impl<T: CType> Linear<T> {
    pub fn new(start: T, step: f64, size: usize) -> Result<Self, Error> {
        programs::constructors::range(T::TYPE).map(|program| Self {
            start,
            step,
            size,
            program,
        })
    }
}

impl<T: Send + Sync> Op for Linear<T> {
    fn size(&self) -> usize {
        self.size
    }
}

impl<T: CType> Enqueue<OpenCL, T> for Linear<T> {
    type Buffer = Buffer<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let queue = OpenCL::queue(self.size, None, None)?;

        let buffer = Buffer::builder()
            .queue(queue.clone())
            .len(self.size)
            .build()?;

        let kernel = Kernel::builder()
            .name("range")
            .queue(queue)
            .program(&self.program)
            .global_work_size(self.size)
            .arg(self.start)
            .arg(self.step)
            .arg(&buffer)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(buffer)
    }
}

impl<T: CType> ReadValue<OpenCL, T> for Linear<T> {
    fn read_value(&self, offset: usize) -> Result<T, Error> {
        Ok(self.start + T::from_f64((offset as f64) * self.step))
    }
}

pub struct RandomNormal {
    program: Program,
    size: usize,
}

impl RandomNormal {
    pub fn new(size: usize) -> Result<Self, Error> {
        programs::constructors::random_normal().map(|program| Self { program, size })
    }
}

impl Op for RandomNormal {
    fn size(&self) -> usize {
        self.size
    }
}

impl Enqueue<OpenCL, f32> for RandomNormal {
    type Buffer = Buffer<f32>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let queue = OpenCL::queue(self.size, None, None)?;
        let seed: u32 = rand::thread_rng().gen();

        let buffer = Buffer::builder()
            .queue(queue.clone())
            .len(WG_SIZE * self.size().div_ceil(WG_SIZE))
            .build()?;

        let kernel = Kernel::builder()
            .name("random_normal")
            .queue(queue.clone())
            .program(&self.program)
            .global_work_size(buffer.len())
            .local_work_size(WG_SIZE)
            .arg(u64::try_from(seed).expect("seed"))
            .arg(&buffer)
            .arg_local::<f32>(WG_SIZE)
            .build()?;

        unsafe { kernel.enq()? }

        if buffer.len() == self.size {
            Ok(buffer)
        } else {
            let output = Buffer::builder().queue(queue).len(self.size).build()?;

            buffer.copy(&output, Some(0), Some(self.size)).enq()?;

            Ok(output)
        }
    }
}

impl ReadValue<OpenCL, f32> for RandomNormal {
    fn read_value(&self, _offset: usize) -> Result<f32, Error> {
        Err(Error::Bounds(
            "cannot read an individual value from a random normal distribution".to_string(),
        ))
    }
}

pub struct RandomUniform {
    program: Program,
    size: usize,
}

impl RandomUniform {
    pub fn new(size: usize) -> Result<Self, Error> {
        programs::constructors::random_uniform().map(|program| Self { program, size })
    }
}

impl Op for RandomUniform {
    fn size(&self) -> usize {
        self.size
    }
}

impl Enqueue<OpenCL, f32> for RandomUniform {
    type Buffer = Buffer<f32>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let queue = OpenCL::queue(self.size, None, None)?;
        let seed: u32 = random();

        let output = Buffer::builder()
            .queue(queue.clone())
            .len(self.size)
            .build()?;

        let kernel = Kernel::builder()
            .name("random_uniform")
            .queue(queue)
            .program(&self.program)
            .global_work_size(output.len())
            .arg(seed as u64)
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }
}

impl ReadValue<OpenCL, f32> for RandomUniform {
    fn read_value(&self, _offset: usize) -> Result<f32, Error> {
        Ok(random())
    }
}

pub struct Slice<A, T> {
    access: A,
    spec: SliceSpec,
    read: Program,
    write: Option<Program>,
    dtype: PhantomData<T>,
}

impl<A, T: CType> Slice<A, T> {
    pub fn new(access: A, shape: &[usize], range: Range) -> Result<Self, Error> {
        let source_strides = strides_for(shape, shape.len()).collect();
        let spec = SliceSpec::new(range, source_strides);

        let read = programs::slice::read_slice(T::TYPE, spec.clone())?;

        Ok(Self {
            access,
            spec,
            read,
            write: None,
            dtype: PhantomData,
        })
    }
}

impl<A: Send + Sync, T: Send + Sync> Op for Slice<A, T> {
    fn size(&self) -> usize {
        self.spec.size()
    }
}

impl<A: Access<T>, T: CType> Enqueue<OpenCL, T> for Slice<A, T> {
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

impl<A: Access<T>, T: CType> ReadValue<OpenCL, T> for Slice<A, T> {
    fn read_value(&self, offset: usize) -> Result<T, Error> {
        self.access.read_value(self.spec.source_offset(offset))
    }
}

impl<'a, B, T> Write<'a, OpenCL, T> for Slice<AccessBuffer<B>, T>
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
            let program = programs::slice::write_to_slice(T::TYPE, self.spec.clone())?;

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
    op: fn(IT) -> OT,
    dtype: PhantomData<(IT, OT)>,
}

impl<A, T: CType> Unary<A, T, T> {
    pub fn ln(access: A) -> Result<Self, Error> {
        let program = programs::elementwise::unary(T::Float::TYPE, T::TYPE, T::TYPE, "_log")?;

        Ok(Self {
            access,
            program,
            op: |n| T::from_float(n.to_float().ln()),
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

impl<A, IT, OT> Enqueue<OpenCL, OT> for Unary<A, IT, OT>
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

impl<A: Access<IT>, IT: CType, OT: CType> ReadValue<OpenCL, OT> for Unary<A, IT, OT> {
    fn read_value(&self, offset: usize) -> Result<OT, Error> {
        self.access.read_value(offset).map(|n| (self.op)(n))
    }
}

pub struct View<A, T> {
    access: A,
    program: Program,
    size: usize,
    spec: ViewSpec,
    dtype: PhantomData<T>,
}

impl<A, T> View<A, T>
where
    T: CType,
{
    pub fn new(access: A, shape: Shape, broadcast: Shape, strides: Strides) -> Result<Self, Error> {
        let size = broadcast.iter().product();
        let source_strides = strides_for(&shape, shape.len()).collect();
        let spec = ViewSpec::new(broadcast, strides, source_strides);

        let program = programs::view::view(T::TYPE, spec.clone())?;

        Ok(Self {
            access,
            program,
            size,
            spec,
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

impl<A: Access<T>, T: CType> Enqueue<OpenCL, T> for View<A, T> {
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

impl<A: Access<T>, T: CType> ReadValue<OpenCL, T> for View<A, T> {
    fn read_value(&self, offset: usize) -> Result<T, Error> {
        self.access.read_value(self.spec.source_offset(offset))
    }
}
