use std::borrow::BorrowMut;
use std::marker::PhantomData;
use std::ops::{Add, Sub};

use ocl::{Buffer, Kernel, Program, Queue};
use rand::{random, Rng};

use crate::access::{Access, AccessBuffer, AccessMut};
use crate::ops::{Op, ReadValue, ReduceAll, SliceSpec, ViewSpec, Write};
use crate::{strides_for, Axes, CType, Enqueue, Error, Float, Range, Shape, Strides};

use super::platform::OpenCL;
use super::{programs, CLConverter, TILE_SIZE, WG_SIZE};

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

pub struct CompareScalar<A, T> {
    access: A,
    scalar: T,
    program: Program,
    op: fn(T, T) -> bool,
}

impl<A, T> CompareScalar<A, T>
where
    T: CType,
{
    fn new(
        access: A,
        scalar: T,
        program: &'static str,
        op: fn(T, T) -> bool,
    ) -> Result<Self, Error> {
        programs::elementwise::compare(T::TYPE, program)
            .map(|program| Self {
                access,
                scalar,
                program,
                op,
            })
            .map_err(Error::from)
    }

    pub fn eq(access: A, scalar: T) -> Result<Self, Error> {
        Self::new(access, scalar, "eq", |l, r| l == r)
    }

    pub fn ge(access: A, scalar: T) -> Result<Self, Error> {
        Self::new(access, scalar, "ge", |l, r| l >= r)
    }

    pub fn gt(access: A, scalar: T) -> Result<Self, Error> {
        Self::new(access, scalar, "gt", |l, r| l > r)
    }

    pub fn le(access: A, scalar: T) -> Result<Self, Error> {
        Self::new(access, scalar, "le", |l, r| l <= r)
    }

    pub fn lt(access: A, scalar: T) -> Result<Self, Error> {
        Self::new(access, scalar, "lt", |l, r| l < r)
    }

    pub fn ne(access: A, scalar: T) -> Result<Self, Error> {
        Self::new(access, scalar, "ne", |l, r| l != r)
    }
}

impl<A: Access<T>, T: CType> Op for CompareScalar<A, T> {
    fn size(&self) -> usize {
        self.access.size()
    }
}

impl<A: Access<T>, T: CType> Enqueue<OpenCL, u8> for CompareScalar<A, T> {
    type Buffer = Buffer<u8>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let input = self.access.read()?.to_cl()?;

        let queue = OpenCL::queue(input.len(), input.default_queue(), None)?;

        let output = Buffer::builder()
            .queue(queue.clone())
            .len(input.len())
            .build()?;

        let kernel = Kernel::builder()
            .name("compare_scalar")
            .program(&self.program)
            .queue(queue)
            .global_work_size(input.len())
            .arg(&*input)
            .arg(self.scalar)
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }
}

impl<A: Access<T>, T: CType> ReadValue<OpenCL, u8> for CompareScalar<A, T> {
    fn read_value(&self, offset: usize) -> Result<u8, Error> {
        self.access
            .read_value(offset)
            .map(|n| if (self.op)(n, self.scalar) { 1 } else { 0 })
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

impl<L, R, T> Op for Dual<L, R, T>
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

pub struct MatMul<L, R, T> {
    left: L,
    right: R,
    batch_size: usize,
    dims: [usize; 3],
    padded: [usize; 3],
    pad_matrices: Program,
    matmul: Program,
    dtype: PhantomData<T>,
}

impl<L, R, T> MatMul<L, R, T>
where
    T: CType,
{
    pub fn new(left: L, right: R, dims: [usize; 4]) -> Result<Self, Error> {
        let pad_matrices = programs::linalg::pad_matrices(T::TYPE)?;
        let matmul = programs::linalg::matmul(T::TYPE)?;

        let [batch_size, a, b, c] = dims;
        assert!(batch_size > 0);

        let dims = [a, b, c];

        let padded = [
            pad_dim(a, TILE_SIZE),
            pad_dim(b, TILE_SIZE),
            pad_dim(c, TILE_SIZE),
        ];

        Ok(Self {
            left,
            right,
            batch_size,
            dims,
            padded,
            pad_matrices,
            matmul,
            dtype: PhantomData,
        })
    }

    fn matmul(&self, left: &Buffer<T>, right: &Buffer<T>) -> Result<Buffer<T>, Error> {
        let [a, b, c] = self.padded;

        assert_eq!(self.batch_size * a * b, left.len());
        assert_eq!(self.batch_size * b * c, right.len());

        let queue = OpenCL::queue(
            (self.batch_size * a * c) / (TILE_SIZE * TILE_SIZE), // global work size
            left.default_queue(),
            right.default_queue(),
        )?;

        let output = Buffer::builder()
            .queue(queue.clone())
            .len(a * c * self.batch_size)
            .build()?;

        let dims = [a as u64, b as u64, c as u64, self.batch_size as u64];

        let kernel = Kernel::builder()
            .name("matmul")
            .program(&self.matmul)
            .queue(queue)
            .global_work_size((self.batch_size, a / TILE_SIZE, c / TILE_SIZE))
            .arg(ocl::core::Ulong4::from(dims))
            .arg(b / TILE_SIZE)
            .arg(left)
            .arg(right)
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }

    fn pad_matrices<'a>(
        &self,
        batch: &Buffer<T>,
        dims_in: [usize; 2],
        dims_out: [usize; 2],
    ) -> Result<Buffer<T>, Error> {
        if dims_in == dims_out {
            return Ok(batch.clone());
        }

        assert_eq!(batch.len(), self.batch_size * dims_in[0] * dims_in[1]);

        let queue = OpenCL::queue(batch.len(), batch.default_queue(), None)?;

        let output = Buffer::builder()
            .queue(queue.clone())
            .len(self.batch_size * dims_out[0] * dims_out[1])
            .build()?;

        let kernel = Kernel::builder()
            .name("pad_matrices")
            .program(&self.pad_matrices)
            .queue(queue)
            .global_work_size(batch.len())
            .arg(self.batch_size)
            .arg(dims_in[1] as u64)
            .arg(dims_out[1] as u64)
            .arg(batch)
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output.into())
    }
}

impl<L, R, T> Op for MatMul<L, R, T>
where
    L: Send + Sync,
    R: Send + Sync,
    T: Send + Sync,
{
    fn size(&self) -> usize {
        self.batch_size * self.dims[0] * self.dims[2]
    }
}

impl<L, R, T> Enqueue<OpenCL, T> for MatMul<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Buffer = Buffer<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let [a, b, c] = self.dims;
        let [a_pad, b_pad, c_pad] = self.padded;

        let left = self.left.read()?.to_cl()?;
        let right = self.right.read()?.to_cl()?;

        assert_eq!(self.batch_size * a * b, left.len());
        assert_eq!(self.batch_size * b * c, right.len());

        let left = self.pad_matrices(&*left, [a, b], [a_pad, b_pad])?;
        let right = self.pad_matrices(&*right, [b, c], [b_pad, c_pad])?;

        let product = self.matmul(&left, &right)?;

        self.pad_matrices(&product, [a_pad, c_pad], [a, c])
    }
}

impl<L, R, T> ReadValue<OpenCL, T> for MatMul<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    fn read_value(&self, _offset: usize) -> Result<T, Error> {
        Err(Error::Bounds(
            "reading an individual value from a matrix multiplication is not implemented"
                .to_string(),
        ))
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

pub struct Reduce<A, T: CType> {
    access: A,
    stride: usize,
    fold: Program,
    reduce: Program,
    reduce_all: fn(OpenCL, AccessBuffer<Buffer<T>>) -> Result<T, Error>,
    id: T,
}

impl<A, T: CType> Reduce<A, T> {
    fn new(
        access: A,
        stride: usize,
        reduce: &'static str,
        reduce_all: fn(OpenCL, AccessBuffer<Buffer<T>>) -> Result<T, Error>,
        id: T,
    ) -> Result<Self, Error> {
        let fold = programs::reduce::fold_axis(T::TYPE, reduce)?;
        let reduce = programs::reduce::reduce_axis(T::TYPE, reduce)?;

        Ok(Self {
            access,
            stride,
            fold,
            reduce,
            reduce_all,
            id,
        })
    }

    pub fn max(access: A, stride: usize) -> Result<Self, Error> {
        Self::new(
            access,
            stride,
            "max",
            <OpenCL as ReduceAll<AccessBuffer<Buffer<T>>, T>>::max,
            T::MIN,
        )
    }

    pub fn min(access: A, stride: usize) -> Result<Self, Error> {
        Self::new(
            access,
            stride,
            "min",
            <OpenCL as ReduceAll<AccessBuffer<Buffer<T>>, T>>::min,
            T::MAX,
        )
    }

    pub fn product(access: A, stride: usize) -> Result<Self, Error> {
        Self::new(
            access,
            stride,
            "mul",
            <OpenCL as ReduceAll<AccessBuffer<Buffer<T>>, T>>::product,
            T::ONE,
        )
    }

    pub fn sum(access: A, stride: usize) -> Result<Self, Error> {
        Self::new(
            access,
            stride,
            "add",
            <OpenCL as ReduceAll<AccessBuffer<Buffer<T>>, T>>::sum,
            T::ZERO,
        )
    }

    fn fold(
        &self,
        queue: Queue,
        input: &Buffer<T>,
        reduce_dim: usize,
        target_dim: usize,
    ) -> Result<Buffer<T>, Error> {
        let output_size = (input.len() / reduce_dim) * target_dim;

        let output = Buffer::builder()
            .queue(queue.clone())
            .len(output_size)
            .build()?;

        let kernel = Kernel::builder()
            .name("fold_axis")
            .program(&self.fold)
            .queue(queue)
            .global_work_size(output_size)
            .arg(reduce_dim as u64)
            .arg(target_dim as u64)
            .arg(self.id)
            .arg(input)
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }

    fn reduce(
        &self,
        queue: Queue,
        input: &Buffer<T>,
        stride: usize,
        wg_size: usize,
    ) -> Result<Buffer<T>, Error> {
        let output = Buffer::builder()
            .queue(queue.clone())
            .len(input.len() / stride)
            .build()?;

        let kernel = Kernel::builder()
            .name("reduce_axis")
            .program(&self.reduce)
            .queue(queue.clone())
            .local_work_size(wg_size)
            .global_work_size(input.len())
            .arg(self.id)
            .arg(input)
            .arg(&output)
            .arg_local::<T>(wg_size)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }
}

impl<A: Access<T>, T: CType> Op for Reduce<A, T> {
    fn size(&self) -> usize {
        debug_assert_eq!(self.access.size() % self.stride, 0);
        self.access.size() / self.stride
    }
}

impl<A: Access<T>, T: CType> Enqueue<OpenCL, T> for Reduce<A, T> {
    type Buffer = Buffer<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let input = self.access.read()?.to_cl()?;
        let queue = OpenCL::queue(input.len(), input.default_queue(), None)?;
        let output_size = input.len() / self.stride;

        let mut stride = self.stride;

        if stride < WG_SIZE {
            return self.fold(queue, &*input, stride, 1);
        }

        let log = (stride as f32).log(WG_SIZE as f32).fract();
        let target_dim = WG_SIZE.pow(log as u32);
        let mut buffer = self.fold(queue.clone(), &*input, stride, target_dim)?;

        stride = target_dim;
        debug_assert_eq!(output_size * stride, buffer.len());

        while buffer.len() > output_size {
            let wg_size = if stride < WG_SIZE {
                stride
            } else {
                debug_assert_eq!(stride % WG_SIZE, 0);
                WG_SIZE
            };

            buffer = self.reduce(queue.clone(), &buffer, stride, wg_size)?;
            stride /= wg_size;
            debug_assert_eq!(output_size * stride, buffer.len());
        }

        Ok(buffer)
    }
}

impl<A: Access<T>, T: CType> ReadValue<OpenCL, T> for Reduce<A, T> {
    fn read_value(&self, offset: usize) -> Result<T, Error> {
        let input = self.access.read()?.to_cl()?;
        let slice = input.create_sub_buffer(None, offset, offset + self.stride)?;
        (self.reduce_all)(OpenCL, AccessBuffer::from(slice))
    }
}

pub struct Slice<A, T> {
    access: A,
    spec: SliceSpec,
    read: Program,
    write: Option<Program>,
    write_value: Option<Program>,
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
            write_value: None,
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
    B: BorrowMut<Buffer<T>>,
    T: CType,
    AccessBuffer<B>: AccessMut<'a, T>,
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
            .global_work_size(source.len())
            .arg(source)
            .arg(&*data)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(())
    }

    fn write_value(&'a mut self, value: T) -> Result<(), Error> {
        let size_hint = self.size();
        let source = self.access.as_ref().into_inner();
        let queue = OpenCL::queue(size_hint, source.default_queue(), None)?;

        if self.write.is_none() {
            let program = programs::slice::write_value_to_slice(T::TYPE, self.spec.clone())?;
            self.write_value = Some(program);
        }

        let kernel = Kernel::builder()
            .name("write_slice_value")
            .program(self.write_value.as_ref().expect("CL write op"))
            .queue(queue)
            .global_work_size(source.len())
            .arg(source)
            .arg(value)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(())
    }

    fn write_value_at(&'a mut self, offset: usize, value: T) -> Result<(), Error> {
        self.access
            .borrow_mut()
            .write_value_at(self.spec.source_offset(offset), value)
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
    fn new(access: A, spec: ViewSpec) -> Result<Self, Error> {
        let size = spec.shape.iter().product();
        let program = programs::view::view(T::TYPE, spec.clone())?;

        Ok(Self {
            access,
            program,
            size,
            spec,
            dtype: PhantomData,
        })
    }

    pub fn broadcast(
        access: A,
        shape: Shape,
        broadcast: Shape,
        strides: Strides,
    ) -> Result<Self, Error> {
        let source_strides = strides_for(&shape, shape.len()).collect();
        let spec = ViewSpec::new(broadcast, strides, source_strides);
        Self::new(access, spec)
    }

    pub fn transpose(access: A, shape: Shape, axes: Axes) -> Result<Self, Error> {
        let source_strides = strides_for(&shape, shape.len()).collect();
        let shape = axes.iter().copied().map(|x| shape[x]).collect::<Shape>();
        let strides = strides_for(&shape, shape.len()).collect();
        let spec = ViewSpec::new(shape, strides, source_strides);
        Self::new(access, spec)
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

#[inline]
fn pad_dim(dim: usize, size: usize) -> usize {
    size * dim.div_ceil(size)
}
