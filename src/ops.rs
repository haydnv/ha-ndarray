use std::cmp::{PartialEq, PartialOrd};
use std::f32::consts::PI;
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Rem, Sub};
use std::sync::Arc;

use rand::Rng;
use rayon::prelude::*;

#[cfg(feature = "opencl")]
use super::cl_programs;
use super::{
    Buffer, CDatatype, Context, Error, Float, Log, NDArray, NDArrayRead, Queue, SliceConverter,
    Trig,
};

pub trait Op: Send + Sync {
    type Out: CDatatype;

    fn context(&self) -> &Context;

    fn enqueue(&self, queue: &Queue) -> Result<Buffer<Self::Out>, Error> {
        // TODO: there must be a better way to do this
        #[cfg(feature = "opencl")]
        if queue.cl_queue.is_some() {
            return self.enqueue_cl(queue).map(Buffer::CL);
        }

        self.enqueue_cpu(queue).map(Buffer::Host)
    }

    fn enqueue_cpu(&self, queue: &Queue) -> Result<Vec<Self::Out>, Error>;

    #[cfg(feature = "opencl")]
    fn enqueue_cl(&self, queue: &Queue) -> Result<ocl::Buffer<Self::Out>, Error>;
}

impl<O: Op + ?Sized> Op for Arc<O> {
    type Out = O::Out;

    fn context(&self) -> &Context {
        (**self).context()
    }

    fn enqueue_cpu(&self, queue: &Queue) -> Result<Vec<Self::Out>, Error> {
        (**self).enqueue_cpu(queue)
    }

    #[cfg(feature = "opencl")]
    fn enqueue_cl(&self, queue: &Queue) -> Result<ocl::Buffer<Self::Out>, Error> {
        (**self).enqueue_cl(queue)
    }
}

impl<O: Op + ?Sized> Op for Box<O> {
    type Out = O::Out;

    fn context(&self) -> &Context {
        (**self).context()
    }

    fn enqueue_cpu(&self, queue: &Queue) -> Result<Vec<Self::Out>, Error> {
        (**self).enqueue_cpu(queue)
    }

    #[cfg(feature = "opencl")]
    fn enqueue_cl(&self, queue: &Queue) -> Result<ocl::Buffer<Self::Out>, Error> {
        (**self).enqueue_cl(queue)
    }
}

// constructors

#[derive(Clone)]
pub struct Range<T> {
    context: Context,
    size: usize,
    start: T,
    step: f64,
    #[cfg(feature = "opencl")]
    kernel_op: ocl::Program,
}

impl<T: CDatatype + fmt::Display> Range<T> {
    pub fn new(start: T, stop: T, size: usize) -> Result<Self, Error> {
        let context = Context::default()?;
        Self::with_context(context, start, stop, size)
    }

    pub fn with_context(context: Context, start: T, stop: T, size: usize) -> Result<Self, Error> {
        let step = if start < stop {
            Ok((stop - start).to_f64() / (size as f64))
        } else {
            Err(Error::Bounds(format!("invalid range: [{start}..{stop})")))
        }?;

        #[cfg(feature = "opencl")]
        let kernel_op = cl_programs::range::<T>(&context)?;

        Ok(Self {
            context,
            size,
            start,
            step,
            #[cfg(feature = "opencl")]
            kernel_op,
        })
    }
}

impl<T: CDatatype> Op for Range<T> {
    type Out = T;

    fn context(&self) -> &Context {
        &self.context
    }

    fn enqueue_cpu(&self, _queue: &Queue) -> Result<Vec<Self::Out>, Error> {
        let start = self.start.to_f64();

        let buffer = (0..self.size)
            .into_par_iter()
            .map(|i| i as f64)
            .map(|i| i * self.step)
            .map(|o| start.to_f64() + o)
            .map(T::from_f64)
            .collect();

        Ok(buffer)
    }

    #[cfg(feature = "opencl")]
    fn enqueue_cl(&self, queue: &Queue) -> Result<ocl::Buffer<Self::Out>, Error> {
        let cl_queue = queue.cl_queue.as_ref().expect("queue");

        let buffer = ocl::Buffer::builder()
            .queue(cl_queue.clone())
            .len(self.size)
            .build()?;

        let kernel = ocl::Kernel::builder()
            .name("range")
            .queue(cl_queue.clone())
            .program(&self.kernel_op)
            .global_work_size(buffer.len())
            .arg(self.step)
            .arg(&buffer)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(buffer)
    }
}

#[derive(Clone)]
pub struct RandomNormal {
    context: Context,
    size: usize,
    #[cfg(feature = "opencl")]
    kernel_op: ocl::Program,
}

impl RandomNormal {
    pub fn new(size: usize) -> Result<Self, Error> {
        let context = Context::default()?;
        Self::with_context(context, size)
    }

    pub fn with_context(context: Context, size: usize) -> Result<Self, Error> {
        #[cfg(feature = "opencl")]
        let kernel_op = cl_programs::random_normal(&context)?;

        Ok(Self {
            context,
            size,
            #[cfg(feature = "opencl")]
            kernel_op,
        })
    }
}

impl Op for RandomNormal {
    type Out = f32;

    fn context(&self) -> &Context {
        &self.context
    }

    fn enqueue_cpu(&self, _queue: &Queue) -> Result<Vec<Self::Out>, Error> {
        let mut u = vec![
            0.0f32;
            if self.size % 2 == 0 {
                self.size
            } else {
                self.size + 1
            }
        ];

        rand::thread_rng().fill(&mut u[..]);

        let mut output = u
            .par_chunks_exact(2)
            .map(|u| {
                let [u1, u2]: [f32; 2] = u.try_into().expect("u");
                let r = (u1.ln() * -2.).sqrt();
                let theta = 2. * PI * u2;
                [r * theta.cos(), r * theta.sin()]
            })
            .flatten()
            .collect::<Vec<f32>>();

        if output.len() > self.size {
            output.pop();
        }

        debug_assert_eq!(output.len(), self.size);

        Ok(output)
    }

    #[cfg(feature = "opencl")]
    fn enqueue_cl(&self, queue: &Queue) -> Result<ocl::Buffer<Self::Out>, Error> {
        use crate::div_ceil;
        use cl_programs::WG_SIZE;

        let cl_queue = queue.cl_queue.as_ref().expect("queue");
        let seed: u32 = rand::thread_rng().gen();

        let buffer = ocl::Buffer::builder()
            .queue(cl_queue.clone())
            .len(WG_SIZE * div_ceil(self.size, WG_SIZE))
            .build()?;

        let kernel = ocl::Kernel::builder()
            .name("random_normal")
            .queue(cl_queue.clone())
            .program(&self.kernel_op)
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
            let output = ocl::Buffer::builder()
                .queue(cl_queue.clone())
                .len(self.size)
                .build()?;

            buffer.copy(&output, Some(0), Some(self.size)).enq()?;

            Ok(output)
        }
    }
}

#[derive(Clone)]
pub struct RandomUniform {
    context: Context,
    size: usize,
    #[cfg(feature = "opencl")]
    kernel_op: ocl::Program,
}

impl RandomUniform {
    pub fn new(size: usize) -> Result<Self, Error> {
        let context = Context::default()?;
        Self::with_context(context, size)
    }

    pub fn with_context(context: Context, size: usize) -> Result<Self, Error> {
        #[cfg(feature = "opencl")]
        let kernel_op = cl_programs::random_uniform(&context)?;

        Ok(Self {
            context,
            size,
            #[cfg(feature = "opencl")]
            kernel_op,
        })
    }
}

impl Op for RandomUniform {
    type Out = f32;

    fn context(&self) -> &Context {
        &self.context
    }

    fn enqueue_cpu(&self, _queue: &Queue) -> Result<Vec<Self::Out>, Error> {
        let mut data = vec![0.; self.size];
        rand::thread_rng().fill(&mut data[..]);
        Ok(data)
    }

    #[cfg(feature = "opencl")]
    fn enqueue_cl(&self, queue: &Queue) -> Result<ocl::Buffer<Self::Out>, Error> {
        let seed: u32 = rand::thread_rng().gen();
        let cl_queue = queue.cl_queue.as_ref().expect("queue");

        let output = ocl::Buffer::builder()
            .queue(cl_queue.clone())
            .len(self.size)
            .build()?;

        let kernel = ocl::Kernel::builder()
            .name("random_uniform")
            .queue(cl_queue.clone())
            .program(&self.kernel_op)
            .global_work_size(output.len())
            .arg(u64::try_from(seed).expect("seed"))
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }
}

// arithmetic

#[derive(Clone)]
pub struct ArrayDual<T, L, R> {
    left: L,
    right: R,
    cpu_op: fn(T, T) -> T,
    #[cfg(feature = "opencl")]
    kernel_op: ocl::Program,
}

impl<T: CDatatype, L: NDArray, R: NDArray> ArrayDual<T, L, R> {
    fn new(
        left: L,
        right: R,
        cpu_op: fn(T, T) -> T,
        #[allow(unused_variables)] kernel_op: &'static str,
    ) -> Result<Self, Error> {
        #[cfg(feature = "opencl")]
        let kernel_op = cl_programs::elementwise_dual::<T, T>(kernel_op, left.context())?;

        Ok(Self {
            left,
            right,
            cpu_op,
            #[cfg(feature = "opencl")]
            kernel_op,
        })
    }

    pub fn add(left: L, right: R) -> Result<Self, Error> {
        Self::new(left, right, Add::add, "add")
    }

    pub fn div(left: L, right: R) -> Result<Self, Error> {
        Self::new(left, right, Div::div, "div")
    }

    pub fn mul(left: L, right: R) -> Result<Self, Error> {
        Self::new(left, right, Mul::mul, "mul")
    }

    pub fn rem(left: L, right: R) -> Result<Self, Error> {
        Self::new(left, right, Rem::rem, "fmod")
    }

    pub fn sub(left: L, right: R) -> Result<Self, Error> {
        Self::new(left, right, Sub::sub, "sub")
    }
}

impl<T: CDatatype, L: NDArrayRead<DType = T>, R: NDArrayRead<DType = T>> Op for ArrayDual<T, L, R> {
    type Out = T;

    fn context(&self) -> &Context {
        self.left.context()
    }

    fn enqueue_cpu(&self, queue: &Queue) -> Result<Vec<T>, Error> {
        let (left, right) = try_join_read(&self.left, &self.right, queue)?;
        debug_assert_eq!(left.len(), right.len());

        let output = left
            .as_ref()
            .par_iter()
            .copied()
            .zip(right.as_ref().par_iter().copied())
            .map(|(l, r)| (self.cpu_op)(l, r))
            .collect();

        Ok(output)
    }

    #[cfg(feature = "opencl")]
    fn enqueue_cl(&self, queue: &Queue) -> Result<ocl::Buffer<T>, Error> {
        let right_queue = queue.split(self.right.size())?;
        let right = self.right.to_cl_buffer(&right_queue)?;
        let left = self.left.to_cl_buffer(queue)?;
        debug_assert_eq!(left.len(), right.len());

        let cl_queue = left.as_ref().default_queue().expect("left queue").clone();

        let output = ocl::Buffer::builder()
            .queue(cl_queue.clone())
            .len(left.len())
            .build()?;

        let kernel = ocl::Kernel::builder()
            .name("elementwise_dual")
            .program(&self.kernel_op)
            .queue(cl_queue)
            .global_work_size(left.len())
            .arg(left.as_ref())
            .arg(right.as_ref())
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }
}

#[derive(Clone)]
pub struct ArrayDualFloat<T: CDatatype, L, R> {
    left: L,
    right: R,
    cpu_op: fn(T, T::Float) -> T,
    #[cfg(feature = "opencl")]
    kernel_op: ocl::Program,
}

impl<T: CDatatype, L: NDArray<DType = T>, R: NDArray<DType = T::Float>> ArrayDualFloat<T, L, R> {
    #[allow(unused_variables)]
    fn new(
        left: L,
        right: R,
        cpu_op: fn(T, T::Float) -> T,
        kernel_op: &'static str,
    ) -> Result<Self, Error> {
        #[cfg(feature = "opencl")]
        let kernel_op = cl_programs::elementwise_dual::<T, T::Float>(kernel_op, left.context())?;

        Ok(Self {
            left,
            right,
            cpu_op,
            #[cfg(feature = "opencl")]
            kernel_op,
        })
    }

    pub fn log(left: L, right: R) -> Result<Self, Error> {
        Self::new(
            left,
            right,
            |l, r| T::from_float(l.to_float().log(r)),
            "log_",
        )
    }

    pub fn pow(left: L, right: R) -> Result<Self, Error> {
        Self::new(
            left,
            right,
            |l, r| T::from_float(l.to_float().pow(r)),
            "pow_",
        )
    }
}

impl<T, L, R> Op for ArrayDualFloat<T, L, R>
where
    T: CDatatype,
    L: NDArrayRead<DType = T>,
    R: NDArrayRead<DType = T::Float>,
{
    type Out = T;

    fn context(&self) -> &Context {
        self.left.context()
    }

    fn enqueue_cpu(&self, queue: &Queue) -> Result<Vec<Self::Out>, Error> {
        let (left, right) = try_join_read(&self.left, &self.right, queue)?;

        let output = left
            .as_ref()
            .par_iter()
            .copied()
            .zip(right.as_ref().par_iter().copied())
            .map(|(l, r)| (self.cpu_op)(l, r))
            .collect();

        Ok(output)
    }

    #[cfg(feature = "opencl")]
    fn enqueue_cl(&self, queue: &Queue) -> Result<ocl::Buffer<Self::Out>, Error> {
        let right_queue = queue.split(self.right.size())?;
        let right = self.right.to_cl_buffer(&right_queue)?;
        let left = self.left.to_cl_buffer(queue)?;
        debug_assert_eq!(left.len(), right.len());

        let cl_queue = left.as_ref().default_queue().expect("left queue").clone();

        let output = ocl::Buffer::builder()
            .queue(cl_queue.clone())
            .len(left.len())
            .build()?;

        let kernel = ocl::Kernel::builder()
            .name("elementwise_dual")
            .program(&self.kernel_op)
            .queue(cl_queue)
            .global_work_size(left.len())
            .arg(left.as_ref())
            .arg(right.as_ref())
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }
}

#[derive(Clone)]
pub struct ArrayScalar<T, A> {
    array: A,
    scalar: T,
    host_op: fn(T, T) -> T,
    #[cfg(feature = "opencl")]
    kernel_op: ocl::Program,
}

impl<T: CDatatype, A: NDArray<DType = T>> ArrayScalar<T, A> {
    #[allow(unused_variables)]
    fn new(
        array: A,
        scalar: T,
        host_op: fn(T, T) -> T,
        kernel_op: &'static str,
    ) -> Result<Self, Error> {
        #[cfg(feature = "opencl")]
        let kernel_op = cl_programs::elementwise_scalar::<T, T>(kernel_op, array.context())?;

        Ok(Self {
            array,
            scalar,
            host_op,
            #[cfg(feature = "opencl")]
            kernel_op,
        })
    }

    pub fn add(left: A, right: T) -> Result<Self, Error> {
        Self::new(left, right, Add::add, "add")
    }

    pub fn div(left: A, right: T) -> Result<Self, Error> {
        Self::new(left, right, Div::div, "div")
    }

    pub fn mul(left: A, right: T) -> Result<Self, Error> {
        Self::new(left, right, Mul::mul, "mul")
    }

    pub fn rem(left: A, right: T) -> Result<Self, Error> {
        Self::new(left, right, Rem::rem, "fmod")
    }

    pub fn sub(left: A, right: T) -> Result<Self, Error> {
        Self::new(left, right, Sub::sub, "sub")
    }
}

impl<T: CDatatype, A: NDArrayRead<DType = T>> Op for ArrayScalar<T, A> {
    type Out = A::DType;

    fn context(&self) -> &Context {
        self.array.context()
    }

    fn enqueue_cpu(&self, queue: &Queue) -> Result<Vec<Self::Out>, Error> {
        let left = self.array.to_host(queue)?;
        let right = self.scalar;

        let output = left
            .as_ref()
            .par_iter()
            .copied()
            .map(|l| (self.host_op)(l, right))
            .collect();

        Ok(output)
    }

    #[cfg(feature = "opencl")]
    fn enqueue_cl(&self, queue: &Queue) -> Result<ocl::Buffer<Self::Out>, Error> {
        let left = self.array.to_cl_buffer(queue)?;
        let right = self.scalar;
        let cl_queue = left.as_ref().default_queue().expect("queue").clone();

        let output = ocl::Buffer::builder()
            .queue(cl_queue.clone())
            .len(left.len())
            .build()?;

        let kernel = ocl::Kernel::builder()
            .name("elementwise_scalar")
            .program(&self.kernel_op)
            .queue(cl_queue)
            .global_work_size(left.len())
            .arg(left.as_ref())
            .arg(right)
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }
}

#[derive(Clone)]
pub struct ArrayScalarFloat<T: CDatatype, A> {
    array: A,
    scalar: T::Float,
    host_op: fn(T, T::Float) -> T,
    #[cfg(feature = "opencl")]
    kernel_op: ocl::Program,
}

impl<T: CDatatype, A: NDArray> ArrayScalarFloat<T, A> {
    #[allow(unused_variables)]
    fn new(
        array: A,
        scalar: T::Float,
        host_op: fn(T, T::Float) -> T,
        kernel_op: &'static str,
    ) -> Result<Self, Error> {
        #[cfg(feature = "opencl")]
        let kernel_op = cl_programs::elementwise_scalar::<T::Float, T>(kernel_op, array.context())?;

        Ok(Self {
            array,
            scalar,
            host_op,
            #[cfg(feature = "opencl")]
            kernel_op,
        })
    }
}

impl<T: CDatatype, A: NDArray> ArrayScalarFloat<T, A> {
    pub fn log(left: A, right: T::Float) -> Result<Self, Error> {
        Self::new(
            left,
            right,
            |l, r| T::from_float(l.to_float().log(r)),
            "log",
        )
    }

    pub fn pow(left: A, right: T::Float) -> Result<Self, Error> {
        Self::new(
            left,
            right,
            |l, r| T::from_float(l.to_float().pow(r)),
            "pow_",
        )
    }
}

impl<T: CDatatype, A: NDArrayRead<DType = T>> Op for ArrayScalarFloat<T, A> {
    type Out = T;

    fn context(&self) -> &Context {
        self.array.context()
    }

    fn enqueue_cpu(&self, queue: &Queue) -> Result<Vec<Self::Out>, Error> {
        let left = self.array.to_host(queue)?;
        let right = self.scalar;

        let output = left
            .as_ref()
            .par_iter()
            .copied()
            .map(|l| (self.host_op)(l, right))
            .collect();

        Ok(output)
    }

    #[cfg(feature = "opencl")]
    fn enqueue_cl(&self, queue: &Queue) -> Result<ocl::Buffer<Self::Out>, Error> {
        let left = self.array.to_cl_buffer(queue)?;
        let right = self.scalar;
        let cl_queue = left.as_ref().default_queue().expect("queue").clone();

        let output = ocl::Buffer::builder()
            .queue(cl_queue.clone())
            .len(left.len())
            .build()?;

        let kernel = ocl::Kernel::builder()
            .name("elementwise_scalar")
            .program(&self.kernel_op)
            .queue(cl_queue)
            .global_work_size(left.len())
            .arg(left.as_ref())
            .arg(right)
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }
}

// linear algebra

#[derive(Clone)]
pub struct MatDiag<A> {
    source: A,
    #[cfg(feature = "opencl")]
    kernel_op: ocl::Program,
}

impl<A: NDArray> MatDiag<A> {
    pub fn new(source: A) -> Result<Self, Error> {
        debug_assert!(source.ndim() >= 2);
        debug_assert_eq!(
            source.shape()[source.ndim() - 1],
            source.shape()[source.ndim() - 2]
        );

        #[cfg(feature = "opencl")]
        let kernel_op = cl_programs::diagonal::<A::DType>(source.context())?;

        Ok(Self {
            source,
            #[cfg(feature = "opencl")]
            kernel_op,
        })
    }
}

impl<A: NDArrayRead> Op for MatDiag<A> {
    type Out = A::DType;

    fn context(&self) -> &Context {
        self.source.context()
    }

    fn enqueue_cpu(&self, queue: &Queue) -> Result<Vec<Self::Out>, Error> {
        let dim = *self.source.shape().last().expect("dim");

        let input = self.source.to_host(queue)?;
        let mut output = Vec::with_capacity(self.source.size() / dim);

        let diagonals = input
            .as_ref()
            .par_chunks_exact(dim * dim)
            .map(|matrix| (0..dim).into_par_iter().map(|i| matrix[(i * dim) + i]))
            .flatten();

        output.par_extend(diagonals);

        Ok(output)
    }

    #[cfg(feature = "opencl")]
    fn enqueue_cl(&self, queue: &Queue) -> Result<ocl::Buffer<Self::Out>, Error> {
        let dim = *self.source.shape().last().expect("dim");
        let batch_size = self
            .source
            .shape()
            .iter()
            .take(self.source.ndim() - 2)
            .product();

        let input = self.source.to_cl_buffer(queue)?;
        let cl_queue = input.as_ref().default_queue().expect("queue");

        let output = ocl::Buffer::builder()
            .queue(cl_queue.clone())
            .len(input.len() / dim)
            .build()?;

        let kernel = ocl::Kernel::builder()
            .name("diagonal")
            .program(&self.kernel_op)
            .queue(cl_queue.clone())
            .global_work_size((batch_size, dim))
            .arg(input.as_ref())
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? };

        Ok(output)
    }
}

#[derive(Clone)]
pub struct MatMul<T, L, R> {
    left: L,
    right: R,
    dtype: PhantomData<T>,
    #[cfg(feature = "opencl")]
    kernel_op: ocl::Program,
}

impl<T, L, R> MatMul<T, L, R>
where
    T: CDatatype,
    L: NDArray<DType = T>,
    R: NDArray<DType = T>,
{
    pub fn new(left: L, right: R) -> Result<Self, Error> {
        debug_assert!(left.ndim() >= 2);
        debug_assert!(right.ndim() >= 2);

        #[cfg(feature = "opencl")]
        let kernel_op = cl_programs::matmul::<T>(left.context())?;

        Ok(Self {
            left,
            right,
            dtype: PhantomData,
            #[cfg(feature = "opencl")]
            kernel_op,
        })
    }

    fn dims(&self) -> [usize; 4] {
        let ndim = self.left.ndim();
        debug_assert_eq!(ndim, self.right.ndim());

        let num_matrices = self.left.shape().iter().take(ndim - 2).product();
        debug_assert_eq!(
            num_matrices,
            self.right.shape().iter().take(ndim - 2).product()
        );

        let a = *self.left.shape().iter().rev().nth(1).expect("a");
        let b = *self.left.shape().last().expect("b");
        let c = *self.right.shape().last().expect("c");
        debug_assert_eq!(b, self.right.shape()[ndim - 2]);
        debug_assert_eq!(num_matrices * a * b, self.left.size());
        debug_assert_eq!(num_matrices * b * c, self.right.size());

        [num_matrices, a, b, c]
    }
}

impl<T, L, R> Op for MatMul<T, L, R>
where
    T: CDatatype,
    L: NDArrayRead<DType = T>,
    R: NDArrayRead<DType = T>,
{
    type Out = T;

    fn context(&self) -> &Context {
        self.left.context()
    }

    fn enqueue_cpu(&self, queue: &Queue) -> Result<Vec<Self::Out>, Error> {
        let [num_matrices, a, b, c] = self.dims();

        let (left, right) = try_join_read(&self.left, &self.right, queue)?;

        let left = left.as_ref();
        let right = right.as_ref();

        // transpose the right matrices
        let right_size = b * c;
        let right_matrices = (0..num_matrices).into_par_iter().map(|n| {
            let start = n * right_size;
            let stop = start + right_size;
            let mut right_t = vec![T::zero(); right_size];
            transpose::transpose(&right[start..stop], &mut right_t[..], b, c);
            right_t
        });

        let left_size = a * b;
        let left_matrices = left.par_chunks_exact(left_size);

        let output_size = a * c;
        let mut output = Vec::<T>::with_capacity(num_matrices * output_size);
        let output_matrices = left_matrices
            .zip(right_matrices)
            .map(|(lm, rm)| {
                let mut out = Vec::<T>::with_capacity(output_size);

                let product = lm
                    .par_chunks_exact(b)
                    .map(|row| {
                        rm.par_chunks_exact(b).map(move |col| {
                            // chunk the dot product to encourage the compiler to vectorize
                            let col = col.chunks(8).map(|cc| cc.into_iter().copied());

                            row.chunks(8)
                                .zip(col)
                                .map(|(rc, cc)| {
                                    rc.into_iter().copied().zip(cc).map(|(r, c)| r * c).sum()
                                })
                                .sum::<T>()
                        })
                    })
                    .flatten();

                out.par_extend(product);
                out
            })
            .flatten();

        output.par_extend(output_matrices);

        debug_assert_eq!(output.len(), num_matrices * output_size);

        Ok(output)
    }

    #[cfg(feature = "opencl")]
    fn enqueue_cl(&self, queue: &Queue) -> Result<ocl::Buffer<Self::Out>, Error> {
        use crate::div_ceil;
        use cl_programs::TILE_SIZE;

        let [num_matrices, a, b, c] = self.dims();

        let right_queue = queue.split(self.right.size())?;
        let right = self.right.to_cl_buffer(&right_queue)?;
        let left = self.left.to_cl_buffer(queue)?;

        let cl_queue = left.as_ref().default_queue().expect("left queue");

        assert!(num_matrices > 0);
        assert_eq!(num_matrices * a * b, left.len());
        assert_eq!(num_matrices * b * c, right.len());

        let dims = [a as u64, b as u64, c as u64, num_matrices as u64];

        let output = ocl::Buffer::builder()
            .queue(cl_queue.clone())
            .len(a * c * num_matrices)
            .build()?;

        let kernel = ocl::Kernel::builder()
            .name("matmul")
            .program(&self.kernel_op)
            .queue(cl_queue.clone())
            .global_work_size((num_matrices, div_ceil(a, TILE_SIZE), div_ceil(c, TILE_SIZE)))
            .arg(ocl::core::Ulong4::from(dims))
            .arg(div_ceil(b, TILE_SIZE))
            .arg(left.as_ref())
            .arg(right.as_ref())
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }
}

// comparison

#[derive(Clone)]
pub struct ArrayBoolean<T, L, R> {
    left: L,
    right: R,
    host_cmp: fn(T, T) -> bool,
    #[cfg(feature = "opencl")]
    kernel_cmp: ocl::Program,
}

impl<T: CDatatype, L: NDArray<DType = T>, R: NDArray<DType = T>> ArrayBoolean<T, L, R> {
    #[allow(unused_variables)]
    fn new(
        left: L,
        right: R,
        host_cmp: fn(T, T) -> bool,
        kernel_cmp: &'static str,
    ) -> Result<Self, Error> {
        debug_assert_eq!(left.shape(), right.shape());

        #[cfg(feature = "opencl")]
        let kernel_cmp = cl_programs::elementwise_boolean::<T>(kernel_cmp, left.context())?;

        Ok(Self {
            left,
            right,
            host_cmp,
            #[cfg(feature = "opencl")]
            kernel_cmp,
        })
    }

    pub fn and(left: L, right: R) -> Result<Self, Error> {
        fn and<T: CDatatype>(l: T, r: T) -> bool {
            (l != T::zero()) && (r != T::zero())
        }

        Self::new(left, right, and, "&&")
    }

    pub fn or(left: L, right: R) -> Result<Self, Error> {
        fn or<T: CDatatype>(l: T, r: T) -> bool {
            (l != T::zero()) || (r != T::zero())
        }

        Self::new(left, right, or, "||")
    }

    pub fn xor(left: L, right: R) -> Result<Self, Error> {
        fn xor<T: CDatatype>(l: T, r: T) -> bool {
            (l != T::zero()) ^ (r != T::zero())
        }

        Self::new(left, right, xor, "^")
    }
}

impl<T: CDatatype, L, R> Op for ArrayBoolean<T, L, R>
where
    L: NDArrayRead<DType = T>,
    R: NDArrayRead<DType = T>,
{
    type Out = u8;

    fn context(&self) -> &Context {
        self.left.context()
    }

    fn enqueue_cpu(&self, queue: &Queue) -> Result<Vec<Self::Out>, Error> {
        let (left, right) = try_join_read(&self.left, &self.right, queue)?;

        let output = left
            .as_ref()
            .par_iter()
            .copied()
            .zip(right.as_ref().par_iter().copied())
            .map(|(l, r)| (self.host_cmp)(l, r))
            .map(|cmp| if cmp { 1 } else { 0 })
            .collect();

        Ok(output)
    }

    #[cfg(feature = "opencl")]
    fn enqueue_cl(&self, queue: &Queue) -> Result<ocl::Buffer<Self::Out>, Error> {
        let right_queue = queue.split(self.right.size())?;
        let right = self.right.to_cl_buffer(&right_queue)?;

        let left = self.left.to_cl_buffer(queue)?;
        debug_assert_eq!(left.len(), right.len());

        let cl_queue = left.as_ref().default_queue().expect("queue");

        let output = ocl::Buffer::builder()
            .queue(cl_queue.clone())
            .len(left.len())
            .build()?;

        let kernel = ocl::Kernel::builder()
            .name("elementwise_boolean")
            .program(&self.kernel_cmp)
            .queue(cl_queue.clone())
            .global_work_size(output.len())
            .arg(left.as_ref())
            .arg(right.as_ref())
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? };

        Ok(output)
    }
}

#[derive(Clone)]
pub struct ArrayBooleanScalar<L, T> {
    left: L,
    right: T,
    host_cmp: fn(T, T) -> bool,
    #[cfg(feature = "opencl")]
    kernel_cmp: ocl::Program,
}

impl<L: NDArray, T: CDatatype> ArrayBooleanScalar<L, T> {
    #[allow(unused_variables)]
    fn new(
        left: L,
        right: T,
        host_cmp: fn(T, T) -> bool,
        kernel_cmp: &'static str,
    ) -> Result<Self, Error> {
        #[cfg(feature = "opencl")]
        let kernel_cmp = cl_programs::scalar_boolean::<T>(kernel_cmp, left.context())?;

        Ok(Self {
            left,
            right,
            host_cmp,
            #[cfg(feature = "opencl")]
            kernel_cmp,
        })
    }

    pub fn and(left: L, right: T) -> Result<Self, Error> {
        fn and<T: CDatatype>(l: T, r: T) -> bool {
            (l != T::zero()) && (r != T::zero())
        }

        Self::new(left, right, and, "&&")
    }

    pub fn or(left: L, right: T) -> Result<Self, Error> {
        fn or<T: CDatatype>(l: T, r: T) -> bool {
            (l != T::zero()) || (r != T::zero())
        }

        Self::new(left, right, or, "||")
    }

    pub fn xor(left: L, right: T) -> Result<Self, Error> {
        fn xor<T: CDatatype>(l: T, r: T) -> bool {
            (l != T::zero()) ^ (r != T::zero())
        }

        Self::new(left, right, xor, "^")
    }
}

impl<L: NDArrayRead<DType = T>, T: CDatatype> Op for ArrayBooleanScalar<L, T> {
    type Out = u8;

    fn context(&self) -> &Context {
        self.left.context()
    }

    fn enqueue_cpu(&self, queue: &Queue) -> Result<Vec<Self::Out>, Error> {
        let left = self.left.to_host(queue)?;

        let output = left
            .as_ref()
            .par_iter()
            .copied()
            .map(|l| (self.host_cmp)(l, self.right))
            .map(|cmp| if cmp { 1 } else { 0 })
            .collect();

        Ok(output)
    }

    #[cfg(feature = "opencl")]
    fn enqueue_cl(&self, queue: &Queue) -> Result<ocl::Buffer<Self::Out>, Error> {
        let left = self.left.to_cl_buffer(queue)?;
        let cl_queue = left.as_ref().default_queue().expect("queue");

        let output = ocl::Buffer::builder()
            .queue(cl_queue.clone())
            .len(left.len())
            .build()?;

        let kernel = ocl::Kernel::builder()
            .name("scalar_boolean")
            .program(&self.kernel_cmp)
            .queue(cl_queue.clone())
            .global_work_size(output.len())
            .arg(left.as_ref())
            .arg(self.right)
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? };

        Ok(output)
    }
}

#[derive(Clone)]
pub struct ArrayCompare<T, L, R> {
    left: L,
    right: R,
    host_cmp: fn(&T, &T) -> bool,
    #[cfg(feature = "opencl")]
    kernel_cmp: ocl::Program,
}

impl<T: CDatatype, L: NDArray<DType = T>, R: NDArray<DType = T>> ArrayCompare<T, L, R> {
    #[allow(unused_variables)]
    fn new(
        left: L,
        right: R,
        host_cmp: fn(&T, &T) -> bool,
        kernel_cmp: &'static str,
    ) -> Result<Self, Error> {
        debug_assert_eq!(left.shape(), right.shape());

        #[cfg(feature = "opencl")]
        let kernel_cmp = cl_programs::elementwise_cmp::<T>(kernel_cmp, left.context())?;

        Ok(Self {
            left,
            right,
            host_cmp,
            #[cfg(feature = "opencl")]
            kernel_cmp,
        })
    }

    pub fn eq(left: L, right: R) -> Result<Self, Error> {
        Self::new(left, right, PartialEq::eq, "==")
    }

    pub fn gt(left: L, right: R) -> Result<Self, Error> {
        Self::new(left, right, PartialOrd::gt, ">")
    }

    pub fn ge(left: L, right: R) -> Result<Self, Error> {
        Self::new(left, right, PartialOrd::ge, ">=")
    }

    pub fn lt(left: L, right: R) -> Result<Self, Error> {
        Self::new(left, right, PartialOrd::lt, "<")
    }

    pub fn le(left: L, right: R) -> Result<Self, Error> {
        Self::new(left, right, PartialOrd::le, "<=")
    }

    pub fn ne(left: L, right: R) -> Result<Self, Error> {
        Self::new(left, right, PartialEq::ne, "!=")
    }
}

impl<T, L, R> Op for ArrayCompare<T, L, R>
where
    T: CDatatype,
    L: NDArrayRead<DType = T>,
    R: NDArrayRead<DType = T>,
{
    type Out = u8;

    fn context(&self) -> &Context {
        self.left.context()
    }

    fn enqueue_cpu(&self, queue: &Queue) -> Result<Vec<Self::Out>, Error> {
        let (left, right) = try_join_read(&self.left, &self.right, queue)?;
        debug_assert_eq!(left.len(), right.len());

        let output = left
            .as_ref()
            .par_iter()
            .zip(right.as_ref().par_iter())
            .map(|(l, r)| (self.host_cmp)(l, r))
            .map(|cmp| if cmp { 1 } else { 0 })
            .collect();

        Ok(output)
    }

    #[cfg(feature = "opencl")]
    fn enqueue_cl(&self, queue: &Queue) -> Result<ocl::Buffer<Self::Out>, Error> {
        let right_queue = queue.split(self.right.size())?;
        let right = self.right.to_cl_buffer(&right_queue)?;
        let left = self.left.to_cl_buffer(queue)?;
        debug_assert_eq!(left.len(), right.len());

        let cl_queue = left.as_ref().default_queue().expect("left queue").clone();

        let output = ocl::Buffer::builder()
            .queue(cl_queue.clone())
            .len(left.len())
            .build()?;

        let kernel = ocl::Kernel::builder()
            .name("elementwise_cmp")
            .program(&self.kernel_cmp)
            .queue(cl_queue)
            .global_work_size(output.len())
            .arg(left.as_ref())
            .arg(right.as_ref())
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }
}

#[derive(Clone)]
pub struct ArrayCompareScalar<T, A> {
    array: A,
    scalar: T,
    host_cmp: fn(&T, &T) -> bool,
    #[cfg(feature = "opencl")]
    kernel_cmp: ocl::Program,
}

impl<T: CDatatype, A: NDArray> ArrayCompareScalar<T, A> {
    #[allow(unused_variables)]
    fn new(
        array: A,
        scalar: T,
        host_cmp: fn(&T, &T) -> bool,
        kernel_cmp: &'static str,
    ) -> Result<Self, Error> {
        #[cfg(feature = "opencl")]
        let kernel_cmp = cl_programs::scalar_cmp::<T>(kernel_cmp, array.context())?;

        Ok(Self {
            array,
            scalar,
            host_cmp,
            #[cfg(feature = "opencl")]
            kernel_cmp,
        })
    }

    pub fn eq(array: A, scalar: T) -> Result<Self, Error> {
        Self::new(array, scalar, PartialEq::eq, "==")
    }

    pub fn gt(array: A, scalar: T) -> Result<Self, Error> {
        Self::new(array, scalar, PartialOrd::gt, ">")
    }

    pub fn ge(array: A, scalar: T) -> Result<Self, Error> {
        Self::new(array, scalar, PartialOrd::ge, ">=")
    }

    pub fn lt(array: A, scalar: T) -> Result<Self, Error> {
        Self::new(array, scalar, PartialOrd::lt, "<")
    }

    pub fn le(array: A, scalar: T) -> Result<Self, Error> {
        Self::new(array, scalar, PartialOrd::le, "<=")
    }

    pub fn ne(array: A, scalar: T) -> Result<Self, Error> {
        Self::new(array, scalar, PartialEq::ne, "!=")
    }
}

impl<T: CDatatype, A: NDArrayRead<DType = T>> Op for ArrayCompareScalar<T, A> {
    type Out = u8;

    fn context(&self) -> &Context {
        self.array.context()
    }

    fn enqueue_cpu(&self, queue: &Queue) -> Result<Vec<Self::Out>, Error> {
        let input = self.array.to_host(queue)?;

        let output = input
            .as_ref()
            .par_iter()
            .map(|n| (self.host_cmp)(n, &self.scalar))
            .map(|cmp| if cmp { 1 } else { 0 })
            .collect();

        Ok(output)
    }

    #[cfg(feature = "opencl")]
    fn enqueue_cl(&self, queue: &Queue) -> Result<ocl::Buffer<Self::Out>, Error> {
        let input = self.array.to_cl_buffer(queue)?;
        let cl_queue = input.as_ref().default_queue().expect("queue").clone();

        let output = ocl::Buffer::builder()
            .queue(cl_queue.clone())
            .len(input.len())
            .build()?;

        let kernel = ocl::Kernel::builder()
            .name("scalar_cmp")
            .program(&self.kernel_cmp)
            .queue(cl_queue)
            .global_work_size(input.len())
            .arg(input.as_ref())
            .arg(self.scalar)
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }
}

// reduction

#[derive(Copy, Clone)]
pub struct ArrayReduceAxis<T, A> {
    source: A,
    axis: usize,
    host_reduce: fn(T, T) -> T,
    #[allow(unused)]
    kernel_reduce: &'static str,
}

impl<T: CDatatype, A: NDArray<DType = T>> ArrayReduceAxis<T, A> {
    fn new(
        source: A,
        axis: usize,
        host_reduce: fn(T, T) -> T,
        kernel_reduce: &'static str,
    ) -> Self {
        debug_assert!(axis < source.ndim());

        Self {
            source,
            axis,
            host_reduce,
            kernel_reduce,
        }
    }

    pub fn max(source: A, axis: usize) -> Self {
        fn max<T: PartialOrd>(l: T, r: T) -> T {
            if r > l {
                r
            } else {
                l
            }
        }

        Self::new(source, axis, max, "max")
    }

    pub fn min(source: A, axis: usize) -> Self {
        fn min<T: PartialOrd>(l: T, r: T) -> T {
            if r < l {
                r
            } else {
                l
            }
        }

        Self::new(source, axis, min, "min")
    }

    pub fn product(source: A, axis: usize) -> Self {
        Self::new(source, axis, Mul::mul, "mul")
    }

    pub fn sum(source: A, axis: usize) -> Self {
        Self::new(source, axis, Add::add, "add")
    }
}

impl<T: CDatatype, A: NDArrayRead<DType = T>> Op for ArrayReduceAxis<T, A> {
    type Out = T;

    fn context(&self) -> &Context {
        self.source.context()
    }

    fn enqueue_cpu(&self, queue: &Queue) -> Result<Vec<Self::Out>, Error> {
        let input = self.source.to_host(queue)?;
        debug_assert!(!input.as_ref().is_empty());

        let reduce_dim = self.source.shape()[self.axis];

        let output = input
            .as_ref()
            .par_chunks_exact(reduce_dim)
            .map(|chunk| {
                // encourage the compiler to vectorize the reduction
                let reduced = chunk
                    .chunks(8)
                    .map(|cc| {
                        let reduced = cc
                            .into_iter()
                            .copied()
                            .reduce(self.host_reduce)
                            .expect("reduce chunk");

                        reduced
                    })
                    .reduce(self.host_reduce)
                    .expect("reduce");

                reduced
            })
            .collect();

        Ok(output)
    }

    #[cfg(feature = "opencl")]
    fn enqueue_cl(&self, queue: &Queue) -> Result<ocl::Buffer<Self::Out>, Error> {
        let input = self.source.to_cl_buffer(queue)?;
        let cl_queue = input.as_ref().default_queue().expect("queue").clone();
        let output = cl_programs::reduce_axis(
            A::DType::zero(),
            self.kernel_reduce,
            cl_queue,
            input.as_ref(),
            self.source.shape(),
            self.axis,
        )?;

        Ok(output)
    }
}

// other unary ops

#[derive(Clone)]
pub struct ArrayCast<A, O> {
    source: A,
    dtype: PhantomData<O>,
    #[cfg(feature = "opencl")]
    kernel_op: ocl::Program,
}

impl<A: NDArray, O: CDatatype> ArrayCast<A, O> {
    pub fn new(source: A) -> Result<Self, Error> {
        #[cfg(feature = "opencl")]
        let kernel_op = cl_programs::cast::<A::DType, O>(source.context())?;

        Ok(Self {
            source,
            dtype: PhantomData,
            #[cfg(feature = "opencl")]
            kernel_op,
        })
    }
}

impl<A: NDArrayRead, O: CDatatype> Op for ArrayCast<A, O> {
    type Out = O;

    fn context(&self) -> &Context {
        self.source.context()
    }

    fn enqueue_cpu(&self, queue: &Queue) -> Result<Vec<Self::Out>, Error> {
        let input = self.source.to_host(queue)?;

        let output = input
            .as_ref()
            .par_iter()
            .copied()
            .map(|n| n.to_f64())
            .map(|f| O::from_f64(f))
            .collect();

        Ok(output)
    }

    #[cfg(feature = "opencl")]
    fn enqueue_cl(&self, queue: &Queue) -> Result<ocl::Buffer<Self::Out>, Error> {
        let input = self.source.to_cl_buffer(queue)?;
        let cl_queue = input.as_ref().default_queue().expect("queue");

        let output = ocl::Buffer::builder()
            .queue(cl_queue.clone())
            .len(input.len())
            .build()?;

        let kernel = ocl::Kernel::builder()
            .name("cast")
            .program(&self.kernel_op)
            .queue(cl_queue.clone())
            .global_work_size(input.len())
            .arg(input.as_ref())
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? };

        Ok(output)
    }
}

#[derive(Clone)]
pub struct ArrayUnary<IT, OT, A> {
    array: A,
    host_op: fn(IT) -> OT,
    #[cfg(feature = "opencl")]
    kernel_op: ocl::Program,
}

impl<IT: CDatatype, OT: CDatatype, A: NDArray> ArrayUnary<IT, OT, A> {
    #[allow(unused_variables)]
    fn new(array: A, host_op: fn(IT) -> OT, kernel_op: &'static str) -> Result<Self, Error> {
        #[cfg(feature = "opencl")]
        let kernel_op = cl_programs::unary::<IT, OT>(kernel_op, array.context())?;

        Ok(Self {
            array,
            host_op,
            #[cfg(feature = "opencl")]
            kernel_op,
        })
    }
}

impl<T: CDatatype, A: NDArray> ArrayUnary<T, T, A> {
    pub fn abs(array: A) -> Result<Self, Error> {
        // TODO: replace "fabs" with "abs" for integer types
        Self::new(array, T::abs, "fabs")
    }

    pub fn ln(array: A) -> Result<Self, Error> {
        // TODO: replace "logf" with "log" for integer types
        Self::new(array, |n| T::from_float(n.to_float().ln()), "logf")
    }

    pub fn exp(array: A) -> Result<Self, Error> {
        Self::new(array, |n| T::from_float(n.to_float().exp()), "exp")
    }

    pub fn round(array: A) -> Result<Self, Error> {
        Self::new(array, T::round, "round")
    }
}

impl<T: CDatatype, A: NDArray> ArrayUnary<T, T::Float, A> {
    // TODO: replace "asinf" with "asin" for integer types
    pub fn asin(array: A) -> Result<Self, Error> {
        Self::new(array, |n| n.to_float().asin(), "asinf")
    }

    // TODO: replace "sinf" with "sin" for integer types
    pub fn sin(array: A) -> Result<Self, Error> {
        Self::new(array, |n| n.to_float().sin(), "sinf")
    }

    // TODO: replace "sinhf" with "sinh" for integer types
    pub fn sinh(array: A) -> Result<Self, Error> {
        Self::new(array, |n| n.to_float().sinh(), "sinhf")
    }

    // TODO: replace "acosf" with "acos" for integer types
    pub fn acos(array: A) -> Result<Self, Error> {
        Self::new(array, |n| n.to_float().acos(), "acosf")
    }

    // TODO: replace "cosf" with "cos" for integer types
    pub fn cos(array: A) -> Result<Self, Error> {
        Self::new(array, |n| n.to_float().cos(), "cosf")
    }

    // TODO: replace "coshf" with "cosh" for integer types
    pub fn cosh(array: A) -> Result<Self, Error> {
        Self::new(array, |n| n.to_float().cosh(), "coshf")
    }

    // TODO: replace "atanf" with "atan" for integer types
    pub fn atan(array: A) -> Result<Self, Error> {
        Self::new(array, |n| n.to_float().atan(), "atanf")
    }

    // TODO: replace "tanf" with "tan" for integer types
    pub fn tan(array: A) -> Result<Self, Error> {
        Self::new(array, |n| n.to_float().tan(), "tanf")
    }

    // TODO: replace "tanhf" with "tanh" for integer types
    pub fn tanh(array: A) -> Result<Self, Error> {
        Self::new(array, |n| n.to_float().tanh(), "tanhf")
    }
}

impl<T: CDatatype, A: NDArray> ArrayUnary<T, T::Neg, A> {
    pub fn neg(array: A) -> Result<Self, Error> {
        Self::new(array, T::neg, "-")
    }
}

impl<IT: CDatatype, A: NDArray> ArrayUnary<IT, u8, A> {
    pub fn not(array: A) -> Result<Self, Error> {
        Self::new(array, IT::not, "!")
    }
}

impl<IT: Float, A: NDArray> ArrayUnary<IT, u8, A> {
    pub fn inf(array: A) -> Result<Self, Error> {
        Self::new(array, IT::is_inf, "isinf")
    }

    pub fn nan(array: A) -> Result<Self, Error> {
        Self::new(array, IT::is_nan, "isnan")
    }
}

impl<IT: CDatatype, OT: CDatatype, A: NDArrayRead<DType = IT>> Op for ArrayUnary<IT, OT, A> {
    type Out = OT;

    fn context(&self) -> &Context {
        self.array.context()
    }

    fn enqueue_cpu(&self, queue: &Queue) -> Result<Vec<Self::Out>, Error> {
        let input = self.array.to_host(queue)?;
        let output = input
            .as_ref()
            .par_iter()
            .copied()
            .map(|n| (self.host_op)(n))
            .collect();

        Ok(output)
    }

    #[cfg(feature = "opencl")]
    fn enqueue_cl(&self, queue: &Queue) -> Result<ocl::Buffer<Self::Out>, Error> {
        let input = self.array.to_cl_buffer(queue)?;
        let cl_queue = input.as_ref().default_queue().expect("queue").clone();

        let output = ocl::Buffer::builder()
            .queue(cl_queue.clone())
            .len(input.len())
            .build()?;

        let kernel = ocl::Kernel::builder()
            .name("unary")
            .program(&self.kernel_op)
            .queue(cl_queue)
            .global_work_size(output.len())
            .arg(input.as_ref())
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }
}

// gather ops

#[derive(Clone)]
pub struct GatherCond<A, T, L, R> {
    cond: A,
    then: L,
    or_else: R,
    dtype: PhantomData<T>,
    #[cfg(feature = "opencl")]
    kernel_op: ocl::Program,
}

impl<A: NDArray<DType = u8>, T: CDatatype, L, R> GatherCond<A, T, L, R> {
    pub fn new(cond: A, then: L, or_else: R) -> Result<Self, Error> {
        #[cfg(feature = "opencl")]
        let kernel_op = cl_programs::gather_cond::<T>(cond.context())?;

        Ok(Self {
            cond,
            then,
            or_else,
            dtype: PhantomData,
            #[cfg(feature = "opencl")]
            kernel_op,
        })
    }
}

impl<A, T, L, R> Op for GatherCond<A, T, L, R>
where
    A: NDArrayRead<DType = u8>,
    T: CDatatype,
    L: NDArrayRead<DType = T>,
    R: NDArrayRead<DType = T>,
{
    type Out = T;

    fn context(&self) -> &Context {
        self.cond.context()
    }

    fn enqueue_cpu(&self, queue: &Queue) -> Result<Vec<Self::Out>, Error> {
        let (cond, (left, right)) = try_join(
            || self.cond.to_host(queue),
            || try_join_read(&self.then, &self.or_else, queue),
        )?;

        debug_assert_eq!(cond.len(), left.len());
        debug_assert_eq!(cond.len(), right.len());

        let lr = left
            .as_ref()
            .par_iter()
            .copied()
            .zip(right.as_ref().par_iter().copied());

        let output = cond
            .as_ref()
            .par_iter()
            .copied()
            .zip(lr)
            .map(
                |(when, (then, or_else))| {
                    if when == 0 {
                        or_else
                    } else {
                        then
                    }
                },
            )
            .collect();

        Ok(output)
    }

    #[cfg(feature = "opencl")]
    fn enqueue_cl(&self, queue: &Queue) -> Result<ocl::Buffer<Self::Out>, Error> {
        let cond = self.cond.to_cl_buffer(queue)?;
        let then = self.then.to_cl_buffer(&queue.split(self.then.size())?)?;
        let or_else = self
            .or_else
            .to_cl_buffer(&queue.split(self.or_else.size())?)?;

        debug_assert_eq!(cond.len(), then.len());
        debug_assert_eq!(cond.len(), or_else.len());

        let cl_queue = cond.as_ref().default_queue().expect("queue");

        let output = ocl::Buffer::builder()
            .queue(cl_queue.clone())
            .len(cond.len())
            .build()?;

        let kernel = ocl::Kernel::builder()
            .name("gather_cond")
            .queue(cl_queue.clone())
            .program(&self.kernel_op)
            .global_work_size(cond.len())
            .arg(cond.as_ref())
            .arg(then.as_ref())
            .arg(or_else.as_ref())
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }
}

#[inline]
fn try_join<LFn, LRT, RFn, RRT>(left: LFn, right: RFn) -> Result<(LRT, RRT), Error>
where
    LFn: FnOnce() -> Result<LRT, Error> + Send + Sync,
    RFn: FnOnce() -> Result<RRT, Error> + Send + Sync,
    LRT: Send + Sync,
    RRT: Send + Sync,
{
    let (left, right) = rayon::join(left, right);
    Ok((left?, right?))
}

#[inline]
fn try_join_read<'a, L: NDArrayRead, R: NDArrayRead>(
    left: &'a L,
    right: &'a R,
    queue: &Queue,
) -> Result<(SliceConverter<'a, L::DType>, SliceConverter<'a, R::DType>), Error> {
    try_join(|| left.to_host(queue), || right.to_host(queue))
}
