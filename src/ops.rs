use std::cmp::{PartialEq, PartialOrd};
use std::f32::consts::PI;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Rem, Sub};

use rand::Rng;
use rayon::prelude::*;

#[cfg(feature = "opencl")]
use super::cl_programs;
use super::{Buffer, CDatatype, Context, DeviceQueue, Error, Float, NDArray, NDArrayRead, Queue};

pub trait Op: Send + Sync {
    type Out: CDatatype;

    fn context(&self) -> &Context;

    fn enqueue(&self, queue: &Queue) -> Result<Buffer<Self::Out>, Error> {
        match queue.device_queue() {
            DeviceQueue::Host => self.enqueue_cpu(queue).map(Buffer::Host),
            #[cfg(feature = "opencl")]
            DeviceQueue::CL(_) => self.enqueue_cl(queue).map(Buffer::CL),
        }
    }

    fn enqueue_cpu(&self, queue: &Queue) -> Result<Vec<Self::Out>, Error>;

    #[cfg(feature = "opencl")]
    fn enqueue_cl(&self, queue: &Queue) -> Result<ocl::Buffer<Self::Out>, Error>;
}

// constructors

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
        #[cfg(feature = "opencl")]
        let kernel_op = cl_programs::random_normal(&context)?;

        Ok(Self {
            context,
            size,
            #[cfg(feature = "opencl")]
            kernel_op,
        })
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

        let cl_queue = self.context.cl_queue(queue.device_queue(), self.size)?;
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
        #[cfg(feature = "opencl")]
        let kernel_op = cl_programs::random_uniform(&context)?;

        Ok(Self {
            context,
            size,
            #[cfg(feature = "opencl")]
            kernel_op,
        })
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
        let cl_queue = self.context.cl_queue(queue.device_queue(), self.size)?;

        let output = ocl::Buffer::builder()
            .queue(cl_queue.clone())
            .len(self.size)
            .build()?;

        let kernel = ocl::Kernel::builder()
            .name("random_uniform")
            .queue(cl_queue)
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
        kernel_op: &'static str,
    ) -> Result<Self, Error> {
        #[cfg(feature = "opencl")]
        let kernel_op = cl_programs::elementwise_inplace::<T>(kernel_op, left.context())?;

        Ok(Self {
            left,
            right,
            cpu_op,
            #[cfg(feature = "opencl")]
            kernel_op,
        })
    }

    pub fn add(left: L, right: R) -> Result<Self, Error> {
        Self::new(left, right, Add::add, "+")
    }

    pub fn div(left: L, right: R) -> Result<Self, Error> {
        Self::new(left, right, Div::div, "/")
    }

    pub fn mul(left: L, right: R) -> Result<Self, Error> {
        Self::new(left, right, Mul::mul, "*")
    }

    pub fn rem(left: L, right: R) -> Result<Self, Error> {
        Self::new(left, right, Rem::rem, "%")
    }

    pub fn sub(left: L, right: R) -> Result<Self, Error> {
        Self::new(left, right, Sub::sub, "-")
    }
}

impl<T: CDatatype, L: NDArrayRead<DType = T>, R: NDArrayRead<DType = T>> Op for ArrayDual<T, L, R> {
    type Out = T;

    fn context(&self) -> &Context {
        self.left.context()
    }

    fn enqueue_cpu(&self, queue: &Queue) -> Result<Vec<T>, Error> {
        let left = self.left.to_vec(queue)?;
        let right = self.right.to_vec(queue)?;
        debug_assert_eq!(left.len(), right.len());

        let output = left
            .into_par_iter()
            .zip(right.into_par_iter())
            .map(|(l, r)| (self.cpu_op)(l, r))
            .collect();

        Ok(output)
    }

    #[cfg(feature = "opencl")]
    fn enqueue_cl(&self, queue: &Queue) -> Result<ocl::Buffer<T>, Error> {
        let right_queue = queue.context().queue(self.right.size())?;
        let right = self.right.to_cl_buffer(&right_queue)?;
        let left = self.left.to_cl_buffer(queue)?;
        debug_assert_eq!(left.len(), right.len());

        let cl_queue = left.default_queue().expect("left queue").clone();

        let kernel = ocl::Kernel::builder()
            .name("elementwise_inplace")
            .program(&self.kernel_op)
            .queue(cl_queue)
            .global_work_size(left.len())
            .arg(&left)
            .arg(right)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(left)
    }
}

#[derive(Copy, Clone)]
pub struct ArrayDualFloat<L, R> {
    left: L,
    right: R,
    op: &'static str,
}

impl<L, R> ArrayDualFloat<L, R> {
    fn new(left: L, right: R, op: &'static str) -> Self {
        Self { left, right, op }
    }

    pub fn log(left: L, right: R) -> Self {
        todo!()
    }

    pub fn pow(left: L, right: R) -> Self {
        todo!()
    }
}

impl<T, L, R> Op for ArrayDualFloat<L, R>
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
        todo!()
    }

    #[cfg(feature = "opencl")]
    fn enqueue_cl(&self, queue: &Queue) -> Result<ocl::Buffer<Self::Out>, Error> {
        let right_queue = queue.context().queue(self.right.size())?;
        let right = self.right.to_cl_buffer(&right_queue)?;
        let left = self.left.to_cl_buffer(queue)?;

        todo!()
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
        let left = self.array.to_vec(queue)?;
        let right = self.scalar;

        let output = left
            .into_par_iter()
            // chunk the input to encourage the compiler to vectorize
            .chunks(8)
            .map(|chunk| {
                chunk
                    .into_iter()
                    .map(|l| (self.host_op)(l, right))
                    .collect::<Vec<T>>()
            })
            .flatten()
            .collect();

        Ok(output)
    }

    #[cfg(feature = "opencl")]
    fn enqueue_cl(&self, queue: &Queue) -> Result<ocl::Buffer<Self::Out>, Error> {
        let left = self.array.to_cl_buffer(queue)?;
        let right = self.scalar;
        let cl_queue = left.default_queue().expect("queue").clone();

        let kernel = ocl::Kernel::builder()
            .name("elementwise_scalar")
            .program(&self.kernel_op)
            .queue(cl_queue)
            .global_work_size(left.len())
            .arg(&left)
            .arg(right)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(left)
    }
}

#[derive(Clone)]
pub struct ArrayScalarFloat<T, A> {
    array: A,
    scalar: f64,
    host_op: fn(T, f64) -> T,
    #[cfg(feature = "opencl")]
    kernel_op: ocl::Program,
}

impl<T: CDatatype, A: NDArray> ArrayScalarFloat<T, A> {
    fn new(
        array: A,
        scalar: f64,
        host_op: fn(T, f64) -> T,
        kernel_op: &'static str,
    ) -> Result<Self, Error> {
        #[cfg(feature = "opencl")]
        let kernel_op = cl_programs::elementwise_scalar::<f64, T>(kernel_op, array.context())?;

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
    pub fn log(left: A, right: f64) -> Result<Self, Error> {
        Self::new(left, right, T::log, "log")
    }

    pub fn pow(left: A, right: f64) -> Result<Self, Error> {
        Self::new(left, right, T::pow, "pow_")
    }
}

impl<T: CDatatype, A: NDArrayRead<DType = T>> Op for ArrayScalarFloat<T, A> {
    type Out = T;

    fn context(&self) -> &Context {
        self.array.context()
    }

    fn enqueue_cpu(&self, queue: &Queue) -> Result<Vec<Self::Out>, Error> {
        let left = self.array.to_vec(queue)?;
        let right = self.scalar;

        let output = left
            .into_par_iter()
            .map(|l| (self.host_op)(l, right))
            .collect();

        Ok(output)
    }

    #[cfg(feature = "opencl")]
    fn enqueue_cl(&self, queue: &Queue) -> Result<ocl::Buffer<Self::Out>, Error> {
        let left = self.array.to_cl_buffer(queue)?;
        let right = self.scalar;
        let cl_queue = left.default_queue().expect("queue").clone();

        let kernel = ocl::Kernel::builder()
            .name("elementwise_scalar")
            .program(&self.kernel_op)
            .queue(cl_queue)
            .global_work_size(left.len())
            .arg(&left)
            .arg(right)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(left)
    }
}

// linear algebra

#[derive(Copy, Clone)]
pub struct MatDiag<A> {
    source: A,
}

#[derive(Clone)]
pub struct MatMul<'a, T, L, R> {
    left: &'a L,
    right: &'a R,
    dtype: PhantomData<T>,
    #[cfg(feature = "opencl")]
    kernel_op: ocl::Program,
}

impl<'a, T: CDatatype, L: NDArray<DType = T>, R: NDArray<DType = T>> MatMul<'a, T, L, R> {
    pub fn new(left: &'a L, right: &'a R) -> Result<Self, Error> {
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

        [num_matrices, a, b, c]
    }
}

impl<'a, T, L, R> Op for MatMul<'a, T, L, R>
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

        let right_queue = queue.context().queue(self.right.size())?;
        let right = self.right.to_vec(&right_queue)?;
        let left = self.left.to_vec(queue)?;

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

        let right_queue = queue.context().queue(self.right.size())?;
        let right = self.right.to_cl_buffer(&right_queue)?;
        let left = self.left.to_cl_buffer(queue)?;

        let cl_queue = left.default_queue().expect("left queue");

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
            .arg(&left)
            .arg(&right)
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }
}

// comparison

#[derive(Clone)]
pub struct ArrayBoolean<'a, T, L, R> {
    left: &'a L,
    right: &'a R,
    host_cmp: fn(T, T) -> bool,
    #[cfg(feature = "opencl")]
    kernel_cmp: ocl::Program,
}

impl<'a, T: CDatatype, L: NDArray<DType = T>, R: NDArray<DType = T>> ArrayBoolean<'a, T, L, R> {
    fn new(
        left: &'a L,
        right: &'a R,
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

    pub fn and(left: &'a L, right: &'a R) -> Result<Self, Error> {
        fn and<T: CDatatype>(l: T, r: T) -> bool {
            (l != T::zero()) && (r != T::zero())
        }

        Self::new(left, right, and, "&&")
    }

    pub fn or(left: &'a L, right: &'a R) -> Result<Self, Error> {
        fn or<T: CDatatype>(l: T, r: T) -> bool {
            (l != T::zero()) || (r != T::zero())
        }

        Self::new(left, right, or, "||")
    }

    pub fn xor(left: &'a L, right: &'a R) -> Result<Self, Error> {
        fn xor<T: CDatatype>(l: T, r: T) -> bool {
            (l != T::zero()) ^ (r != T::zero())
        }

        Self::new(left, right, xor, "^")
    }
}

impl<'a, T: CDatatype, L, R> Op for ArrayBoolean<'a, T, L, R>
where
    L: NDArrayRead<DType = T>,
    R: NDArrayRead<DType = T>,
{
    type Out = u8;

    fn context(&self) -> &Context {
        self.left.context()
    }

    fn enqueue_cpu(&self, queue: &Queue) -> Result<Vec<Self::Out>, Error> {
        let right_queue = queue.context().queue(self.right.size())?;
        let right = self.right.to_vec(&right_queue)?;
        let left = self.left.to_vec(queue)?;

        let output = left
            .into_par_iter()
            .zip(right.into_par_iter())
            .map(|(l, r)| (self.host_cmp)(l, r))
            .map(|cmp| if cmp { 1 } else { 0 })
            .collect();

        Ok(output)
    }

    #[cfg(feature = "opencl")]
    fn enqueue_cl(&self, queue: &Queue) -> Result<ocl::Buffer<Self::Out>, Error> {
        let right_queue = queue.context().queue(self.right.size())?;
        let right = self.right.to_cl_buffer(&right_queue)?;

        let left = self.left.to_cl_buffer(queue)?;
        debug_assert_eq!(left.len(), right.len());

        let cl_queue = left.default_queue().expect("queue");

        let output = ocl::Buffer::builder()
            .queue(cl_queue.clone())
            .len(left.len())
            .build()?;

        let kernel = ocl::Kernel::builder()
            .name("elementwise_boolean")
            .program(&self.kernel_cmp)
            .queue(cl_queue.clone())
            .global_work_size(output.len())
            .arg(&left)
            .arg(&right)
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? };

        Ok(output)
    }
}

#[derive(Clone)]
pub struct ArrayCompare<'a, T, L, R> {
    left: &'a L,
    right: &'a R,
    host_cmp: fn(&T, &T) -> bool,
    #[cfg(feature = "opencl")]
    kernel_cmp: ocl::Program,
}

impl<'a, T: CDatatype, L: NDArray<DType = T>, R: NDArray<DType = T>> ArrayCompare<'a, T, L, R> {
    fn new(
        left: &'a L,
        right: &'a R,
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

    pub fn eq(left: &'a L, right: &'a R) -> Result<Self, Error> {
        Self::new(left, right, PartialEq::eq, "==")
    }

    pub fn gt(left: &'a L, right: &'a R) -> Result<Self, Error> {
        Self::new(left, right, PartialOrd::gt, ">")
    }

    pub fn ge(left: &'a L, right: &'a R) -> Result<Self, Error> {
        Self::new(left, right, PartialOrd::ge, ">=")
    }

    pub fn lt(left: &'a L, right: &'a R) -> Result<Self, Error> {
        Self::new(left, right, PartialOrd::lt, "<")
    }

    pub fn le(left: &'a L, right: &'a R) -> Result<Self, Error> {
        Self::new(left, right, PartialOrd::le, "<=")
    }

    pub fn ne(left: &'a L, right: &'a R) -> Result<Self, Error> {
        Self::new(left, right, PartialEq::ne, "!=")
    }
}

impl<'a, T, L, R> Op for ArrayCompare<'a, T, L, R>
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
        let right_queue = queue.context().queue(self.right.size())?;
        let right = self.right.to_vec(&right_queue)?;
        let left = self.left.to_vec(queue)?;
        debug_assert_eq!(left.len(), right.len());

        let output = left
            .par_iter()
            .zip(right.par_iter())
            .map(|(l, r)| (self.host_cmp)(l, r))
            .map(|cmp| if cmp { 1 } else { 0 })
            .collect();

        Ok(output)
    }

    #[cfg(feature = "opencl")]
    fn enqueue_cl(&self, queue: &Queue) -> Result<ocl::Buffer<Self::Out>, Error> {
        let right_queue = queue.context().queue(self.right.size())?;
        let right = self.right.to_cl_buffer(&right_queue)?;
        let left = self.left.to_cl_buffer(queue)?;
        debug_assert_eq!(left.len(), right.len());

        let cl_queue = left.default_queue().expect("left queue").clone();

        let output = ocl::Buffer::builder()
            .queue(cl_queue.clone())
            .len(left.len())
            .build()?;

        let kernel = ocl::Kernel::builder()
            .name("elementwise_cmp")
            .program(&self.kernel_cmp)
            .queue(cl_queue)
            .global_work_size(output.len())
            .arg(&left)
            .arg(&right)
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }
}

#[derive(Clone)]
pub struct ArrayCompareScalar<'a, T, A> {
    array: &'a A,
    scalar: T,
    host_cmp: fn(&T, &T) -> bool,
    #[cfg(feature = "opencl")]
    kernel_cmp: ocl::Program,
}

impl<'a, T: CDatatype, A: NDArray> ArrayCompareScalar<'a, T, A> {
    fn new(
        array: &'a A,
        scalar: T,
        host_cmp: fn(&T, &T) -> bool,
        kernel_cmp: &'static str,
    ) -> Result<Self, Error> {
        Ok(Self {
            array,
            scalar,
            host_cmp,
            #[cfg(feature = "opencl")]
            kernel_cmp: cl_programs::scalar_cmp::<T>(kernel_cmp, array.context())?,
        })
    }

    pub fn eq(array: &'a A, scalar: T) -> Result<Self, Error> {
        Self::new(array, scalar, PartialEq::eq, "==")
    }

    pub fn gt(array: &'a A, scalar: T) -> Result<Self, Error> {
        Self::new(array, scalar, PartialOrd::gt, ">")
    }

    pub fn ge(array: &'a A, scalar: T) -> Result<Self, Error> {
        Self::new(array, scalar, PartialOrd::ge, ">=")
    }

    pub fn lt(array: &'a A, scalar: T) -> Result<Self, Error> {
        Self::new(array, scalar, PartialOrd::lt, "<")
    }

    pub fn le(array: &'a A, scalar: T) -> Result<Self, Error> {
        Self::new(array, scalar, PartialOrd::le, "<=")
    }

    pub fn ne(array: &'a A, scalar: T) -> Result<Self, Error> {
        Self::new(array, scalar, PartialEq::ne, "!=")
    }
}

impl<'a, T: CDatatype, A: NDArrayRead<DType = T>> Op for ArrayCompareScalar<'a, T, A> {
    type Out = u8;

    fn context(&self) -> &Context {
        self.array.context()
    }

    fn enqueue_cpu(&self, queue: &Queue) -> Result<Vec<Self::Out>, Error> {
        let input = self.array.to_vec(queue)?;

        let output = input
            .par_iter()
            .map(|n| (self.host_cmp)(n, &self.scalar))
            .map(|cmp| if cmp { 1 } else { 0 })
            .collect();

        Ok(output)
    }

    #[cfg(feature = "opencl")]
    fn enqueue_cl(&self, queue: &Queue) -> Result<ocl::Buffer<Self::Out>, Error> {
        let input = self.array.to_cl_buffer(queue)?;
        let cl_queue = input.default_queue().expect("queue").clone();

        let output = ocl::Buffer::builder()
            .queue(cl_queue.clone())
            .len(input.len())
            .build()?;

        let kernel = ocl::Kernel::builder()
            .name("scalar_cmp")
            .program(&self.kernel_cmp)
            .queue(cl_queue)
            .global_work_size(input.len())
            .arg(&input)
            .arg(self.scalar)
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }
}

// reduction

#[derive(Copy, Clone)]
pub struct ArrayReduceAxis<'a, T, A> {
    source: &'a A,
    axis: usize,
    host_reduce: fn(T, T) -> T,
    kernel_reduce: &'static str,
}

impl<'a, T: CDatatype, A: NDArray<DType = T>> ArrayReduceAxis<'a, T, A> {
    fn new(
        source: &'a A,
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

    pub fn sum(source: &'a A, axis: usize) -> Self {
        Self::new(source, axis, Add::add, "add")
    }
}

impl<'a, T: CDatatype, A: NDArrayRead<DType = T>> Op for ArrayReduceAxis<'a, T, A> {
    type Out = T;

    fn context(&self) -> &Context {
        self.source.context()
    }

    fn enqueue_cpu(&self, queue: &Queue) -> Result<Vec<Self::Out>, Error> {
        let input = self.source.to_vec(queue)?;

        debug_assert!(!input.is_empty());

        let reduce_dim = self.source.shape()[self.axis];

        let output = input
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
        let cl_queue = input.default_queue().expect("queue").clone();
        let output = cl_programs::reduce_axis(
            A::DType::zero(),
            self.kernel_reduce,
            cl_queue,
            input,
            self.source.shape(),
            self.axis,
        )?;

        Ok(output)
    }
}

// other unary ops

#[derive(Clone)]
pub struct ArrayCast<'a, A, O> {
    source: &'a A,
    dtype: PhantomData<O>,
    #[cfg(feature = "opencl")]
    kernel_op: ocl::Program,
}

impl<'a, A: NDArray, O: CDatatype> ArrayCast<'a, A, O> {
    pub fn new(source: &'a A) -> Result<Self, Error> {
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

impl<'a, A: NDArrayRead, O: CDatatype> Op for ArrayCast<'a, A, O> {
    type Out = O;

    fn context(&self) -> &Context {
        self.source.context()
    }

    fn enqueue_cpu(&self, queue: &Queue) -> Result<Vec<Self::Out>, Error> {
        let input = self.source.to_vec(queue)?;

        let output = input
            .into_par_iter()
            .map(|n| n.to_f64())
            .map(|float| O::from_f64(float))
            .collect();

        Ok(output)
    }

    #[cfg(feature = "opencl")]
    fn enqueue_cl(&self, queue: &Queue) -> Result<ocl::Buffer<Self::Out>, Error> {
        let input = self.source.to_cl_buffer(queue)?;
        let cl_queue = input.default_queue().expect("queue");

        let output = ocl::Buffer::builder()
            .queue(cl_queue.clone())
            .len(input.len())
            .build()?;

        let kernel = ocl::Kernel::builder()
            .name("cast")
            .program(&self.kernel_op)
            .queue(cl_queue.clone())
            .global_work_size(input.len())
            .arg(input)
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

    pub fn exp(array: A) -> Result<Self, Error> {
        Self::new(array, T::exp, "exp")
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
        let input = self.array.to_vec(queue)?;
        let output = input.into_par_iter().map(|n| (self.host_op)(n)).collect();

        Ok(output)
    }

    #[cfg(feature = "opencl")]
    fn enqueue_cl(&self, queue: &Queue) -> Result<ocl::Buffer<Self::Out>, Error> {
        let input = self.array.to_cl_buffer(queue)?;
        let cl_queue = input.default_queue().expect("queue").clone();

        let output = ocl::Buffer::builder()
            .queue(cl_queue.clone())
            .len(input.len())
            .build()?;

        let kernel = ocl::Kernel::builder()
            .name("unary")
            .program(&self.kernel_op)
            .queue(cl_queue)
            .global_work_size(output.len())
            .arg(&input)
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }
}
