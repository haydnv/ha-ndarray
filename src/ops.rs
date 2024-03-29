//! N-dimensional array [`Op`]s

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
    offset_of, strides_for, Buffer, CDatatype, Context, Error, Float, Log, NDArray, NDArrayMath,
    NDArrayRead, NDArrayTransform, Queue, Shape, SliceConverter, Trig,
};

/// An n-dimensional array [`Op`]
pub trait Op: Send + Sync {
    /// The output data type of this [`Op`]
    type Out: CDatatype;

    /// The execution context of this [`Op`]
    fn context(&self) -> &Context;

    /// Enqueue this [`Op`] for execution.
    fn enqueue(&self, queue: &Queue) -> Result<Buffer<Self::Out>, Error> {
        // TODO: there must be a better way to do this
        #[cfg(feature = "opencl")]
        if queue.cl_queue.is_some() {
            return self.enqueue_cl(queue).map(Buffer::CL);
        }

        self.enqueue_cpu(queue).map(Buffer::Host)
    }

    /// Enqueue this [`Op`] on the host CPU.
    fn enqueue_cpu(&self, queue: &Queue) -> Result<Vec<Self::Out>, Error>;

    /// Enqueue this [`Op`] on an OpenCL device.
    #[cfg(feature = "opencl")]
    fn enqueue_cl(&self, queue: &Queue) -> Result<ocl::Buffer<Self::Out>, Error>;

    /// Read the result of this [`Op`] at a single `coord`.
    fn read_value(&self, coord: &[usize]) -> Result<Self::Out, Error>;
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

    fn read_value(&self, coord: &[usize]) -> Result<Self::Out, Error> {
        (**self).read_value(coord)
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

    fn read_value(&self, coord: &[usize]) -> Result<Self::Out, Error> {
        (**self).read_value(coord)
    }
}

// constructors

/// A range constructor
#[derive(Clone)]
pub struct Range<T> {
    context: Context,
    shape: Shape,
    start: T,
    step: f64,
    #[cfg(feature = "opencl")]
    cl_op: ocl::Program,
}

impl<T: CDatatype + fmt::Display> Range<T> {
    /// Initialize a new [`Range`] constructor.
    pub fn new(start: T, stop: T, shape: Shape) -> Result<Self, Error> {
        let context = Context::default()?;
        Self::with_context(context, start, stop, shape)
    }

    /// Initialize a new [`Range`] constructor with the given [`Context`].
    pub fn with_context(context: Context, start: T, stop: T, shape: Shape) -> Result<Self, Error> {
        let size = shape.iter().product::<usize>();
        let step = if start < stop {
            Ok((stop - start).to_f64() / (size as f64))
        } else {
            Err(Error::Bounds(format!("invalid range: [{start}..{stop})")))
        }?;

        #[cfg(feature = "opencl")]
        let cl_op = cl_programs::range::<T>(&context)?;

        Ok(Self {
            context,
            shape,
            start,
            step,
            #[cfg(feature = "opencl")]
            cl_op,
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
        let size = self.shape.iter().product::<usize>();

        let buffer = (0..size)
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
            .len(self.shape.iter().product::<usize>())
            .build()?;

        let kernel = ocl::Kernel::builder()
            .name("range")
            .queue(cl_queue.clone())
            .program(&self.cl_op)
            .global_work_size(buffer.len())
            .arg(self.step)
            .arg(&buffer)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(buffer)
    }

    fn read_value(&self, coord: &[usize]) -> Result<Self::Out, Error> {
        if coord.len() == self.shape.len() && coord.iter().zip(&self.shape).all(|(i, dim)| i < dim)
        {
            let offset = offset_of(coord, &self.shape);

            Ok(T::from_f64(
                self.start.to_f64() + (self.step * offset as f64),
            ))
        } else {
            Err(Error::Bounds(format!(
                "range constructor with shape {:?} does not contain {:?}",
                self.shape, coord
            )))
        }
    }
}

/// A random normal constructor
#[derive(Clone)]
pub struct RandomNormal {
    context: Context,
    size: usize,
    #[cfg(feature = "opencl")]
    cl_op: ocl::Program,
}

impl RandomNormal {
    /// Initialize a new [`RandomNormal`] constructor.
    pub fn new(size: usize) -> Result<Self, Error> {
        let context = Context::default()?;
        Self::with_context(context, size)
    }

    /// Initialize a new [`RandomNormal`] constructor with the given [`Context`].
    pub fn with_context(context: Context, size: usize) -> Result<Self, Error> {
        #[cfg(feature = "opencl")]
        let cl_op = cl_programs::random_normal(&context)?;

        Ok(Self {
            context,
            size,
            #[cfg(feature = "opencl")]
            cl_op,
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
            .program(&self.cl_op)
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

    fn read_value(&self, _coord: &[usize]) -> Result<Self::Out, Error> {
        Err(Error::Bounds(format!(
            "cannot read an individual value of a random normal distribution"
        )))
    }
}

/// A random uniform constructor
#[derive(Clone)]
pub struct RandomUniform {
    context: Context,
    shape: Shape,
    #[cfg(feature = "opencl")]
    cl_op: ocl::Program,
}

impl RandomUniform {
    /// Initialize a new [`RandomUniform`] constructor.
    pub fn new(shape: Shape) -> Result<Self, Error> {
        let context = Context::default()?;
        Self::with_context(context, shape)
    }

    /// Initialize a new [`RandomUniform`] constructor with the given [`Context`].
    pub fn with_context(context: Context, shape: Shape) -> Result<Self, Error> {
        #[cfg(feature = "opencl")]
        let cl_op = cl_programs::random_uniform(&context)?;

        Ok(Self {
            context,
            shape,
            #[cfg(feature = "opencl")]
            cl_op,
        })
    }
}

impl Op for RandomUniform {
    type Out = f32;

    fn context(&self) -> &Context {
        &self.context
    }

    fn enqueue_cpu(&self, _queue: &Queue) -> Result<Vec<Self::Out>, Error> {
        let size = self.shape.iter().product();
        let mut data = vec![0.; size];
        rand::thread_rng().fill(&mut data[..]);
        Ok(data)
    }

    #[cfg(feature = "opencl")]
    fn enqueue_cl(&self, queue: &Queue) -> Result<ocl::Buffer<Self::Out>, Error> {
        let seed: u32 = rand::thread_rng().gen();
        let cl_queue = queue.cl_queue.as_ref().expect("queue");

        let size = self.shape.iter().product::<usize>();
        let output = ocl::Buffer::builder()
            .queue(cl_queue.clone())
            .len(size)
            .build()?;

        let kernel = ocl::Kernel::builder()
            .name("random_uniform")
            .queue(cl_queue.clone())
            .program(&self.cl_op)
            .global_work_size(output.len())
            .arg(u64::try_from(seed).expect("seed"))
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }

    fn read_value(&self, coord: &[usize]) -> Result<Self::Out, Error> {
        if coord.len() == self.shape.len() && coord.iter().zip(&self.shape).all(|(i, dim)| i < dim)
        {
            Ok(rand::thread_rng().gen())
        } else {
            Err(Error::Bounds(format!(
                "range constructor for shape {:?} does not contain {:?}",
                self.shape, coord
            )))
        }
    }
}

// arithmetic

/// A dual-array [`Op`]
#[derive(Clone)]
pub struct ArrayDual<T, L, R> {
    left: L,
    right: R,
    cpu_op: fn(T, T) -> T,
    #[cfg(feature = "opencl")]
    cl_op: ocl::Program,
}

impl<T: CDatatype, L: NDArray, R: NDArray> ArrayDual<T, L, R> {
    fn new(
        left: L,
        right: R,
        cpu_op: fn(T, T) -> T,
        #[allow(unused_variables)] cl_op: &'static str,
    ) -> Result<Self, Error> {
        #[cfg(feature = "opencl")]
        let cl_op = cl_programs::elementwise_dual::<T, T>(cl_op, left.context())?;

        Ok(Self {
            left,
            right,
            cpu_op,
            #[cfg(feature = "opencl")]
            cl_op,
        })
    }

    /// Initialize an addition [`Op`].
    pub fn add(left: L, right: R) -> Result<Self, Error> {
        Self::new(left, right, Add::add, "add")
    }

    /// Initialize a division [`Op`] which will return an error if `right` contains zeros.
    pub fn checked_div(left: L, right: R) -> Result<Self, Error> {
        Self::new(
            left,
            right,
            |l, r| if r == T::zero() { T::zero() } else { l / r },
            "checked_div",
        )
    }

    /// Initialize a division [`Op`] with undefined behavior if `right` contains zeros.
    pub fn div(left: L, right: R) -> Result<Self, Error> {
        Self::new(left, right, Div::div, "div")
    }

    /// Initialize a multiplication [`Op`].
    pub fn mul(left: L, right: R) -> Result<Self, Error> {
        Self::new(left, right, Mul::mul, "mul")
    }

    /// Initialize a modulo [`Op`].
    pub fn rem(left: L, right: R) -> Result<Self, Error> {
        Self::new(left, right, Rem::rem, "rem")
    }

    /// Initialize a subtraction [`Op`].
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
            .program(&self.cl_op)
            .queue(cl_queue)
            .global_work_size(left.len())
            .arg(left.as_ref())
            .arg(right.as_ref())
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }

    fn read_value(&self, coord: &[usize]) -> Result<Self::Out, Error> {
        let (left, right) = try_join(
            || self.left.read_value(coord),
            || self.right.read_value(coord),
        )?;

        Ok((self.cpu_op)(left, right))
    }
}

/// A dual floating-point array [`Op`]
#[derive(Clone)]
pub struct ArrayDualFloat<T: CDatatype, L, R> {
    left: L,
    right: R,
    cpu_op: fn(T, T::Float) -> T,
    #[cfg(feature = "opencl")]
    cl_op: ocl::Program,
}

impl<T: CDatatype, L: NDArray<DType = T>, R: NDArray<DType = T::Float>> ArrayDualFloat<T, L, R> {
    #[allow(unused_variables)]
    fn new(
        left: L,
        right: R,
        cpu_op: fn(T, T::Float) -> T,
        cl_op: &'static str,
    ) -> Result<Self, Error> {
        #[cfg(feature = "opencl")]
        let cl_op = cl_programs::elementwise_dual::<T, T::Float>(cl_op, left.context())?;

        Ok(Self {
            left,
            right,
            cpu_op,
            #[cfg(feature = "opencl")]
            cl_op,
        })
    }

    /// Initialize a new logarithm [`Op`].
    pub fn log(left: L, right: R) -> Result<Self, Error> {
        Self::new(
            left,
            right,
            |l, r| T::from_float(l.to_float().log(r)),
            "log_",
        )
    }

    /// Initialize a new exponentiation [`Op`].
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
            .program(&self.cl_op)
            .queue(cl_queue)
            .global_work_size(left.len())
            .arg(left.as_ref())
            .arg(right.as_ref())
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }

    fn read_value(&self, coord: &[usize]) -> Result<Self::Out, Error> {
        let (left, right) = try_join(
            || self.left.read_value(coord),
            || self.right.read_value(coord),
        )?;

        Ok((self.cpu_op)(left, right))
    }
}

/// An array [`Op`] with a scalar argument
#[derive(Clone)]
pub struct ArrayScalar<T, A> {
    array: A,
    scalar: T,
    cpu_op: fn(T, T) -> T,
    #[cfg(feature = "opencl")]
    cl_op: ocl::Program,
}

impl<T: CDatatype, A: NDArray<DType = T>> ArrayScalar<T, A> {
    #[allow(unused_variables)]
    fn new(array: A, scalar: T, cpu_op: fn(T, T) -> T, cl_op: &'static str) -> Result<Self, Error> {
        #[cfg(feature = "opencl")]
        let cl_op = cl_programs::elementwise_scalar::<T, T>(cl_op, array.context())?;

        Ok(Self {
            array,
            scalar,
            cpu_op,
            #[cfg(feature = "opencl")]
            cl_op,
        })
    }

    /// Initialize a new scalar addition [`Op`].
    pub fn add(left: A, right: T) -> Result<Self, Error> {
        Self::new(left, right, Add::add, "add")
    }

    /// Initialize a new scalar division [`Op`].
    pub fn div(left: A, right: T) -> Result<Self, Error> {
        Self::new(left, right, Div::div, "div")
    }

    /// Initialize a new scalar multiplication [`Op`].
    pub fn mul(left: A, right: T) -> Result<Self, Error> {
        Self::new(left, right, Mul::mul, "mul")
    }

    /// Initialize a new scalar modulo [`Op`].
    pub fn rem(left: A, right: T) -> Result<Self, Error> {
        Self::new(left, right, Rem::rem, "rem")
    }

    /// Initialize a new scalar subtraction [`Op`].
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
            .map(|l| (self.cpu_op)(l, right))
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
            .program(&self.cl_op)
            .queue(cl_queue)
            .global_work_size(left.len())
            .arg(left.as_ref())
            .arg(right)
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }

    fn read_value(&self, coord: &[usize]) -> Result<Self::Out, Error> {
        let left = self.array.read_value(coord)?;
        let right = self.scalar;
        Ok((self.cpu_op)(left, right))
    }
}

/// An array [`Op`] with a scalar floating-point argument
#[derive(Clone)]
pub struct ArrayScalarFloat<T: CDatatype, A> {
    array: A,
    scalar: T::Float,
    cpu_op: fn(T, T::Float) -> T,
    #[cfg(feature = "opencl")]
    cl_op: ocl::Program,
}

impl<T: CDatatype, A: NDArray> ArrayScalarFloat<T, A> {
    #[allow(unused_variables)]
    fn new(
        array: A,
        scalar: T::Float,
        cpu_op: fn(T, T::Float) -> T,
        cl_op: &'static str,
    ) -> Result<Self, Error> {
        #[cfg(feature = "opencl")]
        let cl_op = cl_programs::elementwise_scalar::<T::Float, T>(cl_op, array.context())?;

        Ok(Self {
            array,
            scalar,
            cpu_op,
            #[cfg(feature = "opencl")]
            cl_op,
        })
    }
}

impl<T: CDatatype, A: NDArray> ArrayScalarFloat<T, A> {
    /// Initialize a new logarithm [`Op`] with a scalar base.
    pub fn log(left: A, right: T::Float) -> Result<Self, Error> {
        Self::new(
            left,
            right,
            |l, r| T::from_float(l.to_float().log(r)),
            "log",
        )
    }

    /// Initialize a new exponentiation [`Op`] with a scalar exponent.
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
            .map(|l| (self.cpu_op)(l, right))
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
            .program(&self.cl_op)
            .queue(cl_queue)
            .global_work_size(left.len())
            .arg(left.as_ref())
            .arg(right)
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }

    fn read_value(&self, coord: &[usize]) -> Result<Self::Out, Error> {
        let left = self.array.read_value(coord)?;
        let right = self.scalar;
        Ok((self.cpu_op)(left, right))
    }
}

// linear algebra

/// A matrix diagonal read [`Op`]
#[derive(Clone)]
pub struct MatDiag<A> {
    source: A,
    #[cfg(feature = "opencl")]
    cl_op: ocl::Program,
}

impl<A: NDArray> MatDiag<A> {
    /// Initialize a new matrix diagonal read [`Op`].
    pub fn new(source: A) -> Result<Self, Error> {
        debug_assert!(source.ndim() >= 2);
        debug_assert_eq!(
            source.shape()[source.ndim() - 1],
            source.shape()[source.ndim() - 2]
        );

        #[cfg(feature = "opencl")]
        let cl_op = cl_programs::diagonal::<A::DType>(source.context())?;

        Ok(Self {
            source,
            #[cfg(feature = "opencl")]
            cl_op,
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
            .program(&self.cl_op)
            .queue(cl_queue.clone())
            .global_work_size((batch_size, dim))
            .arg(input.as_ref())
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? };

        Ok(output)
    }

    fn read_value(&self, coord: &[usize]) -> Result<Self::Out, Error> {
        let mut source_coord = Vec::with_capacity(coord.len() + 1);
        source_coord.extend_from_slice(coord);
        source_coord.push(coord[coord.len() - 1]);
        self.source.read_value(&source_coord)
    }
}

/// A matrix multiplication [`Op`]
#[derive(Clone)]
pub struct MatMul<T, L, R> {
    left: L,
    right: R,
    dtype: PhantomData<T>,
    #[cfg(feature = "opencl")]
    cl_op: ocl::Program,
}

impl<T, L, R> MatMul<T, L, R>
where
    T: CDatatype,
    L: NDArray<DType = T>,
    R: NDArray<DType = T>,
{
    /// Initialize a new matrix multiplication [`Op`].
    pub fn new(left: L, right: R) -> Result<Self, Error> {
        debug_assert!(left.ndim() >= 2);
        debug_assert!(right.ndim() >= 2);

        #[cfg(feature = "opencl")]
        let cl_op = cl_programs::matmul::<T>(left.context())?;

        Ok(Self {
            left,
            right,
            dtype: PhantomData,
            #[cfg(feature = "opencl")]
            cl_op,
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
    L: NDArrayRead<DType = T> + NDArrayTransform,
    R: NDArrayRead<DType = T> + NDArrayTransform,
    L::Slice: NDArrayMath,
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
            transpose::transpose(&right[start..stop], &mut right_t[..], c, b);
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
            .program(&self.cl_op)
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

    fn read_value(&self, coord: &[usize]) -> Result<Self::Out, Error> {
        Err(Error::Bounds(format!(
            "reading the value at {coord:?} from a maxtrix multiplication is not implemented"
        )))
    }
}

// comparison

/// An array comparison [`Op`]
#[derive(Clone)]
pub struct ArrayBoolean<T, L, R> {
    left: L,
    right: R,
    cpu_op: fn(T, T) -> bool,
    #[cfg(feature = "opencl")]
    cl_op: ocl::Program,
}

impl<T: CDatatype, L: NDArray<DType = T>, R: NDArray<DType = T>> ArrayBoolean<T, L, R> {
    #[allow(unused_variables)]
    fn new(
        left: L,
        right: R,
        cpu_op: fn(T, T) -> bool,
        cl_op: &'static str,
    ) -> Result<Self, Error> {
        debug_assert_eq!(left.shape(), right.shape());

        #[cfg(feature = "opencl")]
        let cl_op = cl_programs::elementwise_boolean::<T>(cl_op, left.context())?;

        Ok(Self {
            left,
            right,
            cpu_op,
            #[cfg(feature = "opencl")]
            cl_op,
        })
    }

    /// Initialize a new boolean and [`Op`].
    pub fn and(left: L, right: R) -> Result<Self, Error> {
        fn and<T: CDatatype>(l: T, r: T) -> bool {
            (l != T::zero()) && (r != T::zero())
        }

        Self::new(left, right, and, "&&")
    }

    /// Initialize a new boolean or [`Op`].
    pub fn or(left: L, right: R) -> Result<Self, Error> {
        fn or<T: CDatatype>(l: T, r: T) -> bool {
            (l != T::zero()) || (r != T::zero())
        }

        Self::new(left, right, or, "||")
    }

    /// Initialize a new boolean xor [`Op`].
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
            .map(|(l, r)| (self.cpu_op)(l, r))
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
            .program(&self.cl_op)
            .queue(cl_queue.clone())
            .global_work_size(output.len())
            .arg(left.as_ref())
            .arg(right.as_ref())
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? };

        Ok(output)
    }

    fn read_value(&self, coord: &[usize]) -> Result<Self::Out, Error> {
        let (left, right) = try_join(
            || self.left.read_value(coord),
            || self.right.read_value(coord),
        )?;

        if (self.cpu_op)(left, right) {
            Ok(1)
        } else {
            Ok(0)
        }
    }
}

/// A boolean array [`Op`] with a scalar argument
#[derive(Clone)]
pub struct ArrayBooleanScalar<L, T> {
    left: L,
    right: T,
    cpu_op: fn(T, T) -> bool,
    #[cfg(feature = "opencl")]
    cl_op: ocl::Program,
}

impl<L: NDArray, T: CDatatype> ArrayBooleanScalar<L, T> {
    #[allow(unused_variables)]
    fn new(
        left: L,
        right: T,
        cpu_op: fn(T, T) -> bool,
        cl_op: &'static str,
    ) -> Result<Self, Error> {
        #[cfg(feature = "opencl")]
        let cl_op = cl_programs::scalar_boolean::<T>(cl_op, left.context())?;

        Ok(Self {
            left,
            right,
            cpu_op,
            #[cfg(feature = "opencl")]
            cl_op,
        })
    }

    /// Initialize a new boolean and [`Op`] with a scalar argument.
    pub fn and(left: L, right: T) -> Result<Self, Error> {
        fn and<T: CDatatype>(l: T, r: T) -> bool {
            (l != T::zero()) && (r != T::zero())
        }

        Self::new(left, right, and, "&&")
    }

    /// Initialize a new boolean or [`Op`] with a scalar argument.
    pub fn or(left: L, right: T) -> Result<Self, Error> {
        fn or<T: CDatatype>(l: T, r: T) -> bool {
            (l != T::zero()) || (r != T::zero())
        }

        Self::new(left, right, or, "||")
    }

    /// Initialize a new boolean xor [`Op`] with a scalar argument.
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
            .map(|l| (self.cpu_op)(l, self.right))
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
            .program(&self.cl_op)
            .queue(cl_queue.clone())
            .global_work_size(output.len())
            .arg(left.as_ref())
            .arg(self.right)
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? };

        Ok(output)
    }

    fn read_value(&self, coord: &[usize]) -> Result<Self::Out, Error> {
        let left = self.left.read_value(coord)?;
        let right = self.right;

        if (self.cpu_op)(left, right) {
            Ok(1)
        } else {
            Ok(0)
        }
    }
}

/// An array comparison [`Op`]
#[derive(Clone)]
pub struct ArrayCompare<T, L, R> {
    left: L,
    right: R,
    cpu_op: fn(&T, &T) -> bool,
    #[cfg(feature = "opencl")]
    cl_op: ocl::Program,
}

impl<T: CDatatype, L: NDArray<DType = T>, R: NDArray<DType = T>> ArrayCompare<T, L, R> {
    #[allow(unused_variables)]
    fn new(
        left: L,
        right: R,
        cpu_op: fn(&T, &T) -> bool,
        cl_op: &'static str,
    ) -> Result<Self, Error> {
        debug_assert_eq!(left.shape(), right.shape());

        #[cfg(feature = "opencl")]
        let cl_op = cl_programs::elementwise_cmp::<T>(cl_op, left.context())?;

        Ok(Self {
            left,
            right,
            cpu_op,
            #[cfg(feature = "opencl")]
            cl_op,
        })
    }

    /// Initialize a new equality comparison [`Op`].
    pub fn eq(left: L, right: R) -> Result<Self, Error> {
        Self::new(left, right, PartialEq::eq, "==")
    }

    /// Initialize a new greater-than comparison [`Op`].
    pub fn gt(left: L, right: R) -> Result<Self, Error> {
        Self::new(left, right, PartialOrd::gt, ">")
    }

    /// Initialize a new equal-or-greater-than comparison [`Op`].
    pub fn ge(left: L, right: R) -> Result<Self, Error> {
        Self::new(left, right, PartialOrd::ge, ">=")
    }

    /// Initialize a new less-than comparison [`Op`].
    pub fn lt(left: L, right: R) -> Result<Self, Error> {
        Self::new(left, right, PartialOrd::lt, "<")
    }

    /// Initialize a new equal-or-less-than comparison [`Op`].
    pub fn le(left: L, right: R) -> Result<Self, Error> {
        Self::new(left, right, PartialOrd::le, "<=")
    }

    /// Initialize a new not-equal comparison [`Op`].
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
            .map(|(l, r)| (self.cpu_op)(l, r))
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
            .program(&self.cl_op)
            .queue(cl_queue)
            .global_work_size(output.len())
            .arg(left.as_ref())
            .arg(right.as_ref())
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }

    fn read_value(&self, coord: &[usize]) -> Result<Self::Out, Error> {
        let (left, right) = try_join(
            || self.left.read_value(coord),
            || self.right.read_value(coord),
        )?;

        if (self.cpu_op)(&left, &right) {
            Ok(1)
        } else {
            Ok(0)
        }
    }
}

/// An array comparison [`Op`] with a scalar argument
#[derive(Clone)]
pub struct ArrayCompareScalar<T, A> {
    array: A,
    scalar: T,
    cpu_op: fn(&T, &T) -> bool,
    #[cfg(feature = "opencl")]
    cl_op: ocl::Program,
}

impl<T: CDatatype, A: NDArray> ArrayCompareScalar<T, A> {
    #[allow(unused_variables)]
    fn new(
        array: A,
        scalar: T,
        cpu_op: fn(&T, &T) -> bool,
        cl_op: &'static str,
    ) -> Result<Self, Error> {
        #[cfg(feature = "opencl")]
        let cl_op = cl_programs::scalar_cmp::<T>(cl_op, array.context())?;

        Ok(Self {
            array,
            scalar,
            cpu_op,
            #[cfg(feature = "opencl")]
            cl_op,
        })
    }

    /// Initialize a new equality comparison [`Op`].
    pub fn eq(array: A, scalar: T) -> Result<Self, Error> {
        Self::new(array, scalar, PartialEq::eq, "==")
    }

    /// Initialize a new greater-than comparison [`Op`].
    pub fn gt(array: A, scalar: T) -> Result<Self, Error> {
        Self::new(array, scalar, PartialOrd::gt, ">")
    }

    /// Initialize a new equal-or-greater-than comparison [`Op`].
    pub fn ge(array: A, scalar: T) -> Result<Self, Error> {
        Self::new(array, scalar, PartialOrd::ge, ">=")
    }

    /// Initialize a new less-than comparison [`Op`].
    pub fn lt(array: A, scalar: T) -> Result<Self, Error> {
        Self::new(array, scalar, PartialOrd::lt, "<")
    }

    /// Initialize a new equal-or-less-than comparison [`Op`].
    pub fn le(array: A, scalar: T) -> Result<Self, Error> {
        Self::new(array, scalar, PartialOrd::le, "<=")
    }

    /// Initialize a new not-equal comparison [`Op`].
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
            .map(|n| (self.cpu_op)(n, &self.scalar))
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
            .program(&self.cl_op)
            .queue(cl_queue)
            .global_work_size(input.len())
            .arg(input.as_ref())
            .arg(self.scalar)
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }

    fn read_value(&self, coord: &[usize]) -> Result<Self::Out, Error> {
        let left = self.array.read_value(coord)?;
        let right = self.scalar;

        if (self.cpu_op)(&left, &right) {
            Ok(1)
        } else {
            Ok(0)
        }
    }
}

// reduction

/// An array reduction [`Op`]
#[derive(Copy, Clone)]
pub struct ArrayReduceAxes<T, A> {
    source: A,
    stride: usize,
    id: T,
    cpu_op: fn(T, T) -> T,
    #[allow(unused)]
    cl_op: &'static str,
}

impl<T: CDatatype, A: NDArray<DType = T>> ArrayReduceAxes<T, A> {
    fn new(source: A, stride: usize, id: T, cpu_op: fn(T, T) -> T, cl_op: &'static str) -> Self {
        Self {
            source,
            stride,
            id,
            cpu_op,
            cl_op,
        }
    }

    /// Initialize a new reduce-max [`Op`].
    pub fn max(source: A, stride: usize) -> Self {
        fn max<T: PartialOrd>(l: T, r: T) -> T {
            if r > l {
                r
            } else {
                l
            }
        }

        Self::new(source, stride, T::min(), max, "max")
    }

    /// Initialize a new reduce-min [`Op`].
    pub fn min(source: A, stride: usize) -> Self {
        fn min<T: PartialOrd>(l: T, r: T) -> T {
            if r < l {
                r
            } else {
                l
            }
        }

        Self::new(source, stride, T::max(), min, "min")
    }

    /// Initialize a new product-reduce [`Op`].
    pub fn product(source: A, stride: usize) -> Self {
        Self::new(source, stride, T::one(), Mul::mul, "mul")
    }

    /// Initialize a new sum-reduce [`Op`].
    pub fn sum(source: A, stride: usize) -> Self {
        Self::new(source, stride, T::zero(), Add::add, "add")
    }
}

impl<T, A> Op for ArrayReduceAxes<T, A>
where
    T: CDatatype,
    A: NDArrayRead<DType = T>,
{
    type Out = T;

    fn context(&self) -> &Context {
        self.source.context()
    }

    fn enqueue_cpu(&self, queue: &Queue) -> Result<Vec<Self::Out>, Error> {
        let input = self.source.to_host(queue)?;
        debug_assert!(!input.as_ref().is_empty());

        let output = input
            .as_ref()
            .par_chunks_exact(self.stride)
            .map(|chunk| {
                // encourage the compiler to vectorize the reduction
                let reduced = chunk
                    .chunks(8)
                    .map(|cc| {
                        let reduced = cc
                            .into_iter()
                            .copied()
                            .reduce(self.cpu_op)
                            .expect("reduce chunk");

                        reduced
                    })
                    .reduce(self.cpu_op)
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
            self.cl_op,
            cl_queue,
            input.as_ref(),
            self.source.shape(),
            self.stride,
        )?;

        Ok(output)
    }

    fn read_value(&self, coord: &[usize]) -> Result<Self::Out, Error> {
        let start = offset_of(coord, self.source.shape());
        let stop = start + self.stride;

        let strides = strides_for(self.source.shape(), self.source.ndim());

        (start..stop)
            .into_par_iter()
            .map(|offset| {
                strides
                    .iter()
                    .zip(self.source.shape())
                    .map(|(stride, dim)| (offset / stride) % dim)
                    .collect::<Vec<usize>>()
            })
            .map(|source_coord| self.source.read_value(&source_coord))
            .try_reduce(|| self.id, |r, v| Ok((self.cpu_op)(r, v)))
    }
}

// other unary ops

/// A type cast [`Op`]
#[derive(Clone)]
pub struct ArrayCast<A, O> {
    source: A,
    dtype: PhantomData<O>,
    #[cfg(feature = "opencl")]
    cl_op: ocl::Program,
}

impl<A: NDArray, O: CDatatype> ArrayCast<A, O> {
    /// Initialize a new type-cast [`Op`]
    pub fn new(source: A) -> Result<Self, Error> {
        #[cfg(feature = "opencl")]
        let cl_op = cl_programs::cast::<A::DType, O>(source.context())?;

        Ok(Self {
            source,
            dtype: PhantomData,
            #[cfg(feature = "opencl")]
            cl_op,
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
            .name("cast_dtype")
            .program(&self.cl_op)
            .queue(cl_queue.clone())
            .global_work_size(input.len())
            .arg(input.as_ref())
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? };

        Ok(output)
    }

    fn read_value(&self, coord: &[usize]) -> Result<Self::Out, Error> {
        let value = self.source.read_value(coord)?;
        Ok(O::from_f64(value.to_f64()))
    }
}

/// A unary array [`Op`]
#[derive(Clone)]
pub struct ArrayUnary<IT, OT, A> {
    array: A,
    cpu_op: fn(IT) -> OT,
    #[cfg(feature = "opencl")]
    cl_op: ocl::Program,
}

impl<IT: CDatatype, OT: CDatatype, A: NDArray> ArrayUnary<IT, OT, A> {
    #[allow(unused_variables)]
    fn new(array: A, cpu_op: fn(IT) -> OT, cl_op: &'static str) -> Result<Self, Error> {
        #[cfg(feature = "opencl")]
        let cl_op = cl_programs::unary::<IT, OT>(cl_op, array.context())?;

        Ok(Self {
            array,
            cpu_op,
            #[cfg(feature = "opencl")]
            cl_op,
        })
    }
}

impl<T: CDatatype, A: NDArray> ArrayUnary<T, T, A> {
    /// Initialize a new absolute value [`Op`].
    pub fn abs(array: A) -> Result<Self, Error> {
        // TODO: replace "fabs" with "abs" for integer types
        Self::new(array, T::abs, "fabs")
    }

    /// Initialize a new natural log [`Op`].
    pub fn ln(array: A) -> Result<Self, Error> {
        Self::new(array, |n| T::from_float(n.to_float().ln()), "_log")
    }

    /// Initialize a new exponentiation [`Op`].
    pub fn exp(array: A) -> Result<Self, Error> {
        Self::new(array, |n| T::from_float(n.to_float().exp()), "exp")
    }

    /// Initialize a new rounding [`Op`].
    pub fn round(array: A) -> Result<Self, Error> {
        Self::new(array, T::round, "round")
    }
}

impl<T: CDatatype, A: NDArray> ArrayUnary<T, T::Float, A> {
    /// Initialize a new arcsine [`Op`].
    pub fn asin(array: A) -> Result<Self, Error> {
        Self::new(array, |n| n.to_float().asin(), "asin")
    }

    /// Initialize a new sine [`Op`].
    pub fn sin(array: A) -> Result<Self, Error> {
        Self::new(array, |n| n.to_float().sin(), "sin")
    }

    /// Initialize a new hyperbolic sine [`Op`].
    pub fn sinh(array: A) -> Result<Self, Error> {
        Self::new(array, |n| n.to_float().sinh(), "sinh")
    }

    /// Initialize a new arccosine [`Op`].
    pub fn acos(array: A) -> Result<Self, Error> {
        Self::new(array, |n| n.to_float().acos(), "acos")
    }

    /// Initialize a new cosine [`Op`].
    pub fn cos(array: A) -> Result<Self, Error> {
        Self::new(array, |n| n.to_float().cos(), "cos")
    }

    /// Initialize a new hyperbolic cosine [`Op`].
    pub fn cosh(array: A) -> Result<Self, Error> {
        Self::new(array, |n| n.to_float().cosh(), "cosh")
    }

    /// Initialize a new arctangent [`Op`].
    pub fn atan(array: A) -> Result<Self, Error> {
        Self::new(array, |n| n.to_float().atan(), "atan")
    }

    /// Initialize a new tangent [`Op`].
    pub fn tan(array: A) -> Result<Self, Error> {
        Self::new(array, |n| n.to_float().tan(), "tan")
    }

    /// Initialize a new hyperbolic tangent [`Op`].
    pub fn tanh(array: A) -> Result<Self, Error> {
        Self::new(array, |n| n.to_float().tanh(), "tanh")
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
            .map(|n| (self.cpu_op)(n))
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
            .program(&self.cl_op)
            .queue(cl_queue)
            .global_work_size(output.len())
            .arg(input.as_ref())
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }

    fn read_value(&self, coord: &[usize]) -> Result<Self::Out, Error> {
        let value = self.array.read_value(coord)?;
        Ok((self.cpu_op)(value))
    }
}

// gather ops

/// A conditional selection (boolean logic) [`Op`]
#[derive(Clone)]
pub struct GatherCond<A, T, L, R> {
    cond: A,
    then: L,
    or_else: R,
    dtype: PhantomData<T>,
    #[cfg(feature = "opencl")]
    cl_op: ocl::Program,
}

impl<A: NDArray<DType = u8>, T: CDatatype, L, R> GatherCond<A, T, L, R> {
    /// Initialize a new conditional selection [`Op`].
    pub fn new(cond: A, then: L, or_else: R) -> Result<Self, Error> {
        #[cfg(feature = "opencl")]
        let cl_op = cl_programs::gather_cond::<T>(cond.context())?;

        Ok(Self {
            cond,
            then,
            or_else,
            dtype: PhantomData,
            #[cfg(feature = "opencl")]
            cl_op,
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

    // TODO: this should only resolve elements present in the output
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
            .map(|when| when != 0)
            .zip(lr)
            .map(|(when, (then, or_else))| if when { then } else { or_else })
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
            .program(&self.cl_op)
            .global_work_size(cond.len())
            .arg(cond.as_ref())
            .arg(then.as_ref())
            .arg(or_else.as_ref())
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }

    fn read_value(&self, coord: &[usize]) -> Result<Self::Out, Error> {
        if self.cond.read_value(coord)? != 0 {
            self.then.read_value(coord)
        } else {
            self.or_else.read_value(coord)
        }
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
