use std::cmp::{PartialEq, PartialOrd};
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Rem, Sub};

use rayon::prelude::*;

use super::kernels;
use super::{Buffer, CDatatype, Error, Float, NDArrayRead, Queue};

pub trait Op: Send + Sync {
    type Out: CDatatype;

    fn enqueue(&self, queue: Queue) -> Result<Buffer<Self::Out>, Error>;
}

// arithmetic

#[derive(Copy, Clone)]
pub struct ArrayDual<T, L, R> {
    left: L,
    right: R,
    cpu_op: fn(T, T) -> T,
    kernel_op: &'static str,
}

impl<T: CDatatype, L, R> ArrayDual<T, L, R> {
    fn new(left: L, right: R, cpu_op: fn(T, T) -> T, kernel_op: &'static str) -> Self {
        Self {
            left,
            right,
            cpu_op,
            kernel_op,
        }
    }

    pub fn add(left: L, right: R) -> Self {
        Self::new(left, right, Add::add, "+")
    }

    pub fn div(left: L, right: R) -> Self {
        Self::new(left, right, Div::div, "/")
    }

    pub fn mul(left: L, right: R) -> Self {
        Self::new(left, right, Mul::mul, "*")
    }

    pub fn rem(left: L, right: R) -> Self {
        Self::new(left, right, Rem::rem, "%")
    }

    pub fn sub(left: L, right: R) -> Self {
        Self::new(left, right, Sub::sub, "-")
    }
}

impl<T: CDatatype, L: NDArrayRead<DType = T>, R: NDArrayRead<DType = T>> Op for ArrayDual<T, L, R> {
    type Out = T;

    fn enqueue(&self, queue: Queue) -> Result<Buffer<Self::Out>, Error> {
        debug_assert_eq!(self.left.size(), self.right.size());

        let right_queue = queue.context().queue(self.right.size())?;
        let right = self.right.read(right_queue)?;
        let left = self.left.read(queue)?;

        let output = match sync_device(left, right)? {
            Buffers::CL(left, right) => {
                let right_cl_queue = right.default_queue().expect("right queue").clone();
                let event = right_cl_queue.enqueue_marker::<ocl::Event>(None)?;

                let cl_queue = left.default_queue().expect("left queue").clone();

                let output =
                    kernels::elementwise_inplace(self.kernel_op, cl_queue, left, &right, &event)?;

                Buffer::CL(output)
            }
            Buffers::Host(left, right) => {
                debug_assert_eq!(left.len(), right.len());

                let output = left
                    .into_par_iter()
                    .zip(right.into_par_iter())
                    .map(|(l, r)| (self.cpu_op)(l, r))
                    .collect();

                Buffer::Host(output)
            }
        };

        Ok(output)
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

    fn enqueue(&self, queue: Queue) -> Result<Buffer<Self::Out>, Error> {
        debug_assert_eq!(self.left.size(), self.right.size());

        let right_queue = queue.context().queue(self.right.size())?;
        let right = self.right.read(right_queue)?;
        let left = self.left.read(queue)?;

        todo!()
    }
}

#[derive(Copy, Clone)]
pub struct ArrayScalar<T, A> {
    array: A,
    scalar: T,
    host_op: fn(T, T) -> T,
    kernel_op: &'static str,
}

impl<T: CDatatype, A> ArrayScalar<T, A> {
    fn new(array: A, scalar: T, host_op: fn(T, T) -> T, kernel_op: &'static str) -> Self {
        Self {
            array,
            scalar,
            host_op,
            kernel_op,
        }
    }

    pub fn add(left: A, right: T) -> Self {
        Self::new(left, right, Add::add, "add")
    }

    pub fn div(left: A, right: T) -> Self {
        Self::new(left, right, Div::div, "div")
    }

    pub fn mul(left: A, right: T) -> Self {
        Self::new(left, right, Mul::mul, "mul")
    }

    pub fn rem(left: A, right: T) -> Self {
        Self::new(left, right, Rem::rem, "fmod")
    }

    pub fn sub(left: A, right: T) -> Self {
        Self::new(left, right, Sub::sub, "sub")
    }
}

impl<T: CDatatype, A: NDArrayRead<DType = T>> Op for ArrayScalar<T, A> {
    type Out = A::DType;

    fn enqueue(&self, queue: Queue) -> Result<Buffer<Self::Out>, Error> {
        let left = self.array.read(queue)?;
        let right = self.scalar;

        let output = match left {
            Buffer::CL(left) => {
                let cl_queue = left.default_queue().expect("queue").clone();
                let output = kernels::elementwise_scalar(self.kernel_op, cl_queue, left, right)?;
                Buffer::CL(output)
            }
            Buffer::Host(left) => {
                let output = left
                    .into_par_iter()
                    .chunks(8)
                    .map(|chunk| {
                        chunk
                            .into_iter()
                            .map(|l| (self.host_op)(l, self.scalar))
                            .collect::<Vec<T>>()
                    })
                    .flatten()
                    .collect();

                Buffer::Host(output)
            }
        };

        Ok(output)
    }
}

#[derive(Copy, Clone)]
pub struct ArrayScalarFloat<T, A> {
    array: A,
    scalar: f64,
    host_op: fn(T, f64) -> T,
    kernel_op: &'static str,
}

impl<T, A> ArrayScalarFloat<T, A> {
    fn new(array: A, scalar: f64, host_op: fn(T, f64) -> T, kernel_op: &'static str) -> Self {
        Self {
            array,
            scalar,
            host_op,
            kernel_op,
        }
    }
}

impl<T: CDatatype, A> ArrayScalarFloat<T, A> {
    pub fn log(left: A, right: f64) -> Self {
        Self::new(left, right, T::log, "log")
    }

    pub fn pow(left: A, right: f64) -> Self {
        Self::new(left, right, T::pow, "pow_")
    }
}

impl<T: CDatatype, A: NDArrayRead<DType = T>> Op for ArrayScalarFloat<T, A> {
    type Out = T;

    fn enqueue(&self, queue: Queue) -> Result<Buffer<Self::Out>, Error> {
        let left = self.array.read(queue)?;
        let right = self.scalar;

        let output = match left {
            Buffer::CL(left) => {
                let cl_queue = left.default_queue().expect("queue").clone();
                let output = kernels::elementwise_scalar(self.kernel_op, cl_queue, left, right)?;
                Buffer::CL(output)
            }
            Buffer::Host(left) => {
                let output = left
                    .into_par_iter()
                    .map(|l| (self.host_op)(l, self.scalar))
                    .collect();

                Buffer::Host(output)
            }
        };

        Ok(output)
    }
}

// linear algebra

#[derive(Copy, Clone)]
pub struct MatDiag<A> {
    source: A,
}

#[derive(Copy, Clone)]
pub struct MatMul<'a, L, R> {
    left: &'a L,
    right: &'a R,
}

impl<'a, L, R> MatMul<'a, L, R> {
    pub fn new(left: &'a L, right: &'a R) -> Self {
        Self { left, right }
    }
}

impl<'a, T: CDatatype, L: NDArrayRead<DType = T>, R: NDArrayRead<DType = T>> Op
    for MatMul<'a, L, R>
{
    type Out = T;

    fn enqueue(&self, queue: Queue) -> Result<Buffer<T>, Error> {
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

        let right_queue = queue.context().queue(ndim * a * c)?;
        let right = self.right.read(right_queue)?;
        let left = self.left.read(queue)?;

        let output = match sync_device(left, right)? {
            Buffers::CL(left, right) => {
                let right_cl_queue = right.default_queue().expect("right queue").clone();
                let event = right_cl_queue.enqueue_marker::<ocl::Event>(None)?;
                let cl_queue = left.default_queue().expect("left queue").clone();

                let output =
                    kernels::matmul(cl_queue, left, right, num_matrices, (a, b, c), event)?;

                Buffer::CL(output)
            }
            Buffers::Host(left, right) => {
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
                                            rc.into_iter()
                                                .copied()
                                                .zip(cc)
                                                .map(|(r, c)| r * c)
                                                .sum()
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

                Buffer::Host(output)
            }
        };

        Ok(output)
    }
}

// comparison

#[derive(Copy, Clone)]
pub struct ArrayBoolean<'a, T, L, R> {
    left: &'a L,
    right: &'a R,
    host_cmp: fn(T, T) -> bool,
    kernel_cmp: &'static str,
}

impl<'a, T: CDatatype, L, R> ArrayBoolean<'a, T, L, R> {
    fn new(
        left: &'a L,
        right: &'a R,
        host_cmp: fn(T, T) -> bool,
        kernel_cmp: &'static str,
    ) -> Self {
        Self {
            left,
            right,
            host_cmp,
            kernel_cmp,
        }
    }

    pub fn and(left: &'a L, right: &'a R) -> Self {
        fn and<T: CDatatype>(l: T, r: T) -> bool {
            (l != T::zero()) && (r != T::zero())
        }

        Self::new(left, right, and, "&&")
    }

    pub fn or(left: &'a L, right: &'a R) -> Self {
        fn or<T: CDatatype>(l: T, r: T) -> bool {
            (l != T::zero()) || (r != T::zero())
        }

        Self::new(left, right, or, "||")
    }

    pub fn xor(left: &'a L, right: &'a R) -> Self {
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

    fn enqueue(&self, queue: Queue) -> Result<Buffer<Self::Out>, Error> {
        debug_assert_eq!(self.left.shape(), self.right.shape());

        let right_queue = queue.context().queue(self.left.size())?;
        let right = self.right.read(right_queue)?;
        let left = self.left.read(queue)?;

        let output = match sync_device(left, right)? {
            Buffers::CL(left, right) => {
                let right_cl_queue = right.default_queue().expect("right queue").clone();
                let event = right_cl_queue.enqueue_marker::<ocl::Event>(None)?;
                let cl_queue = left.default_queue().expect("left queue").clone();

                let output =
                    kernels::elementwise_boolean(self.kernel_cmp, cl_queue, &left, &right, &event)?;

                Buffer::CL(output)
            }
            Buffers::Host(left, right) => {
                let output = left
                    .into_par_iter()
                    .zip(right.into_par_iter())
                    .map(|(l, r)| (self.host_cmp)(l, r))
                    .map(|cmp| if cmp { 1 } else { 0 })
                    .collect();

                Buffer::Host(output)
            }
        };

        Ok(output)
    }
}

#[derive(Copy, Clone)]
pub struct ArrayCompare<'a, T, L, R> {
    left: &'a L,
    right: &'a R,
    host_cmp: fn(&T, &T) -> bool,
    kernel_cmp: &'static str,
}

impl<'a, T: CDatatype, L, R> ArrayCompare<'a, T, L, R> {
    fn new(
        left: &'a L,
        right: &'a R,
        host_cmp: fn(&T, &T) -> bool,
        kernel_cmp: &'static str,
    ) -> Self {
        Self {
            left,
            right,
            host_cmp,
            kernel_cmp,
        }
    }

    pub fn eq(left: &'a L, right: &'a R) -> Self {
        Self::new(left, right, PartialEq::eq, "==")
    }

    pub fn gt(left: &'a L, right: &'a R) -> Self {
        Self::new(left, right, PartialOrd::gt, ">")
    }

    pub fn gte(left: &'a L, right: &'a R) -> Self {
        Self::new(left, right, PartialOrd::ge, ">=")
    }

    pub fn lt(left: &'a L, right: &'a R) -> Self {
        Self::new(left, right, PartialOrd::lt, "<")
    }

    pub fn lte(left: &'a L, right: &'a R) -> Self {
        Self::new(left, right, PartialOrd::le, "<=")
    }

    pub fn ne(left: &'a L, right: &'a R) -> Self {
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

    fn enqueue(&self, queue: Queue) -> Result<Buffer<Self::Out>, Error> {
        debug_assert_eq!(self.left.shape(), self.right.shape());

        let right_queue = queue.context().queue(self.right.size())?;
        let right = self.right.read(right_queue)?;
        let left = self.left.read(queue)?;

        let output = match sync_device(left, right)? {
            Buffers::CL(left, right) => {
                let right_cl_queue = right.default_queue().expect("right queue").clone();
                let event = right_cl_queue.enqueue_marker::<ocl::Event>(None)?;
                let cl_queue = left.default_queue().expect("left queue").clone();

                let output =
                    kernels::elementwise_cmp(self.kernel_cmp, cl_queue, &left, &right, &event)?;

                Buffer::CL(output)
            }
            Buffers::Host(left, right) => {
                debug_assert_eq!(left.len(), right.len());

                let output = left
                    .par_iter()
                    .zip(right.par_iter())
                    .map(|(l, r)| (self.host_cmp)(l, r))
                    .map(|cmp| if cmp { 1 } else { 0 })
                    .collect();

                Buffer::Host(output)
            }
        };

        Ok(output)
    }
}

#[derive(Copy, Clone)]
pub struct ArrayCompareScalar<'a, T, A> {
    array: &'a A,
    scalar: T,
    host_cmp: fn(&T, &T) -> bool,
    kernel_cmp: &'static str,
}

impl<'a, T: CDatatype, A> ArrayCompareScalar<'a, T, A> {
    fn new(
        array: &'a A,
        scalar: T,
        host_cmp: fn(&T, &T) -> bool,
        kernel_cmp: &'static str,
    ) -> Self {
        Self {
            array,
            scalar,
            host_cmp,
            kernel_cmp,
        }
    }

    pub fn eq(array: &'a A, scalar: T) -> Self {
        Self::new(array, scalar, PartialEq::eq, "==")
    }

    pub fn gt(array: &'a A, scalar: T) -> Self {
        Self::new(array, scalar, PartialOrd::gt, ">")
    }

    pub fn gte(array: &'a A, scalar: T) -> Self {
        Self::new(array, scalar, PartialOrd::ge, ">=")
    }

    pub fn lt(array: &'a A, scalar: T) -> Self {
        Self::new(array, scalar, PartialOrd::lt, "<")
    }

    pub fn lte(array: &'a A, scalar: T) -> Self {
        Self::new(array, scalar, PartialOrd::le, "<=")
    }

    pub fn ne(array: &'a A, scalar: T) -> Self {
        Self::new(array, scalar, PartialEq::ne, "!=")
    }
}

impl<'a, T: CDatatype, A: NDArrayRead<DType = T>> Op for ArrayCompareScalar<'a, T, A> {
    type Out = u8;

    fn enqueue(&self, queue: Queue) -> Result<Buffer<Self::Out>, Error> {
        let input = self.array.read(queue)?;

        let output = match input {
            Buffer::CL(input) => {
                let cl_queue = input.default_queue().expect("queue").clone();
                let output = kernels::scalar_cmp(self.kernel_cmp, cl_queue, &input, self.scalar)?;
                Buffer::CL(output)
            }
            Buffer::Host(input) => {
                let output = input
                    .par_iter()
                    .map(|n| (self.host_cmp)(n, &self.scalar))
                    .map(|cmp| if cmp { 1 } else { 0 })
                    .collect();

                Buffer::Host(output)
            }
        };

        Ok(output)
    }
}

// reduction

#[derive(Copy, Clone)]
pub struct ArrayReduce<'a, T, A> {
    source: &'a A,
    axis: usize,
    host_reduce: fn(T, T) -> T,
    kernel_reduce: &'static str,
}

impl<'a, T: CDatatype, A> ArrayReduce<'a, T, A> {
    fn new(
        source: &'a A,
        axis: usize,
        host_reduce: fn(T, T) -> T,
        kernel_reduce: &'static str,
    ) -> Self {
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

impl<'a, T: CDatatype, A: NDArrayRead<DType = T>> Op for ArrayReduce<'a, T, A> {
    type Out = T;

    fn enqueue(&self, queue: Queue) -> Result<Buffer<T>, Error> {
        debug_assert!(self.axis < self.source.ndim());
        let shape = self.source.shape().to_vec();
        let input = self.source.read(queue)?;

        let output = match input {
            Buffer::CL(input) => {
                let cl_queue = input.default_queue().expect("queue").clone();
                let output = kernels::reduce_axis(
                    A::DType::zero(),
                    self.kernel_reduce,
                    cl_queue,
                    input,
                    shape,
                    self.axis,
                )?;

                Buffer::CL(output)
            }
            Buffer::Host(input) => {
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

                Buffer::Host(output)
            }
        };

        Ok(output)
    }
}

// other unary ops

#[derive(Copy, Clone)]
pub struct ArrayCast<'a, A, O> {
    source: &'a A,
    dtype: PhantomData<O>,
}

impl<'a, A, O> ArrayCast<'a, A, O> {
    pub fn new(source: &'a A) -> Self {
        Self {
            source,
            dtype: PhantomData,
        }
    }
}

impl<'a, A: NDArrayRead, O: CDatatype> Op for ArrayCast<'a, A, O> {
    type Out = O;

    fn enqueue(&self, queue: Queue) -> Result<Buffer<Self::Out>, Error> {
        let output = match self.source.read(queue)? {
            Buffer::CL(input) => {
                let cl_queue = input.default_queue().expect("queue").clone();
                let output = kernels::cast(cl_queue, &input)?;
                Buffer::CL(output)
            }
            Buffer::Host(input) => {
                let output = input
                    .into_par_iter()
                    .map(|n| n.to_f64())
                    .map(|float| O::from_f64(float))
                    .collect();

                Buffer::Host(output)
            }
        };

        Ok(output)
    }
}

#[derive(Copy, Clone)]
pub struct ArrayUnary<IT, OT, A> {
    array: A,
    host_op: fn(IT) -> OT,
    kernel_op: &'static str,
}

impl<IT, OT, A> ArrayUnary<IT, OT, A> {
    fn new(array: A, host_op: fn(IT) -> OT, kernel_op: &'static str) -> Self {
        Self {
            array,
            host_op,
            kernel_op,
        }
    }
}

impl<T: CDatatype, A> ArrayUnary<T, T, A> {
    pub fn abs(array: A) -> Self {
        // TODO: replace "fabs" with "abs" for integer types
        Self::new(array, T::abs, "fabs")
    }

    pub fn exp(array: A) -> Self {
        Self::new(array, T::exp, "exp")
    }
}

impl<T: CDatatype, A> ArrayUnary<T, T::Neg, A> {
    pub fn neg(array: A) -> Self {
        Self::new(array, T::neg, "-")
    }
}

impl<IT: CDatatype, A> ArrayUnary<IT, u8, A> {
    pub fn not(array: A) -> Self {
        Self::new(array, IT::not, "!")
    }
}

impl<IT: Float, A> ArrayUnary<IT, u8, A> {
    pub fn inf(array: A) -> Self {
        Self::new(array, IT::is_inf, "isinf")
    }

    pub fn nan(array: A) -> Self {
        Self::new(array, IT::is_nan, "isnan")
    }
}

impl<IT: CDatatype, OT: CDatatype, A: NDArrayRead<DType = IT>> Op for ArrayUnary<IT, OT, A> {
    type Out = OT;

    fn enqueue(&self, queue: Queue) -> Result<Buffer<Self::Out>, Error> {
        let input = self.array.read(queue)?;

        let output = match input {
            Buffer::CL(input) => {
                let cl_queue = input.default_queue().expect("queue").clone();
                let output = kernels::unary(self.kernel_op, cl_queue, &input)?;
                Buffer::CL(output)
            }
            Buffer::Host(input) => {
                let output = input.into_par_iter().map(|n| (self.host_op)(n)).collect();
                Buffer::Host(output)
            },
        };

        Ok(output)
    }
}

enum Buffers<LT: CDatatype, RT: CDatatype> {
    CL(ocl::Buffer<LT>, ocl::Buffer<RT>),
    Host(Vec<LT>, Vec<RT>),
}

#[inline]
fn sync_device<LT: CDatatype, RT: CDatatype>(
    left: Buffer<LT>,
    right: Buffer<RT>,
) -> Result<Buffers<LT, RT>, ocl::Error> {
    match (left, right) {
        (Buffer::Host(left), Buffer::Host(right)) => Ok(Buffers::Host(left, right)),
        (Buffer::CL(left), Buffer::CL(right)) => Ok(Buffers::CL(left, right)),
        (Buffer::Host(left), Buffer::CL(right)) => {
            let cl_queue = right.default_queue().expect("queue").clone();
            let left_cl = ocl::Buffer::builder()
                .queue(cl_queue)
                .len(left.len())
                .copy_host_slice(&left[..])
                .build()?;

            Ok(Buffers::CL(left_cl, right))
        }
        (Buffer::CL(left), Buffer::Host(right)) => {
            let cl_queue = left.default_queue().expect("queue").clone();
            let right_cl = ocl::Buffer::builder()
                .queue(cl_queue)
                .len(left.len())
                .copy_host_slice(&right[..])
                .build()?;

            Ok(Buffers::CL(left, right_cl))
        }
    }
}
