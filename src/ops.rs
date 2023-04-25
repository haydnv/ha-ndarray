use std::marker::PhantomData;

use super::kernels;
use super::{Buffer, CDatatype, Error, NDArrayRead, Queue};

pub trait Op {
    type Out: CDatatype;

    fn enqueue(&self, queue: Queue) -> Result<Buffer<Self::Out>, Error>;
}

// arithmetic

#[derive(Copy, Clone)]
pub struct ArrayDual<L, R> {
    left: L,
    right: R,
    op: &'static str,
}

impl<L, R> ArrayDual<L, R> {
    fn new(left: L, right: R, op: &'static str) -> Self {
        Self { left, right, op }
    }

    pub fn add(left: L, right: R) -> Self {
        Self::new(left, right, "+")
    }

    pub fn div(left: L, right: R) -> Self {
        Self::new(left, right, "/")
    }

    pub fn mul(left: L, right: R) -> Self {
        Self::new(left, right, "*")
    }

    pub fn rem(left: L, right: R) -> Self {
        Self::new(left, right, "%")
    }

    pub fn sub(left: L, right: R) -> Self {
        Self::new(left, right, "-")
    }
}

impl<T: CDatatype, L: NDArrayRead<DType = T>, R: NDArrayRead<DType = T>> Op for ArrayDual<L, R> {
    type Out = T;

    fn enqueue(&self, queue: Queue) -> Result<Buffer<Self::Out>, Error> {
        assert_eq!(self.left.size(), self.right.size());

        let right_queue = queue.context().queue(self.right.size())?;
        let right = self.right.read(right_queue)?;
        let left = self.left.read(queue)?;

        let output = match (left, right) {
            (Buffer::CL(left), Buffer::CL(right)) => {
                let right_cl_queue = right.default_queue().expect("right queue").clone();
                let event = right_cl_queue.enqueue_marker::<ocl::Event>(None)?;

                let cl_queue = left.default_queue().expect("left queue").clone();

                let output = kernels::elementwise_inplace(self.op, cl_queue, left, &right, &event)?;

                output.into()
            }
            (_, _) => todo!("dual array ops on CPU"),
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
    op: &'static str,
}

impl<T, A> ArrayScalar<T, A> {
    fn new(array: A, scalar: T, op: &'static str) -> Self {
        Self { array, scalar, op }
    }

    pub fn add(left: A, right: T) -> Self {
        Self::new(left, right, "add")
    }

    pub fn div(left: A, right: T) -> Self {
        Self::new(left, right, "div")
    }

    pub fn mul(left: A, right: T) -> Self {
        Self::new(left, right, "mul")
    }

    pub fn rem(left: A, right: T) -> Self {
        Self::new(left, right, "fmod")
    }

    pub fn sub(left: A, right: T) -> Self {
        Self::new(left, right, "sub")
    }
}

impl<A> ArrayScalar<f64, A> {
    pub fn log(left: A, right: f64) -> Self {
        Self::new(left, right, "log")
    }

    pub fn pow(left: A, right: f64) -> Self {
        Self::new(left, right, "pow_")
    }
}

impl<T: CDatatype, A: NDArrayRead> Op for ArrayScalar<T, A> {
    type Out = A::DType;

    fn enqueue(&self, queue: Queue) -> Result<Buffer<Self::Out>, Error> {
        let left = self.array.read(queue)?;
        let right = self.scalar;

        let output = match left {
            Buffer::CL(left) => {
                let cl_queue = left.default_queue().expect("queue").clone();
                let output = kernels::elementwise_scalar(self.op, cl_queue, left, right)?;
                output.into()
            }
            Buffer::Host(_) => todo!("array scalar ops on CPU"),
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

        let num_matrices = self.left.shape()[..ndim - 2].iter().product();
        let a = *self.left.shape().iter().nth(ndim - 2).expect("a");
        let b = *self.left.shape().last().expect("b");
        let c = *self.right.shape().last().expect("c");
        debug_assert_eq!(b, self.right.shape()[ndim - 2]);

        let right_queue = queue.context().queue(ndim * a * c)?;
        let right = self.right.read(right_queue)?;
        let left = self.left.read(queue)?;

        let output = match (left, right) {
            (Buffer::CL(left), Buffer::CL(right)) => {
                let right_cl_queue = right.default_queue().expect("right queue").clone();
                let event = right_cl_queue.enqueue_marker::<ocl::Event>(None)?;
                let cl_queue = left.default_queue().expect("left queue").clone();

                let output =
                    kernels::matmul(cl_queue, left, right, num_matrices, (a, b, c), event)?;

                output.into()
            }
            (_, _) => todo!("matmul on CPU"),
        };

        Ok(output)
    }
}

// comparison

#[derive(Copy, Clone)]
pub struct ArrayBoolean<'a, L, R> {
    left: &'a L,
    right: &'a R,
    cmp: &'static str,
}

impl<'a, L, R> ArrayBoolean<'a, L, R> {
    fn new(left: &'a L, right: &'a R, cmp: &'static str) -> Self {
        Self { left, right, cmp }
    }

    pub fn and(left: &'a L, right: &'a R) -> Self {
        Self::new(left, right, "&&")
    }

    pub fn or(left: &'a L, right: &'a R) -> Self {
        Self::new(left, right, "||")
    }

    pub fn xor(left: &'a L, right: &'a R) -> Self {
        Self::new(left, right, "^")
    }
}

impl<'a, L, R> Op for ArrayBoolean<'a, L, R>
where
    L: NDArrayRead,
    R: NDArrayRead<DType = L::DType>,
{
    type Out = u8;

    fn enqueue(&self, queue: Queue) -> Result<Buffer<Self::Out>, Error> {
        debug_assert_eq!(self.left.shape(), self.right.shape());

        let right_queue = queue.context().queue(self.left.size())?;
        let right = self.right.read(right_queue)?;
        let left = self.left.read(queue)?;

        let output = match (left, right) {
            (Buffer::CL(left), Buffer::CL(right)) => {
                let right_cl_queue = right.default_queue().expect("right queue").clone();
                let event = right_cl_queue.enqueue_marker::<ocl::Event>(None)?;
                let cl_queue = left.default_queue().expect("left queue").clone();

                let output =
                    kernels::elementwise_boolean(self.cmp, cl_queue, &left, &right, &event)?;

                output.into()
            }
            (_, _) => todo!("array comparison on CPU"),
        };

        Ok(output)
    }
}

// TODO: remove the lifetime parameter
#[derive(Copy, Clone)]
pub struct ArrayCompare<'a, L, R> {
    left: &'a L,
    right: &'a R,
    cmp: &'static str,
}

impl<'a, L, R> ArrayCompare<'a, L, R> {
    fn new(left: &'a L, right: &'a R, cmp: &'static str) -> Self {
        Self { left, right, cmp }
    }

    pub fn eq(left: &'a L, right: &'a R) -> Self {
        Self::new(left, right, "==")
    }

    pub fn gt(left: &'a L, right: &'a R) -> Self {
        Self::new(left, right, ">")
    }

    pub fn gte(left: &'a L, right: &'a R) -> Self {
        Self::new(left, right, ">=")
    }

    pub fn lt(left: &'a L, right: &'a R) -> Self {
        Self::new(left, right, "<")
    }

    pub fn lte(left: &'a L, right: &'a R) -> Self {
        Self::new(left, right, "<=")
    }

    pub fn ne(left: &'a L, right: &'a R) -> Self {
        Self::new(left, right, "!=")
    }
}

impl<'a, T, L, R> Op for ArrayCompare<'a, L, R>
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

        let output = match (left, right) {
            (Buffer::CL(left), Buffer::CL(right)) => {
                let right_cl_queue = right.default_queue().expect("right queue").clone();
                let event = right_cl_queue.enqueue_marker::<ocl::Event>(None)?;
                let cl_queue = left.default_queue().expect("left queue").clone();
                let output = kernels::elementwise_cmp(self.cmp, cl_queue, &left, &right, &event)?;
                output.into()
            }
            (_, _) => todo!("compare arrays on CPU"),
        };

        Ok(output)
    }
}

#[derive(Copy, Clone)]
pub struct ArrayCompareScalar<'a, A, T> {
    array: &'a A,
    scalar: T,
    cmp: &'static str,
}

impl<'a, A, T> ArrayCompareScalar<'a, A, T> {
    fn new(array: &'a A, scalar: T, cmp: &'static str) -> Self {
        Self { array, scalar, cmp }
    }

    pub fn eq(array: &'a A, scalar: T) -> Self {
        Self::new(array, scalar, "==")
    }

    pub fn gt(array: &'a A, scalar: T) -> Self {
        Self::new(array, scalar, ">")
    }

    pub fn gte(array: &'a A, scalar: T) -> Self {
        Self::new(array, scalar, ">=")
    }

    pub fn lt(array: &'a A, scalar: T) -> Self {
        Self::new(array, scalar, "<")
    }

    pub fn lte(array: &'a A, scalar: T) -> Self {
        Self::new(array, scalar, "<=")
    }

    pub fn ne(array: &'a A, scalar: T) -> Self {
        Self::new(array, scalar, "!=")
    }
}

impl<'a, A: NDArrayRead> Op for ArrayCompareScalar<'a, A, A::DType> {
    type Out = u8;

    fn enqueue(&self, queue: Queue) -> Result<Buffer<Self::Out>, Error> {
        let input = self.array.read(queue)?;

        let output = match input {
            Buffer::CL(input) => {
                let cl_queue = input.default_queue().expect("queue").clone();
                let output = kernels::scalar_cmp(self.cmp, cl_queue, &input, self.scalar)?;
                output.into()
            }
            Buffer::Host(_) => todo!("compare array with scalar on CPU"),
        };

        Ok(output)
    }
}

// reduction

#[derive(Copy, Clone)]
pub struct ArrayReduce<'a, A> {
    source: &'a A,
    axis: usize,
    reduce: &'static str,
}

impl<'a, A> ArrayReduce<'a, A> {
    fn new(source: &'a A, axis: usize, reduce: &'static str) -> Self {
        Self {
            source,
            axis,
            reduce,
        }
    }

    pub fn sum(source: &'a A, axis: usize) -> Self {
        Self::new(source, axis, "add")
    }
}

impl<'a, A: NDArrayRead> Op for ArrayReduce<'a, A> {
    type Out = A::DType;

    fn enqueue(&self, queue: Queue) -> Result<Buffer<Self::Out>, Error> {
        debug_assert!(self.axis < self.source.ndim());
        let shape = self.source.shape().to_vec();
        let input = (&self.source).read(queue)?;

        let output = match input {
            Buffer::CL(input) => {
                let cl_queue = input.default_queue().expect("queue").clone();
                let output = kernels::reduce_axis(
                    A::DType::zero(),
                    self.reduce,
                    cl_queue,
                    input,
                    shape,
                    self.axis,
                )?;

                output.into()
            }
            Buffer::Host(_) => todo!("reduce axis on CPU"),
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
                output.into()
            }
            Buffer::Host(_) => todo!("cast array on CPU"),
        };

        Ok(output)
    }
}

#[derive(Copy, Clone)]
pub struct ArrayUnary<A> {
    array: A,
    op: &'static str,
}

impl<A> ArrayUnary<A> {
    fn new(array: A, op: &'static str) -> Self {
        Self { array, op }
    }

    // TODO: replace with abs for integer types
    pub fn abs(array: A) -> Self {
        Self::new(array, "fabs")
    }

    pub fn exp(array: A) -> Self {
        Self::new(array, "exp")
    }

    pub fn inf(array: A) -> Self {
        Self::new(array, "isinf")
    }

    pub fn nan(array: A) -> Self {
        Self::new(array, "isnan")
    }

    pub fn neg(array: A) -> Self {
        Self::new(array, "-")
    }

    pub fn not(array: A) -> Self {
        Self::new(array, "!")
    }
}

// TODO: can this be an in-place operation?
impl<A: NDArrayRead> Op for ArrayUnary<A> {
    type Out = A::DType;

    fn enqueue(&self, queue: Queue) -> Result<Buffer<Self::Out>, Error> {
        let input = self.array.read(queue)?;

        let output = match input {
            Buffer::CL(input) => {
                let cl_queue = input.default_queue().expect("queue").clone();
                let output = kernels::unary(self.op, cl_queue, input)?;
                output.into()
            }
            Buffer::Host(_) => todo!("unary array ops on CPU"),
        };

        Ok(output)
    }
}
