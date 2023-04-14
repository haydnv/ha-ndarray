use std::marker::PhantomData;

use ocl::{Buffer, Event, OclPrm, Queue};

use super::{autoqueue, kernels, ArrayBase, ArrayOp, CDatatype, Error, NDArray, NDArrayRead};

pub trait Op {
    type Out: CDatatype;

    fn enqueue(&self, queue: Queue) -> Result<Buffer<Self::Out>, Error>;
}

// constructors

pub struct ArrayRandom {
    size: u64,
}

pub struct MatEye {
    count: u64,
    size: u64,
}

// arithmetic

pub struct ArrayAdd<L, R> {
    left: L,
    right: R,
}

impl<L, R> ArrayAdd<L, R> {
    pub fn new(left: L, right: R) -> Self {
        Self { left, right }
    }

    fn enqueue<T, LA, RA>(left: LA, right: RA, queue: Queue) -> Result<Buffer<T>, Error>
    where
        T: CDatatype,
        LA: NDArrayRead<T>,
        RA: NDArrayRead<T>,
    {
        let right_queue = autoqueue(Some(queue.context()))?;
        let right = right.read(queue.clone())?;
        let event = right_queue.enqueue_marker::<Event>(None)?;

        let left = left.read(queue.clone())?;

        kernels::elementwise_inplace("+=", queue, left, &right, &event).map_err(Error::from)
    }
}

impl<T: CDatatype> Op for ArrayAdd<ArrayBase<T>, ArrayBase<T>> {
    type Out = T;

    fn enqueue(&self, queue: Queue) -> Result<Buffer<Self::Out>, Error> {
        Self::enqueue(&self.left, &self.right, queue)
    }
}

impl<'a, T: CDatatype> Op for ArrayAdd<&'a ArrayBase<T>, &'a ArrayBase<T>> {
    type Out = T;

    fn enqueue(&self, queue: Queue) -> Result<Buffer<Self::Out>, Error> {
        Self::enqueue(self.left, self.right, queue)
    }
}

pub struct ArrayDiv<L, R> {
    left: L,
    right: R,
}

pub struct ArrayMul<L, R> {
    left: L,
    right: R,
}

pub struct ArrayMod<L, R> {
    left: L,
    right: R,
}

pub struct ArraySub<L, R> {
    left: L,
    right: R,
}

// linear algebra

pub struct MatDiag<A> {
    source: A,
}

pub struct MatMul<L, R> {
    left: L,
    right: R,
}

// comparison

pub struct ArrayCompare<L, R> {
    left: L,
    right: R,
    cmp: &'static str,
}

impl<L, R> ArrayCompare<L, R> {
    fn new(left: L, right: R, cmp: &'static str) -> Self {
        Self { left, right, cmp }
    }

    pub fn eq(left: L, right: R) -> Self {
        Self::new(left, right, "==")
    }

    pub fn gt(left: L, right: R) -> Self {
        Self::new(left, right, ">")
    }

    pub fn gte(left: L, right: R) -> Self {
        Self::new(left, right, ">=")
    }

    pub fn lt(left: L, right: R) -> Self {
        Self::new(left, right, "<")
    }

    pub fn lte(left: L, right: R) -> Self {
        Self::new(left, right, "<=")
    }

    pub fn ne(left: L, right: R) -> Self {
        Self::new(left, right, "!=")
    }

    fn enqueue<T, LA, RA>(
        cmp: &'static str,
        queue: Queue,
        left: LA,
        right: RA,
    ) -> Result<Buffer<u8>, Error>
    where
        T: CDatatype,
        LA: NDArrayRead<T>,
        RA: NDArrayRead<T>,
    {
        assert_eq!(left.shape(), right.shape());

        let right_queue = autoqueue(Some(queue.context()))?;
        let right = right.read(right_queue.clone())?;
        let event = right_queue.enqueue_marker::<Event>(None)?;

        let left = left.read(queue.clone())?;

        kernels::elementwise_cmp(cmp, queue, &left, &right, &event).map_err(Error::from)
    }
}

impl<T: CDatatype> Op for ArrayCompare<ArrayBase<T>, ArrayBase<T>> {
    type Out = u8;

    fn enqueue(&self, queue: Queue) -> Result<Buffer<u8>, Error> {
        Self::enqueue(self.cmp, queue, &self.left, &self.right)
    }
}

impl<'a, T: CDatatype> Op for ArrayCompare<ArrayBase<T>, &'a ArrayBase<T>> {
    type Out = u8;

    fn enqueue(&self, queue: Queue) -> Result<Buffer<u8>, Error> {
        Self::enqueue(self.cmp, queue, &self.left, self.right)
    }
}

impl<'a, T: CDatatype> Op for ArrayCompare<&'a ArrayBase<T>, ArrayBase<T>> {
    type Out = u8;

    fn enqueue(&self, queue: Queue) -> Result<Buffer<u8>, Error> {
        Self::enqueue(self.cmp, queue, self.left, &self.right)
    }
}

impl<'a, T: CDatatype> Op for ArrayCompare<&'a ArrayBase<T>, &'a ArrayBase<T>> {
    type Out = u8;

    fn enqueue(&self, queue: Queue) -> Result<Buffer<u8>, Error> {
        Self::enqueue(self.cmp, queue, self.left, self.right)
    }
}

impl<T: CDatatype, O: Op<Out = T>> Op for ArrayCompare<ArrayBase<T>, ArrayOp<O>> {
    type Out = u8;

    fn enqueue(&self, queue: Queue) -> Result<Buffer<u8>, Error> {
        Self::enqueue(self.cmp, queue, &self.left, &self.right)
    }
}

impl<'a, T: CDatatype, O: Op<Out = T>> Op for ArrayCompare<ArrayBase<T>, &'a ArrayOp<O>> {
    type Out = u8;

    fn enqueue(&self, queue: Queue) -> Result<Buffer<u8>, Error> {
        Self::enqueue(self.cmp, queue, &self.left, self.right)
    }
}

impl<'a, T: CDatatype, O: Op<Out = T>> Op for ArrayCompare<&'a ArrayBase<T>, ArrayOp<O>> {
    type Out = u8;

    fn enqueue(&self, queue: Queue) -> Result<Buffer<u8>, Error> {
        Self::enqueue(self.cmp, queue, self.left, &self.right)
    }
}

impl<'a, T: CDatatype, O: Op<Out = T>> Op for ArrayCompare<&'a ArrayBase<T>, &'a ArrayOp<O>> {
    type Out = u8;

    fn enqueue(&self, queue: Queue) -> Result<Buffer<u8>, Error> {
        Self::enqueue(self.cmp, queue, self.left, self.right)
    }
}

pub struct ArrayCompareScalar<A, T> {
    array: A,
    scalar: T,
    cmp: &'static str,
}

impl<A, T> ArrayCompareScalar<A, T> {
    fn new(array: A, scalar: T, cmp: &'static str) -> Self {
        Self { array, scalar, cmp }
    }

    pub fn eq(array: A, scalar: T) -> Self {
        Self::new(array, scalar, "==")
    }

    pub fn gt(array: A, scalar: T) -> Self {
        Self::new(array, scalar, ">")
    }

    pub fn gte(array: A, scalar: T) -> Self {
        Self::new(array, scalar, ">=")
    }

    pub fn lt(array: A, scalar: T) -> Self {
        Self::new(array, scalar, "<")
    }

    pub fn lte(array: A, scalar: T) -> Self {
        Self::new(array, scalar, "<=")
    }

    pub fn ne(array: A, scalar: T) -> Self {
        Self::new(array, scalar, "!=")
    }

    fn enqueue<LA>(
        cmp: &'static str,
        queue: Queue,
        array: LA,
        scalar: T,
    ) -> Result<Buffer<u8>, Error>
    where
        T: CDatatype,
        LA: NDArrayRead<T>,
    {
        let input = array.read(queue.clone())?;
        kernels::scalar_cmp(cmp, queue, &input, scalar).map_err(Error::from)
    }
}

impl<T: CDatatype> Op for ArrayCompareScalar<ArrayBase<T>, T> {
    type Out = u8;

    fn enqueue(&self, queue: Queue) -> Result<Buffer<u8>, Error> {
        Self::enqueue(self.cmp, queue, &self.array, self.scalar)
    }
}

impl<'a, T: CDatatype> Op for ArrayCompareScalar<&'a ArrayBase<T>, T> {
    type Out = u8;

    fn enqueue(&self, queue: Queue) -> Result<Buffer<u8>, Error> {
        Self::enqueue(self.cmp, queue, self.array, self.scalar)
    }
}

impl<O: Op> Op for ArrayCompareScalar<ArrayOp<O>, O::Out> {
    type Out = u8;

    fn enqueue(&self, queue: Queue) -> Result<Buffer<u8>, Error> {
        Self::enqueue(self.cmp, queue, &self.array, self.scalar)
    }
}

impl<'a, O: Op> Op for ArrayCompareScalar<&'a ArrayOp<O>, O::Out> {
    type Out = u8;

    fn enqueue(&self, queue: Queue) -> Result<Buffer<u8>, Error> {
        Self::enqueue(self.cmp, queue, self.array, self.scalar)
    }
}

// reduction

pub struct ArrayAll<A> {
    source: A,
}

pub struct ArrayAny<A> {
    source: A,
}

pub struct ArrayMax<A> {
    source: A,
}

pub struct ArrayMin<A> {
    source: A,
}

pub struct ArrayProduct<A> {
    source: A,
}

pub struct ArraySum<A> {
    source: A,
    axis: usize,
}

impl<A> ArraySum<A> {
    pub fn new(source: A, axis: usize) -> Self {
        Self { source, axis }
    }
}

impl<'a, T: CDatatype> Op for ArraySum<&'a ArrayBase<T>> {
    type Out = T;

    fn enqueue(&self, queue: Queue) -> Result<Buffer<Self::Out>, Error> {
        assert!(self.axis < self.source.ndim());

        let shape = self.source.shape().to_vec();
        let input = (&self.source).read(queue.clone())?;

        kernels::reduce_axis(T::zero(), "+=", queue, input, shape, self.axis).map_err(Error::from)
    }
}

impl<O: Op> Op for ArraySum<ArrayOp<O>> {
    type Out = O::Out;

    fn enqueue(&self, queue: Queue) -> Result<Buffer<Self::Out>, Error> {
        todo!()
    }
}

// other unary ops

pub struct ArrayCast<A, O> {
    source: A,
    dtype: PhantomData<O>,
}
