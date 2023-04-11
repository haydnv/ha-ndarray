use std::marker::PhantomData;

use ocl::{Buffer, Queue};

pub mod kernels;

use super::array::ArrayBase;
use super::{CDatatype, Error, NDArray, NDArrayRead};

pub trait Op {
    type Out: CDatatype;

    fn enqueue(
        &self,
        queue: Queue,
        output: Option<Buffer<Self::Out>>,
    ) -> Result<Buffer<Self::Out>, Error>;
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

pub struct ArrayEq<L, R> {
    left: L,
    right: R,
}

impl<L, R> ArrayEq<L, R> {
    pub fn new(left: L, right: R) -> Self {
        Self { left, right }
    }

    fn enqueue<T, LA, RA>(
        queue: Queue,
        left: LA,
        right: RA,
        output: Option<Buffer<u8>>,
    ) -> Result<Buffer<u8>, Error>
    where
        T: CDatatype,
        LA: NDArrayRead<T>,
        RA: NDArrayRead<T>,
    {
        assert_eq!(left.shape(), right.shape());

        let output = if let Some(output) = output {
            output
        } else {
            Buffer::builder()
                .queue(queue.clone())
                .len(left.size())
                .build()?
        };

        let left = left.read(queue.clone(), None)?;
        let right = right.read(queue.clone(), None)?;

        kernels::elementwise("==", queue, &left, &right, output).map_err(Error::from)
    }
}

impl<T: CDatatype> Op for ArrayEq<ArrayBase<T>, ArrayBase<T>> {
    type Out = u8;

    fn enqueue(&self, queue: Queue, output: Option<Buffer<u8>>) -> Result<Buffer<u8>, Error> {
        Self::enqueue(queue, &self.left, &self.right, output)
    }
}

impl<'a, T: CDatatype> Op for ArrayEq<ArrayBase<T>, &'a ArrayBase<T>> {
    type Out = u8;

    fn enqueue(&self, queue: Queue, output: Option<Buffer<u8>>) -> Result<Buffer<u8>, Error> {
        Self::enqueue(queue, &self.left, self.right, output)
    }
}

impl<'a, T: CDatatype> Op for ArrayEq<&'a ArrayBase<T>, ArrayBase<T>> {
    type Out = u8;

    fn enqueue(&self, queue: Queue, output: Option<Buffer<u8>>) -> Result<Buffer<u8>, Error> {
        Self::enqueue(queue, self.left, &self.right, output)
    }
}

impl<'a, T: CDatatype> Op for ArrayEq<&'a ArrayBase<T>, &'a ArrayBase<T>> {
    type Out = u8;

    fn enqueue(&self, queue: Queue, output: Option<Buffer<u8>>) -> Result<Buffer<u8>, Error> {
        Self::enqueue(queue, self.left, self.right, output)
    }
}

pub struct ArrayGT<L, R> {
    left: L,
    right: R,
}

pub struct ArrayGTE<L, R> {
    left: L,
    right: R,
}

pub struct ArrayLT<L, R> {
    left: L,
    right: R,
}

pub struct ArrayLTE<L, R> {
    left: L,
    right: R,
}

pub struct ArrayNE<L, R> {
    left: L,
    right: R,
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
}

// other unary ops

pub struct ArrayCast<A, O> {
    source: A,
    dtype: PhantomData<O>,
}
