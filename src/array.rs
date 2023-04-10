use ocl::{Buffer, OclPrm};

use super::{AxisBound, NDArray, Shape};

pub struct ArrayBase<T: OclPrm> {
    buffer: Buffer<T>,
    shape: Shape,
}

impl<T: OclPrm> ArrayBase<T> {
    pub fn concatenate(arrays: Vec<Array<T>>, axis: usize) -> Self {
        todo!()
    }

    pub fn constant(value: T, shape: Shape) -> Self {
        todo!()
    }
}

impl<T: OclPrm> NDArray for ArrayBase<T> {
    fn shape(&self) -> &[u64] {
        &self.shape
    }
}

pub struct ArrayOp<Op> {
    op: Op,
    shape: Shape,
}

impl<Op> NDArray for ArrayOp<Op> {
    fn shape(&self) -> &[u64] {
        &self.shape
    }
}

pub struct ArraySlice<A> {
    source: A,
    bounds: Vec<AxisBound>,
    shape: Shape,
}

impl<A> NDArray for ArraySlice<A> {
    fn shape(&self) -> &[u64] {
        &self.shape
    }
}

pub struct ArrayView<A> {
    source: A,
    strides: Vec<u64>,
    shape: Shape,
}

impl<A> NDArray for ArrayView<A> {
    fn shape(&self) -> &[u64] {
        &self.shape
    }
}

pub enum Array<T: OclPrm> {
    Base(ArrayBase<T>),
    Slice(ArraySlice<Box<Self>>),
    View(ArrayView<Box<Self>>),
    Op(ArrayOp<Box<dyn super::ops::Op<T>>>),
}

impl<T: OclPrm> NDArray for Array<T> {
    fn shape(&self) -> &[u64] {
        match self {
            Self::Base(base) => base.shape(),
            Self::Slice(slice) => slice.shape(),
            Self::View(view) => view.shape(),
            Self::Op(op) => op.shape(),
        }
    }
}
