use std::fmt;

use ocl::{Buffer, OclPrm};

use super::{AxisBound, NDArray, Shape};

pub enum Error {
    Bounds(String),
    Platform(ocl::Error),
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Bounds(cause) => f.write_str(cause),
            Self::Platform(cause) => cause.fmt(f),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Bounds(cause) => f.write_str(cause),
            Self::Platform(cause) => cause.fmt(f),
        }
    }
}

impl std::error::Error for Error {}

pub struct ArrayBase<T: OclPrm> {
    buffer: Buffer<T>,
    shape: Shape,
}

impl<T: OclPrm> ArrayBase<T> {
    pub fn concatenate(arrays: Vec<Array<T>>, axis: usize) -> Result<Self, Error> {
        todo!()
    }

    pub fn constant(value: T, shape: Shape) -> Result<Self, Error> {
        todo!()
    }

    pub fn eye(shape: Shape) -> Result<Self, Error> {
        let ndim = shape.len();
        if shape.len() < 2 || shape[ndim - 1] != shape[ndim - 2] {
            Err(Error::Bounds(format!(
                "invalid shape for identity matrix: {:?}",
                shape
            )))
        } else {
            todo!()
        }
    }

    pub fn random(shape: Shape) -> Result<Self, Error> {
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
