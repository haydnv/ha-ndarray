use std::fmt;
use std::ops::{Add, Div, Mul, Rem, Sub};

use ocl::{Buffer, OclPrm, Queue};

use super::ops::*;
use super::{
    AxisBound, CDatatype, Error, MatrixMath, NDArray, NDArrayCompare, NDArrayCompareScalar,
    NDArrayRead, NDArrayReduce, NDArrayTransform, Shape,
};

#[derive(Clone)]
pub struct ArrayBase<T> {
    data: Vec<T>,
    shape: Shape,
}

impl<T: OclPrm + CDatatype> ArrayBase<T> {
    pub fn concatenate(arrays: Vec<Array<T>>, axis: usize) -> Result<Self, Error> {
        todo!()
    }

    pub fn constant(shape: Shape, value: T) -> Self {
        let size = shape.iter().product();
        Self {
            data: vec![value; size],
            shape,
        }
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

    pub fn from_vec(shape: Shape, data: Vec<T>) -> Result<Self, Error> {
        let size = shape.iter().product();
        if data.len() == size {
            Ok(Self { shape, data })
        } else {
            Err(Error::Bounds(format!(
                "{} data were provided for an array of size {}",
                data.len(),
                size
            )))
        }
    }

    pub fn random(shape: Shape) -> Result<Self, Error> {
        todo!()
    }

    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    pub fn to_vec(&self) -> Vec<T> {
        self.data.to_vec()
    }
}

impl<T> NDArray for ArrayBase<T> {
    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl<'a, T> NDArray for &'a ArrayBase<T> {
    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl<T: CDatatype> NDArrayTransform for ArrayBase<T> {
    type Slice = ArraySlice<Self>;
    type View = ArrayView<Self>;

    fn broadcast(&self, shape: Shape) -> Result<ArrayView<Self>, Error> {
        todo!()
    }

    fn expand_dims<Dims: IntoIterator<Item = usize>>(
        &self,
        dims: Dims,
    ) -> Result<ArrayView<Self>, Error> {
        todo!()
    }

    fn transpose(&self, axes: Option<Vec<usize>>) -> Result<ArrayView<Self>, Error> {
        todo!()
    }

    fn reshape(&self, shape: Shape) -> Result<ArrayView<Self>, Error> {
        todo!()
    }

    fn slice<Bounds: IntoIterator<Item = AxisBound>>(
        &self,
        bounds: Bounds,
    ) -> Result<ArraySlice<Self>, Error> {
        todo!()
    }
}

macro_rules! impl_op {
    ($op:ident, $name:ident, $t:ty, $o:ty) => {
        impl<T, O> $op<$o> for $t {
            type Output = ArrayOp<ArrayDual<Self, $o>>;

            fn $name(self, rhs: $o) -> Self::Output {
                let shape = self.shape().to_vec();
                assert_eq!(shape, rhs.shape());

                let op = ArrayDual::$name(self, rhs);
                ArrayOp { op, shape }
            }
        }

        impl<'a, T, O> $op<&'a $o> for $t {
            type Output = ArrayOp<ArrayDual<Self, &'a $o>>;

            fn $name(self, rhs: &'a $o) -> Self::Output {
                let shape = self.shape().to_vec();
                assert_eq!(shape, rhs.shape());

                let op = ArrayDual::$name(self, rhs);
                ArrayOp { op, shape }
            }
        }

        impl<'a, T, O> $op<$o> for &'a $t {
            type Output = ArrayOp<ArrayDual<Self, $o>>;

            fn $name(self, rhs: $o) -> Self::Output {
                let shape = self.shape().to_vec();
                assert_eq!(shape, rhs.shape());

                let op = ArrayDual::$name(self, rhs);
                ArrayOp { op, shape }
            }
        }

        impl<'a, T, O> $op<&'a $o> for &'a $t {
            type Output = ArrayOp<ArrayDual<Self, &'a $o>>;

            fn $name(self, rhs: &'a $o) -> Self::Output {
                let shape = self.shape().to_vec();
                assert_eq!(shape, rhs.shape());

                let op = ArrayDual::$name(self, rhs);
                ArrayOp { op, shape }
            }
        }
    };
}

impl_op!(Add, add, ArrayBase<T>, ArrayBase<O>);
impl_op!(Div, div, ArrayBase<T>, ArrayBase<O>);
impl_op!(Mul, mul, ArrayBase<T>, ArrayBase<O>);
impl_op!(Rem, rem, ArrayBase<T>, ArrayBase<O>);
impl_op!(Sub, sub, ArrayBase<T>, ArrayBase<O>);

impl_op!(Add, add, ArrayBase<T>, ArrayOp<O>);
impl_op!(Div, div, ArrayBase<T>, ArrayOp<O>);
impl_op!(Mul, mul, ArrayBase<T>, ArrayOp<O>);
impl_op!(Rem, rem, ArrayBase<T>, ArrayOp<O>);
impl_op!(Sub, sub, ArrayBase<T>, ArrayOp<O>);

impl_op!(Add, add, ArrayBase<T>, ArraySlice<O>);
impl_op!(Div, div, ArrayBase<T>, ArraySlice<O>);
impl_op!(Mul, mul, ArrayBase<T>, ArraySlice<O>);
impl_op!(Rem, rem, ArrayBase<T>, ArraySlice<O>);
impl_op!(Sub, sub, ArrayBase<T>, ArraySlice<O>);

impl_op!(Add, add, ArrayBase<T>, ArrayView<O>);
impl_op!(Div, div, ArrayBase<T>, ArrayView<O>);
impl_op!(Mul, mul, ArrayBase<T>, ArrayView<O>);
impl_op!(Rem, rem, ArrayBase<T>, ArrayView<O>);
impl_op!(Sub, sub, ArrayBase<T>, ArrayView<O>);

impl<T: CDatatype, A: NDArrayRead<T>> MatrixMath<T, A> for ArrayBase<T> {}

impl<T: CDatatype> NDArrayCompare<T, Self> for ArrayBase<T> {}

impl<T: CDatatype, Op: super::ops::Op<Out = T>> NDArrayCompare<T, ArrayOp<Op>> for ArrayBase<T> {}

impl<T: CDatatype> NDArrayCompareScalar<T> for ArrayBase<T> {}

impl<T: CDatatype> NDArrayRead<T> for ArrayBase<T> {
    fn read(&self, queue: Queue) -> Result<Buffer<T>, Error> {
        let buffer = Buffer::<T>::builder()
            .queue(queue)
            .len(self.size())
            .build()
            .map_err(Error::from)?;

        buffer.write(&self.data).enq()?;

        Ok(buffer)
    }
}

impl<T: CDatatype> NDArrayReduce<T> for ArrayBase<T> {}

impl<T: CDatatype> fmt::Debug for ArrayBase<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} array with shape {:?}", T::TYPE_STR, self.shape)
    }
}

pub struct ArrayOp<Op> {
    op: Op,
    shape: Shape,
}

impl<Op> ArrayOp<Op> {
    pub fn new(op: Op, shape: Shape) -> Self {
        Self { op, shape }
    }
}

impl<Op> NDArray for ArrayOp<Op> {
    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl<Op: super::ops::Op> NDArrayCompareScalar<Op::Out> for ArrayOp<Op> {}

impl<Op: super::ops::Op> NDArrayRead<Op::Out> for ArrayOp<Op> {
    fn read(&self, queue: Queue) -> Result<Buffer<Op::Out>, Error> {
        self.op.enqueue(queue)
    }
}

impl<Op: super::ops::Op> NDArrayReduce<Op::Out> for ArrayOp<Op> {}

impl<Op: super::ops::Op> fmt::Debug for ArrayOp<Op> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{} array result with shape {:?}",
            Op::Out::TYPE_STR,
            self.shape
        )
    }
}

pub struct ArraySlice<A> {
    source: A,
    bounds: Vec<AxisBound>,
    shape: Shape,
}

impl<A> NDArray for ArraySlice<A> {
    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

pub struct ArrayView<A> {
    source: A,
    strides: Vec<u64>,
    shape: Shape,
}

impl<A> NDArray for ArrayView<A> {
    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl<T: CDatatype, A: NDArrayRead<T>> NDArrayRead<T> for ArrayView<A> {
    fn read(&self, queue: Queue) -> Result<Buffer<T>, Error> {
        todo!()
    }
}

impl<A: NDArray> NDArrayTransform for ArrayView<A> {
    type Slice = ArraySlice<Self>;
    type View = Self;

    fn broadcast(&self, shape: Shape) -> Result<Self::View, Error> {
        todo!()
    }

    fn expand_dims<Dims: IntoIterator<Item = usize>>(
        &self,
        dims: Dims,
    ) -> Result<Self::View, Error> {
        todo!()
    }

    fn transpose(&self, axes: Option<Vec<usize>>) -> Result<Self::View, Error> {
        todo!()
    }

    fn reshape(&self, shape: Shape) -> Result<Self::View, Error> {
        todo!()
    }

    fn slice<Bounds: IntoIterator<Item = AxisBound>>(
        &self,
        bounds: Bounds,
    ) -> Result<Self::Slice, Error> {
        todo!()
    }
}

pub enum Array<T: OclPrm> {
    Base(ArrayBase<T>),
    Slice(ArraySlice<Box<Self>>),
    View(ArrayView<Box<Self>>),
    Op(ArrayOp<Box<dyn super::ops::Op<Out = T>>>),
}

impl<T: OclPrm> NDArray for Array<T> {
    fn shape(&self) -> &[usize] {
        match self {
            Self::Base(base) => base.shape(),
            Self::Slice(slice) => slice.shape(),
            Self::View(view) => view.shape(),
            Self::Op(op) => op.shape(),
        }
    }
}
