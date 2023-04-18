use std::fmt;
use std::ops::{Add, Div, Mul, Rem, Sub};
use std::sync::{Arc, RwLock};

use ocl::{Buffer, OclPrm, Queue};

use super::kernels;
use super::ops::*;
use super::{
    AxisBound, CDatatype, Error, MatrixMath, NDArray, NDArrayCompare, NDArrayCompareScalar,
    NDArrayRead, NDArrayReduce, NDArrayTransform, Shape,
};

#[derive(Clone)]
pub struct ArrayBase<T> {
    data: Arc<RwLock<Vec<T>>>,
    shape: Shape,
}

impl<T: OclPrm + CDatatype> ArrayBase<T> {
    fn new(shape: Shape, data: Vec<T>) -> Self {
        Self {
            data: Arc::new(RwLock::new(data)),
            shape,
        }
    }

    pub fn concatenate(arrays: Vec<Array<T>>, axis: usize) -> Result<Self, Error> {
        todo!()
    }

    pub fn constant(shape: Shape, value: T) -> Self {
        let size = shape.iter().product();
        Self::new(shape, vec![value; size])
    }

    pub fn from_vec(shape: Shape, data: Vec<T>) -> Result<Self, Error> {
        let size = shape.iter().product();
        if data.len() == size {
            Ok(Self::new(shape, data))
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

    pub fn to_vec(&self) -> Vec<T> {
        let data = self.data.read().expect("array data");
        data.to_vec()
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
        if shape.len() < self.ndim() {
            return Err(Error::Bounds(format!(
                "cannot broadcast {:?} into {:?}",
                self.shape, shape
            )));
        }

        for (dim, bdim) in self.shape.iter().zip(&shape[shape.len() - self.ndim()..]) {
            if dim == bdim || *dim == 1 {
                // ok
            } else {
                return Err(Error::Bounds(format!(
                    "cannot broadcast dimension {} into {}",
                    dim, bdim
                )));
            }
        }

        let strides = strides_for(&self.shape, shape.len());

        Ok(ArrayView::new(self.clone(), shape, strides))
    }

    fn transpose(&self, axes: Option<Vec<usize>>) -> Result<ArrayView<Self>, Error> {
        todo!()
    }

    fn reshape(&self, shape: Shape) -> Result<Self, Error> {
        if shape.iter().product::<usize>() == self.size() {
            Ok(Self {
                shape,
                data: self.data.clone(),
            })
        } else {
            Err(Error::Bounds(format!(
                "cannot reshape from {:?} to {:?}",
                self.shape, shape
            )))
        }
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
        let data = self.data.read().expect("array data");

        let buffer = Buffer::<T>::builder()
            .queue(queue)
            .len(self.size())
            .build()
            .map_err(Error::from)?;

        buffer.write(data.as_slice()).enq()?;

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
    shape: Shape,
    strides: Vec<usize>,
}

impl<A> ArrayView<A> {
    fn new(source: A, shape: Shape, strides: Vec<usize>) -> Self {
        Self {
            source,
            shape,
            strides,
        }
    }
}

impl<A> NDArray for ArrayView<A> {
    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl<T: CDatatype, A: NDArrayRead<T>> NDArrayRead<T> for ArrayView<A> {
    fn read(&self, queue: Queue) -> Result<Buffer<T>, Error> {
        let buffer = self.source.read(queue.clone())?;
        let strides = strides_for(&self.shape, self.ndim());

        if self.size() == self.source.size() {
            kernels::reorder_inplace(queue, buffer, &self.shape, &self.strides, &strides)
                .map_err(Error::from)
        } else {
            kernels::reorder(queue, buffer, &self.shape, &strides, &self.strides)
                .map_err(Error::from)
        }
    }
}

impl<A: NDArray> NDArrayTransform for ArrayView<A> {
    type Slice = ArraySlice<Self>;
    type View = Self;

    fn broadcast(&self, shape: Shape) -> Result<Self::View, Error> {
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

#[inline]
fn strides_for(shape: &[usize], ndim: usize) -> Vec<usize> {
    debug_assert!(ndim >= shape.len());

    let mut strides = vec![0; ndim];

    let offset = ndim - shape.len();
    let mut stride = 1;
    for (x, dim) in shape.iter().enumerate().rev() {
        if *dim == 1 {
            strides[offset + x] = 0;
        } else {
            strides[offset + x] = stride;
            stride *= dim;
        }
    }

    strides
}
