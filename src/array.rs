use std::cmp::Ordering;
use std::ops::{Add, Div, Mul, Neg, Not, Rem, Sub};
use std::sync::{Arc, RwLock};
use std::{fmt, iter};

use ocl::{Buffer, OclPrm, Queue};
use rand::Rng;

use super::kernels;
use super::ops::*;
use super::{
    autoqueue, AxisBound, CDatatype, Error, MatrixMath, NDArray, NDArrayBoolean, NDArrayCompare,
    NDArrayCompareScalar, NDArrayExp, NDArrayMath, NDArrayMathScalar, NDArrayNumeric, NDArrayRead,
    NDArrayReduce, NDArrayTransform, NDArrayWrite, Shape,
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

    pub fn to_vec(&self) -> Vec<T> {
        let data = self.data.read().expect("array data");
        data.to_vec()
    }
}

impl ArrayBase<f32> {
    pub fn random_normal(shape: Shape, seed: Option<usize>) -> Result<Self, Error> {
        let seed = seed.unwrap_or_else(|| {
            let mut rng = rand::thread_rng();
            rng.gen()
        });

        let size = shape.iter().product();
        let queue = autoqueue(None)?;
        let buffer = kernels::random_normal(queue.clone(), seed, size)?;

        let mut data = vec![0.; size];
        buffer.read(&mut data[..]).enq()?;

        queue.finish()?;

        Self::from_vec(shape, data)
    }

    pub fn random_uniform(shape: Shape, seed: Option<usize>) -> Result<Self, Error> {
        let seed = seed.unwrap_or_else(|| {
            let mut rng = rand::thread_rng();
            rng.gen()
        });

        let size = shape.iter().product();
        let queue = autoqueue(None)?;
        let buffer = kernels::random_uniform(queue.clone(), seed, size)?;

        let mut data = vec![0.; size];
        buffer.read(&mut data[..]).enq()?;

        queue.finish()?;

        Self::from_vec(shape, data)
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

impl<T: CDatatype> NDArrayExp for ArrayBase<T> {}

impl<T: CDatatype> NDArrayTransform for ArrayBase<T> {
    type Broadcast = ArrayView<Self>;
    type Expand = Self;
    type Reshape = Self;
    type Slice = ArraySlice<Self>;
    type Transpose = ArrayView<Self>;

    fn broadcast(&self, shape: Shape) -> Result<ArrayView<Self>, Error> {
        ArrayView::broadcast(self.clone(), shape)
    }

    fn expand_dim(&self, axis: usize) -> Result<Self::Expand, Error> {
        if axis > self.ndim() {
            return Err(Error::Bounds(format!(
                "cannot expand axis {} of {:?}",
                axis, self
            )));
        }

        let mut shape = Vec::with_capacity(self.ndim() + 1);
        shape.extend_from_slice(&self.shape);
        shape.insert(axis, 1);

        let data = self.data.clone();

        Ok(Self { data, shape })
    }

    fn expand_dims(&self, axes: Vec<usize>) -> Result<Self::Expand, Error> {
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

    fn slice(&self, bounds: Vec<AxisBound>) -> Result<ArraySlice<Self>, Error> {
        ArraySlice::new(self.clone(), bounds)
    }

    fn transpose(&self, axes: Option<Vec<usize>>) -> Result<ArrayView<Self>, Error> {
        let axes = if let Some(axes) = axes {
            if axes.len() == self.ndim() && (0..self.ndim()).into_iter().all(|x| axes.contains(&x))
            {
                Ok(axes)
            } else {
                Err(Error::Bounds(format!(
                    "invalid permutation {:?} for shape {:?}",
                    axes, self.shape
                )))
            }
        } else {
            Ok((0..self.ndim()).into_iter().rev().collect())
        }?;

        let shape = axes.iter().copied().map(|x| self.shape[x]).collect();

        let source_strides = strides_for(&self.shape, self.ndim());
        let strides = axes.into_iter().map(|x| source_strides[x]).collect();

        Ok(ArrayView::new(self.clone(), shape, strides))
    }
}

impl<A: NDArrayRead> NDArrayBoolean<A> for ArrayBase<A::Out> {}

impl<T: CDatatype> NDArrayMath<ArrayBase<f64>> for ArrayBase<T> {}

impl<T: CDatatype, Op: super::ops::Op<Out = f64>> NDArrayMath<ArrayOp<Op>> for ArrayBase<T> {}

impl<T: CDatatype> NDArrayNumeric for ArrayBase<T> {}

impl<T: CDatatype, A: NDArrayRead<Out = f64>> NDArrayMath<ArraySlice<A>> for ArrayBase<T> {}

impl<T: CDatatype, A: NDArrayRead<Out = f64>> NDArrayMath<ArrayView<A>> for ArrayBase<T> {}

impl<T: CDatatype> NDArrayMathScalar for ArrayBase<T> {}

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

macro_rules! impl_base_scalar_op {
    ($op:ident, $name:ident) => {
        impl<T: CDatatype> $op<T> for ArrayBase<T> {
            type Output = ArrayOp<ArrayScalar<T, Self>>;

            fn $name(self, rhs: T) -> Self::Output {
                let shape = self.shape.to_vec();
                let op = ArrayScalar::$name(self, rhs);
                ArrayOp::new(op, shape)
            }
        }
    };
}

impl_base_scalar_op!(Add, add);
impl_base_scalar_op!(Div, div);
impl_base_scalar_op!(Mul, mul);
impl_base_scalar_op!(Rem, rem);
impl_base_scalar_op!(Sub, sub);

impl<T: CDatatype> Neg for ArrayBase<T> {
    type Output = ArrayOp<ArrayUnary<Self>>;

    fn neg(self) -> Self::Output {
        let shape = self.shape.to_vec();
        let op = ArrayUnary::neg(self);
        ArrayOp::new(op, shape)
    }
}

impl<T: CDatatype> Not for ArrayBase<T> {
    type Output = ArrayOp<ArrayUnary<Self>>;

    fn not(self) -> Self::Output {
        let shape = self.shape.to_vec();
        let op = ArrayUnary::not(self);
        ArrayOp::new(op, shape)
    }
}

impl<A: NDArrayRead> MatrixMath<A> for ArrayBase<A::Out> {}

impl<T: CDatatype, A: NDArrayRead<Out = T>> NDArrayCompare<A> for ArrayBase<T> {}

impl<T: CDatatype> NDArrayCompareScalar<T> for ArrayBase<T> {}

impl<T: CDatatype> NDArrayRead for ArrayBase<T> {
    type Out = T;

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

impl<A: NDArrayRead + fmt::Debug> NDArrayWrite<A> for ArrayBase<A::Out> {
    fn write(&self, other: &A) -> Result<(), Error> {
        if self.shape == other.shape() {
            let queue = autoqueue(None)?;
            let buffer = other.read(queue)?;
            let mut data = self.data.write().expect("data");
            buffer.read(&mut data[..]).enq().map_err(Error::from)
        } else {
            Err(Error::Bounds(format!(
                "cannot write {:?} to {:?}",
                other, self
            )))
        }
    }
}

impl<T: CDatatype> NDArrayReduce for ArrayBase<T> {}

impl<T: CDatatype> fmt::Debug for ArrayBase<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} array with shape {:?}", T::TYPE_STR, self.shape)
    }
}

#[derive(Clone)]
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

impl<Op: super::ops::Op> NDArrayRead for ArrayOp<Op> {
    type Out = Op::Out;

    fn read(&self, queue: Queue) -> Result<Buffer<Op::Out>, Error> {
        self.op.enqueue(queue)
    }
}

impl<Op: super::ops::Op> NDArrayTransform for ArrayOp<Op>
where
    Self: Clone,
    Op: Clone,
{
    type Broadcast = ArrayView<Self>;
    type Expand = Self;
    type Reshape = Self;
    type Slice = ArraySlice<Self>;
    type Transpose = ArrayView<Self>;

    fn broadcast(&self, shape: Shape) -> Result<Self::Broadcast, Error> {
        ArrayView::broadcast(self.clone(), shape)
    }

    fn expand_dim(&self, axis: usize) -> Result<Self::Expand, Error> {
        if axis > self.ndim() {
            return Err(Error::Bounds(format!(
                "cannot expand axis {} of {:?}",
                axis, self
            )));
        }

        let mut shape = Vec::with_capacity(self.ndim() + 1);
        shape.extend_from_slice(&self.shape);
        shape.insert(axis, 1);

        Ok(Self {
            op: self.op.clone(),
            shape,
        })
    }

    fn expand_dims(&self, axes: Vec<usize>) -> Result<Self::Expand, Error> {
        todo!()
    }

    fn reshape(&self, shape: Shape) -> Result<Self::Reshape, Error> {
        todo!()
    }

    fn slice(&self, bounds: Vec<AxisBound>) -> Result<Self::Slice, Error> {
        ArraySlice::new(self.clone(), bounds)
    }

    fn transpose(&self, axes: Option<Vec<usize>>) -> Result<Self::Transpose, Error> {
        ArrayView::transpose(self.clone(), axes)
    }
}

impl<Op: super::ops::Op> NDArrayCompareScalar<Op::Out> for ArrayOp<Op> {}

impl<Op: super::ops::Op> NDArrayExp for ArrayOp<Op> where Self: Clone {}

impl<Op: super::ops::Op> NDArrayNumeric for ArrayOp<Op> where Self: Clone {}

impl<Op: super::ops::Op> NDArrayReduce for ArrayOp<Op> where Self: Clone {}

impl<Op: super::ops::Op, O: NDArrayRead<Out = Op::Out>> MatrixMath<O> for ArrayOp<Op> {}

impl<A, Op> NDArrayBoolean<A> for ArrayOp<Op>
where
    A: NDArrayRead,
    Op: super::ops::Op<Out = A::Out>,
{
}

impl<Op: super::ops::Op> NDArrayMathScalar for ArrayOp<Op> where Self: Clone {}

impl_op!(Add, add, ArrayOp<T>, ArrayBase<O>);
impl_op!(Div, div, ArrayOp<T>, ArrayBase<O>);
impl_op!(Mul, mul, ArrayOp<T>, ArrayBase<O>);
impl_op!(Rem, rem, ArrayOp<T>, ArrayBase<O>);
impl_op!(Sub, sub, ArrayOp<T>, ArrayBase<O>);

impl_op!(Add, add, ArrayOp<T>, ArrayOp<O>);
impl_op!(Div, div, ArrayOp<T>, ArrayOp<O>);
impl_op!(Mul, mul, ArrayOp<T>, ArrayOp<O>);
impl_op!(Rem, rem, ArrayOp<T>, ArrayOp<O>);
impl_op!(Sub, sub, ArrayOp<T>, ArrayOp<O>);

impl_op!(Add, add, ArrayOp<T>, ArraySlice<O>);
impl_op!(Div, div, ArrayOp<T>, ArraySlice<O>);
impl_op!(Mul, mul, ArrayOp<T>, ArraySlice<O>);
impl_op!(Rem, rem, ArrayOp<T>, ArraySlice<O>);
impl_op!(Sub, sub, ArrayOp<T>, ArraySlice<O>);

impl_op!(Add, add, ArrayOp<T>, ArrayView<O>);
impl_op!(Div, div, ArrayOp<T>, ArrayView<O>);
impl_op!(Mul, mul, ArrayOp<T>, ArrayView<O>);
impl_op!(Rem, rem, ArrayOp<T>, ArrayView<O>);
impl_op!(Sub, sub, ArrayOp<T>, ArrayView<O>);

macro_rules! impl_op_scalar_op {
    ($op:ident, $name:ident) => {
        impl<T: CDatatype, Op: super::ops::Op<Out = T>> $op<T> for ArrayOp<Op> {
            type Output = ArrayOp<ArrayScalar<Op::Out, Self>>;

            fn $name(self, rhs: Op::Out) -> Self::Output {
                let shape = self.shape.to_vec();
                let op = ArrayScalar::$name(self, rhs);
                ArrayOp::new(op, shape)
            }
        }
    };
}

impl_op_scalar_op!(Add, add);
impl_op_scalar_op!(Mul, mul);
impl_op_scalar_op!(Div, div);
impl_op_scalar_op!(Rem, rem);
impl_op_scalar_op!(Sub, sub);

impl<Op: super::ops::Op> Neg for ArrayOp<Op> {
    type Output = ArrayOp<ArrayUnary<Self>>;

    fn neg(self) -> Self::Output {
        let shape = self.shape.to_vec();
        let op = ArrayUnary::neg(self);
        ArrayOp::new(op, shape)
    }
}

impl<Op: super::ops::Op> Not for ArrayOp<Op> {
    type Output = ArrayOp<ArrayUnary<Self>>;

    fn not(self) -> Self::Output {
        let shape = self.shape.to_vec();
        let op = ArrayUnary::not(self);
        ArrayOp::new(op, shape)
    }
}

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

#[derive(Clone)]
pub struct ArraySlice<A> {
    source: A,
    bounds: Vec<AxisBound>,
    shape: Shape,
}

impl<A: NDArray> ArraySlice<A> {
    pub fn new(source: A, mut bounds: Vec<AxisBound>) -> Result<Self, Error> {
        if bounds.len() > source.ndim() {
            return Err(Error::Bounds(format!(
                "shape {:?} does not support slice bounds {:?}",
                source.shape(),
                bounds
            )));
        }

        for (bound, dim) in bounds.iter().zip(source.shape()) {
            match bound {
                AxisBound::At(i) => check_bound(i, dim, true)?,
                AxisBound::In(start, stop, _step) => {
                    check_bound(start, dim, false)?;
                    check_bound(stop, dim, false)?;
                }
                AxisBound::Of(indices) => {
                    for i in indices {
                        check_bound(i, dim, true)?;
                    }
                }
            }
        }

        let tail_bounds = source
            .shape()
            .iter()
            .rev()
            .take(source.ndim() - bounds.len())
            .copied()
            .map(|dim| AxisBound::In(0, dim, 1))
            .rev();

        bounds.extend(tail_bounds);

        debug_assert_eq!(source.ndim(), bounds.len());

        let shape = bounds
            .iter()
            .map(|bound| bound.size())
            .filter(|size| *size > 0)
            .collect();

        Ok(Self {
            source,
            bounds,
            shape,
        })
    }
}

impl<A> NDArray for ArraySlice<A> {
    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl<A: NDArrayRead> NDArrayRead for ArraySlice<A> {
    type Out = A::Out;

    fn read(&self, queue: Queue) -> Result<Buffer<Self::Out>, Error> {
        let buffer = self.source.read(queue.clone())?;
        let strides = strides_for(self.shape(), self.ndim());
        let source_strides = strides_for(self.source.shape(), self.source.ndim());

        kernels::slice(
            queue,
            &buffer,
            self.shape(),
            &strides,
            &self.bounds,
            &source_strides,
        )
        .map_err(Error::from)
    }
}

impl<A: NDArray + fmt::Debug> NDArrayTransform for ArraySlice<A>
where
    Self: Clone,
{
    type Broadcast = ArrayView<Self>;
    type Expand = ArrayView<Self>;
    type Reshape = ArrayView<Self>;
    type Slice = Self;
    type Transpose = ArrayView<Self>;

    fn broadcast(&self, shape: Shape) -> Result<Self::Broadcast, Error> {
        todo!()
    }

    fn expand_dim(&self, axis: usize) -> Result<Self::Expand, Error> {
        if axis > self.ndim() {
            return Err(Error::Bounds(format!(
                "cannot expand axis {} of {:?}",
                axis, self
            )));
        }

        let mut shape = Vec::with_capacity(self.ndim() + 1);
        shape.extend_from_slice(&self.shape);
        shape.insert(axis, 1);

        let strides = strides_for(&shape, shape.len());

        Ok(ArrayView::new(self.clone(), shape, strides))
    }

    fn expand_dims(&self, axes: Vec<usize>) -> Result<Self::Expand, Error> {
        todo!()
    }

    fn reshape(&self, shape: Shape) -> Result<ArrayView<Self>, Error> {
        todo!()
    }

    fn slice(&self, bounds: Vec<AxisBound>) -> Result<Self::Slice, Error> {
        todo!()
    }

    fn transpose(&self, axes: Option<Vec<usize>>) -> Result<Self::Transpose, Error> {
        todo!()
    }
}

impl<A, O> NDArrayBoolean<O> for ArraySlice<A>
where
    A: NDArrayRead,
    O: NDArrayRead<Out = A::Out>,
{
}

impl<A: NDArray> NDArrayExp for ArraySlice<A> where Self: Clone {}

impl<T, A, O> MatrixMath<O> for ArraySlice<A>
where
    T: CDatatype,
    A: NDArrayRead<Out = T>,
    O: NDArrayRead<Out = T>,
{
}

impl<A: NDArrayRead> NDArrayMathScalar for ArraySlice<A> where Self: Clone {}

impl<A: NDArrayRead> NDArrayNumeric for ArraySlice<A> where Self: Clone {}

impl_op!(Add, add, ArraySlice<T>, ArrayBase<O>);
impl_op!(Div, div, ArraySlice<T>, ArrayBase<O>);
impl_op!(Mul, mul, ArraySlice<T>, ArrayBase<O>);
impl_op!(Rem, rem, ArraySlice<T>, ArrayBase<O>);
impl_op!(Sub, sub, ArraySlice<T>, ArrayBase<O>);

impl_op!(Add, add, ArraySlice<T>, ArrayOp<O>);
impl_op!(Div, div, ArraySlice<T>, ArrayOp<O>);
impl_op!(Mul, mul, ArraySlice<T>, ArrayOp<O>);
impl_op!(Rem, rem, ArraySlice<T>, ArrayOp<O>);
impl_op!(Sub, sub, ArraySlice<T>, ArrayOp<O>);

impl_op!(Add, add, ArraySlice<T>, ArraySlice<O>);
impl_op!(Div, div, ArraySlice<T>, ArraySlice<O>);
impl_op!(Mul, mul, ArraySlice<T>, ArraySlice<O>);
impl_op!(Rem, rem, ArraySlice<T>, ArraySlice<O>);
impl_op!(Sub, sub, ArraySlice<T>, ArraySlice<O>);

impl_op!(Add, add, ArraySlice<T>, ArrayView<O>);
impl_op!(Div, div, ArraySlice<T>, ArrayView<O>);
impl_op!(Mul, mul, ArraySlice<T>, ArrayView<O>);
impl_op!(Rem, rem, ArraySlice<T>, ArrayView<O>);
impl_op!(Sub, sub, ArraySlice<T>, ArrayView<O>);

macro_rules! impl_slice_scalar_op {
    ($op:ident, $name:ident) => {
        impl<T: CDatatype, Op: super::ops::Op<Out = T>> $op<T> for ArraySlice<Op> {
            type Output = ArrayOp<ArrayScalar<Op::Out, Self>>;

            fn $name(self, rhs: Op::Out) -> Self::Output {
                let shape = self.shape.to_vec();
                let op = ArrayScalar::$name(self, rhs);
                ArrayOp::new(op, shape)
            }
        }
    };
}

impl_slice_scalar_op!(Add, add);
impl_slice_scalar_op!(Div, div);
impl_slice_scalar_op!(Mul, mul);
impl_slice_scalar_op!(Rem, rem);
impl_slice_scalar_op!(Sub, sub);

impl<A: NDArrayRead> Neg for ArraySlice<A> {
    type Output = ArrayOp<ArrayUnary<Self>>;

    fn neg(self) -> Self::Output {
        let shape = self.shape.to_vec();
        let op = ArrayUnary::neg(self);
        ArrayOp::new(op, shape)
    }
}

impl<A: NDArrayRead> Not for ArraySlice<A> {
    type Output = ArrayOp<ArrayUnary<Self>>;

    fn not(self) -> Self::Output {
        let shape = self.shape.to_vec();
        let op = ArrayUnary::not(self);
        ArrayOp::new(op, shape)
    }
}

impl<A: fmt::Debug> fmt::Debug for ArraySlice<A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "slice of {:?} with shape {:?}", self.source, self.shape)
    }
}

#[derive(Clone)]
pub struct ArrayView<A> {
    source: A,
    shape: Shape,
    strides: Vec<usize>,
}

impl<A: NDArray> ArrayView<A> {
    fn new(source: A, shape: Shape, strides: Vec<usize>) -> Self {
        Self {
            source,
            shape,
            strides,
        }
    }

    fn broadcast(source: A, shape: Shape) -> Result<Self, Error> {
        if shape.len() < source.ndim() {
            return Err(Error::Bounds(format!(
                "cannot broadcast {:?} into {:?}",
                source.shape(),
                shape
            )));
        }

        for (dim, bdim) in source
            .shape()
            .iter()
            .zip(&shape[shape.len() - source.ndim()..])
        {
            if dim == bdim || *dim == 1 {
                // ok
            } else {
                return Err(Error::Bounds(format!(
                    "cannot broadcast dimension {} into {}",
                    dim, bdim
                )));
            }
        }

        let strides = strides_for(source.shape(), shape.len());

        Ok(Self::new(source, shape, strides))
    }

    fn transpose(source: A, axes: Option<Vec<usize>>) -> Result<Self, Error>
    where
        A: fmt::Debug,
    {
        let axes = if let Some(axes) = axes {
            if axes.len() == source.ndim() && (0..source.ndim()).all(|x| axes.contains(&x)) {
                Ok(axes)
            } else {
                Err(Error::Bounds(format!(
                    "cannot transpose axes {:?} of {:?}",
                    axes, source
                )))
            }
        } else {
            Ok((0..source.ndim()).into_iter().rev().collect())
        }?;

        let source_strides = strides_for(source.shape(), source.ndim());

        let mut shape = Vec::with_capacity(source.ndim());
        let mut strides = Vec::with_capacity(source.ndim());
        for x in axes {
            shape.push(source.shape()[x]);
            strides.push(source_strides[x]);
        }

        debug_assert!(!shape.iter().any(|dim| *dim == 0));

        Ok(Self {
            source,
            shape,
            strides,
        })
    }
}

impl<A> NDArray for ArrayView<A> {
    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl<A: NDArrayRead> NDArrayRead for ArrayView<A> {
    type Out = A::Out;

    fn read(&self, queue: Queue) -> Result<Buffer<Self::Out>, Error> {
        let buffer = self.source.read(queue.clone())?;
        let strides = strides_for(&self.shape, self.ndim());

        if self.size() == self.source.size() {
            kernels::reorder_inplace(queue, buffer, &self.shape, &strides, &self.strides)
                .map_err(Error::from)
        } else {
            kernels::reorder(queue, buffer, &self.shape, &strides, &self.strides)
                .map_err(Error::from)
        }
    }
}

impl<A, O> NDArrayBoolean<O> for ArrayView<A>
where
    A: NDArrayRead,
    O: NDArrayRead<Out = A::Out>,
{
}

impl<A: NDArray> NDArrayExp for ArrayView<A> where Self: Clone {}

impl<A: NDArrayRead> NDArrayMathScalar for ArrayView<A> where Self: Clone {}

impl<A: NDArrayRead> NDArrayNumeric for ArrayView<A> where Self: Clone {}

impl_op!(Add, add, ArrayView<T>, ArrayBase<O>);
impl_op!(Div, div, ArrayView<T>, ArrayBase<O>);
impl_op!(Mul, mul, ArrayView<T>, ArrayBase<O>);
impl_op!(Rem, rem, ArrayView<T>, ArrayBase<O>);
impl_op!(Sub, sub, ArrayView<T>, ArrayBase<O>);

impl_op!(Add, add, ArrayView<T>, ArrayOp<O>);
impl_op!(Div, div, ArrayView<T>, ArrayOp<O>);
impl_op!(Mul, mul, ArrayView<T>, ArrayOp<O>);
impl_op!(Rem, rem, ArrayView<T>, ArrayOp<O>);
impl_op!(Sub, sub, ArrayView<T>, ArrayOp<O>);

impl_op!(Add, add, ArrayView<T>, ArraySlice<O>);
impl_op!(Div, div, ArrayView<T>, ArraySlice<O>);
impl_op!(Mul, mul, ArrayView<T>, ArraySlice<O>);
impl_op!(Rem, rem, ArrayView<T>, ArraySlice<O>);
impl_op!(Sub, sub, ArrayView<T>, ArraySlice<O>);

impl_op!(Add, add, ArrayView<T>, ArrayView<O>);
impl_op!(Div, div, ArrayView<T>, ArrayView<O>);
impl_op!(Mul, mul, ArrayView<T>, ArrayView<O>);
impl_op!(Rem, rem, ArrayView<T>, ArrayView<O>);
impl_op!(Sub, sub, ArrayView<T>, ArrayView<O>);

macro_rules! impl_view_scalar_op {
    ($op:ident, $name:ident) => {
        impl<T: CDatatype, Op: super::ops::Op<Out = T>> $op<T> for ArrayView<Op> {
            type Output = ArrayOp<ArrayScalar<Op::Out, Self>>;

            fn $name(self, rhs: Op::Out) -> Self::Output {
                let shape = self.shape.to_vec();
                let op = ArrayScalar::$name(self, rhs);
                ArrayOp::new(op, shape)
            }
        }
    };
}

impl_view_scalar_op!(Add, add);
impl_view_scalar_op!(Div, div);
impl_view_scalar_op!(Mul, mul);
impl_view_scalar_op!(Rem, rem);
impl_view_scalar_op!(Sub, sub);

impl<A: NDArrayRead> Neg for ArrayView<A> {
    type Output = ArrayOp<ArrayUnary<Self>>;

    fn neg(self) -> Self::Output {
        let shape = self.shape.to_vec();
        let op = ArrayUnary::neg(self);
        ArrayOp::new(op, shape)
    }
}

impl<A: NDArrayRead> Not for ArrayView<A> {
    type Output = ArrayOp<ArrayUnary<Self>>;

    fn not(self) -> Self::Output {
        let shape = self.shape.to_vec();
        let op = ArrayUnary::not(self);
        ArrayOp::new(op, shape)
    }
}

impl<A: NDArray + fmt::Debug> NDArrayTransform for ArrayView<A> {
    type Broadcast = Self;
    type Expand = Self;
    type Reshape = ArrayView<Self>;
    type Slice = ArraySlice<Self>;
    type Transpose = Self;

    fn broadcast(&self, shape: Shape) -> Result<Self::Broadcast, Error> {
        todo!()
    }

    fn expand_dim(&self, axis: usize) -> Result<Self::Expand, Error> {
        todo!()
    }

    fn expand_dims(&self, axes: Vec<usize>) -> Result<Self::Expand, Error> {
        todo!()
    }

    fn reshape(&self, shape: Shape) -> Result<Self::Reshape, Error> {
        todo!()
    }

    fn slice(&self, bounds: Vec<AxisBound>) -> Result<Self::Slice, Error> {
        todo!()
    }

    fn transpose(&self, axes: Option<Vec<usize>>) -> Result<Self::Transpose, Error> {
        todo!()
    }
}

impl<T, A, O> MatrixMath<O> for ArrayView<A>
where
    T: CDatatype,
    A: NDArrayRead<Out = T>,
    O: NDArrayRead<Out = T>,
{
}

impl<A: fmt::Debug> fmt::Debug for ArrayView<A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "view of {:?} with shape {:?}", self.source, self.shape)
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
fn check_bound(i: &usize, dim: &usize, is_index: bool) -> Result<(), Error> {
    match i.cmp(dim) {
        Ordering::Less => Ok(()),
        Ordering::Equal if !is_index => Ok(()),
        Ordering::Greater | Ordering::Equal => Err(Error::Bounds(format!(
            "index {i} is out of bounds for dimension {dim}"
        ))),
    }
}

#[inline]
fn strides_for(shape: &[usize], ndim: usize) -> Vec<usize> {
    debug_assert!(ndim >= shape.len());

    let zeros = iter::repeat(0).take(ndim - shape.len());

    let strides = shape.iter().enumerate().map(|(x, dim)| {
        if *dim == 1 {
            0
        } else {
            shape.iter().rev().take(shape.len() - 1 - x).product()
        }
    });

    zeros.chain(strides).collect()
}
