use std::cmp::Ordering;
use std::ops::{Add, Div, Mul, Neg, Not, Rem, Sub};
use std::sync::{Arc, RwLock};
use std::{fmt, iter};

use rayon::prelude::*;

use super::ops::*;
use super::{
    strides_for, AsBuffer, AxisBound, Buffer, BufferConverter, BufferConverterMut, BufferInstance,
    BufferRead, BufferWrite, CDatatype, Context, Error, NDArray, NDArrayRead, NDArrayTransform,
    NDArrayWrite, Queue, Shape,
};

pub enum Array<T: CDatatype> {
    Base(ArrayBase<Box<dyn BufferRead<DType = T>>>),
    Op(ArrayOp<Arc<dyn super::ops::Op<Out = T>>>),
    Slice(Box<ArraySlice<Self>>),
    View(Box<ArrayView<Self>>),
}

macro_rules! array_dispatch {
    ($this:ident, $var:ident, $call:expr) => {
        match $this {
            Self::Base($var) => $call,
            Self::Op($var) => $call,
            Self::Slice($var) => $call,
            Self::View($var) => $call,
        }
    };
}

impl<T: CDatatype> NDArray for Array<T> {
    type DType = T;

    fn context(&self) -> &Context {
        array_dispatch!(self, this, this.context())
    }

    fn shape(&self) -> &[usize] {
        array_dispatch!(self, this, this.shape())
    }
}

impl<T: CDatatype> NDArrayRead for Array<T> {
    fn read(&self, queue: &Queue) -> Result<BufferConverter<Self::DType>, Error> {
        array_dispatch!(self, this, this.read(queue))
    }
}

impl<T: CDatatype> NDArrayTransform for Array<T> {
    type Broadcast = Self;
    type Expand = Self;
    type Reshape = Self;
    type Slice = Self;
    type Transpose = Self;

    fn broadcast(self, shape: Shape) -> Result<Self, Error> {
        array_dispatch!(self, this, this.broadcast(shape).map(Self::from))
    }

    fn expand_dims(self, axes: Vec<usize>) -> Result<Self, Error> {
        array_dispatch!(self, this, this.expand_dims(axes).map(Self::from))
    }

    fn reshape(self, shape: Shape) -> Result<Self, Error> {
        array_dispatch!(self, this, this.reshape(shape).map(Self::from))
    }

    fn slice(self, bounds: Vec<AxisBound>) -> Result<Self, Error> {
        array_dispatch!(self, this, this.slice(bounds).map(Self::from))
    }

    fn transpose(self, axes: Option<Vec<usize>>) -> Result<Self, Error> {
        array_dispatch!(self, this, this.transpose(axes).map(Self::from))
    }
}

#[cfg(feature = "freqfs")]
impl<FE, T> From<ArrayBase<freqfs::FileReadGuardOwned<FE, Buffer<T>>>> for Array<T>
where
    FE: Send + Sync + 'static,
    T: CDatatype,
{
    fn from(base: ArrayBase<freqfs::FileReadGuardOwned<FE, Buffer<T>>>) -> Self {
        Self::Base(ArrayBase {
            context: base.context,
            shape: base.shape,
            data: Box::new(base.data),
        })
    }
}

#[cfg(feature = "freqfs")]
impl<FE, T> From<ArrayBase<freqfs::FileWriteGuardOwned<FE, Buffer<T>>>> for Array<T>
where
    FE: Send + Sync + 'static,
    T: CDatatype,
{
    fn from(base: ArrayBase<freqfs::FileWriteGuardOwned<FE, Buffer<T>>>) -> Self {
        Self::Base(ArrayBase {
            context: base.context,
            shape: base.shape,
            data: Box::new(base.data),
        })
    }
}

impl<T: CDatatype> fmt::Debug for Array<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        array_dispatch!(self, this, this.fmt(f))
    }
}

#[derive(Clone)]
pub struct ArrayBase<Buf> {
    context: Context,
    shape: Shape,
    data: Buf,
}

impl<Buf> ArrayBase<Buf> {
    fn new_inner(context: Context, shape: Shape, size: usize, data: Buf) -> Result<Self, Error> {
        if shape.iter().product::<usize>() == size {
            Ok(Self {
                context,
                data,
                shape,
            })
        } else {
            Err(Error::Bounds(format!(
                "expected {} elements for shape {:?} but found {}",
                shape.iter().product::<usize>(),
                shape,
                size,
            )))
        }
    }

    pub fn into_inner(self) -> Buf {
        self.data
    }
}

macro_rules! construct_array {
    ($buf:ty) => {
        impl<T: CDatatype> ArrayBase<$buf> {
            pub fn new(shape: Shape, data: $buf) -> Result<Self, Error> {
                Context::default().and_then(|cxt| Self::with_context(cxt, shape, data))
            }

            pub fn with_context(context: Context, shape: Shape, data: $buf) -> Result<Self, Error> {
                Self::new_inner(context, shape, data.len(), data)
            }
        }
    };
}

construct_array!(Vec<T>);
construct_array!(Arc<Vec<T>>);
#[cfg(feature = "opencl")]
construct_array!(ocl::Buffer<T>);
#[cfg(feature = "opencl")]
construct_array!(Arc<ocl::Buffer<T>>);
construct_array!(Buffer<T>);
construct_array!(Arc<Buffer<T>>);

macro_rules! construct_array_lock {
    ($buf:ty) => {
        impl<T: CDatatype> ArrayBase<Arc<RwLock<$buf>>> {
            pub fn new(shape: Shape, data: $buf) -> Result<Self, Error> {
                let context = Context::default()?;
                let size = data.len();
                let data = Arc::new(RwLock::new(data));
                Self::new_inner(context, shape, size, data)
            }

            pub fn with_context(context: Context, shape: Shape, data: $buf) -> Result<Self, Error> {
                let size = data.len();
                let data = Arc::new(RwLock::new(data));
                Self::new_inner(context, shape, size, data)
            }
        }
    };
}

construct_array_lock!(Vec<T>);
#[cfg(feature = "opencl")]
construct_array_lock!(ocl::Buffer<T>);
construct_array_lock!(Buffer<T>);

#[cfg(feature = "freqfs")]
impl<FE, T: CDatatype> ArrayBase<freqfs::FileReadGuardOwned<FE, Buffer<T>>> {
    pub fn new(
        shape: Shape,
        data: freqfs::FileReadGuardOwned<FE, Buffer<T>>,
    ) -> Result<Self, Error> {
        Context::default().and_then(|context| Self::with_context(context, shape, data))
    }

    pub fn with_context(
        context: Context,
        shape: Shape,
        data: freqfs::FileReadGuardOwned<FE, Buffer<T>>,
    ) -> Result<Self, Error> {
        Self::new_inner(context, shape, data.len(), data)
    }
}

#[cfg(feature = "freqfs")]
impl<FE: Send + Sync, T: CDatatype> ArrayBase<freqfs::FileWriteGuardOwned<FE, Buffer<T>>> {
    pub fn new(
        shape: Shape,
        data: freqfs::FileWriteGuardOwned<FE, Buffer<T>>,
    ) -> Result<Self, Error> {
        Context::default().and_then(|context| Self::with_context(context, shape, data))
    }

    pub fn with_context(
        context: Context,
        shape: Shape,
        data: freqfs::FileWriteGuardOwned<FE, Buffer<T>>,
    ) -> Result<Self, Error> {
        Self::new_inner(context, shape, data.len(), data)
    }
}

impl<T: CDatatype> ArrayBase<Vec<T>> {
    pub fn copy<O: NDArrayRead<DType = T>>(other: &O) -> Result<Self, Error> {
        let context = other.context().clone();
        let queue = Queue::new(context.clone(), other.size())?;

        let shape = other.shape().to_vec();
        let data = other.to_host(&queue)?.into_vec();

        Ok(Self {
            context,
            data,
            shape,
        })
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data
    }
}

impl<T: CDatatype> ArrayBase<Arc<Vec<T>>> {
    pub fn copy<O: NDArrayRead<DType = T>>(other: &O) -> Result<Self, Error> {
        let context = other.context().clone();
        let queue = Queue::new(context.clone(), other.size())?;

        let shape = other.shape().to_vec();
        let data = other.to_host(&queue)?;
        let data = Arc::new(data.into_vec());

        Ok(Self {
            context,
            data,
            shape,
        })
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data
    }
}

impl<T: CDatatype> ArrayBase<Arc<RwLock<Vec<T>>>> {
    pub fn copy<O: NDArrayRead<DType = T>>(other: &O) -> Result<Self, Error> {
        let context = other.context().clone();
        let queue = Queue::new(context.clone(), other.size())?;

        let shape = other.shape().to_vec();
        let data = other.to_host(&queue)?;
        let data = Arc::new(RwLock::new(data.into_vec()));

        Ok(Self {
            context,
            data,
            shape,
        })
    }
}

#[cfg(feature = "opencl")]
impl<T: CDatatype> ArrayBase<ocl::Buffer<T>> {
    pub fn copy<O: NDArrayRead<DType = T>>(other: &O) -> Result<Self, Error> {
        let context = other.context().clone();
        let queue = Queue::new(context.clone(), other.size())?;

        let shape = other.shape().to_vec();
        let data = other.to_cl_buffer(&queue)?.into_buffer()?;

        Ok(Self {
            context,
            data,
            shape,
        })
    }
}

#[cfg(feature = "opencl")]
impl<T: CDatatype> ArrayBase<Arc<ocl::Buffer<T>>> {
    pub fn copy<O: NDArrayRead<DType = T>>(other: &O) -> Result<Self, Error> {
        let context = other.context().clone();
        let queue = Queue::new(context.clone(), other.size())?;

        let shape = other.shape().to_vec();
        let data = other.to_cl_buffer(&queue)?;
        let data = data.into_buffer().map(Arc::new)?;

        Ok(Self {
            context,
            data,
            shape,
        })
    }
}

#[cfg(feature = "opencl")]
impl<T: CDatatype> ArrayBase<Arc<RwLock<ocl::Buffer<T>>>> {
    pub fn copy<O: NDArrayRead<DType = T>>(other: &O) -> Result<Self, Error> {
        let context = other.context().clone();
        let queue = Queue::new(context.clone(), other.size())?;

        let shape = other.shape().to_vec();
        let data = other.to_cl_buffer(&queue)?;
        let data = data.into_buffer().map(RwLock::new).map(Arc::new)?;

        Ok(Self {
            context,
            data,
            shape,
        })
    }
}

impl<T: CDatatype> ArrayBase<Buffer<T>> {
    pub fn copy<O: NDArrayRead<DType = T>>(other: &O) -> Result<Self, Error> {
        let context = other.context().clone();
        let queue = Queue::new(context.clone(), other.size())?;

        let shape = other.shape().to_vec();
        let data = other.read(&queue)?;
        let data = data.into_buffer()?;

        Ok(Self {
            context,
            data,
            shape,
        })
    }
}

impl<T: CDatatype> ArrayBase<Arc<Buffer<T>>> {
    pub fn copy<O: NDArrayRead<DType = T>>(other: &O) -> Result<Self, Error> {
        let context = other.context().clone();
        let queue = Queue::new(context.clone(), other.size())?;

        let shape = other.shape().to_vec();
        let data = other.read(&queue)?;
        let data = data.into_buffer().map(Arc::new)?;

        Ok(Self {
            context,
            data,
            shape,
        })
    }
}

impl<T: CDatatype> ArrayBase<Arc<RwLock<Buffer<T>>>> {
    pub fn copy<O: NDArrayRead<DType = T>>(other: &O) -> Result<Self, Error> {
        let context = other.context().clone();
        let queue = Queue::new(context.clone(), other.size())?;

        let shape = other.shape().to_vec();
        let data = other.read(&queue)?;
        let data = data.into_buffer().map(RwLock::new).map(Arc::new)?;

        Ok(Self {
            context,
            data,
            shape,
        })
    }
}

impl<Buf: BufferInstance> NDArray for ArrayBase<Buf> {
    type DType = Buf::DType;

    fn context(&self) -> &Context {
        &self.context
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl<T: CDatatype> NDArrayRead for ArrayBase<Box<dyn BufferRead<DType = T>>> {
    fn read(&self, _queue: &Queue) -> Result<BufferConverter<Self::DType>, Error> {
        Ok(self.data.read())
    }
}

impl<T: CDatatype> NDArrayRead for ArrayBase<Vec<T>> {
    fn read(&self, _queue: &Queue) -> Result<BufferConverter<T>, Error> {
        Ok(BufferConverter::from(&self.data[..]))
    }
}

impl<'a, T: CDatatype> NDArrayWrite<'a> for ArrayBase<Vec<T>> {
    fn write<O: NDArrayRead<DType = T>>(&'a mut self, other: &O) -> Result<(), Error> {
        if self.shape == other.shape() {
            let queue = Queue::new(self.context().clone(), self.size())?;
            let buffer = other.read(&queue)?;
            self.data.write(buffer)
        } else {
            Err(Error::Bounds(format!(
                "cannot write {:?} to {:?}",
                other, self
            )))
        }
    }

    fn write_value(&'a mut self, value: T) -> Result<(), Error> {
        self.data.write_value(value)
    }

    fn write_value_at(&'a mut self, coord: &[usize], value: T) -> Result<(), Error> {
        validate_coord(self, coord)?;
        let offset = offset_of(coord, &self.shape);
        self.data.write_value_at(offset, value)
    }
}

impl<T: CDatatype> NDArrayRead for ArrayBase<Arc<Vec<T>>> {
    fn read(&self, _queue: &Queue) -> Result<BufferConverter<T>, Error> {
        Ok(BufferConverter::from(&self.data[..]))
    }
}

impl<T: CDatatype> NDArrayRead for ArrayBase<Arc<RwLock<Vec<T>>>> {
    fn read(&self, _queue: &Queue) -> Result<BufferConverter<T>, Error> {
        let data = RwLock::read(&self.data).expect("read buffer");
        Ok(BufferConverter::from(data.to_vec()))
    }
}

impl<'a, T: CDatatype> NDArrayWrite<'a> for ArrayBase<Arc<RwLock<Vec<T>>>> {
    fn write<O: NDArrayRead<DType = T>>(&'a mut self, other: &O) -> Result<(), Error> {
        if self.shape == other.shape() {
            let queue = Queue::new(self.context().clone(), self.size())?;
            let mut data = self.data.write().expect("write buffer");
            other.read(&queue).and_then(|buffer| data.write(buffer))
        } else {
            Err(Error::Bounds(format!(
                "cannot write {:?} to {:?}",
                other, self
            )))
        }
    }

    fn write_value(&'a mut self, value: T) -> Result<(), Error> {
        let mut data = self.data.write().expect("write buffer");
        data.write_value(value)
    }

    fn write_value_at(&'a mut self, coord: &[usize], value: T) -> Result<(), Error> {
        validate_coord(self, coord)?;

        let mut data = self.data.write().expect("write buffer");
        let offset = offset_of(coord, &self.shape);
        data.write_value_at(offset, value)
    }
}

#[cfg(feature = "opencl")]
impl<T: CDatatype> NDArrayRead for ArrayBase<ocl::Buffer<T>> {
    fn read(&self, _queue: &Queue) -> Result<BufferConverter<T>, Error> {
        Ok(BufferConverter::from(&self.data))
    }
}

#[cfg(feature = "opencl")]
impl<'a, T: CDatatype> NDArrayWrite<'a> for ArrayBase<ocl::Buffer<T>> {
    fn write<O: NDArrayRead<DType = T>>(&'a mut self, other: &O) -> Result<(), Error> {
        if self.shape == other.shape() {
            let queue = Queue::new(self.context().clone(), self.size())?;

            other
                .read(&queue)
                .and_then(|buffer| BufferWrite::write(&mut self.data, buffer))
        } else {
            Err(Error::Bounds(format!(
                "cannot write {:?} to {:?}",
                other, self
            )))
        }
    }

    fn write_value(&'a mut self, value: T) -> Result<(), Error> {
        self.data.write_value(value)
    }

    fn write_value_at(&'a mut self, coord: &[usize], value: T) -> Result<(), Error> {
        validate_coord(self, coord)?;
        let offset = offset_of(coord, &self.shape);
        self.data.write_value_at(offset, value)
    }
}

#[cfg(feature = "opencl")]
impl<T: CDatatype> NDArrayRead for ArrayBase<Arc<ocl::Buffer<T>>> {
    fn read(&self, _queue: &Queue) -> Result<BufferConverter<T>, Error> {
        Ok(BufferConverter::from(&*self.data))
    }
}

#[cfg(feature = "opencl")]
impl<T: CDatatype> NDArrayRead for ArrayBase<Arc<RwLock<ocl::Buffer<T>>>> {
    fn read(&self, queue: &Queue) -> Result<BufferConverter<T>, Error> {
        let data = RwLock::read(&self.data).expect("read buffer");

        let cl_queue = queue.cl_queue(data.default_queue());
        let mut copy = ocl::Buffer::builder()
            .queue(cl_queue)
            .len(data.len())
            .build()?;

        data.copy(&mut copy, None, None).enq()?;

        Ok(BufferConverter::from(copy))
    }
}

#[cfg(feature = "opencl")]
impl<'a, T: CDatatype> NDArrayWrite<'a> for ArrayBase<Arc<RwLock<ocl::Buffer<T>>>> {
    fn write<O: NDArrayRead<DType = T>>(&'a mut self, other: &O) -> Result<(), Error> {
        if self.shape == other.shape() {
            let queue = Queue::new(self.context().clone(), self.size())?;
            let mut data = self.data.write().expect("write buffer");

            other
                .read(&queue)
                .and_then(|buffer| BufferWrite::write(&mut *data, buffer))
        } else {
            Err(Error::Bounds(format!(
                "cannot write {:?} to {:?}",
                other, self
            )))
        }
    }

    fn write_value(&'a mut self, value: T) -> Result<(), Error> {
        let mut data = self.data.write().expect("write buffer");
        data.write_value(value)
    }

    fn write_value_at(&'a mut self, coord: &[usize], value: T) -> Result<(), Error> {
        validate_coord(self, coord)?;
        let mut data = self.data.write().expect("write buffer");
        let offset = offset_of(coord, &self.shape);
        data.write_value_at(offset, value)
    }
}

impl<T: CDatatype> NDArrayRead for ArrayBase<Buffer<T>> {
    fn read(&self, _queue: &Queue) -> Result<BufferConverter<T>, Error> {
        Ok(BufferConverter::from(&self.data))
    }
}

impl<'a, T: CDatatype> NDArrayWrite<'a> for ArrayBase<Buffer<T>> {
    fn write<O: NDArrayRead<DType = T>>(&'a mut self, other: &O) -> Result<(), Error> {
        if self.shape == other.shape() {
            let queue = Queue::new(self.context().clone(), self.size())?;
            other
                .read(&queue)
                .and_then(|buffer| self.data.write(buffer))
        } else {
            Err(Error::Bounds(format!(
                "cannot write {:?} to {:?}",
                other, self
            )))
        }
    }

    fn write_value(&'a mut self, value: T) -> Result<(), Error> {
        self.data.write_value(value)
    }

    fn write_value_at(&'a mut self, coord: &[usize], value: T) -> Result<(), Error> {
        validate_coord(self, coord)?;
        let offset = offset_of(coord, &self.shape);
        self.data.write_value_at(offset, value)
    }
}

impl<T: CDatatype> NDArrayRead for ArrayBase<Arc<Buffer<T>>> {
    fn read(&self, _queue: &Queue) -> Result<BufferConverter<T>, Error> {
        Ok(BufferConverter::from(&*self.data))
    }
}

impl<T: CDatatype> NDArrayRead for ArrayBase<Arc<RwLock<Buffer<T>>>> {
    #[allow(unused_variables)]
    fn read(&self, queue: &Queue) -> Result<BufferConverter<T>, Error> {
        let data = RwLock::read(&self.data).expect("read buffer");

        match &*data {
            Buffer::Host(buffer) => Ok(BufferConverter::from(buffer.to_vec())),
            #[cfg(feature = "opencl")]
            Buffer::CL(buffer) => {
                let cl_queue = queue.cl_queue(buffer.default_queue());
                let mut copy = ocl::Buffer::builder()
                    .queue(cl_queue)
                    .len(buffer.len())
                    .build()?;

                buffer.copy(&mut copy, None, None).enq()?;

                Ok(BufferConverter::from(copy))
            }
        }
    }
}

impl<'a, T: CDatatype> NDArrayWrite<'a> for ArrayBase<Arc<RwLock<Buffer<T>>>> {
    fn write<O: NDArrayRead<DType = T>>(&'a mut self, other: &O) -> Result<(), Error> {
        if self.shape == other.shape() {
            let queue = Queue::new(self.context().clone(), self.size())?;
            let mut data = self.data.write().expect("write buffer");
            other.read(&queue).and_then(|buffer| data.write(buffer))
        } else {
            Err(Error::Bounds(format!(
                "cannot write {:?} to {:?}",
                other, self
            )))
        }
    }

    fn write_value(&'a mut self, value: T) -> Result<(), Error> {
        let mut data = self.data.write().expect("write buffer");
        data.write_value(value)
    }

    fn write_value_at(&'a mut self, coord: &[usize], value: T) -> Result<(), Error> {
        validate_coord(self, coord)?;
        let mut data = self.data.write().expect("write buffer");
        let offset = offset_of(coord, &self.shape);
        data.write_value_at(offset, value)
    }
}

#[cfg(feature = "freqfs")]
impl<FE, T> NDArrayRead for ArrayBase<freqfs::FileReadGuardOwned<FE, Buffer<T>>>
where
    FE: Send + Sync,
    T: CDatatype,
{
    fn read(&self, _queue: &Queue) -> Result<BufferConverter<Self::DType>, Error> {
        Ok(self.data.read())
    }
}

#[cfg(feature = "freqfs")]
impl<FE, T> NDArrayRead for ArrayBase<freqfs::FileWriteGuardOwned<FE, Buffer<T>>>
where
    FE: Send + Sync,
    T: CDatatype,
{
    fn read(&self, _queue: &Queue) -> Result<BufferConverter<Self::DType>, Error> {
        Ok(self.data.clone().into())
    }
}

#[cfg(feature = "freqfs")]
impl<'a, FE, T> NDArrayWrite<'a> for ArrayBase<freqfs::FileWriteGuardOwned<FE, Buffer<T>>>
where
    FE: Send + Sync,
    T: CDatatype,
{
    fn write<O: NDArrayRead<DType = T>>(&'a mut self, other: &O) -> Result<(), Error> {
        if self.shape == other.shape() {
            let queue = Queue::new(self.context().clone(), self.size())?;

            other
                .read(&queue)
                .and_then(|buffer| self.data.write(buffer))
        } else {
            Err(Error::Bounds(format!(
                "cannot write {:?} to {:?}",
                other, self
            )))
        }
    }

    fn write_value(&'a mut self, value: T) -> Result<(), Error> {
        self.data.write_value(value)
    }

    fn write_value_at(&'a mut self, coord: &[usize], value: T) -> Result<(), Error> {
        validate_coord(self, coord)?;
        let offset = offset_of(coord, &self.shape);
        self.data.write_value_at(offset, value)
    }
}

impl<Buf: BufferInstance> NDArrayTransform for ArrayBase<Buf>
where
    Self: NDArrayRead,
{
    type Broadcast = ArrayView<Self>;
    type Expand = Self;
    type Reshape = Self;
    type Slice = ArraySlice<Self>;
    type Transpose = ArrayView<Self>;

    fn broadcast(self, shape: Shape) -> Result<ArrayView<Self>, Error> {
        ArrayView::broadcast(self, shape)
    }

    fn expand_dims(self, axes: Vec<usize>) -> Result<Self::Expand, Error> {
        let shape = expand_dims(&self, axes)?;

        Ok(Self {
            context: self.context,
            data: self.data,
            shape,
        })
    }

    fn reshape(self, shape: Shape) -> Result<Self, Error> {
        if shape.iter().product::<usize>() == self.size() {
            Ok(Self {
                context: self.context,
                shape,
                data: self.data,
            })
        } else {
            Err(Error::Bounds(format!(
                "cannot reshape from {:?} to {:?}",
                self.shape, shape
            )))
        }
    }

    fn slice(self, bounds: Vec<AxisBound>) -> Result<ArraySlice<Self>, Error> {
        ArraySlice::new(self, bounds)
    }

    fn transpose(self, axes: Option<Vec<usize>>) -> Result<ArrayView<Self>, Error> {
        let axes = permutation(&self, axes)?;

        let shape = axes.iter().copied().map(|x| self.shape[x]).collect();

        let source_strides = strides_for(&self.shape, self.ndim());
        let strides = axes.into_iter().map(|x| source_strides[x]).collect();

        ArrayView::new(self, shape, strides)
    }
}

macro_rules! impl_base_op {
    ($op:ident, $name:ident) => {
        impl<T, LB, RB> $op<ArrayBase<RB>> for ArrayBase<LB>
        where
            T: CDatatype,
            LB: BufferInstance<DType = T>,
            RB: BufferInstance<DType = T>,
        {
            type Output = ArrayOp<ArrayDual<T, ArrayBase<LB>, ArrayBase<RB>>>;

            fn $name(self, rhs: ArrayBase<RB>) -> Self::Output {
                let shape = self.shape().to_vec();
                assert_eq!(shape, rhs.shape());

                let op = ArrayDual::$name(self, rhs).expect("op");
                ArrayOp { op, shape }
            }
        }
    };
}

impl_base_op!(Add, add);
impl_base_op!(Div, div);
impl_base_op!(Mul, mul);
impl_base_op!(Rem, rem);
impl_base_op!(Sub, sub);

macro_rules! impl_base_dual_op {
    ($op:ident, $name:ident, $o:ty) => {
        impl<T: CDatatype, Buf: BufferInstance<DType = T>, O> $op<$o> for ArrayBase<Buf>
        where
            $o: NDArray<DType = T>,
        {
            type Output = ArrayOp<ArrayDual<T, Self, $o>>;

            fn $name(self, rhs: $o) -> Self::Output {
                let shape = self.shape().to_vec();
                assert_eq!(shape, rhs.shape());

                let op = ArrayDual::$name(self, rhs).expect("op");
                ArrayOp { op, shape }
            }
        }
    };
}

impl_base_dual_op!(Add, add, ArrayOp<O>);
impl_base_dual_op!(Div, div, ArrayOp<O>);
impl_base_dual_op!(Mul, mul, ArrayOp<O>);
impl_base_dual_op!(Rem, rem, ArrayOp<O>);
impl_base_dual_op!(Sub, sub, ArrayOp<O>);

impl_base_dual_op!(Add, add, ArraySlice<O>);
impl_base_dual_op!(Div, div, ArraySlice<O>);
impl_base_dual_op!(Mul, mul, ArraySlice<O>);
impl_base_dual_op!(Rem, rem, ArraySlice<O>);
impl_base_dual_op!(Sub, sub, ArraySlice<O>);

impl_base_dual_op!(Add, add, ArrayView<O>);
impl_base_dual_op!(Div, div, ArrayView<O>);
impl_base_dual_op!(Mul, mul, ArrayView<O>);
impl_base_dual_op!(Rem, rem, ArrayView<O>);
impl_base_dual_op!(Sub, sub, ArrayView<O>);

macro_rules! impl_base_scalar_op {
    ($op:ident, $name:ident) => {
        impl<T: CDatatype, Buf: BufferInstance<DType = T>> $op<T> for ArrayBase<Buf> {
            type Output = ArrayOp<ArrayScalar<T, Self>>;

            fn $name(self, rhs: T) -> Self::Output {
                let shape = self.shape.to_vec();
                let op = ArrayScalar::$name(self, rhs).expect("op");
                ArrayOp::new(shape, op)
            }
        }
    };
}

impl_base_scalar_op!(Add, add);
impl_base_scalar_op!(Div, div);
impl_base_scalar_op!(Mul, mul);
impl_base_scalar_op!(Rem, rem);
impl_base_scalar_op!(Sub, sub);

impl<Buf: BufferInstance> Neg for ArrayBase<Buf> {
    type Output = ArrayOp<ArrayUnary<Buf::DType, <Buf::DType as CDatatype>::Neg, Self>>;

    fn neg(self) -> Self::Output {
        let shape = self.shape.to_vec();
        let op = ArrayUnary::neg(self).expect("op");
        ArrayOp::new(shape, op)
    }
}

impl<Buf: BufferInstance> Not for ArrayBase<Buf> {
    type Output = ArrayOp<ArrayUnary<Buf::DType, u8, Self>>;

    fn not(self) -> Self::Output {
        let shape = self.shape.to_vec();
        let op = ArrayUnary::not(self).expect("op");
        ArrayOp::new(shape, op)
    }
}

impl<T: CDatatype> From<ArrayBase<Vec<T>>> for ArrayBase<Arc<Vec<T>>> {
    fn from(base: ArrayBase<Vec<T>>) -> Self {
        ArrayBase {
            context: base.context,
            data: base.data.into(),
            shape: base.shape,
        }
    }
}

impl<T: CDatatype> From<ArrayBase<Vec<T>>> for ArrayBase<Box<dyn BufferRead<DType = T>>> {
    fn from(base: ArrayBase<Vec<T>>) -> Self {
        ArrayBase {
            context: base.context,
            data: Box::new(Arc::new(base.data)),
            shape: base.shape,
        }
    }
}

impl<T: CDatatype> From<ArrayBase<Vec<T>>> for ArrayBase<Buffer<T>> {
    fn from(base: ArrayBase<Vec<T>>) -> Self {
        ArrayBase {
            context: base.context,
            data: base.data.into(),
            shape: base.shape,
        }
    }
}

#[cfg(feature = "opencl")]
impl<T: CDatatype> From<ArrayBase<ocl::Buffer<T>>> for ArrayBase<Arc<ocl::Buffer<T>>> {
    fn from(base: ArrayBase<ocl::Buffer<T>>) -> Self {
        ArrayBase {
            context: base.context,
            data: base.data.into(),
            shape: base.shape,
        }
    }
}

#[cfg(feature = "opencl")]
impl<T: CDatatype> From<ArrayBase<ocl::Buffer<T>>> for ArrayBase<Buffer<T>> {
    fn from(base: ArrayBase<ocl::Buffer<T>>) -> Self {
        ArrayBase {
            context: base.context,
            data: base.data.into(),
            shape: base.shape,
        }
    }
}

impl<T: CDatatype> From<ArrayBase<Vec<T>>> for ArrayBase<Arc<Buffer<T>>> {
    fn from(base: ArrayBase<Vec<T>>) -> Self {
        ArrayBase {
            context: base.context,
            data: Arc::new(Buffer::Host(base.data)),
            shape: base.shape,
        }
    }
}

#[cfg(feature = "opencl")]
impl<T: CDatatype> From<ArrayBase<ocl::Buffer<T>>> for ArrayBase<Arc<Buffer<T>>> {
    fn from(base: ArrayBase<ocl::Buffer<T>>) -> Self {
        ArrayBase {
            context: base.context,
            data: Arc::new(Buffer::CL(base.data)),
            shape: base.shape,
        }
    }
}

#[cfg(feature = "opencl")]
impl<T: CDatatype> From<ArrayBase<ocl::Buffer<T>>> for ArrayBase<Box<dyn BufferRead<DType = T>>> {
    fn from(base: ArrayBase<ocl::Buffer<T>>) -> Self {
        ArrayBase {
            context: base.context,
            data: Box::new(Arc::new(base.data)),
            shape: base.shape,
        }
    }
}

impl<T: CDatatype> From<ArrayBase<Buffer<T>>> for ArrayBase<Arc<Buffer<T>>> {
    fn from(base: ArrayBase<Buffer<T>>) -> Self {
        ArrayBase {
            context: base.context,
            data: base.data.into(),
            shape: base.shape,
        }
    }
}

impl<T: CDatatype> From<ArrayBase<Vec<T>>> for Array<T> {
    fn from(base: ArrayBase<Vec<T>>) -> Self {
        Self::Base(base.into())
    }
}

#[cfg(feature = "opencl")]
impl<T: CDatatype> From<ArrayBase<ocl::Buffer<T>>> for Array<T> {
    fn from(base: ArrayBase<ocl::Buffer<T>>) -> Self {
        Self::Base(base.into())
    }
}

impl<T: CDatatype> From<ArrayBase<Buffer<T>>> for Array<T> {
    fn from(base: ArrayBase<Buffer<T>>) -> Self {
        Self::Base(ArrayBase {
            context: base.context,
            shape: base.shape,
            data: Box::new(Arc::new(base.data)),
        })
    }
}

impl<T: CDatatype> From<ArrayBase<Arc<Buffer<T>>>> for Array<T> {
    fn from(base: ArrayBase<Arc<Buffer<T>>>) -> Self {
        Self::Base(ArrayBase {
            context: base.context,
            shape: base.shape,
            data: Box::new(base.data),
        })
    }
}

impl<T: CDatatype> From<ArrayBase<Box<dyn BufferRead<DType = T>>>> for Array<T> {
    fn from(base: ArrayBase<Box<dyn BufferRead<DType = T>>>) -> Self {
        Self::Base(base)
    }
}

impl<Buf: BufferInstance> fmt::Debug for ArrayBase<Buf> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{} array with shape {:?}",
            Buf::DType::TYPE_STR,
            self.shape
        )
    }
}

#[derive(Clone)]
pub struct ArrayOp<Op> {
    op: Op,
    shape: Shape,
}

impl<Op> ArrayOp<Op> {
    pub fn new(shape: Shape, op: Op) -> Self {
        Self { op, shape }
    }
}

impl<Op: super::ops::Op> NDArray for ArrayOp<Op> {
    type DType = Op::Out;

    fn context(&self) -> &Context {
        self.op.context()
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl<Op: super::ops::Op> NDArrayRead for ArrayOp<Op> {
    fn read(&self, queue: &Queue) -> Result<BufferConverter<Op::Out>, Error> {
        self.op.enqueue(queue).map(BufferConverter::from)
    }
}

impl<Op: super::ops::Op> NDArrayTransform for ArrayOp<Op> {
    type Broadcast = ArrayView<Self>;
    type Expand = Self;
    type Reshape = Self;
    type Slice = ArraySlice<Self>;
    type Transpose = ArrayView<Self>;

    fn broadcast(self, shape: Shape) -> Result<Self::Broadcast, Error> {
        ArrayView::broadcast(self, shape)
    }

    fn expand_dims(self, axes: Vec<usize>) -> Result<Self::Expand, Error> {
        let shape = expand_dims(&self, axes)?;
        self.reshape(shape)
    }

    fn reshape(self, shape: Shape) -> Result<Self::Reshape, Error> {
        if shape.iter().product::<usize>() == self.size() {
            Ok(Self { shape, op: self.op })
        } else {
            Err(Error::Bounds(format!(
                "cannot reshape {:?} into {:?} (wrong size)",
                self, shape
            )))
        }
    }

    fn slice(self, bounds: Vec<AxisBound>) -> Result<Self::Slice, Error> {
        ArraySlice::new(self, bounds)
    }

    fn transpose(self, axes: Option<Vec<usize>>) -> Result<Self::Transpose, Error> {
        let axes = permutation(&self, axes)?;
        let shape = axes.iter().copied().map(|x| self.shape[x]).collect();
        let strides = strides_for(self.shape(), self.ndim());
        let strides = axes.into_iter().map(|x| strides[x]).collect();
        ArrayView::new(self, shape, strides)
    }
}

macro_rules! impl_op_dual_op {
    ($op:ident, $name:ident, $o:ty) => {
        impl<T: CDatatype, Op: super::ops::Op<Out = T>, O> $op<$o> for ArrayOp<Op>
        where
            $o: NDArray<DType = T>,
        {
            type Output = ArrayOp<ArrayDual<T, Self, $o>>;

            fn $name(self, rhs: $o) -> Self::Output {
                let shape = self.shape().to_vec();
                assert_eq!(shape, rhs.shape());

                let op = ArrayDual::$name(self, rhs).expect("op");
                ArrayOp { op, shape }
            }
        }
    };
}

impl_op_dual_op!(Add, add, ArrayBase<O>);
impl_op_dual_op!(Div, div, ArrayBase<O>);
impl_op_dual_op!(Mul, mul, ArrayBase<O>);
impl_op_dual_op!(Rem, rem, ArrayBase<O>);
impl_op_dual_op!(Sub, sub, ArrayBase<O>);

impl_op_dual_op!(Add, add, ArrayOp<O>);
impl_op_dual_op!(Div, div, ArrayOp<O>);
impl_op_dual_op!(Mul, mul, ArrayOp<O>);
impl_op_dual_op!(Rem, rem, ArrayOp<O>);
impl_op_dual_op!(Sub, sub, ArrayOp<O>);

impl_op_dual_op!(Add, add, ArraySlice<O>);
impl_op_dual_op!(Div, div, ArraySlice<O>);
impl_op_dual_op!(Mul, mul, ArraySlice<O>);
impl_op_dual_op!(Rem, rem, ArraySlice<O>);
impl_op_dual_op!(Sub, sub, ArraySlice<O>);

impl_op_dual_op!(Add, add, ArrayView<O>);
impl_op_dual_op!(Div, div, ArrayView<O>);
impl_op_dual_op!(Mul, mul, ArrayView<O>);
impl_op_dual_op!(Rem, rem, ArrayView<O>);
impl_op_dual_op!(Sub, sub, ArrayView<O>);

macro_rules! impl_op_scalar_op {
    ($op:ident, $name:ident) => {
        impl<T: CDatatype, Op: super::ops::Op<Out = T>> $op<T> for ArrayOp<Op> {
            type Output = ArrayOp<ArrayScalar<Op::Out, Self>>;

            fn $name(self, rhs: Op::Out) -> Self::Output {
                let shape = self.shape.to_vec();
                let op = ArrayScalar::$name(self, rhs).expect("op");
                ArrayOp::new(shape, op)
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
    type Output = ArrayOp<ArrayUnary<Op::Out, <Op::Out as CDatatype>::Neg, Self>>;

    fn neg(self) -> Self::Output {
        let shape = self.shape.to_vec();
        let op = ArrayUnary::neg(self).expect("op");
        ArrayOp::new(shape, op)
    }
}

impl<Op: super::ops::Op> Not for ArrayOp<Op> {
    type Output = ArrayOp<ArrayUnary<Op::Out, u8, Self>>;

    fn not(self) -> Self::Output {
        let shape = self.shape.to_vec();
        let op = ArrayUnary::not(self).expect("op");
        ArrayOp::new(shape, op)
    }
}

impl<Op: super::ops::Op + 'static> From<ArrayOp<Op>> for Array<Op::Out> {
    fn from(op: ArrayOp<Op>) -> Self {
        Self::Op(ArrayOp {
            op: Arc::new(op.op),
            shape: op.shape,
        })
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
    strides: Vec<usize>,
    source_strides: Vec<usize>,
    #[cfg(feature = "opencl")]
    kernel_read_op: ocl::Program,
    #[cfg(feature = "opencl")]
    kernel_write_op: ocl::Program,
    #[cfg(feature = "opencl")]
    kernel_write_value_op: ocl::Program,
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
            .collect::<Vec<usize>>();

        let strides = strides_for(&shape, shape.len());
        let source_strides = strides_for(source.shape(), source.ndim());

        #[cfg(feature = "opencl")]
        let kernel_read_op = crate::cl_programs::read_slice::<A::DType>(
            source.context(),
            &shape,
            &strides,
            &bounds,
            &source_strides,
        )?;

        #[cfg(feature = "opencl")]
        let kernel_write_op = crate::cl_programs::write_to_slice::<A::DType>(
            source.context(),
            &shape,
            &strides,
            &bounds,
            &source_strides,
        )?;

        #[cfg(feature = "opencl")]
        let kernel_write_value_op = crate::cl_programs::write_value_to_slice::<A::DType>(
            source.context(),
            &shape,
            &strides,
            &bounds,
            &source_strides,
        )?;

        Ok(Self {
            source,
            bounds,
            shape,
            strides,
            source_strides,
            #[cfg(feature = "opencl")]
            kernel_read_op,
            #[cfg(feature = "opencl")]
            kernel_write_op,
            #[cfg(feature = "opencl")]
            kernel_write_value_op,
        })
    }

    fn source_offset(
        offset: usize,
        strides: &[usize],
        shape: &[usize],
        source_strides: &[usize],
        bounds: &[AxisBound],
    ) -> usize {
        let coord = strides
            .iter()
            .zip(shape)
            .map(|(stride, dim)| (offset / stride) % dim)
            .collect::<Vec<usize>>();

        let mut offset = 0;
        let mut x = 0;
        for (stride, bound) in source_strides.iter().zip(bounds.iter()) {
            let i = match bound {
                AxisBound::At(i) => *i,
                AxisBound::In(start, stop, step) => {
                    let i = start + (coord[x] * step);
                    debug_assert!(i < *stop);
                    x += 1;
                    i
                }
                AxisBound::Of(indices) => {
                    let i = indices[coord[x]];
                    x += 1;
                    i
                }
            };

            offset += i * stride;
        }

        offset
    }

    fn read_vec(&self, source: &[A::DType]) -> Result<Vec<A::DType>, Error> {
        let output = (0..self.size())
            .into_par_iter()
            .map(|offset_out| {
                let offset_in = Self::source_offset(
                    offset_out,
                    &self.strides,
                    &self.shape,
                    &self.source_strides,
                    &self.bounds,
                );

                source[offset_in]
            })
            .collect();

        Ok(output)
    }

    fn write_slice<Data>(
        size: usize,
        strides: &[usize],
        shape: &[usize],
        source_strides: &[usize],
        bounds: &[AxisBound],
        source: &mut [A::DType],
        data: Data,
    ) -> Result<(), Error>
    where
        Data: Iterator<Item = A::DType>,
    {
        for (offset, value) in (0..size).into_iter().zip(data) {
            let source_offset = Self::source_offset(offset, strides, shape, source_strides, bounds);
            source[source_offset] = value;
        }

        Ok(())
    }

    #[cfg(feature = "opencl")]
    fn read_cl(&self, source: &ocl::Buffer<A::DType>) -> Result<ocl::Buffer<A::DType>, Error> {
        let cl_queue = source.default_queue().expect("queue").clone();

        let output = ocl::Buffer::builder()
            .queue(cl_queue.clone())
            .len(self.size())
            .build()?;

        let kernel = ocl::Kernel::builder()
            .name("slice")
            .program(&self.kernel_read_op)
            .queue(cl_queue)
            .global_work_size(output.len())
            .arg(source)
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }

    #[cfg(feature = "opencl")]
    fn write_cl(
        &self,
        source: &mut ocl::Buffer<A::DType>,
        data: &ocl::Buffer<A::DType>,
    ) -> Result<(), Error> {
        let cl_queue = source.default_queue().expect("queue").clone();

        let kernel = ocl::Kernel::builder()
            .name("write_slice")
            .program(&self.kernel_write_op)
            .queue(cl_queue)
            .global_work_size(data.len())
            .arg(source)
            .arg(data)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(())
    }

    #[cfg(feature = "opencl")]
    fn write_cl_value(
        &self,
        source: &mut ocl::Buffer<A::DType>,
        value: A::DType,
    ) -> Result<(), Error> {
        let cl_queue = source.default_queue().expect("queue").clone();

        let kernel = ocl::Kernel::builder()
            .name("write_slice")
            .program(&self.kernel_write_value_op)
            .queue(cl_queue)
            .global_work_size(self.size())
            .arg(source)
            .arg(value)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(())
    }
}

impl<A: NDArray> NDArray for ArraySlice<A> {
    type DType = A::DType;

    fn context(&self) -> &Context {
        self.source.context()
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl<A: NDArrayRead> NDArrayRead for ArraySlice<A> {
    fn read(&self, queue: &Queue) -> Result<BufferConverter<Self::DType>, Error> {
        let source_queue = Queue::new(queue.context().clone(), self.source.size())?;

        match self.source.read(&source_queue)? {
            BufferConverter::Host(source) => {
                self.read_vec(source.as_ref()).map(BufferConverter::from)
            }
            #[cfg(feature = "opencl")]
            BufferConverter::CL(source) => self.read_cl(source.as_ref()).map(BufferConverter::from),
        }
    }
}

impl<'a, Buf: BufferWrite> NDArrayWrite<'a> for ArraySlice<ArrayBase<Buf>>
where
    ArrayBase<Buf>: AsBuffer<DType = <ArrayBase<Buf> as NDArray>::DType>,
{
    fn write<O: NDArrayRead<DType = Self::DType>>(&'a mut self, other: &O) -> Result<(), Error> {
        let size = self.size();
        let queue = Queue::new(self.context().clone(), size)?;
        let that = other.read(&queue)?;

        match self.source.as_buffer_mut() {
            BufferConverterMut::Host(mut this) => {
                let that = that.to_slice()?;

                Self::write_slice(
                    size,
                    &self.strides,
                    &self.shape,
                    &self.source_strides,
                    &self.bounds,
                    this.as_mut(),
                    that.as_ref().into_iter().copied(),
                )
            }
            #[cfg(feature = "opencl")]
            BufferConverterMut::CL(mut this) => {
                let that = that.to_cl(&queue)?;
                self.write_cl(this.as_mut(), that.as_ref())
            },
        }
    }

    fn write_value(&'a mut self, value: Self::DType) -> Result<(), Error> {
        let size = self.size();

        match self.source.as_buffer_mut() {
            BufferConverterMut::Host(mut this) => Self::write_slice(
                size,
                &self.strides,
                &self.shape,
                &self.source_strides,
                &self.bounds,
                this.as_mut(),
                iter::repeat(value).take(size),
            ),
            #[cfg(feature = "opencl")]
            BufferConverterMut::CL(mut this) => self.write_cl_value(this.as_mut(), value),
        }
    }

    fn write_value_at(&'a mut self, coord: &[usize], value: Self::DType) -> Result<(), Error> {
        let offset = offset_of(coord, &self.shape);
        let source_offset = Self::source_offset(
            offset,
            &self.strides,
            &self.shape,
            &self.source_strides,
            &self.bounds,
        );

        match self.source.as_buffer_mut() {
            BufferConverterMut::Host(mut slice) => {
                slice.as_mut()[source_offset] = value;
                Ok(())
            }
            #[cfg(feature = "opencl")]
            BufferConverterMut::CL(mut buffer) => {
                buffer.as_mut().write_value_at(source_offset, value)
            }
        }
    }
}

impl<A: NDArrayRead + fmt::Debug> NDArrayTransform for ArraySlice<A> {
    type Broadcast = ArrayView<Self>;
    type Expand = ArrayView<Self>;
    type Reshape = ArrayView<Self>;
    type Slice = ArraySlice<Self>;
    type Transpose = ArrayView<Self>;

    fn broadcast(self, shape: Shape) -> Result<Self::Broadcast, Error> {
        ArrayView::broadcast(self, shape)
    }

    fn expand_dims(self, axes: Vec<usize>) -> Result<Self::Expand, Error> {
        let shape = expand_dims(&self, axes)?;
        let strides = strides_for(&shape, shape.len());
        ArrayView::new(self, shape, strides)
    }

    fn reshape(self, shape: Shape) -> Result<ArrayView<Self>, Error> {
        if shape.iter().product::<usize>() == self.size() {
            let strides = strides_for(&shape, shape.len());
            ArrayView::new(self, shape, strides)
        } else {
            Err(Error::Bounds(format!(
                "cannot reshape {:?} into {:?}",
                self, shape
            )))
        }
    }

    fn slice(self, bounds: Vec<AxisBound>) -> Result<Self::Slice, Error> {
        ArraySlice::new(self, bounds)
    }

    fn transpose(self, axes: Option<Vec<usize>>) -> Result<Self::Transpose, Error> {
        let axes = permutation(&self, axes)?;
        let shape = axes.iter().copied().map(|x| self.shape[x]).collect();
        let strides = axes.into_iter().map(|x| self.strides[x]).collect();
        ArrayView::new(self, shape, strides)
    }
}

macro_rules! impl_slice_dual_op {
    ($op:ident, $name:ident, $o:ty) => {
        impl<T: CDatatype, A: NDArray<DType = T>, O> $op<$o> for ArraySlice<A>
        where
            $o: NDArray<DType = T>,
        {
            type Output = ArrayOp<ArrayDual<T, Self, $o>>;

            fn $name(self, rhs: $o) -> Self::Output {
                let shape = self.shape().to_vec();
                assert_eq!(shape, rhs.shape());

                let op = ArrayDual::$name(self, rhs).expect("op");
                ArrayOp { op, shape }
            }
        }
    };
}

impl_slice_dual_op!(Add, add, ArrayBase<O>);
impl_slice_dual_op!(Div, div, ArrayBase<O>);
impl_slice_dual_op!(Mul, mul, ArrayBase<O>);
impl_slice_dual_op!(Rem, rem, ArrayBase<O>);
impl_slice_dual_op!(Sub, sub, ArrayBase<O>);

impl_slice_dual_op!(Add, add, ArrayOp<O>);
impl_slice_dual_op!(Div, div, ArrayOp<O>);
impl_slice_dual_op!(Mul, mul, ArrayOp<O>);
impl_slice_dual_op!(Rem, rem, ArrayOp<O>);
impl_slice_dual_op!(Sub, sub, ArrayOp<O>);

impl_slice_dual_op!(Add, add, ArraySlice<O>);
impl_slice_dual_op!(Div, div, ArraySlice<O>);
impl_slice_dual_op!(Mul, mul, ArraySlice<O>);
impl_slice_dual_op!(Rem, rem, ArraySlice<O>);
impl_slice_dual_op!(Sub, sub, ArraySlice<O>);

impl_slice_dual_op!(Add, add, ArrayView<O>);
impl_slice_dual_op!(Div, div, ArrayView<O>);
impl_slice_dual_op!(Mul, mul, ArrayView<O>);
impl_slice_dual_op!(Rem, rem, ArrayView<O>);
impl_slice_dual_op!(Sub, sub, ArrayView<O>);

macro_rules! impl_slice_scalar_op {
    ($op:ident, $name:ident) => {
        impl<T: CDatatype, A: NDArray<DType = T>> $op<T> for ArraySlice<A> {
            type Output = ArrayOp<ArrayScalar<T, Self>>;

            fn $name(self, rhs: T) -> Self::Output {
                let shape = self.shape.to_vec();
                let op = ArrayScalar::$name(self, rhs).expect("op");
                ArrayOp::new(shape, op)
            }
        }
    };
}

impl_slice_scalar_op!(Add, add);
impl_slice_scalar_op!(Div, div);
impl_slice_scalar_op!(Mul, mul);
impl_slice_scalar_op!(Rem, rem);
impl_slice_scalar_op!(Sub, sub);

impl<T: CDatatype, A: NDArrayRead<DType = T>> Neg for ArraySlice<A> {
    type Output = ArrayOp<ArrayUnary<T, T::Neg, Self>>;

    fn neg(self) -> Self::Output {
        let shape = self.shape.to_vec();
        let op = ArrayUnary::neg(self).expect("op");
        ArrayOp::new(shape, op)
    }
}

impl<A: NDArrayRead> Not for ArraySlice<A>
where
    Self: NDArray,
{
    type Output = ArrayOp<ArrayUnary<<Self as NDArray>::DType, u8, Self>>;

    fn not(self) -> Self::Output {
        let shape = self.shape.to_vec();
        let op = ArrayUnary::not(self).expect("op");
        ArrayOp::new(shape, op)
    }
}

impl<T: CDatatype, A: Into<Array<T>>> From<ArraySlice<A>> for Array<T> {
    fn from(slice: ArraySlice<A>) -> Self {
        Self::Slice(Box::new(ArraySlice {
            source: slice.source.into(),
            bounds: slice.bounds,
            shape: slice.shape,
            strides: slice.strides,
            source_strides: slice.source_strides,
            #[cfg(feature = "opencl")]
            kernel_read_op: slice.kernel_read_op,
            #[cfg(feature = "opencl")]
            kernel_write_op: slice.kernel_write_op,
            #[cfg(feature = "opencl")]
            kernel_write_value_op: slice.kernel_write_value_op,
        }))
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
    #[cfg(feature = "opencl")]
    kernel_op: ocl::Program,
}

impl<A: NDArray> ArrayView<A> {
    fn new(source: A, shape: Shape, strides: Vec<usize>) -> Result<Self, Error> {
        #[cfg(feature = "opencl")]
        let kernel_op = crate::cl_programs::reorder::<A::DType>(
            source.context(),
            &shape,
            &strides_for(&shape, shape.len()),
            &strides,
        )?;

        Ok(Self {
            source,
            shape,
            strides,
            #[cfg(feature = "opencl")]
            kernel_op,
        })
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

        Self::new(source, shape, strides)
    }

    fn read_vec(&self, source: &[A::DType]) -> Result<Vec<A::DType>, Error> {
        let source_strides = &self.strides;
        let strides = strides_for(self.shape(), self.ndim());
        let dims = self.shape();
        debug_assert_eq!(strides.len(), dims.len());

        let buffer = (0..self.size())
            .into_par_iter()
            .map(|offset| {
                strides
                    .iter()
                    .copied()
                    .zip(dims.iter().copied())
                    .map(|(stride, dim)| {
                        if stride == 0 {
                            0
                        } else {
                            (offset / stride) % dim
                        }
                    }) // coord
                    .zip(source_strides.iter().copied())
                    .map(|(i, source_stride)| i * source_stride) // source offset
                    .sum::<usize>()
            })
            .map(|source_offset| source[source_offset])
            .collect();

        Ok(buffer)
    }

    #[cfg(feature = "opencl")]
    fn read_cl(&self, source: &ocl::Buffer<A::DType>) -> Result<ocl::Buffer<A::DType>, Error> {
        let cl_queue = source.default_queue().expect("queue").clone();

        let output = ocl::Buffer::builder()
            .queue(cl_queue.clone())
            .len(self.size())
            .build()?;

        let kernel = ocl::Kernel::builder()
            .name("reorder")
            .program(&self.kernel_op)
            .queue(cl_queue)
            .global_work_size(output.len())
            .arg(source)
            .arg(&output)
            .build()?;

        unsafe { kernel.enq()? }

        Ok(output)
    }
}

impl<A: NDArray> NDArray for ArrayView<A> {
    type DType = A::DType;

    fn context(&self) -> &Context {
        self.source.context()
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl<A: NDArrayRead> NDArrayRead for ArrayView<A> {
    fn read(&self, queue: &Queue) -> Result<BufferConverter<Self::DType>, Error> {
        let source_queue = Queue::new(queue.context().clone(), self.source.size())?;

        match self.source.read(&source_queue)? {
            BufferConverter::Host(source) => {
                self.read_vec(source.as_ref()).map(BufferConverter::from)
            }
            #[cfg(feature = "opencl")]
            BufferConverter::CL(source) => self.read_cl(source.as_ref()).map(BufferConverter::from),
        }
    }
}

macro_rules! impl_view_dual_op {
    ($op:ident, $name:ident, $o:ty) => {
        impl<T: CDatatype, A: NDArray<DType = T>, O> $op<$o> for ArrayView<A>
        where
            $o: NDArray<DType = T>,
        {
            type Output = ArrayOp<ArrayDual<T, Self, $o>>;

            fn $name(self, rhs: $o) -> Self::Output {
                let shape = self.shape().to_vec();
                assert_eq!(shape, rhs.shape());

                let op = ArrayDual::$name(self, rhs).expect("op");
                ArrayOp { op, shape }
            }
        }
    };
}

impl_view_dual_op!(Add, add, ArrayBase<O>);
impl_view_dual_op!(Div, div, ArrayBase<O>);
impl_view_dual_op!(Mul, mul, ArrayBase<O>);
impl_view_dual_op!(Rem, rem, ArrayBase<O>);
impl_view_dual_op!(Sub, sub, ArrayBase<O>);

impl_view_dual_op!(Add, add, ArrayOp<O>);
impl_view_dual_op!(Div, div, ArrayOp<O>);
impl_view_dual_op!(Mul, mul, ArrayOp<O>);
impl_view_dual_op!(Rem, rem, ArrayOp<O>);
impl_view_dual_op!(Sub, sub, ArrayOp<O>);

impl_view_dual_op!(Add, add, ArraySlice<O>);
impl_view_dual_op!(Div, div, ArraySlice<O>);
impl_view_dual_op!(Mul, mul, ArraySlice<O>);
impl_view_dual_op!(Rem, rem, ArraySlice<O>);
impl_view_dual_op!(Sub, sub, ArraySlice<O>);

impl_view_dual_op!(Add, add, ArrayView<O>);
impl_view_dual_op!(Div, div, ArrayView<O>);
impl_view_dual_op!(Mul, mul, ArrayView<O>);
impl_view_dual_op!(Rem, rem, ArrayView<O>);
impl_view_dual_op!(Sub, sub, ArrayView<O>);

macro_rules! impl_view_scalar_op {
    ($op:ident, $name:ident) => {
        impl<T: CDatatype, A: NDArray<DType = T>> $op<T> for ArrayView<A> {
            type Output = ArrayOp<ArrayScalar<T, Self>>;

            fn $name(self, rhs: T) -> Self::Output {
                let shape = self.shape.to_vec();
                let op = ArrayScalar::$name(self, rhs).expect("op");
                ArrayOp::new(shape, op)
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
    type Output = ArrayOp<ArrayUnary<A::DType, <A::DType as CDatatype>::Neg, Self>>;

    fn neg(self) -> Self::Output {
        let shape = self.shape.to_vec();
        let op = ArrayUnary::neg(self).expect("program");
        ArrayOp::new(shape, op)
    }
}

impl<A: NDArrayRead> Not for ArrayView<A> {
    type Output = ArrayOp<ArrayUnary<A::DType, u8, Self>>;

    fn not(self) -> Self::Output {
        let shape = self.shape.to_vec();
        let op = ArrayUnary::not(self).expect("program");
        ArrayOp::new(shape, op)
    }
}

impl<A: NDArrayRead + fmt::Debug> NDArrayTransform for ArrayView<A> {
    type Broadcast = Self;
    type Expand = Self;
    type Reshape = ArrayView<Self>;
    type Slice = ArraySlice<Self>;
    type Transpose = Self;

    fn broadcast(self, shape: Shape) -> Result<Self::Broadcast, Error> {
        if shape.len() < self.ndim() {
            return Err(Error::Bounds(format!(
                "cannot broadcast {:?} into {:?}",
                self, shape
            )));
        }

        let offset = shape.len() - self.ndim();
        let mut strides = Vec::with_capacity(shape.len());
        strides.extend(iter::repeat(0).take(offset));

        for (x, (dim, stride)) in self.shape().iter().copied().zip(&self.strides).enumerate() {
            if dim == 1 || dim == shape[offset + x] {
                strides.push(*stride);
            } else {
                return Err(Error::Bounds(format!(
                    "cannot broadcast {} into {}",
                    dim,
                    shape[offset + x]
                )));
            }
        }

        debug_assert_eq!(strides.len(), shape.len());

        ArrayView::new(self.source, shape, strides)
    }

    fn expand_dims(self, mut axes: Vec<usize>) -> Result<Self::Expand, Error> {
        axes.sort();

        if axes.last().copied() > Some(self.ndim()) {
            return Err(Error::Bounds(format!(
                "cannot expand axes {:?} of {:?}",
                axes, self
            )));
        }

        let mut shape = Vec::with_capacity(self.ndim() + axes.len());
        shape.extend_from_slice(self.shape());

        let mut strides = Vec::with_capacity(self.ndim() + axes.len());
        strides.extend_from_slice(&self.strides);

        for x in axes.into_iter().rev() {
            shape.insert(x, 1);
            strides.insert(x, 0);
        }

        debug_assert_eq!(shape.len(), strides.len());

        ArrayView::new(self.source, shape, strides)
    }

    fn reshape(self, shape: Shape) -> Result<Self::Reshape, Error> {
        let strides = strides_for(&shape, shape.len());
        ArrayView::new(self, shape, strides)
    }

    fn slice(self, bounds: Vec<AxisBound>) -> Result<Self::Slice, Error> {
        ArraySlice::new(self, bounds)
    }

    fn transpose(self, axes: Option<Vec<usize>>) -> Result<Self::Transpose, Error> {
        let axes = permutation(&self, axes)?;
        let shape = axes.iter().copied().map(|x| self.shape[x]).collect();
        let strides = axes.into_iter().map(|x| self.strides[x]).collect();
        ArrayView::new(self.source, shape, strides)
    }
}

impl<T: CDatatype, A: Into<Array<T>>> From<ArrayView<A>> for Array<T> {
    fn from(view: ArrayView<A>) -> Self {
        Self::View(Box::new(ArrayView {
            source: view.source.into(),
            shape: view.shape,
            strides: view.strides,
            #[cfg(feature = "opencl")]
            kernel_op: view.kernel_op,
        }))
    }
}

impl<A: fmt::Debug> fmt::Debug for ArrayView<A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "view of {:?} with shape {:?}", self.source, self.shape)
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
fn expand_dims<A: NDArray + fmt::Debug>(source: &A, mut axes: Vec<usize>) -> Result<Shape, Error> {
    axes.sort();

    if axes.is_empty() {
        Ok(source.shape().to_vec())
    } else if *axes.last().expect("x") <= source.ndim() {
        let mut shape = Vec::with_capacity(source.ndim() + axes.len());
        shape.extend_from_slice(source.shape());

        for x in axes.into_iter().rev() {
            shape.insert(x, 1);
        }

        Ok(shape)
    } else {
        Err(Error::Bounds(format!(
            "cannot expand axes {:?} of {:?}",
            axes, source
        )))
    }
}

#[inline]
fn offset_of(coord: &[usize], shape: &[usize]) -> usize {
    let strides = shape.iter().enumerate().map(|(x, dim)| {
        if *dim == 1 {
            0
        } else {
            shape.iter().rev().take(shape.len() - 1 - x).product()
        }
    });

    coord
        .iter()
        .copied()
        .zip(strides)
        .map(|(i, dim)| i * dim)
        .sum()
}

#[inline]
fn permutation<A: NDArray + fmt::Debug>(
    source: &A,
    axes: Option<Vec<usize>>,
) -> Result<Vec<usize>, Error> {
    let ndim = source.ndim();

    if let Some(axes) = axes {
        if axes.len() == ndim
            && axes.iter().copied().all(|x| x < ndim)
            && (0..ndim).into_iter().all(|x| axes.contains(&x))
        {
            Ok(axes)
        } else {
            Err(Error::Bounds(format!(
                "invalid permutation for {:?}: {:?}",
                source, axes
            )))
        }
    } else {
        Ok((0..ndim).into_iter().rev().collect())
    }
}

#[inline]
fn validate_coord<A: NDArray + fmt::Debug>(array: &A, coord: &[usize]) -> Result<(), Error> {
    if coord.len() == array.ndim() || coord.iter().zip(array.shape()).all(|(i, dim)| i < dim) {
        Ok(())
    } else {
        Err(Error::Bounds(format!(
            "invalid coordinate for {:?}: {:?}",
            array, coord
        )))
    }
}
