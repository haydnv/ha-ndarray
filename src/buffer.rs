//! The underlying [`Buffer`] types used to store array elements

use std::ops::{Add, Mul};
use std::sync::{Arc, RwLock};
use std::{fmt, iter};

#[cfg(feature = "stream")]
use async_trait::async_trait;
#[cfg(feature = "stream")]
use destream::{de, en};
use get_size::GetSize;
use rayon::prelude::*;

#[cfg(feature = "opencl")]
use super::cl_programs;
use super::{CDatatype, Error, Queue};

/// An array buffer
pub trait BufferInstance: Send + Sync {
    type DType: CDatatype;
}

impl<B: BufferInstance + ?Sized> BufferInstance for Box<B> {
    type DType = B::DType;
}

/// Buffer read methods
pub trait BufferRead: BufferInstance {
    /// Access the elements of this buffer as a [`BufferConverter`].
    fn read(&self) -> BufferConverter<Self::DType>;

    /// Read an individual element from this buffer.
    fn read_value(&self, offset: usize) -> Result<Self::DType, Error>;
}

impl<T: CDatatype> BufferRead for Vec<T> {
    fn read(&self) -> BufferConverter<Self::DType> {
        SliceConverter::Slice(self).into()
    }

    fn read_value(&self, offset: usize) -> Result<Self::DType, Error> {
        Ok(self[offset])
    }
}

impl<T: CDatatype> BufferRead for Arc<Vec<T>> {
    fn read(&self) -> BufferConverter<Self::DType> {
        SliceConverter::Slice(self).into()
    }

    fn read_value(&self, offset: usize) -> Result<Self::DType, Error> {
        Ok(self[offset])
    }
}

#[cfg(feature = "opencl")]
impl<T: CDatatype> BufferRead for ocl::Buffer<T> {
    fn read(&self) -> BufferConverter<Self::DType> {
        CLConverter::Borrowed(self).into()
    }

    fn read_value(&self, offset: usize) -> Result<Self::DType, Error> {
        let mut data = vec![T::zero()];
        let buffer = self.create_sub_buffer(None, offset, 1)?;
        buffer.read(&mut data).enq()?;
        Ok(data[0])
    }
}

#[cfg(feature = "opencl")]
impl<T: CDatatype> BufferRead for Arc<ocl::Buffer<T>> {
    fn read(&self) -> BufferConverter<Self::DType> {
        CLConverter::Borrowed(self).into()
    }

    fn read_value(&self, offset: usize) -> Result<Self::DType, Error> {
        (**self).read_value(offset)
    }
}

impl<T: CDatatype> BufferRead for Buffer<T> {
    fn read(&self) -> BufferConverter<Self::DType> {
        match self {
            Buffer::Host(buffer) => SliceConverter::Slice(buffer).into(),
            #[cfg(feature = "opencl")]
            Buffer::CL(buffer) => CLConverter::Borrowed(buffer).into(),
        }
    }

    fn read_value(&self, offset: usize) -> Result<Self::DType, Error> {
        match self {
            Buffer::Host(buffer) => buffer.read_value(offset),
            #[cfg(feature = "opencl")]
            Buffer::CL(buffer) => buffer.read_value(offset),
        }
    }
}

impl<T: CDatatype> BufferRead for Arc<Buffer<T>> {
    fn read(&self) -> BufferConverter<Self::DType> {
        match &**self {
            Buffer::Host(buffer) => SliceConverter::Slice(buffer).into(),
            #[cfg(feature = "opencl")]
            Buffer::CL(buffer) => CLConverter::Borrowed(buffer).into(),
        }
    }

    fn read_value(&self, offset: usize) -> Result<Self::DType, Error> {
        match &**self {
            Buffer::Host(buffer) => buffer.read_value(offset),
            #[cfg(feature = "opencl")]
            Buffer::CL(buffer) => buffer.read_value(offset),
        }
    }
}

#[cfg(feature = "freqfs")]
impl<FE: Send + Sync, T: CDatatype> BufferRead for freqfs::FileReadGuardOwned<FE, Buffer<T>> {
    fn read(&self) -> BufferConverter<Self::DType> {
        match &**self {
            Buffer::Host(buffer) => SliceConverter::Slice(buffer).into(),
            #[cfg(feature = "opencl")]
            Buffer::CL(buffer) => CLConverter::Borrowed(buffer).into(),
        }
    }

    fn read_value(&self, offset: usize) -> Result<Self::DType, Error> {
        match &**self {
            Buffer::Host(buffer) => buffer.read_value(offset),
            #[cfg(feature = "opencl")]
            Buffer::CL(buffer) => buffer.read_value(offset),
        }
    }
}

#[cfg(feature = "freqfs")]
impl<FE: Send + Sync, T: CDatatype> BufferRead for freqfs::FileWriteGuardOwned<FE, Buffer<T>> {
    fn read(&self) -> BufferConverter<Self::DType> {
        match &**self {
            Buffer::Host(buffer) => SliceConverter::Slice(buffer).into(),
            #[cfg(feature = "opencl")]
            Buffer::CL(buffer) => CLConverter::Borrowed(buffer).into(),
        }
    }

    fn read_value(&self, offset: usize) -> Result<Self::DType, Error> {
        match &**self {
            Buffer::Host(buffer) => buffer.read_value(offset),
            #[cfg(feature = "opencl")]
            Buffer::CL(buffer) => buffer.read_value(offset),
        }
    }
}

/// Buffer write methods
pub trait BufferWrite: BufferInstance {
    /// Overwrite the elements of this buffer with the given `other` elements.
    fn write<'a, O: Into<BufferConverter<'a, Self::DType>>>(
        &mut self,
        other: O,
    ) -> Result<(), Error>;

    /// Overwrite all elements of this buffer with the given scalar `value`.
    fn write_value(&mut self, value: Self::DType) -> Result<(), Error>;

    /// Overwrite the element at `offset` with the given `value`.
    fn write_value_at(&mut self, offset: usize, value: Self::DType) -> Result<(), Error>;
}

impl<T: CDatatype> BufferWrite for Vec<T> {
    fn write<'a, O: Into<BufferConverter<'a, T>>>(&mut self, other: O) -> Result<(), Error> {
        match other.into() {
            BufferConverter::Host(buffer) => {
                self.copy_from_slice(buffer.as_ref());
                Ok(())
            }
            #[cfg(feature = "opencl")]
            BufferConverter::CL(buffer) => buffer
                .as_ref()
                .read(&mut self[..])
                .enq()
                .map_err(Error::from),
        }
    }

    fn write_value(&mut self, value: T) -> Result<(), Error> {
        let len = self.len();
        self.clear();
        self.extend(iter::repeat(value).take(len));
        Ok(())
    }

    fn write_value_at(&mut self, offset: usize, value: T) -> Result<(), Error> {
        if offset < self.len() {
            self[offset] = value;
            Ok(())
        } else {
            Err(Error::Bounds(format!(
                "offset {} is out of bounds for buffer of size {}",
                offset,
                self.len()
            )))
        }
    }
}

#[cfg(feature = "opencl")]
impl<T: CDatatype> BufferWrite for ocl::Buffer<T> {
    fn write<'a, O: Into<BufferConverter<'a, Self::DType>>>(
        &mut self,
        other: O,
    ) -> Result<(), Error> {
        match other.into() {
            BufferConverter::Host(buffer) => ocl::Buffer::write(self, buffer.as_ref())
                .enq()
                .map_err(Error::from),

            #[cfg(feature = "opencl")]
            BufferConverter::CL(buffer) => buffer
                .as_ref()
                .copy(self, None, None)
                .enq()
                .map_err(Error::from),
        }
    }

    fn write_value(&mut self, value: Self::DType) -> Result<(), Error> {
        ocl::Buffer::write(self, &vec![value; self.len()])
            .enq()
            .map_err(Error::from)
    }

    fn write_value_at(&mut self, offset: usize, value: Self::DType) -> Result<(), Error> {
        let data = self.create_sub_buffer(None, offset, 1)?;
        data.write(&vec![value]).enq().map_err(Error::from)
    }
}

#[cfg(feature = "freqfs")]
impl<FE: Send + Sync, T: CDatatype> BufferWrite for freqfs::FileWriteGuardOwned<FE, Buffer<T>> {
    fn write<'a, O: Into<BufferConverter<'a, Self::DType>>>(
        &mut self,
        other: O,
    ) -> Result<(), Error> {
        Buffer::write(&mut *self, other)
    }

    fn write_value(&mut self, value: Self::DType) -> Result<(), Error> {
        Buffer::write_value(&mut *self, value)
    }

    fn write_value_at(&mut self, offset: usize, value: Self::DType) -> Result<(), Error> {
        Buffer::write_value_at(&mut *self, offset, value)
    }
}

/// Buffer reduce operations
pub trait BufferReduce {
    type DType: CDatatype;

    /// Return `true` if all elements in this buffer are non-zero.
    fn all(&self, queue: &Queue) -> Result<bool, Error>;

    /// Return `true` if any elements in this buffer are non-zero.
    fn any(&self, queue: &Queue) -> Result<bool, Error>;

    /// Return the maximum element in this buffer.
    fn max(&self, queue: &Queue) -> Result<Self::DType, Error>;

    /// Return the minimum element in this buffer.
    fn min(&self, queue: &Queue) -> Result<Self::DType, Error>;

    /// Return the product of all elements in this buffer.
    fn product(&self, queue: &Queue) -> Result<Self::DType, Error>;

    /// Return the sum of all elements in this buffer.
    fn sum(&self, queue: &Queue) -> Result<Self::DType, Error>;
}

#[derive(Clone)]
/// A buffer in host memory, either borrowed or owned
pub enum SliceConverter<'a, T> {
    Vec(Vec<T>),
    Slice(&'a [T]),
}

impl<'a, T> SliceConverter<'a, T> {
    /// Return the number of elements in this buffer.
    pub fn len(&self) -> usize {
        match self {
            Self::Vec(vec) => vec.len(),
            Self::Slice(slice) => slice.len(),
        }
    }
}

impl<'a, T: Clone> SliceConverter<'a, T> {
    /// Return this buffer as an owned [`Vec`].
    /// This will allocate a new [`Vec`] only if this buffer is a borrowed slice.
    pub fn into_vec(self) -> Vec<T> {
        match self {
            Self::Vec(vec) => vec,
            Self::Slice(slice) => slice.to_vec(),
        }
    }
}

impl<'a, T> GetSize for SliceConverter<'a, T> {
    fn get_size(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }
}

impl<'a, T> From<&'a Vec<T>> for SliceConverter<'a, T> {
    fn from(slice: &'a Vec<T>) -> Self {
        Self::Slice(slice)
    }
}

impl<'a, T> AsRef<[T]> for SliceConverter<'a, T> {
    fn as_ref(&self) -> &[T] {
        match self {
            Self::Vec(data) => data.as_slice(),
            Self::Slice(slice) => slice,
        }
    }
}

#[cfg(feature = "opencl")]
#[derive(Clone)]
/// A buffer in OpenCL memory
pub enum CLConverter<'a, T: CDatatype> {
    Owned(ocl::Buffer<T>),
    Borrowed(&'a ocl::Buffer<T>),
}

#[cfg(feature = "opencl")]
impl<'a, T: CDatatype> CLConverter<'a, T> {
    /// Return this buffer as an owned [`ocl::Buffer`].
    /// This will allocate a new [`ocl::Buffer`] only if this buffer is borrowed.
    pub fn into_buffer(self) -> Result<ocl::Buffer<T>, Error> {
        match self {
            Self::Owned(buffer) => Ok(buffer),
            Self::Borrowed(buffer) => {
                let cl_queue = buffer.default_queue().expect("OpenCL queue");
                let mut copy = ocl::Buffer::builder()
                    .queue(cl_queue.clone())
                    .len(buffer.len())
                    .build()?;

                buffer.copy(&mut copy, None, None).enq()?;

                Ok(copy)
            }
        }
    }

    /// Return the number of elements in this buffer.
    pub fn len(&self) -> usize {
        match self {
            Self::Owned(buffer) => buffer.len(),
            Self::Borrowed(buffer) => buffer.len(),
        }
    }
}

#[cfg(feature = "opencl")]
impl<'a, T: CDatatype> AsRef<ocl::Buffer<T>> for CLConverter<'a, T> {
    fn as_ref(&self) -> &ocl::Buffer<T> {
        match self {
            Self::Owned(buffer) => &buffer,
            Self::Borrowed(buffer) => buffer,
        }
    }
}

#[cfg(feature = "opencl")]
impl<'a, T: CDatatype> GetSize for CLConverter<'a, T> {
    fn get_size(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }
}

#[derive(Clone)]
/// A sequence of elements in a single contiguous block of memory
pub enum BufferConverter<'a, T: CDatatype> {
    Host(SliceConverter<'a, T>),
    #[cfg(feature = "opencl")]
    CL(CLConverter<'a, T>),
}

impl<'a, T: CDatatype> BufferConverter<'a, T> {
    /// Return an owned [`Buffer`], allocating memory only if this is a borrow.
    pub fn into_buffer(self) -> Result<Buffer<T>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(buffer) => buffer.into_buffer().map(Buffer::CL),
            Self::Host(buffer) => Ok(Buffer::Host(buffer.into_vec())),
        }
    }

    /// Return the number of elements in this [`Buffer`].
    pub fn len(&self) -> usize {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(buffer) => buffer.len(),
            Self::Host(buffer) => buffer.len(),
        }
    }

    #[cfg(feature = "opencl")]
    /// Ensure that this [`Buffer`] is in OpenCL memory by making a copy if necessary.
    pub fn to_cl(self, queue: &Queue) -> Result<CLConverter<'a, T>, Error> {
        match self {
            Self::CL(buffer) => Ok(buffer),
            Self::Host(buffer) => {
                let buffer = buffer.as_ref();

                let cl_queue = if let Some(cl_queue) = &queue.cl_queue {
                    cl_queue.clone()
                } else {
                    let cl_context = queue.context().cl_context();
                    let size_hint = Ord::max(buffer.len(), queue.context.gpu_min);
                    let device = queue
                        .context()
                        .select_device(size_hint)
                        .expect("OpenCL device");

                    ocl::Queue::new(cl_context, device, None)?
                };

                let buffer = ocl::Buffer::builder()
                    .queue(cl_queue)
                    .len(buffer.len())
                    .copy_host_slice(buffer)
                    .build()?;

                Ok(CLConverter::Owned(buffer))
            }
        }
    }

    /// Ensure that this buffer is in host memory by making a copy if necessary.
    pub fn to_slice(self) -> Result<SliceConverter<'a, T>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(buffer) => {
                let buffer = buffer.as_ref();
                let mut copy = vec![T::default(); buffer.len()];
                buffer.read(&mut copy[..]).enq()?;
                Ok(SliceConverter::Vec(copy))
            }
            Self::Host(buffer) => Ok(buffer),
        }
    }
}

impl<'a, T: CDatatype> GetSize for BufferConverter<'a, T> {
    fn get_size(&self) -> usize {
        match self {
            Self::Host(slice) => slice.get_size(),
            #[cfg(feature = "opencl")]
            Self::CL(buffer) => buffer.get_size(),
        }
    }
}

impl<'a, T: CDatatype> From<&'a Vec<T>> for BufferConverter<'a, T> {
    fn from(slice: &'a Vec<T>) -> Self {
        Self::Host(slice.into())
    }
}

macro_rules! buffer_reduce {
    ($this:ident, $var:ident, $call:expr) => {
        match $this {
            Self::Host(buffer) => match buffer {
                SliceConverter::Vec($var) => $call,
                SliceConverter::Slice($var) => $call,
            },
            #[cfg(feature = "opencl")]
            Self::CL(buffer) => match buffer {
                CLConverter::Owned($var) => $call,
                CLConverter::Borrowed($var) => $call,
            },
        }
    };
}

impl<'a, T: CDatatype> BufferReduce for BufferConverter<'a, T> {
    type DType = T;

    fn all(&self, queue: &Queue) -> Result<bool, Error> {
        buffer_reduce!(self, this, this.all(queue))
    }

    fn any(&self, queue: &Queue) -> Result<bool, Error> {
        buffer_reduce!(self, this, this.any(queue))
    }

    fn max(&self, queue: &Queue) -> Result<Self::DType, Error> {
        buffer_reduce!(self, this, this.max(queue))
    }

    fn min(&self, queue: &Queue) -> Result<Self::DType, Error> {
        buffer_reduce!(self, this, this.min(queue))
    }

    fn product(&self, queue: &Queue) -> Result<Self::DType, Error> {
        buffer_reduce!(self, this, this.product(queue))
    }

    fn sum(&self, queue: &Queue) -> Result<Self::DType, Error> {
        buffer_reduce!(self, this, this.sum(queue))
    }
}

impl<'a, T: CDatatype> From<SliceConverter<'a, T>> for BufferConverter<'a, T> {
    fn from(buffer: SliceConverter<'a, T>) -> Self {
        Self::Host(buffer)
    }
}

#[cfg(feature = "opencl")]
impl<'a, T: CDatatype> From<CLConverter<'a, T>> for BufferConverter<'a, T> {
    fn from(buffer: CLConverter<'a, T>) -> Self {
        Self::CL(buffer)
    }
}

impl<'a, T: CDatatype> From<Vec<T>> for BufferConverter<'a, T> {
    fn from(buffer: Vec<T>) -> Self {
        Self::Host(SliceConverter::Vec(buffer))
    }
}

impl<'a, T: CDatatype> From<&'a [T]> for BufferConverter<'a, T> {
    fn from(buffer: &'a [T]) -> Self {
        Self::Host(SliceConverter::Slice(buffer))
    }
}

#[cfg(feature = "opencl")]
impl<'a, T: CDatatype> From<ocl::Buffer<T>> for BufferConverter<'a, T> {
    fn from(buffer: ocl::Buffer<T>) -> Self {
        Self::CL(CLConverter::Owned(buffer))
    }
}

#[cfg(feature = "opencl")]
impl<'a, T: CDatatype> From<&'a ocl::Buffer<T>> for BufferConverter<'a, T> {
    fn from(buffer: &'a ocl::Buffer<T>) -> Self {
        Self::CL(CLConverter::Borrowed(buffer))
    }
}

impl<'a, T: CDatatype> From<Buffer<T>> for BufferConverter<'a, T> {
    fn from(buffer: Buffer<T>) -> Self {
        match buffer {
            Buffer::Host(buffer) => BufferConverter::Host(SliceConverter::Vec(buffer)),
            #[cfg(feature = "opencl")]
            Buffer::CL(buffer) => BufferConverter::CL(CLConverter::Owned(buffer)),
        }
    }
}

impl<'a, T: CDatatype> From<&'a Buffer<T>> for BufferConverter<'a, T> {
    fn from(buffer: &'a Buffer<T>) -> Self {
        match buffer {
            Buffer::Host(buffer) => BufferConverter::Host(SliceConverter::Slice(buffer)),
            #[cfg(feature = "opencl")]
            Buffer::CL(buffer) => BufferConverter::CL(CLConverter::Borrowed(buffer)),
        }
    }
}

/// A mutable accessor for a buffer in host memory.
pub enum SliceConverterMut<'a, T> {
    Vec(Vec<T>),
    Slice(&'a mut [T]),
}

impl<'a, T> From<&'a mut Vec<T>> for SliceConverterMut<'a, T> {
    fn from(slice: &'a mut Vec<T>) -> Self {
        Self::Slice(slice)
    }
}

impl<'a, T> From<SliceConverterMut<'a, T>> for SliceConverter<'a, T> {
    fn from(slice: SliceConverterMut<'a, T>) -> Self {
        match slice {
            SliceConverterMut::Vec(slice) => Self::Vec(slice),
            SliceConverterMut::Slice(slice) => Self::Slice(slice),
        }
    }
}

impl<'a, T> AsMut<[T]> for SliceConverterMut<'a, T> {
    fn as_mut(&mut self) -> &mut [T] {
        match self {
            Self::Vec(data) => &mut data[..],
            Self::Slice(slice) => slice,
        }
    }
}

#[cfg(feature = "opencl")]
/// A mutable accessor for a buffer in OpenCL memory.
pub enum CLConverterMut<'a, T: CDatatype> {
    Owned(ocl::Buffer<T>),
    Borrowed(&'a mut ocl::Buffer<T>),
}

#[cfg(feature = "opencl")]
impl<'a, T: CDatatype> From<&'a mut ocl::Buffer<T>> for CLConverterMut<'a, T> {
    fn from(buffer: &'a mut ocl::Buffer<T>) -> Self {
        Self::Borrowed(buffer)
    }
}

#[cfg(feature = "opencl")]
impl<'a, T: CDatatype> From<CLConverterMut<'a, T>> for CLConverter<'a, T> {
    fn from(buffer: CLConverterMut<'a, T>) -> Self {
        match buffer {
            CLConverterMut::Owned(buffer) => CLConverter::Owned(buffer),
            CLConverterMut::Borrowed(buffer) => CLConverter::Borrowed(buffer),
        }
    }
}

#[cfg(feature = "opencl")]
impl<'a, T: CDatatype> AsMut<ocl::Buffer<T>> for CLConverterMut<'a, T> {
    fn as_mut(&mut self) -> &mut ocl::Buffer<T> {
        match self {
            Self::Owned(buffer) => buffer,
            Self::Borrowed(buffer) => buffer,
        }
    }
}

/// A mutable accessor for a [`Buffer`].
pub enum BufferConverterMut<'a, T: CDatatype> {
    Host(SliceConverterMut<'a, T>),
    #[cfg(feature = "opencl")]
    CL(CLConverterMut<'a, T>),
}

impl<'a, T: CDatatype> From<&'a mut Vec<T>> for BufferConverterMut<'a, T> {
    fn from(buffer: &'a mut Vec<T>) -> Self {
        Self::Host(buffer.into())
    }
}

#[cfg(feature = "opencl")]
impl<'a, T: CDatatype> From<&'a mut ocl::Buffer<T>> for BufferConverterMut<'a, T> {
    fn from(buffer: &'a mut ocl::Buffer<T>) -> Self {
        Self::CL(buffer.into())
    }
}

impl<'a, T: CDatatype> From<&'a mut Buffer<T>> for BufferConverterMut<'a, T> {
    fn from(buffer: &'a mut Buffer<T>) -> Self {
        match buffer {
            Buffer::Host(slice) => Self::Host(slice.into()),
            #[cfg(feature = "opencl")]
            Buffer::CL(buffer) => Self::CL(buffer.into()),
        }
    }
}

impl<'a, T: CDatatype> From<BufferConverterMut<'a, T>> for BufferConverter<'a, T> {
    fn from(buffer: BufferConverterMut<'a, T>) -> Self {
        match buffer {
            BufferConverterMut::Host(slice) => BufferConverter::Host(slice.into()),
            #[cfg(feature = "opencl")]
            BufferConverterMut::CL(buffer) => BufferConverter::CL(buffer.into()),
        }
    }
}

impl<T: CDatatype> BufferInstance for Vec<T> {
    type DType = T;
}

impl<T: CDatatype> BufferReduce for Vec<T> {
    type DType = T;

    fn all(&self, queue: &Queue) -> Result<bool, Error> {
        self.as_slice().all(queue)
    }

    fn any(&self, queue: &Queue) -> Result<bool, Error> {
        self.as_slice().any(queue)
    }

    fn max(&self, queue: &Queue) -> Result<Self::DType, Error> {
        self.as_slice().max(queue)
    }

    fn min(&self, queue: &Queue) -> Result<Self::DType, Error> {
        self.as_slice().min(queue)
    }

    fn product(&self, queue: &Queue) -> Result<Self::DType, Error> {
        self.as_slice().product(queue)
    }

    fn sum(&self, queue: &Queue) -> Result<Self::DType, Error> {
        self.as_slice().sum(queue)
    }
}

impl<T: CDatatype> BufferReduce for [T] {
    type DType = T;

    fn all(&self, _queue: &Queue) -> Result<bool, Error> {
        let zero = Self::DType::zero();
        Ok(self.par_iter().copied().all(|n| n != zero))
    }

    fn any(&self, _queue: &Queue) -> Result<bool, Error> {
        let zero = Self::DType::zero();
        Ok(self.par_iter().copied().any(|n| n != zero))
    }

    fn max(&self, _queue: &Queue) -> Result<Self::DType, Error> {
        let collector = |l, r| {
            if r > l {
                r
            } else {
                l
            }
        };

        Ok(self.par_iter().copied().reduce(T::min, collector))
    }

    fn min(&self, _queue: &Queue) -> Result<Self::DType, Error> {
        let collector = |l, r| {
            if r < l {
                r
            } else {
                l
            }
        };

        Ok(self.par_iter().copied().reduce(T::max, collector))
    }

    fn product(&self, _queue: &Queue) -> Result<Self::DType, Error> {
        Ok(self
            .par_chunks(8)
            .map(|chunk| chunk.into_iter().copied().fold(T::one(), Mul::mul))
            .reduce(T::one, Mul::mul))
    }

    fn sum(&self, _queue: &Queue) -> Result<Self::DType, Error> {
        Ok(self
            .par_chunks(8)
            .map(|chunk| chunk.into_iter().copied().fold(T::zero(), Add::add))
            .reduce(T::zero, Add::add))
    }
}

impl<T: CDatatype> BufferInstance for Arc<Vec<T>> {
    type DType = T;
}

impl<T: CDatatype> BufferInstance for Arc<RwLock<Vec<T>>> {
    type DType = T;
}

#[cfg(feature = "opencl")]
impl<T: CDatatype> BufferInstance for ocl::Buffer<T> {
    type DType = T;
}

#[cfg(feature = "opencl")]
impl<T: CDatatype> BufferReduce for ocl::Buffer<T> {
    type DType = T;

    fn all(&self, queue: &Queue) -> Result<bool, Error> {
        let cl_queue = queue.cl_queue(self.default_queue());
        cl_programs::reduce_all(cl_queue, self).map_err(Error::from)
    }

    fn any(&self, queue: &Queue) -> Result<bool, Error> {
        let cl_queue = queue.cl_queue(self.default_queue());
        cl_programs::reduce_any(cl_queue, self).map_err(Error::from)
    }

    fn max(&self, queue: &Queue) -> Result<Self::DType, Error> {
        let collector = |l, r| {
            if r > l {
                r
            } else {
                l
            }
        };

        let cl_queue = queue.cl_queue(self.default_queue());
        cl_programs::reduce(T::min(), "max", cl_queue, self, collector).map_err(Error::from)
    }

    fn min(&self, queue: &Queue) -> Result<Self::DType, Error> {
        let collector = |l, r| {
            if r < l {
                r
            } else {
                l
            }
        };

        let cl_queue = queue.cl_queue(self.default_queue());
        cl_programs::reduce(T::max(), "min", cl_queue, self, collector).map_err(Error::from)
    }

    fn product(&self, queue: &Queue) -> Result<Self::DType, Error> {
        let cl_queue = queue.cl_queue(self.default_queue());
        cl_programs::reduce(T::one(), "mul", cl_queue, self, Mul::mul).map_err(Error::from)
    }

    fn sum(&self, queue: &Queue) -> Result<Self::DType, Error> {
        let cl_queue = queue.cl_queue(self.default_queue());
        cl_programs::reduce(T::zero(), "add", cl_queue, self, Add::add).map_err(Error::from)
    }
}

#[cfg(feature = "opencl")]
impl<T: CDatatype> BufferInstance for Arc<ocl::Buffer<T>> {
    type DType = T;
}

#[cfg(feature = "opencl")]
impl<T: CDatatype> BufferInstance for Arc<RwLock<ocl::Buffer<T>>> {
    type DType = T;
}

#[cfg(feature = "opencl")]
#[derive(Clone)]
/// A sequence of elements in a single contiguous block in memory
pub enum Buffer<T: CDatatype> {
    Host(Vec<T>),
    CL(ocl::Buffer<T>),
}

#[cfg(not(feature = "opencl"))]
#[derive(Clone)]
/// A sequence of elements in a single contiguous block in memory
pub enum Buffer<T: CDatatype> {
    Host(Vec<T>),
}

impl<T: CDatatype> Buffer<T> {
    /// Return the number of elements in this [`Buffer`].
    pub fn len(&self) -> usize {
        match self {
            Self::Host(buffer) => buffer.len(),
            #[cfg(feature = "opencl")]
            Self::CL(buffer) => buffer.len(),
        }
    }
}

impl<T: CDatatype> GetSize for Buffer<T> {
    fn get_size(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }
}

impl<T: CDatatype> BufferWrite for Buffer<T> {
    fn write<'a, O: Into<BufferConverter<'a, T>>>(&mut self, other: O) -> Result<(), Error> {
        let other = other.into();

        match self {
            Self::Host(this) => match other {
                BufferConverter::Host(that) => this.copy_from_slice(that.as_ref()),
                #[cfg(feature = "opencl")]
                BufferConverter::CL(that) => that.as_ref().read(&mut this[..]).enq()?,
            },
            #[cfg(feature = "opencl")]
            Self::CL(this) => match other {
                BufferConverter::Host(that) => this.write(that.as_ref())?,
                BufferConverter::CL(that) => that.as_ref().copy(this, None, None).enq()?,
            },
        }

        Ok(())
    }

    fn write_value(&mut self, value: T) -> Result<(), Error> {
        match self {
            Self::Host(this) => {
                let len = this.len();
                this.clear();
                this.extend(iter::repeat(value).take(len));
                Ok(())
            }
            #[cfg(feature = "opencl")]
            Self::CL(this) => ocl::Buffer::write(this, &vec![value; this.len()])
                .enq()
                .map_err(Error::from),
        }
    }

    fn write_value_at(&mut self, offset: usize, value: T) -> Result<(), Error> {
        match self {
            Self::Host(this) => {
                this[offset] = value;
                Ok(())
            }
            #[cfg(feature = "opencl")]
            Self::CL(this) => {
                let this = this.create_sub_buffer(None, offset, 1)?;
                this.write(&vec![value]).enq().map_err(Error::from)
            }
        }
    }
}

macro_rules! buffer_dispatch {
    ($this:ident, $var:ident, $call:expr) => {
        match $this {
            Self::Host($var) => $call,
            #[cfg(feature = "opencl")]
            Self::CL($var) => $call,
        }
    };
}

impl<T: CDatatype> BufferInstance for Buffer<T> {
    type DType = T;
}

impl<T: CDatatype> BufferReduce for Buffer<T> {
    type DType = T;

    fn all(&self, queue: &Queue) -> Result<bool, Error> {
        buffer_dispatch!(self, this, BufferReduce::all(this, queue))
    }

    fn any(&self, queue: &Queue) -> Result<bool, Error> {
        buffer_dispatch!(self, this, BufferReduce::any(this, queue))
    }

    fn max(&self, queue: &Queue) -> Result<Self::DType, Error> {
        buffer_dispatch!(self, this, BufferReduce::max(this, queue))
    }

    fn min(&self, queue: &Queue) -> Result<Self::DType, Error> {
        buffer_dispatch!(self, this, BufferReduce::min(this, queue))
    }

    fn product(&self, queue: &Queue) -> Result<Self::DType, Error> {
        buffer_dispatch!(self, this, BufferReduce::product(this, queue))
    }

    fn sum(&self, queue: &Queue) -> Result<Self::DType, Error> {
        buffer_dispatch!(self, this, BufferReduce::sum(this, queue))
    }
}

impl<T: CDatatype> BufferInstance for Arc<Buffer<T>> {
    type DType = T;
}

impl<T: CDatatype> BufferInstance for Arc<RwLock<Buffer<T>>> {
    type DType = T;
}

#[cfg(feature = "freqfs")]
impl<FE: Send + Sync, T: CDatatype> BufferInstance for freqfs::FileReadGuardOwned<FE, Buffer<T>> {
    type DType = T;
}

#[cfg(feature = "freqfs")]
impl<FE: Send + Sync, T: CDatatype> BufferInstance for freqfs::FileWriteGuardOwned<FE, Buffer<T>> {
    type DType = T;
}

#[cfg(feature = "opencl")]
impl<T: CDatatype> From<ocl::Buffer<T>> for Buffer<T> {
    fn from(buffer: ocl::Buffer<T>) -> Self {
        Self::CL(buffer)
    }
}

impl<T: CDatatype> From<Vec<T>> for Buffer<T> {
    fn from(buffer: Vec<T>) -> Self {
        Self::Host(buffer)
    }
}

#[cfg(feature = "stream")]
struct BufferVisitor<T> {
    data: Vec<T>,
}

#[cfg(feature = "stream")]
impl<T> BufferVisitor<T> {
    fn new() -> Self {
        Self { data: Vec::new() }
    }
}

#[cfg(feature = "stream")]
macro_rules! decode_buffer {
    ($t:ty, $name:expr, $decode:ident, $visit:ident, $encode:ident) => {
        #[async_trait]
        impl de::Visitor for BufferVisitor<$t> {
            type Value = Buffer<$t>;

            fn expecting() -> &'static str {
                $name
            }

            async fn $visit<A: de::ArrayAccess<$t>>(
                self,
                mut array: A,
            ) -> Result<Self::Value, A::Error> {
                const BUF_SIZE: usize = 4_096;
                let mut data = self.data;

                let mut buf = [<$t>::zero(); BUF_SIZE];
                loop {
                    let len = array.buffer(&mut buf).await?;
                    if len == 0 {
                        break;
                    } else {
                        data.extend_from_slice(&buf[..len]);
                    }
                }

                Ok(Buffer::Host(data))
            }
        }

        #[async_trait]
        impl de::FromStream for Buffer<$t> {
            type Context = ();

            async fn from_stream<D: de::Decoder>(
                _cxt: (),
                decoder: &mut D,
            ) -> Result<Self, D::Error> {
                decoder.$decode(BufferVisitor::<$t>::new()).await
            }
        }

        impl<'en> en::ToStream<'en> for Buffer<$t> {
            fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
                match self {
                    Self::Host(buffer) => {
                        let chunk = buffer.into_iter().copied();
                        let fut = futures::future::ready(chunk);
                        let stream = futures::stream::once(fut);
                        encoder.$encode(stream)
                    }
                    #[cfg(feature = "opencl")]
                    Self::CL(buffer) => {
                        let mut data = Vec::with_capacity(buffer.len());
                        buffer.read(&mut data).enq().map_err(en::Error::custom)?;
                        encoder.$encode(futures::stream::once(futures::future::ready(data)))
                    }
                }
            }
        }

        impl<'en> en::IntoStream<'en> for Buffer<$t> {
            fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
                match self {
                    Self::Host(buffer) => {
                        encoder.$encode(futures::stream::once(futures::future::ready(buffer)))
                    }
                    #[cfg(feature = "opencl")]
                    Self::CL(buffer) => {
                        let mut data = Vec::with_capacity(buffer.len());
                        buffer.read(&mut data).enq().map_err(en::Error::custom)?;
                        encoder.$encode(futures::stream::once(futures::future::ready(data)))
                    }
                }
            }
        }
    };
}

#[cfg(feature = "stream")]
decode_buffer!(
    u8,
    "byte array",
    decode_array_u8,
    visit_array_u8,
    encode_array_u8
);

#[cfg(feature = "stream")]
decode_buffer!(
    u16,
    "16-bit unsigned int array",
    decode_array_u16,
    visit_array_u16,
    encode_array_u16
);

#[cfg(feature = "stream")]
decode_buffer!(
    u32,
    "32-bit unsigned int array",
    decode_array_u32,
    visit_array_u32,
    encode_array_u32
);

#[cfg(feature = "stream")]
decode_buffer!(
    u64,
    "64-bit unsigned int array",
    decode_array_u64,
    visit_array_u64,
    encode_array_u64
);

#[cfg(feature = "stream")]
decode_buffer!(
    i16,
    "16-bit int array",
    decode_array_i16,
    visit_array_i16,
    encode_array_i16
);

#[cfg(feature = "stream")]
decode_buffer!(
    i32,
    "32-bit int array",
    decode_array_i32,
    visit_array_i32,
    encode_array_i32
);

#[cfg(feature = "stream")]
decode_buffer!(
    i64,
    "64-bit int array",
    decode_array_i64,
    visit_array_i64,
    encode_array_i64
);

#[cfg(feature = "stream")]
decode_buffer!(
    f32,
    "32-bit int array",
    decode_array_f32,
    visit_array_f32,
    encode_array_f32
);

#[cfg(feature = "stream")]
decode_buffer!(
    f64,
    "64-bit int array",
    decode_array_f64,
    visit_array_f64,
    encode_array_f64
);

impl<T: CDatatype + fmt::Debug> fmt::Debug for Buffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Host(buffer) => fmt::Debug::fmt(buffer, f),
            #[cfg(feature = "opencl")]
            Self::CL(buffer) => fmt::Debug::fmt(buffer, f),
        }
    }
}
