use std::fmt;
use std::ops::{Add, Mul};
use std::sync::{Arc, RwLock};

#[cfg(feature = "stream")]
use async_trait::async_trait;
#[cfg(feature = "stream")]
use destream::{de, en};
use rayon::prelude::*;

#[cfg(feature = "opencl")]
use super::cl_programs;
use super::{CDatatype, Error, Queue};

pub trait BufferInstance: Clone + Send + Sync {
    type DType: CDatatype;

    fn size(&self) -> usize;
}

pub trait BufferReduce {
    type DType: CDatatype;

    fn all(&self, queue: &Queue) -> Result<bool, Error>;

    fn any(&self, queue: &Queue) -> Result<bool, Error>;

    fn max(&self, queue: &Queue) -> Result<Self::DType, Error>;

    fn min(&self, queue: &Queue) -> Result<Self::DType, Error>;

    fn product(&self, queue: &Queue) -> Result<Self::DType, Error>;

    fn sum(&self, queue: &Queue) -> Result<Self::DType, Error>;
}

#[derive(Clone)]
pub enum SliceConverter<'a, T> {
    Vec(Vec<T>),
    Slice(&'a [T]),
}

impl<'a, T: Clone> SliceConverter<'a, T> {
    pub fn into_vec(self) -> Vec<T> {
        match self {
            Self::Vec(vec) => vec,
            Self::Slice(slice) => slice.to_vec(),
        }
    }

    pub fn size(&self) -> usize {
        match self {
            Self::Vec(vec) => vec.len(),
            Self::Slice(slice) => slice.len(),
        }
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
pub enum CLConverter<'a, T: CDatatype> {
    Owned(ocl::Buffer<T>),
    Borrowed(&'a ocl::Buffer<T>),
}

#[cfg(feature = "opencl")]
impl<'a, T: CDatatype> CLConverter<'a, T> {
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

#[derive(Clone)]
pub enum BufferConverter<'a, T: CDatatype> {
    Host(SliceConverter<'a, T>),
    #[cfg(feature = "opencl")]
    CL(CLConverter<'a, T>),
}

impl<'a, T: CDatatype> BufferConverter<'a, T> {
    pub fn into_buffer(self) -> Result<Buffer<T>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(buffer) => buffer.into_buffer().map(Buffer::CL),
            Self::Host(buffer) => Ok(Buffer::Host(buffer.into_vec())),
        }
    }

    pub fn size(&self) -> usize {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(buffer) => buffer.len(),
            Self::Host(buffer) => buffer.size(),
        }
    }

    #[cfg(feature = "opencl")]
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

impl<T: CDatatype> BufferInstance for Vec<T> {
    type DType = T;

    fn size(&self) -> usize {
        self.len()
    }
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

    fn size(&self) -> usize {
        self.len()
    }
}

impl<T: CDatatype> BufferInstance for Arc<RwLock<Vec<T>>> {
    type DType = T;

    fn size(&self) -> usize {
        let data = RwLock::read(self).expect("read buffer");
        data.len()
    }
}

#[cfg(feature = "opencl")]
impl<T: CDatatype> BufferInstance for ocl::Buffer<T> {
    type DType = T;

    fn size(&self) -> usize {
        self.len()
    }
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

    fn size(&self) -> usize {
        self.len()
    }
}

#[cfg(feature = "opencl")]
impl<T: CDatatype> BufferInstance for Arc<RwLock<ocl::Buffer<T>>> {
    type DType = T;

    fn size(&self) -> usize {
        let data = RwLock::read(self).expect("read buffer");
        data.len()
    }
}

#[cfg(feature = "opencl")]
#[derive(Clone)]
pub enum Buffer<T: CDatatype> {
    Host(Vec<T>),
    CL(ocl::Buffer<T>),
}

#[cfg(not(feature = "opencl"))]
#[derive(Clone)]
pub enum Buffer<T: CDatatype> {
    Host(Vec<T>),
}

impl<T: CDatatype> Buffer<T> {
    pub fn len(&self) -> usize {
        match self {
            Self::Host(buffer) => buffer.len(),
            #[cfg(feature = "opencl")]
            Self::CL(buffer) => buffer.len(),
        }
    }

    pub fn write<'a, O: Into<BufferConverter<'a, T>>>(&mut self, other: O) -> Result<(), Error> {
        let other = other.into();

        match self {
            Self::Host(this) => match other {
                BufferConverter::Host(that) => this.copy_from_slice(that.as_ref()),
                #[cfg(feature = "opencl")]
                BufferConverter::CL(that) => that.as_ref().read(&mut this[..]).enq()?,
            },
            #[cfg(feature = "opencl")]
            Self::CL(this) => match other {
                BufferConverter::Host(that) => this.write(that.as_ref()).enq()?,
                BufferConverter::CL(that) => that.as_ref().copy(this, None, None).enq()?,
            },
        }

        Ok(())
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

    fn size(&self) -> usize {
        buffer_dispatch!(self, this, this.size())
    }
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

    fn size(&self) -> usize {
        match &**self {
            Buffer::Host(buffer) => buffer.len(),
            #[cfg(feature = "opencl")]
            Buffer::CL(buffer) => buffer.len(),
        }
    }
}

impl<T: CDatatype> BufferInstance for Arc<RwLock<Buffer<T>>> {
    type DType = T;

    fn size(&self) -> usize {
        let data = RwLock::read(self).expect("read buffer");
        data.size()
    }
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
