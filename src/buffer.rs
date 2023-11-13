#[cfg(feature = "stream")]
use destream::{de, en};

#[cfg(feature = "opencl")]
use crate::opencl;
use crate::{host, CType, Error, StackVec};

pub trait BufferInstance<T: CType>: Send + Sync + Sized {
    fn read(&self) -> Result<BufferConverter<T>, Error>;

    fn size(&self) -> usize;
}

#[derive(Clone, Debug)]
pub enum Buffer<T: CType> {
    #[cfg(feature = "opencl")]
    CL(ocl::Buffer<T>),
    Host(host::Buffer<T>),
}

impl<T: CType> BufferInstance<T> for Buffer<T> {
    fn read(&self) -> Result<BufferConverter<T>, Error> {
        Ok(BufferConverter::from(self))
    }

    fn size(&self) -> usize {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(buf) => buf.size(),
            Self::Host(buf) => buf.size(),
        }
    }
}

impl<'a, T: CType> BufferInstance<T> for &'a Buffer<T> {
    fn read(&self) -> Result<BufferConverter<T>, Error> {
        Ok(BufferConverter::from(*self))
    }

    fn size(&self) -> usize {
        BufferInstance::size(*self)
    }
}

#[cfg(feature = "opencl")]
impl<T: CType> From<ocl::Buffer<T>> for Buffer<T> {
    fn from(buf: ocl::Buffer<T>) -> Self {
        Self::CL(buf)
    }
}

impl<T: CType> From<host::Buffer<T>> for Buffer<T> {
    fn from(buf: host::Buffer<T>) -> Self {
        Self::Host(buf)
    }
}

#[derive(Clone)]
/// A sequence of elements in a single contiguous block of memory
pub enum BufferConverter<'a, T: CType> {
    #[cfg(feature = "opencl")]
    CL(opencl::CLConverter<'a, T>),
    Host(host::SliceConverter<'a, T>),
}

impl<'a, T: CType> BufferConverter<'a, T> {
    /// Return an owned [`Buffer`], allocating memory only if this is a borrow.
    pub fn into_buffer(self) -> Result<Buffer<T>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(buffer) => buffer.into_buffer().map(Buffer::CL),
            Self::Host(buffer) => Ok(Buffer::Host(buffer.into_buffer())),
        }
    }

    /// Return the number of elements in this [`Buffer`].
    pub fn size(&self) -> usize {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(buffer) => buffer.size(),
            Self::Host(buffer) => buffer.size(),
        }
    }

    #[cfg(feature = "opencl")]
    /// Ensure that this [`Buffer`] is in OpenCL memory by making a copy if necessary.
    pub fn to_cl(self) -> Result<opencl::CLConverter<'a, T>, ocl::Error> {
        match self {
            Self::CL(buffer) => Ok(buffer),
            Self::Host(buffer) => {
                opencl::OpenCL::copy_into_buffer(buffer.as_ref()).map(opencl::CLConverter::Owned)
            }
        }
    }

    /// Ensure that this buffer is in host memory by making a copy if necessary.
    pub fn to_slice(self) -> Result<host::SliceConverter<'a, T>, Error> {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(buffer) => {
                let buffer = buffer.as_ref();
                let mut copy = vec![T::default(); buffer.len()];
                buffer.read(&mut copy[..]).enq()?;
                Ok(host::SliceConverter::from(copy))
            }
            Self::Host(buffer) => Ok(buffer),
        }
    }
}

impl<T: CType> From<Buffer<T>> for BufferConverter<'static, T> {
    fn from(buf: Buffer<T>) -> Self {
        match buf {
            #[cfg(feature = "opencl")]
            Buffer::CL(buf) => Self::CL(buf.into()),
            Buffer::Host(buf) => Self::Host(buf.into()),
        }
    }
}

impl<'a, T: CType> From<&'a Buffer<T>> for BufferConverter<'a, T> {
    fn from(buf: &'a Buffer<T>) -> Self {
        match buf {
            #[cfg(feature = "opencl")]
            Buffer::CL(buf) => Self::CL(buf.into()),
            Buffer::Host(buf) => Self::Host(buf.into()),
        }
    }
}

impl<T: CType> From<Vec<T>> for BufferConverter<'static, T> {
    fn from(buf: Vec<T>) -> Self {
        Self::Host(buf.into())
    }
}

impl<T: CType> From<StackVec<T>> for BufferConverter<'static, T> {
    fn from(buf: StackVec<T>) -> Self {
        Self::Host(buf.into())
    }
}

impl<'a, T: CType> From<&'a [T]> for BufferConverter<'a, T> {
    fn from(buf: &'a [T]) -> Self {
        Self::Host(buf.into())
    }
}

#[cfg(feature = "opencl")]
impl<T: CType> From<ocl::Buffer<T>> for BufferConverter<'static, T> {
    fn from(buf: ocl::Buffer<T>) -> Self {
        Self::CL(buf.into())
    }
}

#[cfg(feature = "opencl")]
impl<'a, T: CType> From<&'a ocl::Buffer<T>> for BufferConverter<'a, T> {
    fn from(buf: &'a ocl::Buffer<T>) -> Self {
        Self::CL(buf.into())
    }
}

#[cfg(feature = "stream")]
impl<'en, T: CType> en::IntoStream<'en> for Buffer<T>
where
    for<'a> &'a [T]: en::IntoStream<'en>,
{
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream(BufferConverter::from(self), encoder)
    }
}

#[cfg(feature = "stream")]
impl<'en, T: CType> en::ToStream<'en> for Buffer<T>
where
    for<'a> &'a [T]: en::IntoStream<'en>,
{
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream(BufferConverter::from(self), encoder)
    }
}

#[cfg(feature = "stream")]
impl<'en, T: CType> en::IntoStream<'en> for BufferConverter<'en, T>
where
    for<'a> &'a [T]: en::IntoStream<'en>,
{
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        let slice = self.to_slice().map_err(en::Error::custom)?;
        en::IntoStream::<'en>::into_stream(slice.as_ref(), encoder)
    }
}

#[cfg(feature = "stream")]
#[async_trait::async_trait]
impl<T: CType> de::FromStream for Buffer<T>
where
    T: de::FromStream<Context = ()>,
{
    type Context = ();

    async fn from_stream<D: de::Decoder>(cxt: (), decoder: &mut D) -> Result<Self, D::Error> {
        let data = Vec::<T>::from_stream(cxt, decoder).await?;
        Ok(Self::Host(data.into()))
    }
}
