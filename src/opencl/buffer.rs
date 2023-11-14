use ocl::Buffer;

use crate::buffer::{BufferConverter, BufferInstance, BufferMut};
use crate::{CType, Error};

impl<T: CType> BufferInstance<T> for ocl::Buffer<T> {
    fn read(&self) -> Result<BufferConverter<T>, Error> {
        Ok(BufferConverter::CL(self.into()))
    }

    fn size(&self) -> usize {
        self.len()
    }
}

impl<'a, T: CType> BufferMut<'a, T> for ocl::Buffer<T> {
    type Data = CLConverter<'a, T>;

    fn write(&'a mut self, data: Self::Data) -> Result<(), Error> {
        if data.size() == self.size() {
            data.as_ref()
                .copy(self, None, None)
                .enq()
                .map_err(Error::from)
        } else {
            Err(Error::Bounds(format!(
                "cannot overwrite a buffer of size {} with one of size {}",
                self.size(),
                data.size()
            )))
        }
    }
}

impl<'a, T: CType> BufferInstance<T> for &'a ocl::Buffer<T> {
    fn read(&self) -> Result<BufferConverter<T>, Error> {
        Ok(BufferConverter::CL((*self).into()))
    }

    fn size(&self) -> usize {
        self.len()
    }
}

impl<'a, T: CType> BufferInstance<T> for &'a mut ocl::Buffer<T> {
    fn read(&self) -> Result<BufferConverter<T>, Error> {
        Ok(BufferConverter::CL((&**self).into()))
    }

    fn size(&self) -> usize {
        self.len()
    }
}

impl<'a, T: CType> BufferMut<'a, T> for &'a mut ocl::Buffer<T> {
    type Data = &'a ocl::Buffer<T>;

    fn write(&'a mut self, data: Self::Data) -> Result<(), Error> {
        BufferMut::write(&mut **self, data.into())
    }
}

/// A buffer in OpenCL memory
#[derive(Clone)]
pub enum CLConverter<'a, T: CType> {
    Owned(ocl::Buffer<T>),
    Borrowed(&'a ocl::Buffer<T>),
}

#[cfg(feature = "opencl")]
impl<'a, T: CType> CLConverter<'a, T> {
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
    pub fn size(&self) -> usize {
        match self {
            Self::Owned(buffer) => buffer.len(),
            Self::Borrowed(buffer) => buffer.len(),
        }
    }
}

#[cfg(feature = "opencl")]
impl<'a, T: CType> AsRef<ocl::Buffer<T>> for CLConverter<'a, T> {
    fn as_ref(&self) -> &ocl::Buffer<T> {
        match self {
            Self::Owned(buffer) => &buffer,
            Self::Borrowed(buffer) => buffer,
        }
    }
}

impl<T: CType> From<ocl::Buffer<T>> for CLConverter<'static, T> {
    fn from(buf: Buffer<T>) -> Self {
        Self::Owned(buf)
    }
}

impl<'a, T: CType> From<&'a ocl::Buffer<T>> for CLConverter<'a, T> {
    fn from(buf: &'a Buffer<T>) -> Self {
        Self::Borrowed(buf)
    }
}
