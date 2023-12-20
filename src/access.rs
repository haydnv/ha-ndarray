use std::borrow::{Borrow, BorrowMut};
use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::buffer::{BufferConverter, BufferInstance, BufferMut};
use crate::ops::{ReadOp, Write};
use crate::platform::PlatformInstance;
use crate::{Buffer, CType, Error, Platform};

/// A type which allows accessing array data
pub trait Access<T: CType>: Send + Sync {
    /// Read the data of this accessor as a [`BufferConverter`].
    fn read(&self) -> Result<BufferConverter<T>, Error>;

    /// Access a single value.
    fn read_value(&self, offset: usize) -> Result<T, Error>;

    /// Return the data size.
    fn size(&self) -> usize;
}

/// A type which allows accessing array data mutably
pub trait AccessMut<T: CType>: Access<T> + fmt::Debug {
    #[cfg(feature = "opencl")]
    /// Borrow the array data as an [`ocl::Buffer`], or return an error if this not an OpenCL buffer.
    fn cl_buffer(&mut self) -> Result<&mut ocl::Buffer<T>, Error> {
        Err(Error::Unsupported(format!(
            "not an OpenCL buffer: {self:?}"
        )))
    }

    /// Overwrite these data with the given `data`.
    fn write<'a>(&mut self, data: BufferConverter<'a, T>) -> Result<(), Error>;

    /// Overwrite these data with a single value.
    fn write_value(&mut self, value: T) -> Result<(), Error>;

    /// Overwrite a single value.
    fn write_value_at(&mut self, offset: usize, value: T) -> Result<(), Error>;
}

/// A struct which provides n-dimensional access to an underlying [`BufferInstance`]
pub struct AccessBuf<B> {
    buffer: B,
}

impl<B: Clone> Clone for AccessBuf<B> {
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer.clone(),
        }
    }
}

impl<B> AccessBuf<B> {
    /// Construct an [`AccessBuf`] from a mutable reference to this buffer.
    pub fn as_mut<RB: ?Sized>(&mut self) -> AccessBuf<&mut RB>
    where
        B: BorrowMut<RB>,
    {
        AccessBuf {
            buffer: self.buffer.borrow_mut(),
        }
    }

    /// Construct an [`AccessBuf`] from a reference to this buffer.
    pub fn as_ref<RB: ?Sized>(&self) -> AccessBuf<&RB>
    where
        B: Borrow<RB>,
    {
        AccessBuf {
            buffer: self.buffer.borrow(),
        }
    }

    /// Borrow the underlying [`BufferInstance`] of this [`AccessBuf`].
    pub fn inner(&self) -> &B {
        &self.buffer
    }

    /// Borrow the underlying [`BufferInstance`] of this [`AccessBuf`] mutably.
    pub fn inner_mut(&mut self) -> &mut B {
        &mut self.buffer
    }

    /// Destructure this [`AccessBuf`] into its underlying [`BufferInstance`].
    pub fn into_inner(self) -> B {
        self.buffer
    }
}

impl<B> From<B> for AccessBuf<B> {
    fn from(buffer: B) -> Self {
        Self { buffer }
    }
}

impl<T, B> Access<T> for AccessBuf<B>
where
    T: CType,
    B: BufferInstance<T>,
{
    fn read(&self) -> Result<BufferConverter<T>, Error> {
        Ok(self.buffer.read())
    }

    fn read_value(&self, offset: usize) -> Result<T, Error> {
        self.buffer.read_value(offset)
    }

    fn size(&self) -> usize {
        self.buffer.len()
    }
}

impl<T, B> AccessMut<T> for AccessBuf<B>
where
    T: CType,
    B: BufferMut<T>,
{
    #[cfg(feature = "opencl")]
    fn cl_buffer(&mut self) -> Result<&mut ocl::Buffer<T>, Error> {
        self.buffer.cl()
    }

    fn write<'a>(&mut self, data: BufferConverter<'a, T>) -> Result<(), Error> {
        self.buffer.write(data)
    }

    fn write_value(&mut self, value: T) -> Result<(), Error> {
        self.buffer.write_value(value)
    }

    fn write_value_at(&mut self, offset: usize, value: T) -> Result<(), Error> {
        self.buffer.write_value_at(offset, value)
    }
}

impl<B: fmt::Debug> fmt::Debug for AccessBuf<B> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "access {:?}", self.buffer)
    }
}

/// A struct which provides n-dimensional access to the result of an array operation.
pub struct AccessOp<O, P> {
    op: O,
    platform: PhantomData<P>,
}

impl<O, P> AccessOp<O, P> {
    /// Convert the given [`AccessOp`] to a more general type of [`PlatformIntance`].
    pub fn wrap<FO, FP>(access: AccessOp<FO, FP>) -> Self
    where
        FO: Into<O>,
        FP: Into<P>,
    {
        Self {
            op: access.op.into(),
            platform: PhantomData,
        }
    }
}

impl<O, P> From<O> for AccessOp<O, P> {
    fn from(op: O) -> Self {
        Self {
            op,
            platform: PhantomData,
        }
    }
}

impl<O, P, T> Access<T> for AccessOp<O, P>
where
    T: CType,
    O: ReadOp<P, T>,
    P: PlatformInstance,
{
    fn read(&self) -> Result<BufferConverter<'static, T>, Error> {
        self.op.enqueue().map(|buffer| buffer.into())
    }

    fn read_value(&self, offset: usize) -> Result<T, Error> {
        self.op.read_value(offset)
    }

    fn size(&self) -> usize {
        self.op.size()
    }
}

impl<'a, O, P, T> Access<T> for &'a AccessOp<O, P>
where
    T: CType,
    O: ReadOp<P, T>,
    P: PlatformInstance,
    BufferConverter<'static, T>: From<O::Buffer>,
{
    fn read(&self) -> Result<BufferConverter<'static, T>, Error> {
        self.op.enqueue().map(BufferConverter::from)
    }

    fn read_value(&self, offset: usize) -> Result<T, Error> {
        self.op.read_value(offset)
    }

    fn size(&self) -> usize {
        self.op.size()
    }
}

impl<O, P, T> AccessMut<T> for AccessOp<O, P>
where
    T: CType,
    O: ReadOp<P, T> + Write<P, T>,
    P: PlatformInstance,
    BufferConverter<'static, T>: From<O::Buffer>,
{
    fn write<'a>(&mut self, data: BufferConverter<'a, T>) -> Result<(), Error> {
        self.op.write(data)
    }

    fn write_value(&mut self, value: T) -> Result<(), Error> {
        self.op.write_value(value)
    }

    fn write_value_at(&mut self, offset: usize, value: T) -> Result<(), Error> {
        self.op.write_value_at(offset, value)
    }
}

impl<O, P: fmt::Debug> fmt::Debug for AccessOp<O, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "access op {:?} on {:?}",
            std::any::type_name::<O>(),
            self.platform
        )
    }
}

/// A general-purpose implementor of [`Access`] used to elide recursive types.
/// Uses an [`Arc`] so that cloning does not allocate.
#[derive(Clone)]
pub enum Accessor<T: CType> {
    Buffer(Arc<dyn BufferInstance<T>>),
    Op(Arc<dyn ReadOp<Platform, T, Buffer = Buffer<T>>>),
}

impl<T: CType> Access<T> for Accessor<T> {
    fn read(&self) -> Result<BufferConverter<T>, Error> {
        match self {
            Self::Buffer(buf) => Ok(buf.read()),
            Self::Op(op) => op.enqueue().map(BufferConverter::from),
        }
    }

    fn read_value(&self, offset: usize) -> Result<T, Error> {
        match self {
            Self::Buffer(buf) => buf.read_value(offset),
            Self::Op(op) => op.read_value(offset),
        }
    }

    fn size(&self) -> usize {
        match self {
            Self::Buffer(buf) => buf.len(),
            Self::Op(op) => op.size(),
        }
    }
}

impl<T, B> From<AccessBuf<B>> for Accessor<T>
where
    T: CType,
    B: BufferInstance<T> + 'static,
{
    fn from(access: AccessBuf<B>) -> Self {
        Self::Buffer(Arc::new(access.buffer))
    }
}

impl<T, O, P> From<AccessOp<O, P>> for Accessor<T>
where
    T: CType,
    O: ReadOp<Platform, T, Buffer = Buffer<T>> + 'static,
    P: PlatformInstance + Into<Platform>,
{
    fn from(access: AccessOp<O, P>) -> Self {
        let access: AccessOp<O, Platform> = AccessOp::wrap(access);
        let op: Arc<dyn ReadOp<Platform, T, Buffer = Buffer<T>>> = Arc::new(access.op);
        Self::Op(op)
    }
}
