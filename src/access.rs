use std::borrow::{Borrow, BorrowMut};
use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::buffer::{BufferConverter, BufferInstance, BufferMut};
use crate::ops::{ReadOp, Write};
use crate::platform::PlatformInstance;
use crate::{Buffer, Error, Number, Platform};

/// A type which allows accessing array data
pub trait Access<T: Number>: Send + Sync {
    /// Read the data of this accessor as a [`BufferConverter`].
    fn read(&self) -> Result<BufferConverter<T>, Error>;

    /// Access a single value.
    fn read_value(&self, offset: usize) -> Result<T, Error>;

    /// Return the data size.
    fn size(&self) -> usize;
}

/// A type which allows accessing array data mutably
pub trait AccessMut<T: Number>: Access<T> + fmt::Debug {
    #[cfg(feature = "opencl")]
    /// Borrow the array data as an [`ocl::Buffer`], or return an error if this not an OpenCL buffer.
    fn cl_buffer(&mut self) -> Result<&mut ocl::Buffer<T>, Error> {
        Err(Error::unsupported(format!(
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

/// Borrow an accessor immutably
pub trait AccessBorrow<'a, T, B>: Access<T>
where
    T: Number,
    B: Access<T> + 'a,
{
    fn borrow(&'a self) -> B;
}

/// Borrow an accessor mutably
pub trait AccessBorrowMut<'a, T, B>: Access<T>
where
    T: Number,
    B: AccessMut<T> + 'a,
{
    fn borrow_mut(&'a mut self) -> B;
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

impl<'a, T, B, RB> AccessBorrow<'a, T, AccessBuf<&'a RB>> for AccessBuf<B>
where
    T: Number,
    B: BufferInstance<T> + Borrow<RB>,
    &'a RB: BufferInstance<T>,
{
    fn borrow(&'a self) -> AccessBuf<&'a RB> {
        AccessBuf {
            buffer: self.buffer.borrow(),
        }
    }
}

impl<'a, T, B, RB> AccessBorrowMut<'a, T, AccessBuf<&'a mut RB>> for AccessBuf<B>
where
    T: Number,
    B: BufferInstance<T> + BorrowMut<RB>,
    &'a mut RB: BufferMut<T>,
{
    fn borrow_mut(&'a mut self) -> AccessBuf<&'a mut RB> {
        AccessBuf {
            buffer: self.buffer.borrow_mut(),
        }
    }
}

impl<B> From<B> for AccessBuf<B> {
    fn from(buffer: B) -> Self {
        Self { buffer }
    }
}

impl<T, B> Access<T> for AccessBuf<B>
where
    T: Number,
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
    T: Number,
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

impl<'a, T, O, P> AccessBorrow<'a, T, &'a Self> for AccessOp<O, P>
where
    T: Number,
    O: ReadOp<P, T>,
    P: PlatformInstance,
    Self: Access<T>,
    &'a Self: Access<T>,
{
    fn borrow(&'a self) -> &'a Self {
        self
    }
}

impl<'a, T, O, P> AccessBorrowMut<'a, T, &'a mut Self> for AccessOp<O, P>
where
    T: Number,
    O: ReadOp<P, T> + Write<P, T>,
    P: PlatformInstance,
    Self: AccessMut<T>,
    &'a mut Self: AccessMut<T>,
{
    fn borrow_mut(&'a mut self) -> &'a mut Self {
        self
    }
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
    T: Number,
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
    T: Number,
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
    T: Number,
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
pub enum Accessor<'a, T: Number> {
    Buffer(Arc<dyn BufferInstance<T> + 'a>),
    Op(Arc<dyn ReadOp<Platform, T, Buffer = Buffer<T>> + 'a>),
}

impl<'a, T: Number> Access<T> for Accessor<'a, T> {
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

impl<'a, T, B> From<AccessBuf<B>> for Accessor<'a, T>
where
    T: Number,
    B: BufferInstance<T> + 'a,
{
    fn from(access: AccessBuf<B>) -> Self {
        Self::Buffer(Arc::new(access.buffer))
    }
}

impl<'a, T, O, P> From<AccessOp<O, P>> for Accessor<'a, T>
where
    T: Number,
    O: ReadOp<Platform, T, Buffer = Buffer<T>> + 'a,
    P: PlatformInstance + Into<Platform>,
{
    fn from(access: AccessOp<O, P>) -> Self {
        let access: AccessOp<O, Platform> = AccessOp::wrap(access);
        let op: Arc<dyn ReadOp<Platform, T, Buffer = Buffer<T>>> = Arc::new(access.op);
        Self::Op(op)
    }
}
