use std::borrow::{Borrow, BorrowMut};
use std::marker::PhantomData;

use crate::buffer::{BufferConverter, BufferInstance, BufferMut};
use crate::ops::{ReadValue, Write};
use crate::platform::PlatformInstance;
use crate::{Buffer, CType, Error, Platform};

pub trait Access<T: CType>: Send + Sync {
    fn read(&self) -> Result<BufferConverter<T>, Error>;

    fn read_value(&self, offset: usize) -> Result<T, Error>;

    fn size(&self) -> usize;
}

pub trait AccessMut<T: CType>: Access<T> {
    fn write<'a>(&mut self, data: BufferConverter<'a, T>) -> Result<(), Error>;

    fn write_value(&mut self, value: T) -> Result<(), Error>;

    fn write_value_at(&mut self, offset: usize, value: T) -> Result<(), Error>;
}

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
    pub fn as_mut<RB: ?Sized>(&mut self) -> AccessBuf<&mut RB>
    where
        B: BorrowMut<RB>,
    {
        AccessBuf {
            buffer: self.buffer.borrow_mut(),
        }
    }

    pub fn as_ref<RB: ?Sized>(&self) -> AccessBuf<&RB>
    where
        B: Borrow<RB>,
    {
        AccessBuf {
            buffer: self.buffer.borrow(),
        }
    }

    pub fn inner(&self) -> &B {
        &self.buffer
    }

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
        self.buffer.size()
    }
}

impl<T, B> AccessMut<T> for AccessBuf<B>
where
    T: CType,
    B: BufferMut<T>,
{
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

pub struct AccessOp<O, P> {
    op: O,
    platform: PhantomData<P>,
}

impl<O, P> AccessOp<O, P> {
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
    O: ReadValue<P, T>,
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
    O: ReadValue<P, T>,
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
    O: ReadValue<P, T> + Write<P, T>,
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

pub enum Accessor<T: CType> {
    Buffer(Box<dyn BufferInstance<T>>),
    Op(Box<dyn ReadValue<Platform, T, Buffer = Buffer<T>>>),
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
            Self::Buffer(buf) => buf.size(),
            Self::Op(op) => op.size(),
        }
    }
}

impl<T: CType, B: BufferInstance<T> + 'static> From<AccessBuf<B>> for Accessor<T> {
    fn from(access: AccessBuf<B>) -> Self {
        Self::Buffer(Box::new(access.buffer))
    }
}

impl<T, O, P> From<AccessOp<O, P>> for Accessor<T>
where
    T: CType,
    O: ReadValue<Platform, T, Buffer = Buffer<T>> + 'static,
    P: PlatformInstance + Into<Platform>,
{
    fn from(access: AccessOp<O, P>) -> Self {
        let access: AccessOp<O, Platform> = AccessOp::wrap(access);
        let op: Box<dyn ReadValue<Platform, T, Buffer = Buffer<T>>> = Box::new(access.op);
        Self::Op(op)
    }
}
