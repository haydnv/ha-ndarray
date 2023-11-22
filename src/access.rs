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

pub trait AccessMut<'a, T: CType>: Access<T> {
    type Data;

    fn write(&'a mut self, data: Self::Data) -> Result<(), Error>;

    fn write_value(&'a mut self, value: T) -> Result<(), Error>;

    fn write_value_at(&'a mut self, offset: usize, value: T) -> Result<(), Error>;
}

pub struct AccessBuffer<B> {
    buffer: B,
}

impl<B> AccessBuffer<B> {
    pub fn as_mut<RB: ?Sized>(&mut self) -> AccessBuffer<&mut RB>
    where
        B: BorrowMut<RB>,
    {
        AccessBuffer {
            buffer: self.buffer.borrow_mut(),
        }
    }

    pub fn as_ref<RB: ?Sized>(&self) -> AccessBuffer<&RB>
    where
        B: Borrow<RB>,
    {
        AccessBuffer {
            buffer: self.buffer.borrow(),
        }
    }

    pub fn into_inner(self) -> B {
        self.buffer
    }
}

impl<B> From<B> for AccessBuffer<B> {
    fn from(buffer: B) -> Self {
        Self { buffer }
    }
}

impl<T, B> Access<T> for AccessBuffer<B>
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

impl<'a, T, B> AccessMut<'a, T> for AccessBuffer<B>
where
    T: CType,
    B: BufferMut<'a, T>,
{
    type Data = B::Data;

    fn write(&'a mut self, data: Self::Data) -> Result<(), Error> {
        self.buffer.write(data)
    }

    fn write_value(&'a mut self, value: T) -> Result<(), Error> {
        self.buffer.write_value(value)
    }

    fn write_value_at(&'a mut self, offset: usize, value: T) -> Result<(), Error> {
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
        O: From<FO>,
        P: From<FP>,
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

impl<'a, O, P, T> AccessMut<'a, T> for AccessOp<O, P>
where
    T: CType,
    O: ReadValue<P, T> + Write<'a, P, T>,
    P: PlatformInstance,
    BufferConverter<'static, T>: From<O::Buffer>,
{
    type Data = O::Data;

    fn write(&'a mut self, data: Self::Data) -> Result<(), Error> {
        self.op.write(data)
    }

    fn write_value(&'a mut self, value: T) -> Result<(), Error> {
        self.op.write_value(value)
    }

    fn write_value_at(&'a mut self, offset: usize, value: T) -> Result<(), Error> {
        self.op.write_value_at(offset, value)
    }
}

pub enum Accessor<T: CType> {
    Buffer(Buffer<T>),
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

impl<T: CType, B: Into<Buffer<T>>> From<AccessBuffer<B>> for Accessor<T> {
    fn from(access: AccessBuffer<B>) -> Self {
        Self::Buffer(access.buffer.into())
    }
}

impl<T, O, P> From<AccessOp<O, P>> for Accessor<T>
where
    T: CType,
    O: ReadValue<Platform, T, Buffer = Buffer<T>> + Sized + 'static,
    Platform: From<P>,
{
    fn from(access: AccessOp<O, P>) -> Self {
        let access: AccessOp<O, Platform> = AccessOp::wrap(access);
        let op: Box<dyn ReadValue<Platform, T, Buffer = Buffer<T>>> = Box::new(access.op);
        Self::Op(op)
    }
}
