use std::borrow::{Borrow, BorrowMut};
use std::marker::PhantomData;

use crate::buffer::{BufferConverter, BufferInstance};
use crate::ops::{Enqueue, Write};
use crate::platform::PlatformInstance;
use crate::{Buffer, CType, Error, Platform};

pub trait Access<T: CType>: Send + Sync {
    fn read(&self) -> Result<BufferConverter<T>, Error>;

    fn size(&self) -> usize;
}

pub trait AccessMut<'a, T: CType>: Access<T> {
    type Data;

    fn write(&'a mut self, data: Self::Data) -> Result<(), Error>;
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
        self.buffer.read()
    }

    fn size(&self) -> usize {
        self.buffer.size()
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

impl<'a, O, P, T> Access<T> for AccessOp<O, P>
where
    T: CType,
    O: Enqueue<P>,
    P: PlatformInstance,
    BufferConverter<'static, T>: From<O::Buffer>,
{
    fn read(&self) -> Result<BufferConverter<'static, T>, Error> {
        self.op.enqueue().map(BufferConverter::from)
    }

    fn size(&self) -> usize {
        self.op.size()
    }
}

pub struct AccessSlice<A, O, P> {
    source: A,
    read: O,
    platform: PhantomData<P>,
}

impl<A, O, P, T> Access<T> for AccessSlice<A, O, P>
where
    T: CType,
    A: Access<T>,
    O: Enqueue<P>,
    P: PlatformInstance,
    BufferConverter<'static, T>: From<O::Buffer>,
{
    fn read(&self) -> Result<BufferConverter<T>, Error> {
        self.read.enqueue().map(BufferConverter::from)
    }

    fn size(&self) -> usize {
        self.read.size()
    }
}

impl<'a, A, O, P, T> AccessMut<'a, T> for AccessSlice<A, O, P>
where
    T: CType,
    A: Access<T>,
    O: Write<'a, P>,
    P: PlatformInstance,
    BufferConverter<'static, T>: From<O::Buffer>,
{
    type Data = O::Data;

    fn write(&'a mut self, data: Self::Data) -> Result<(), Error> {
        todo!()
    }
}

pub enum Accessor<T: CType> {
    Buffer(Buffer<T>),
    Op(Box<dyn Enqueue<Platform, Buffer = Buffer<T>>>),
}

impl<T: CType> Access<T> for Accessor<T> {
    fn read(&self) -> Result<BufferConverter<T>, Error> {
        match self {
            Self::Buffer(buf) => buf.read(),
            Self::Op(op) => op.enqueue().map(BufferConverter::from),
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

impl<T, O> From<AccessOp<O, Platform>> for Accessor<T>
where
    T: CType,
    O: Enqueue<Platform, Buffer = Buffer<T>> + Sized + 'static,
{
    fn from(access: AccessOp<O, Platform>) -> Self {
        let op: Box<dyn Enqueue<Platform, Buffer = Buffer<T>>> = Box::new(access.op);
        Self::Op(op)
    }
}
