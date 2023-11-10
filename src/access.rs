use std::borrow::Borrow;
use std::marker::PhantomData;

use crate::buffer::{BufferConverter, BufferInstance};
use crate::ops::Enqueue;
use crate::platform::PlatformInstance;
use crate::{Buffer, CType, Error, Platform};

pub trait Access<T: CType>: Send + Sync {
    fn read(&self) -> Result<BufferConverter<T>, Error>;

    fn size(&self) -> usize;
}

pub struct AccessBuffer<B> {
    buffer: B,
}

impl<B> AccessBuffer<B> {
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
