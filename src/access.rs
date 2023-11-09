use std::borrow::Borrow;
use std::marker::PhantomData;

use crate::buffer::{BufferConverter, BufferInstance};
use crate::ops::{Enqueue, Op};
use crate::platform::PlatformInstance;
use crate::{CType, Error, ReadBuf};

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

impl<'a, T, B> ReadBuf<'a, T> for AccessBuffer<B>
where
    T: CType,
    B: BufferInstance<T> + Into<BufferConverter<'a, T>> + 'a,
{
    fn read(self) -> Result<BufferConverter<'a, T>, Error> {
        Ok(self.buffer.into())
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

impl<'a, O, P, T> ReadBuf<'a, T> for AccessOp<O, P>
where
    T: CType,
    O: Enqueue<P, DType = T>,
    P: PlatformInstance,
    O::Buffer: 'a,
    BufferConverter<'a, T>: From<O::Buffer>,
{
    fn read(self) -> Result<BufferConverter<'a, T>, Error> {
        self.op.enqueue().map(BufferConverter::from)
    }

    fn size(&self) -> usize {
        self.op.size()
    }
}

impl<'a, 'b, T, O, P> ReadBuf<'b, T> for &'a AccessOp<O, P>
where
    T: CType,
    O: Send + Sync,
    &'a O: Enqueue<P, DType = T>,
    P: PlatformInstance,
    <&'a O as Enqueue<P>>::Buffer: 'b,
    BufferConverter<'b, T>: From<<&'a O as Enqueue<P>>::Buffer>,
{
    fn read(self) -> Result<BufferConverter<'b, T>, Error> {
        self.op.enqueue().map(BufferConverter::from)
    }

    fn size(&self) -> usize {
        Op::size(&&self.op)
    }
}
