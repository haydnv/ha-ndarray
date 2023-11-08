use std::borrow::Borrow;
use std::marker::PhantomData;

use crate::{BufferInstance, CType, Enqueue, Error, Op, PlatformInstance, ReadBuf};

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

impl<'a, T: CType, B: BufferInstance<T> + 'a> ReadBuf<'a, T> for AccessBuffer<B> {
    type Buffer = B;

    fn read(self) -> Result<B, Error> {
        Ok(self.buffer)
    }

    fn size(&self) -> usize {
        self.buffer.size()
    }
}

pub struct AccessOp<O, P> {
    op: O,
    platform: PhantomData<P>,
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
{
    type Buffer = O::Buffer;

    fn read(self) -> Result<O::Buffer, Error> {
        self.op.enqueue()
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
{
    type Buffer = <&'a O as Enqueue<P>>::Buffer;

    fn read(self) -> Result<<&'a O as Enqueue<P>>::Buffer, Error> {
        self.op.enqueue()
    }

    fn size(&self) -> usize {
        Op::size(&&self.op)
    }
}
