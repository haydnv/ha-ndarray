use crate::{BufferInstance, CType, Enqueue, Error, PlatformInstance, ReadBuf};

pub struct AccessBuffer<B> {
    buffer: B,
}

impl<B> AccessBuffer<B> {
    pub fn as_ref<RB>(&self) -> AccessBuffer<&RB>
    where
        B: AsRef<RB>,
    {
        AccessBuffer {
            buffer: self.buffer.as_ref(),
        }
    }
}

impl<B> From<B> for AccessBuffer<B> {
    fn from(buffer: B) -> Self {
        Self { buffer }
    }
}

impl<T: CType, B: BufferInstance<T>> ReadBuf<T> for AccessBuffer<B> {
    type Buffer = B;

    fn read(self) -> Result<B, Error> {
        Ok(self.buffer)
    }
}

pub struct AccessOp<O, P> {
    op: O,
    platform: P,
}

impl<O, P> AccessOp<O, P> {
    pub fn new(op: O, platform: P) -> Self {
        Self { op, platform }
    }
}

impl<T, O, P> ReadBuf<T> for AccessOp<O, P>
where
    T: CType,
    O: Enqueue<P, DType = T>,
    P: PlatformInstance,
{
    type Buffer = O::Buffer;

    fn read(self) -> Result<O::Buffer, Error> {
        self.op.enqueue(self.platform)
    }
}

impl<'a, T, O, P> ReadBuf<T> for &'a AccessOp<O, P>
where
    T: CType,
    &'a O: Enqueue<P, DType = T>,
    P: PlatformInstance,
{
    type Buffer = <&'a O as Enqueue<P>>::Buffer;

    fn read(self) -> Result<<&'a O as Enqueue<P>>::Buffer, Error> {
        self.op.enqueue(self.platform)
    }
}
