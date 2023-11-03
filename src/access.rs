use crate::{BufferInstance, Enqueue, Error, PlatformInstance, ReadBuf};

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

impl<B: BufferInstance> ReadBuf for AccessBuffer<B> {
    type Buffer = B;

    fn read(self) -> Result<B, Error> {
        Ok(self.buffer)
    }
}

pub struct AccessOp<O, P> {
    op: O,
    platform: P,
}

impl<O, P> ReadBuf for AccessOp<O, P>
where
    O: Enqueue<P>,
    P: PlatformInstance,
{
    type Buffer = O::Buffer;

    fn read(self) -> Result<O::Buffer, Error> {
        self.op.enqueue(self.platform)
    }
}
