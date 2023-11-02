use smallvec::SmallVec;

use crate::{BufferInstance, CType, Enqueue, Error, NDArrayRead, Op, PlatformInstance};

type StackBuf<T> = SmallVec<[T; 64]>;

impl<T> BufferInstance for StackBuf<T> {
    fn size(&self) -> usize {
        self.len()
    }
}

pub struct Stack;

impl PlatformInstance for Stack {}

impl<T> BufferInstance for Vec<T> {
    fn size(&self) -> usize {
        self.len()
    }
}

pub struct Heap;

impl PlatformInstance for Heap {}

pub enum Host {
    Stack(Stack),
    Heal(Heap),
}

struct Dual<L, R, T> {
    left: L,
    right: R,
    zip: fn(T, T) -> T,
}

impl<L, R, T: CType> Op for Dual<L, R, T> {
    type DType = T;
}

impl<L, R, T> Enqueue<Stack> for Dual<L, R, T>
where
    L: NDArrayRead<StackBuf<T>>,
    T: CType,
{
    type Buffer = StackBuf<T>;

    fn enqueue(&self, platform: &Stack) -> Result<Self::Buffer, Error> {
        todo!()
    }
}

impl<L, R, T> Enqueue<Heap> for Dual<L, R, T>
where
    L: NDArrayRead<StackBuf<T>>,
    T: CType,
{
    type Buffer = StackBuf<T>;

    fn enqueue(&self, platform: &Heap) -> Result<Self::Buffer, Error> {
        todo!()
    }
}
