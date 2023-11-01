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

impl<L, R, T> Enqueue<Dual<L, R, T>> for Stack
where
    L: NDArrayRead,
    T: CType,
{
    type Buffer = StackBuf<T>;

    fn enqueue(&self, op: Dual<L, R, T>) -> Result<Self::Buffer, Error> {
        todo!()
    }
}

impl<L, R, T> Enqueue<Dual<L, R, T>> for Heap
where
    L: NDArrayRead,
    T: CType,
{
    type Buffer = StackBuf<T>;

    fn enqueue(&self, op: Dual<L, R, T>) -> Result<Self::Buffer, Error> {
        todo!()
    }
}
