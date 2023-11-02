use smallvec::SmallVec;

use crate::{BufferInstance, CType, Enqueue, Error, NDArrayRead, Op, PlatformInstance};

const VEC_MIN_SIZE: usize = 64;

type StackBuf<T> = SmallVec<[T; VEC_MIN_SIZE]>;

impl<T> BufferInstance for StackBuf<T> {
    fn size(&self) -> usize {
        self.len()
    }
}

pub struct Stack;

impl PlatformInstance for Stack {
    fn select(_size_hint: usize) -> Self {
        Self
    }
}

impl<T> BufferInstance for Vec<T> {
    fn size(&self) -> usize {
        self.len()
    }
}

pub struct Heap;

impl PlatformInstance for Heap {
    fn select(_size_hint: usize) -> Self {
        Self
    }
}

pub enum Host {
    Stack(Stack),
    Heap(Heap),
}

impl PlatformInstance for Host {
    fn select(size_hint: usize) -> Self {
        if size_hint < VEC_MIN_SIZE {
            Self::Stack(Stack)
        } else {
            Self::Heap(Heap)
        }
    }
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
