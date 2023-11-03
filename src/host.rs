use std::ops::Add;

use rayon::join;
use smallvec::SmallVec;

use crate::{CType, Enqueue, Error, Op, PlatformInstance, ReadBuf};

const VEC_MIN_SIZE: usize = 64;

pub type StackBuf<T> = SmallVec<[T; VEC_MIN_SIZE]>;

pub struct Stack;

impl PlatformInstance for Stack {
    fn select(_size_hint: usize) -> Self {
        Self
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

pub struct Dual<L, R, T> {
    left: L,
    right: R,
    zip: fn(T, T) -> T,
}

impl<L, R, T> Op for Dual<L, R, T>
where
    L: Send + Sync,
    R: Send + Sync,
    T: CType,
{
    type DType = T;
}

impl<L, R, T: CType> Dual<L, R, T> {
    pub fn add(left: L, right: R) -> Self {
        Self {
            left,
            right,
            zip: Add::add,
        }
    }
}

impl<L, R, T> Enqueue<Stack> for Dual<L, R, T>
where
    L: ReadBuf<StackBuf<T>>,
    R: ReadBuf<StackBuf<T>>,
    T: CType,
{
    type Buffer = StackBuf<T>;

    fn enqueue(self, _platform: &Stack) -> Result<Self::Buffer, Error> {
        let (left, right) = join(|| self.left.read(), || self.right.read());

        let buf = left?
            .into_iter()
            .zip(right?.into_iter())
            .map(|(l, r)| (self.zip)(l, r))
            .collect();

        Ok(buf)
    }
}

impl<L, R, T> Enqueue<Heap> for Dual<L, R, T>
where
    L: ReadBuf<Vec<T>>,
    R: ReadBuf<Vec<T>>,
    T: CType,
{
    type Buffer = StackBuf<T>;

    fn enqueue(self, platform: &Heap) -> Result<Self::Buffer, Error> {
        todo!()
    }
}
