use crate::{BufferInstance, CType, PlatformInstance, StackVec};

pub const VEC_MIN_SIZE: usize = 64;

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct Stack;

impl PlatformInstance for Stack {
    fn select(_size_hint: usize) -> Self {
        Self
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct Heap;

impl PlatformInstance for Heap {
    fn select(_size_hint: usize) -> Self {
        Self
    }
}

pub enum Buffer<T> {
    Heap(Vec<T>),
    Stack(StackVec<T>),
}

impl<T: CType> BufferInstance<T> for Buffer<T> {
    fn size(&self) -> usize {
        match self {
            Self::Heap(buf) => buf.size(),
            Self::Stack(buf) => buf.size(),
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
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
