use std::borrow::Borrow;

use smallvec::SmallVec;

use crate::{BufferInstance, CType};

use super::VEC_MIN_SIZE;

pub type StackVec<T> = SmallVec<[T; VEC_MIN_SIZE]>;

impl<T: CType> BufferInstance<T> for StackVec<T> {
    fn size(&self) -> usize {
        self.len()
    }
}

impl<T: CType> BufferInstance<T> for Vec<T> {
    fn size(&self) -> usize {
        self.len()
    }
}

impl<'a, T: CType> BufferInstance<T> for &'a [T] {
    fn size(&self) -> usize {
        self.len()
    }
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub enum Buffer<T> {
    Heap(Vec<T>),
    Stack(StackVec<T>),
}

impl<T> Borrow<[T]> for Buffer<T> {
    fn borrow(&self) -> &[T] {
        match self {
            Self::Heap(buf) => buf.borrow(),
            Self::Stack(buf) => buf.borrow(),
        }
    }
}

impl<T: CType> BufferInstance<T> for Buffer<T> {
    fn size(&self) -> usize {
        match self {
            Self::Heap(buf) => buf.size(),
            Self::Stack(buf) => buf.size(),
        }
    }
}

impl<T> From<StackVec<T>> for Buffer<T> {
    fn from(buf: StackVec<T>) -> Self {
        Self::Stack(buf)
    }
}

impl<T> From<Vec<T>> for Buffer<T> {
    fn from(buf: Vec<T>) -> Self {
        Self::Heap(buf)
    }
}
