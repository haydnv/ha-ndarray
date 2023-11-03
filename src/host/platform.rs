use rayon::prelude::*;

use crate::{CType, PlatformInstance};

use super::StackVec;

pub const VEC_MIN_SIZE: usize = 64;

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct Stack;

impl PlatformInstance for Stack {
    fn select(_size_hint: usize) -> Self {
        Self
    }
}

pub trait StackBuf<T: CType> {
    type Iter: Iterator<Item = T>;

    fn read(self) -> Self::Iter;
}

impl<T: CType> StackBuf<T> for StackVec<T> {
    type Iter = <StackVec<T> as IntoIterator>::IntoIter;

    fn read(self) -> Self::Iter {
        self.into_iter()
    }
}

impl<'a, T: CType> StackBuf<T> for &'a [T] {
    type Iter = std::iter::Copied<<&'a [T] as IntoIterator>::IntoIter>;

    fn read(self) -> Self::Iter {
        self.into_iter().copied()
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct Heap;

impl PlatformInstance for Heap {
    fn select(_size_hint: usize) -> Self {
        Self
    }
}

pub trait HeapBuf<T: CType> {
    type Iter: IndexedParallelIterator<Item = T>;

    fn read(self) -> Self::Iter;
}

impl<T: CType> HeapBuf<T> for Vec<T> {
    type Iter = <Vec<T> as IntoParallelIterator>::Iter;

    fn read(self) -> Self::Iter {
        self.into_par_iter()
    }
}

impl<'a, T: CType> HeapBuf<T> for &'a [T] {
    type Iter = rayon::iter::Copied<<&'a [T] as IntoParallelIterator>::Iter>;

    fn read(self) -> Self::Iter {
        self.into_par_iter().copied()
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
