use std::ops::Add;

use rayon::join;
use rayon::prelude::*;
use smallvec::SmallVec;

use crate::{CType, Enqueue, Error, Op, PlatformInstance, ReadBuf};

pub const VEC_MIN_SIZE: usize = 64;

pub type StackVec<T> = SmallVec<[T; VEC_MIN_SIZE]>;

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct Stack;

impl PlatformInstance for Stack {
    fn select(_size_hint: usize) -> Self {
        Self
    }
}

trait StackBuf<T: CType> {
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

trait HeapBuf<T: CType> {
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

pub struct Dual<L, R, T> {
    left: L,
    right: R,
    zip: fn(T, T) -> T,
}

impl<L, R, T> Op for Dual<L, R, T>
where
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
    L: ReadBuf + Send + Sync,
    R: ReadBuf + Send + Sync,
    T: CType,
    L::Buffer: StackBuf<T> + Send + Sync,
    R::Buffer: StackBuf<T> + Send + Sync,
{
    type Buffer = Vec<T>;

    fn enqueue(self, _platform: Stack) -> Result<Self::Buffer, Error> {
        let (left, right) = join(|| self.left.read(), || self.right.read());

        let buf = left?
            .read()
            .zip(right?.read())
            .map(|(l, r)| (self.zip)(l, r))
            .collect();

        Ok(buf)
    }
}

impl<L, R, T> Enqueue<Heap> for Dual<L, R, T>
where
    L: ReadBuf + Send + Sync,
    R: ReadBuf + Send + Sync,
    T: CType,
    L::Buffer: HeapBuf<T> + Send + Sync,
    R::Buffer: HeapBuf<T> + Send + Sync,
{
    type Buffer = Vec<T>;

    fn enqueue(self, _platform: Heap) -> Result<Self::Buffer, Error> {
        let (left, right) = join(|| self.left.read(), || self.right.read());

        let buf = left?
            .read()
            .zip(right?.read())
            .map(|(l, r)| (self.zip)(l, r))
            .collect();

        Ok(buf)
    }
}
