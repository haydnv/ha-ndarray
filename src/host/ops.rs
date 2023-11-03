use std::ops::Add;

use rayon::join;
use rayon::prelude::*;

use crate::{CType, Enqueue, Error, Op, ReadBuf};

use super::platform::{Heap, HeapBuf, Stack, StackBuf};

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
    L: ReadBuf<T> + Send + Sync,
    R: ReadBuf<T> + Send + Sync,
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
    L: ReadBuf<T> + Send + Sync,
    R: ReadBuf<T> + Send + Sync,
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
