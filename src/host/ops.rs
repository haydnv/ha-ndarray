use std::ops::{Add, Sub};

use rayon::join;
use rayon::prelude::*;

use crate::{CType, Enqueue, Error, Host, Op, ReadBuf};

use super::platform::{Heap, Stack};
use super::{Buffer, SliceConverter, VEC_MIN_SIZE};

pub struct Compare<L, R, T> {
    left: L,
    right: R,
    zip: fn(T, T) -> u8,
}

impl<'a, L, R, T> Op for Compare<L, R, T>
where
    L: ReadBuf<'a, T>,
    R: ReadBuf<'a, T>,
    T: CType,
{
    type DType = u8;

    fn size(&self) -> usize {
        self.left.size()
    }
}

impl<L, R, T: CType> Compare<L, R, T> {
    pub fn eq(left: L, right: R) -> Self {
        Self {
            left,
            right,
            zip: |l, r| if l == r { 1 } else { 0 },
        }
    }
}

impl<'a, L, R, T> Enqueue<Stack> for Compare<L, R, T>
where
    L: ReadBuf<'a, T>,
    R: ReadBuf<'a, T>,
    T: CType,
    L::Buffer: Into<SliceConverter<'a, T>>,
    R::Buffer: Into<SliceConverter<'a, T>>,
{
    type Buffer = Vec<u8>;

    fn enqueue(self) -> Result<Self::Buffer, Error> {
        let (left, right) = join(|| self.left.read(), || self.right.read());

        let buf = left?
            .into()
            .as_ref()
            .iter()
            .copied()
            .zip(right?.into().as_ref().iter().copied())
            .map(|(l, r)| (self.zip)(l, r))
            .collect();

        Ok(buf)
    }
}

impl<'a, L, R, T> Enqueue<Heap> for Compare<L, R, T>
where
    L: ReadBuf<'a, T>,
    R: ReadBuf<'a, T>,
    T: CType,
    L::Buffer: Into<SliceConverter<'a, T>>,
    R::Buffer: Into<SliceConverter<'a, T>>,
{
    type Buffer = Vec<u8>;

    fn enqueue(self) -> Result<Self::Buffer, Error> {
        let (left, right) = join(|| self.left.read(), || self.right.read());

        let buf = left?
            .into()
            .as_ref()
            .into_par_iter()
            .copied()
            .zip(right?.into().as_ref().into_par_iter().copied())
            .map(|(l, r)| (self.zip)(l, r))
            .collect();

        Ok(buf)
    }
}

pub struct Dual<L, R, T> {
    left: L,
    right: R,
    zip: fn(T, T) -> T,
}

impl<'a, L, R, T> Op for Dual<L, R, T>
where
    L: ReadBuf<'a, T>,
    R: ReadBuf<'a, T>,
    T: CType,
{
    type DType = T;

    fn size(&self) -> usize {
        self.left.size()
    }
}

impl<L, R, T: CType> Dual<L, R, T> {
    pub fn add(left: L, right: R) -> Self {
        Self {
            left,
            right,
            zip: Add::add,
        }
    }

    pub fn sub(left: L, right: R) -> Self {
        Self {
            left,
            right,
            zip: Sub::sub,
        }
    }
}

impl<'a, L, R, T> Enqueue<Stack> for Dual<L, R, T>
where
    L: ReadBuf<'a, T>,
    R: ReadBuf<'a, T>,
    T: CType,
    L::Buffer: Into<SliceConverter<'a, T>>,
    R::Buffer: Into<SliceConverter<'a, T>>,
{
    type Buffer = Vec<T>;

    fn enqueue(self) -> Result<Self::Buffer, Error> {
        let (left, right) = join(|| self.left.read(), || self.right.read());

        let buf = left?
            .into()
            .as_ref()
            .iter()
            .copied()
            .zip(right?.into().as_ref().iter().copied())
            .map(|(l, r)| (self.zip)(l, r))
            .collect();

        Ok(buf)
    }
}

impl<'a, L, R, T> Enqueue<Heap> for Dual<L, R, T>
where
    L: ReadBuf<'a, T>,
    R: ReadBuf<'a, T>,
    T: CType,
    L::Buffer: Into<SliceConverter<'a, T>>,
    R::Buffer: Into<SliceConverter<'a, T>>,
{
    type Buffer = Vec<T>;

    fn enqueue(self) -> Result<Self::Buffer, Error> {
        let (left, right) = join(|| self.left.read(), || self.right.read());

        let buf = left?
            .into()
            .as_ref()
            .into_par_iter()
            .copied()
            .zip(right?.into().as_ref().into_par_iter().copied())
            .map(|(l, r)| (self.zip)(l, r))
            .collect();

        Ok(buf)
    }
}

impl<'a, L, R, T> Enqueue<Host> for Dual<L, R, T>
where
    L: ReadBuf<'a, T>,
    R: ReadBuf<'a, T>,
    T: CType,
    L::Buffer: Into<SliceConverter<'a, T>>,
    R::Buffer: Into<SliceConverter<'a, T>>,
{
    type Buffer = Buffer<T>;

    fn enqueue(self) -> Result<Self::Buffer, Error> {
        if self.size() < VEC_MIN_SIZE {
            Enqueue::<Stack>::enqueue(self).map(Buffer::from)
        } else {
            Enqueue::<Heap>::enqueue(self).map(Buffer::from)
        }
    }
}
