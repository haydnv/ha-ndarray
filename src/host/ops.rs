use std::ops::{Add, Sub};

use rayon::join;
use rayon::prelude::*;

use crate::{BufferConverter, CType, Enqueue, Error, Host, Op, ReadBuf, StackVec};

use super::buffer::Buffer;
use super::platform::{Heap, Stack};
use super::VEC_MIN_SIZE;

pub struct Compare<L, R, T> {
    left: L,
    right: R,
    cmp: fn(T, T) -> u8,
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
            cmp: |l, r| if l == r { 1 } else { 0 },
        }
    }
}

impl<'a, L, R, T> Enqueue<Stack> for Compare<L, R, T>
where
    L: ReadBuf<'a, T>,
    R: ReadBuf<'a, T>,
    T: CType,
{
    type Buffer = StackVec<u8>;

    fn enqueue(self) -> Result<Self::Buffer, Error> {
        let (left, right) = try_join(self.left, self.right)?;
        exec_dual(self.cmp, left, right)
    }
}

impl<'a, L, R, T> Enqueue<Heap> for Compare<L, R, T>
where
    L: ReadBuf<'a, T>,
    R: ReadBuf<'a, T>,
    T: CType,
{
    type Buffer = Vec<u8>;

    fn enqueue(self) -> Result<Self::Buffer, Error> {
        let (left, right) = try_join(self.left, self.right)?;
        exec_dual_parallel(self.cmp, left, right)
    }
}

impl<'a, L, R, T> Enqueue<Host> for Compare<L, R, T>
where
    L: ReadBuf<'a, T>,
    R: ReadBuf<'a, T>,
    T: CType,
{
    type Buffer = Buffer<u8>;

    fn enqueue(self) -> Result<Self::Buffer, Error> {
        if self.size() < VEC_MIN_SIZE {
            Enqueue::<Stack>::enqueue(self).map(Buffer::Stack)
        } else {
            Enqueue::<Heap>::enqueue(self).map(Buffer::Heap)
        }
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
{
    type Buffer = StackVec<T>;

    fn enqueue(self) -> Result<Self::Buffer, Error> {
        let (left, right) = try_join(self.left, self.right)?;
        exec_dual(self.zip, left, right)
    }
}

impl<'a, L, R, T> Enqueue<Heap> for Dual<L, R, T>
where
    L: ReadBuf<'a, T>,
    R: ReadBuf<'a, T>,
    T: CType,
{
    type Buffer = Vec<T>;

    fn enqueue(self) -> Result<Self::Buffer, Error> {
        let (left, right) = try_join(self.left, self.right)?;
        exec_dual_parallel(self.zip, left, right)
    }
}

impl<'a, L, R, T> Enqueue<Host> for Dual<L, R, T>
where
    L: ReadBuf<'a, T>,
    R: ReadBuf<'a, T>,
    T: CType,
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

fn exec_dual<IT, OT>(
    zip: fn(IT, IT) -> OT,
    left: BufferConverter<IT>,
    right: BufferConverter<IT>,
) -> Result<StackVec<OT>, Error>
where
    IT: CType,
    OT: CType,
{
    let left = left.to_slice()?;
    let right = right.to_slice()?;

    let output = left
        .as_ref()
        .into_iter()
        .copied()
        .zip(right.as_ref().into_iter().copied())
        .map(|(l, r)| (zip)(l, r))
        .collect();

    Ok(output)
}

fn exec_dual_parallel<IT, OT>(
    zip: fn(IT, IT) -> OT,
    left: BufferConverter<IT>,
    right: BufferConverter<IT>,
) -> Result<Vec<OT>, Error>
where
    IT: CType,
    OT: CType,
{
    let left = left.to_slice()?;
    let right = right.to_slice()?;

    let output = left
        .as_ref()
        .into_par_iter()
        .copied()
        .zip(right.as_ref().into_par_iter().copied())
        .map(|(l, r)| (zip)(l, r))
        .collect();

    Ok(output)
}

#[inline]
fn try_join<'a, L, R, T>(
    left: L,
    right: R,
) -> Result<(BufferConverter<'a, T>, BufferConverter<'a, T>), Error>
where
    L: ReadBuf<'a, T>,
    R: ReadBuf<'a, T>,
    T: CType,
{
    let (l, r) = join(|| left.read(), || right.read());

    Ok((l?, r?))
}
