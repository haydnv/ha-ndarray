use std::marker::PhantomData;
use std::ops::{Add, Sub};

use rayon::join;
use rayon::prelude::*;

use crate::access::Access;
use crate::array::Array;
use crate::buffer::BufferConverter;
use crate::{strides_for, CType, Enqueue, Error, Op, Shape, StackVec, Strides};

use super::buffer::Buffer;
use super::platform::{Heap, Host, Stack};
use super::VEC_MIN_SIZE;

pub struct Compare<L, R, T> {
    left: L,
    right: R,
    cmp: fn(T, T) -> u8,
}

impl<L, R, T> Op for Compare<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
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

impl<L, R, T> Enqueue<Stack> for Compare<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Buffer = StackVec<u8>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let (left, right) = try_join(&self.left, &self.right)?;
        exec_dual(self.cmp, left, right)
    }
}

impl<L, R, T> Enqueue<Heap> for Compare<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Buffer = Vec<u8>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let (left, right) = try_join(&self.left, &self.right)?;
        exec_dual_parallel(self.cmp, left, right)
    }
}

impl<L, R, T> Enqueue<Host> for Compare<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Buffer = Buffer<u8>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
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

impl<L, R, T> Op for Dual<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
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

impl<L, R, T> Enqueue<Stack> for Dual<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Buffer = StackVec<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let (left, right) = try_join(&self.left, &self.right)?;
        exec_dual(self.zip, left, right)
    }
}

impl<L, R, T> Enqueue<Heap> for Dual<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Buffer = Vec<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let (left, right) = try_join(&self.left, &self.right)?;
        exec_dual_parallel(self.zip, left, right)
    }
}

impl<L, R, T> Enqueue<Host> for Dual<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Buffer = Buffer<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        if self.size() < VEC_MIN_SIZE {
            Enqueue::<Stack>::enqueue(self).map(Buffer::from)
        } else {
            Enqueue::<Heap>::enqueue(self).map(Buffer::from)
        }
    }
}

struct ViewSpec<T> {
    shape: Shape,
    strides: Strides,
    source_strides: Strides,
    dtype: PhantomData<T>,
}

impl<T: CType> ViewSpec<T> {
    fn invert_offset(&self, offset: usize) -> usize {
        debug_assert!(offset < self.shape.iter().product::<usize>());

        self.strides
            .iter()
            .copied()
            .zip(self.shape.iter().copied())
            .map(|(stride, dim)| {
                if stride == 0 {
                    0
                } else {
                    (offset / stride) % dim
                }
            }) // coord
            .zip(self.source_strides.iter().copied())
            .map(|(i, source_stride)| i * source_stride) // source offset
            .sum::<usize>()
    }

    fn read(&self, source: BufferConverter<T>) -> Result<StackVec<T>, Error> {
        let source = source.to_slice()?;
        let source = source.as_ref();

        let buffer = (0..self.shape.iter().product())
            .into_iter()
            .map(|offset| self.invert_offset(offset))
            .map(|source_offset| source[source_offset])
            .collect();

        Ok(buffer)
    }

    fn read_parallel(&self, source: BufferConverter<T>) -> Result<Vec<T>, Error> {
        let source = source.to_slice()?;
        let source = source.as_ref();

        let buffer = (0..self.shape.iter().product())
            .into_par_iter()
            .map(|offset| self.invert_offset(offset))
            .map(|source_offset| source[source_offset])
            .collect();

        Ok(buffer)
    }
}

pub struct View<A, T> {
    access: A,
    spec: ViewSpec<T>,
}

impl<A, T> View<A, T>
where
    A: Access<T>,
    T: CType,
{
    pub fn new<P>(array: Array<T, A, P>, shape: Shape, strides: Strides) -> Self {
        let source_strides = strides_for(array.shape(), array.ndim());

        Self {
            access: array.into_inner(),
            spec: ViewSpec {
                shape,
                strides,
                source_strides,
                dtype: PhantomData,
            },
        }
    }
}

impl<A, T> Op for View<A, T>
where
    A: Access<T>,
    T: CType,
{
    fn size(&self) -> usize {
        self.spec.shape.iter().product()
    }
}

impl<A, T> Enqueue<Stack> for View<A, T>
where
    A: Access<T>,
    T: CType,
{
    type Buffer = StackVec<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        self.access.read().and_then(|source| self.spec.read(source))
    }
}

impl<A, T> Enqueue<Heap> for View<A, T>
where
    A: Access<T>,
    T: CType,
{
    type Buffer = Vec<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        self.access
            .read()
            .and_then(|source| self.spec.read_parallel(source))
    }
}

impl<A, T> Enqueue<Host> for View<A, T>
where
    A: Access<T>,
    T: CType,
{
    type Buffer = Buffer<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        if self.size() < VEC_MIN_SIZE {
            Enqueue::<Stack>::enqueue(self).map(Buffer::Stack)
        } else {
            Enqueue::<Heap>::enqueue(self).map(Buffer::Heap)
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
    left: &'a L,
    right: &'a R,
) -> Result<(BufferConverter<'a, T>, BufferConverter<'a, T>), Error>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    let (l, r) = join(|| left.read(), || right.read());

    Ok((l?, r?))
}
