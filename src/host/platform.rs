use rayon::prelude::*;

use crate::access::{Access, AccessOp};
use crate::buffer::BufferConverter;
use crate::ops::{ElementwiseCompare, ElementwiseDual, ElementwiseUnary, Reduce, Transform};
use crate::platform::{Convert, PlatformInstance};
use crate::{CType, Error, Range, Shape};

use super::buffer::SliceConverter;
use super::ops::*;

pub const VEC_MIN_SIZE: usize = 64;

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct Stack;

impl PlatformInstance for Stack {
    fn select(_size_hint: usize) -> Self {
        Self
    }
}

impl<A, T> Reduce<A, T> for Stack
where
    A: Access<T>,
    T: CType,
{
    fn all(self, access: A) -> Result<bool, Error> {
        access
            .read()
            .and_then(|buf| buf.to_slice())
            .map(|slice| slice.as_ref().iter().copied().all(|n| n != T::ZERO))
    }

    fn any(self, access: A) -> Result<bool, Error> {
        access
            .read()
            .and_then(|buf| buf.to_slice())
            .map(|slice| slice.as_ref().iter().copied().any(|n| n != T::ZERO))
    }

    fn sum(self, access: A) -> Result<T, Error> {
        access
            .read()
            .and_then(|buf| buf.to_slice())
            .map(|slice| slice.as_ref().iter().copied().sum())
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct Heap;

impl PlatformInstance for Heap {
    fn select(_size_hint: usize) -> Self {
        Self
    }
}

impl<A, T> Reduce<A, T> for Heap
where
    A: Access<T>,
    T: CType,
{
    fn all(self, access: A) -> Result<bool, Error> {
        access.read().and_then(|buf| buf.to_slice()).map(|slice| {
            slice
                .as_ref()
                .into_par_iter()
                .copied()
                .all(|n| n != T::ZERO)
        })
    }

    fn any(self, access: A) -> Result<bool, Error> {
        access.read().and_then(|buf| buf.to_slice()).map(|slice| {
            slice
                .as_ref()
                .into_par_iter()
                .copied()
                .any(|n| n != T::ZERO)
        })
    }

    fn sum(self, access: A) -> Result<T, Error> {
        access
            .read()
            .and_then(|buf| buf.to_slice())
            .map(|slice| slice.as_ref().into_par_iter().copied().sum())
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

impl<'a, T: CType> Convert<'a, T> for Host {
    type Buffer = SliceConverter<'a, T>;

    fn convert(&self, buffer: BufferConverter<'a, T>) -> Result<Self::Buffer, Error> {
        buffer.to_slice()
    }
}

impl From<Heap> for Host {
    fn from(heap: Heap) -> Self {
        Self::Heap(heap)
    }
}

impl From<Stack> for Host {
    fn from(stack: Stack) -> Self {
        Self::Stack(stack)
    }
}

impl<'a, L, R, T> ElementwiseCompare<L, R, T> for Host
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Op = Compare<L, R, T>;

    fn eq(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Compare::eq(left, right).into())
    }
}

impl<L, R, T> ElementwiseDual<L, R, T> for Host
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Op = Dual<L, R, T>;

    fn add(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Dual::add(left, right).into())
    }

    fn sub(self, left: L, right: R) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Dual::sub(left, right).into())
    }
}

impl<A: Access<T>, T: CType> ElementwiseUnary<A, T> for Host {
    type Op = Unary<A, T, T>;

    fn ln(self, access: A) -> Result<AccessOp<Self::Op, Self>, Error> {
        Ok(Unary::ln(access).into())
    }
}

impl<A: Access<T>, T: CType> Reduce<A, T> for Host {
    fn all(self, access: A) -> Result<bool, Error> {
        match self {
            Self::Heap(heap) => heap.all(access),
            Self::Stack(stack) => stack.all(access),
        }
    }

    fn any(self, access: A) -> Result<bool, Error> {
        match self {
            Self::Heap(heap) => heap.all(access),
            Self::Stack(stack) => stack.all(access),
        }
    }

    fn sum(self, access: A) -> Result<T, Error> {
        match self {
            Self::Heap(heap) => heap.sum(access),
            Self::Stack(stack) => stack.sum(access),
        }
    }
}

impl<'a, A, T> Transform<A, T> for Host
where
    A: Access<T>,
    T: CType,
{
    type Broadcast = View<A, T>;
    type Slice = Slice<A, T>;

    fn broadcast(
        self,
        array: A,
        shape: Shape,
        broadcast: Shape,
    ) -> Result<AccessOp<Self::Broadcast, Self>, Error> {
        Ok(View::new(array, shape, broadcast).into())
    }

    fn slice(
        self,
        access: A,
        shape: &[usize],
        range: Range,
    ) -> Result<AccessOp<Self::Slice, Self>, Error> {
        Ok(Slice::new(access, shape, range).into())
    }
}
