use rayon::prelude::*;

use crate::access::{Access, AccessOp};
use crate::array::Array;
use crate::ops::{ElementwiseCompare, ElementwiseDual, Reduce, Transform};
use crate::platform::PlatformInstance;
use crate::{strides_for, CType, Error, Shape};

use super::ops::*;

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

impl<'a, L, R, T> ElementwiseCompare<L, R, T> for Host
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Output = Compare<L, R, T>;

    fn eq(self, left: L, right: R) -> Result<AccessOp<Self::Output, Self>, Error> {
        Ok(Compare::eq(left, right).into())
    }
}

impl<'a, L, R, T> ElementwiseDual<L, R, T> for Host
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Output = Dual<L, R, T>;

    fn add(self, left: L, right: R) -> Result<AccessOp<Self::Output, Self>, Error> {
        Ok(Dual::add(left, right).into())
    }

    fn sub(self, left: L, right: R) -> Result<AccessOp<Self::Output, Self>, Error> {
        Ok(Dual::sub(left, right).into())
    }
}

impl<'a, A, T> Reduce<A, T> for Host
where
    A: Access<T>,
    T: CType,
{
    fn all(self, access: A) -> Result<bool, Error> {
        access.read().and_then(|buf| buf.to_slice()).map(|slice| {
            if slice.size() < VEC_MIN_SIZE {
                slice.as_ref().into_iter().copied().all(|n| n != T::ZERO)
            } else {
                slice
                    .as_ref()
                    .into_par_iter()
                    .copied()
                    .all(|n| n != T::ZERO)
            }
        })
    }

    fn any(self, access: A) -> Result<bool, Error> {
        access.read().and_then(|buf| buf.to_slice()).map(|slice| {
            if slice.size() < VEC_MIN_SIZE {
                slice.as_ref().into_iter().copied().any(|n| n != T::ZERO)
            } else {
                slice
                    .as_ref()
                    .into_par_iter()
                    .copied()
                    .any(|n| n != T::ZERO)
            }
        })
    }
}

impl<'a, A, T> Transform<A, T> for Host
where
    A: Access<T>,
    T: CType,
{
    type Broadcast = View<A, T>;

    fn broadcast(
        self,
        array: Array<T, A, Self>,
        shape: &[usize],
    ) -> Result<AccessOp<Self::Broadcast, Self>, Error> {
        let strides = strides_for(array.shape(), shape.len());
        Ok(View::new(array, Shape::from_slice(shape), strides).into())
    }
}
