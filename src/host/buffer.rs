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

#[derive(Clone)]
/// A buffer in host memory, either borrowed or owned
pub enum SliceConverter<'a, T> {
    Heap(Vec<T>),
    Stack(StackVec<T>),
    Slice(&'a [T]),
}

impl<'a, T> SliceConverter<'a, T> {
    /// Return the number of elements in this buffer.
    pub fn size(&self) -> usize {
        match self {
            Self::Heap(vec) => vec.len(),
            Self::Stack(vec) => vec.len(),
            Self::Slice(slice) => slice.len(),
        }
    }
}

impl<'a, T: Copy> SliceConverter<'a, T> {
    /// Return this buffer as an owned [`Vec`].
    /// This will allocate a new [`Vec`] if this buffer is a [`StackVec`] or borrowed slice.
    pub fn into_vec(self) -> Vec<T> {
        match self {
            Self::Heap(vec) => vec,
            Self::Stack(vec) => vec.into_vec(),
            Self::Slice(slice) => slice.to_vec(),
        }
    }

    /// Return this buffer as an owned [`StackVec`].
    pub fn into_stackvec(self) -> StackVec<T> {
        match self {
            Self::Heap(vec) => vec.into(),
            Self::Stack(vec) => vec,
            Self::Slice(slice) => StackVec::from_slice(slice),
        }
    }

    /// Return this buffer as an owned host [`Buffer`].
    pub fn into_buffer(self) -> Buffer<T> {
        match self {
            Self::Heap(vec) => Buffer::Heap(vec),
            Self::Stack(vec) => Buffer::Stack(vec),
            Self::Slice(slice) => {
                if slice.len() < VEC_MIN_SIZE {
                    Buffer::Stack(StackVec::from_slice(slice))
                } else {
                    Buffer::Heap(slice.to_vec())
                }
            }
        }
    }
}

impl<T> From<StackVec<T>> for SliceConverter<'static, T> {
    fn from(vec: StackVec<T>) -> Self {
        Self::Stack(vec)
    }
}

impl<T> From<Vec<T>> for SliceConverter<'static, T> {
    fn from(vec: Vec<T>) -> Self {
        Self::Heap(vec)
    }
}

impl<'a, T> From<&'a [T]> for SliceConverter<'a, T> {
    fn from(slice: &'a [T]) -> Self {
        Self::Slice(slice)
    }
}

impl<'a, T> AsRef<[T]> for SliceConverter<'a, T> {
    fn as_ref(&self) -> &[T] {
        match self {
            Self::Heap(data) => data.as_slice(),
            Self::Stack(data) => data.as_slice(),
            Self::Slice(slice) => slice,
        }
    }
}
