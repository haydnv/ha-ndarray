use std::borrow::{Borrow, BorrowMut};
use std::fmt;
use std::marker::PhantomData;

use crate::access::*;
use crate::buffer::BufferInstance;
use crate::ops::*;
use crate::platform::PlatformInstance;
use crate::{shape, AxisRange, CType, Convert, Error, Range, Shape};

pub struct Array<T, A, P> {
    shape: Shape,
    access: A,
    platform: P,
    dtype: PhantomData<T>,
}

impl<T, A, P> Array<T, A, P> {
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn into_inner(self) -> A {
        self.access
    }
}

// constructors
impl<T, B, P> Array<T, AccessBuffer<B>, P>
where
    T: CType,
    B: BufferInstance<T>,
    P: PlatformInstance,
{
    pub fn new(buffer: B, shape: Shape) -> Result<Self, Error> {
        if shape.iter().product::<usize>() == buffer.size() {
            let platform = P::select(buffer.size());
            let access = buffer.into();

            Ok(Self {
                shape,
                access,
                platform,
                dtype: PhantomData,
            })
        } else {
            Err(Error::Bounds(format!(
                "cannot construct an array with shape {shape:?} from a buffer of size {}",
                buffer.size()
            )))
        }
    }
}

// op constructors
impl<T: CType, P: PlatformInstance> Array<T, AccessOp<P::Range, P>, P>
where
    P: Construct<T>,
{
    pub fn range(start: T, stop: T, size: usize) -> Result<Self, Error> {
        let platform = P::select(size);
        let shape = shape![size];

        platform.range(start, stop, size).map(|access| Self {
            shape,
            access,
            platform,
            dtype: PhantomData,
        })
    }
}

impl<P: PlatformInstance> Array<f32, AccessOp<P::Normal, P>, P>
where
    P: Random,
{
    pub fn random_normal(size: usize) -> Result<Self, Error> {
        let platform = P::select(size);
        let shape = shape![size];

        platform.random_normal(size).map(|access| Self {
            shape,
            access,
            platform,
            dtype: PhantomData,
        })
    }
}

impl<P: PlatformInstance> Array<f32, AccessOp<P::Uniform, P>, P>
where
    P: Random,
{
    pub fn random_uniform(size: usize) -> Result<Self, Error> {
        let platform = P::select(size);
        let shape = shape![size];

        platform.random_uniform(size).map(|access| Self {
            shape,
            access,
            platform,
            dtype: PhantomData,
        })
    }
}

// references
impl<T, B, P> Array<T, AccessBuffer<B>, P>
where
    T: CType,
    B: BufferInstance<T>,
    P: PlatformInstance,
{
    pub fn as_mut<RB: ?Sized>(&mut self) -> Array<T, AccessBuffer<&mut RB>, P>
    where
        B: BorrowMut<RB>,
    {
        Array {
            shape: Shape::from_slice(&self.shape),
            access: self.access.as_mut(),
            platform: self.platform,
            dtype: PhantomData,
        }
    }

    pub fn as_ref<RB: ?Sized>(&self) -> Array<T, AccessBuffer<&RB>, P>
    where
        B: Borrow<RB>,
    {
        Array {
            shape: Shape::from_slice(&self.shape),
            access: self.access.as_ref(),
            platform: self.platform,
            dtype: PhantomData,
        }
    }
}

// references
impl<T, O, P> Array<T, AccessOp<O, P>, P>
where
    T: CType,
    O: Enqueue<P>,
    P: PlatformInstance,
{
    pub fn as_mut<'a>(&'a mut self) -> Array<T, &'a mut AccessOp<O, P>, P>
    where
        O: Write<'a, P>,
    {
        Array {
            shape: Shape::from_slice(&self.shape),
            access: &mut self.access,
            platform: self.platform,
            dtype: PhantomData,
        }
    }

    pub fn as_ref(&self) -> Array<T, &AccessOp<O, P>, P> {
        Array {
            shape: Shape::from_slice(&self.shape),
            access: &self.access,
            platform: self.platform,
            dtype: PhantomData,
        }
    }
}

// write ops
impl<T, L, P> Array<T, L, P>
where
    T: CType,
{
    pub fn write<'a, R>(&'a mut self, other: &'a Array<T, R, P>) -> Result<(), Error>
    where
        L: AccessMut<'a, T>,
        R: Access<T> + 'a,
        P: Convert<'a, T, Buffer = L::Data>,
    {
        same_shape("write", self.shape(), other.shape())?;

        let data = other
            .access
            .read()
            .and_then(|buf| self.platform.convert(buf))?;

        self.access.write(data)
    }
}

// unary ops
impl<T, A, P> Array<T, A, P> {
    // transforms
    pub fn broadcast(self, shape: Shape) -> Result<Array<T, AccessOp<P::Broadcast, P>, P>, Error>
    where
        P: Transform<A, T>,
    {
        if !can_broadcast(self.shape(), &shape) {
            return Err(Error::Bounds(format!(
                "cannot broadcast {self:?} into {shape:?}"
            )));
        }

        let platform = P::select(shape.iter().product());
        let broadcast = Shape::from_slice(&shape);
        let access = platform.broadcast(self.access, self.shape, broadcast)?;

        Ok(Array {
            shape,
            access,
            platform,
            dtype: PhantomData,
        })
    }

    pub fn slice(self, range: Range) -> Result<Array<T, AccessOp<P::Slice, P>, P>, Error>
    where
        P: Transform<A, T>,
    {
        for (dim, range) in self.shape.iter().zip(&range) {
            match range {
                AxisRange::At(i) if i < dim => Ok(()),
                AxisRange::In(start, stop, _step) if start < dim && stop <= dim => Ok(()),
                AxisRange::Of(indices) if indices.iter().all(|i| i < dim) => Ok(()),
                range => Err(Error::Bounds(format!(
                    "invalid range {range:?} for dimension {dim}"
                ))),
            }?;
        }

        let shape = range.iter().filter_map(|ar| ar.size()).collect::<Shape>();
        let platform = P::select(shape.iter().product());
        let access = platform.slice(self.access, &self.shape, range)?;

        Ok(Array {
            shape,
            access,
            platform,
            dtype: PhantomData,
        })
    }

    // math
    pub fn ln(self) -> Result<Array<T, AccessOp<P::Op, P>, P>, Error>
    where
        P: ElementwiseUnary<A, T>,
    {
        self.platform.ln(self.access).map(|access| Array {
            access,
            shape: self.shape,
            platform: self.platform,
            dtype: self.dtype,
        })
    }
}

// array-array ops
impl<T, L, P> Array<T, L, P> {
    // array-array comparison
    pub fn eq<R>(self, other: Array<T, R, P>) -> Result<Array<u8, AccessOp<P::Op, P>, P>, Error>
    where
        P: ElementwiseCompare<L, R, T>,
    {
        same_shape("compare", self.shape(), other.shape())?;

        Ok(Array {
            shape: self.shape,
            access: self.platform.eq(self.access, other.access)?,
            platform: self.platform,
            dtype: PhantomData,
        })
    }

    // array-array arithmetic
    pub fn add<R>(self, other: Array<T, R, P>) -> Result<Array<T, AccessOp<P::Op, P>, P>, Error>
    where
        P: ElementwiseDual<L, R, T>,
    {
        same_shape("add", self.shape(), other.shape())?;

        Ok(Array {
            shape: self.shape,
            access: self.platform.add(self.access, other.access)?,
            platform: self.platform,
            dtype: PhantomData,
        })
    }

    pub fn sub<R>(self, other: Array<T, R, P>) -> Result<Array<T, AccessOp<P::Op, P>, P>, Error>
    where
        P: ElementwiseDual<L, R, T>,
    {
        same_shape("subtract", self.shape(), other.shape())?;

        Ok(Array {
            shape: self.shape,
            access: self.platform.sub(self.access, other.access)?,
            platform: self.platform,
            dtype: PhantomData,
        })
    }
}

impl<'a, T, A, P> Array<T, A, P>
where
    T: CType,
    A: Access<T>,
    P: Reduce<A, T>,
{
    pub fn all(self) -> Result<bool, Error> {
        self.platform.all(self.access)
    }

    pub fn any(self) -> Result<bool, Error> {
        self.platform.any(self.access)
    }

    pub fn sum_all(self) -> Result<T, Error> {
        self.platform.sum(self.access)
    }
}

impl<T, A, P> fmt::Debug for Array<T, A, P> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "a {} array of shape {:?}",
            std::any::type_name::<T>(),
            self.shape
        )
    }
}

#[inline]
fn can_broadcast(left: &[usize], right: &[usize]) -> bool {
    if left.len() < right.len() {
        return can_broadcast(right, left);
    }

    for (l, r) in left.iter().copied().rev().zip(right.iter().copied().rev()) {
        if l == r || l == 1 || r == 1 {
            // pass
        } else {
            return false;
        }
    }

    true
}

#[inline]
fn same_shape(op_name: &'static str, left: &[usize], right: &[usize]) -> Result<(), Error> {
    if left == right {
        Ok(())
    } else if can_broadcast(left, right) {
        Err(Error::Bounds(format!(
            "cannot {op_name} arrays with shapes {left:?} and {right:?} (consider broadcasting)"
        )))
    } else {
        Err(Error::Bounds(format!(
            "cannot {op_name} arrays with shapes {left:?} and {right:?}"
        )))
    }
}
