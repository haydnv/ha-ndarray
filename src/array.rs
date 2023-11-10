use std::borrow::Borrow;
use std::fmt;
use std::marker::PhantomData;

use crate::access::*;
use crate::buffer::{BufferConverter, BufferInstance};
use crate::ops::*;
use crate::platform::{Convert, PlatformInstance};
use crate::{CType, Error, Shape};

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

impl<T, B, P> Array<T, AccessBuffer<B>, P>
where
    T: CType,
    B: BufferInstance<T>,
    P: PlatformInstance,
{
    pub fn new<'a, D>(buffer: D, shape: Shape) -> Result<Self, Error>
    where
        P: Convert<T, Buffer = B>,
        D: Into<BufferConverter<'a, T>>,
    {
        let buffer = buffer.into();

        if shape.iter().product::<usize>() == buffer.size() {
            let platform = P::select(buffer.size());
            let buffer = platform.convert(buffer)?;
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

// unary ops
impl<T, A, P> Array<T, A, P> {
    // transforms
    pub fn broadcast(self, shape: Shape) -> Result<Array<T, AccessOp<P::Broadcast, P>, P>, Error>
    where
        P: Transform<A, T>,
    {
        if shape.len() < self.ndim() {
            return Err(Error::Bounds(format!(
                "cannot broadcast {self:?} into {shape:?}"
            )));
        }

        for (dim, bdim) in self.shape().iter().rev().zip(shape.iter().rev()) {
            if dim == bdim || *dim == 1 {
                // ok
            } else {
                return Err(Error::Bounds(format!(
                    "cannot broadcast dimension {dim} into {bdim}"
                )));
            }
        }

        let platform = self.platform;

        let access = self.platform.broadcast(self, &shape)?;

        Ok(Array {
            shape,
            access,
            platform,
            dtype: PhantomData,
        })
    }
}

// array-array ops
impl<T, L, P> Array<T, L, P> {
    // array-array comparison
    pub fn eq<R>(self, other: Array<T, R, P>) -> Result<Array<u8, AccessOp<P::Output, P>, P>, Error>
    where
        P: ElementwiseCompare<L, R, T>,
    {
        if self.shape == other.shape {
            Ok(Array {
                shape: self.shape,
                access: self.platform.eq(self.access, other.access)?,
                platform: self.platform,
                dtype: PhantomData,
            })
        } else {
            todo!("broadcast")
        }
    }

    // array-array arithmetic
    pub fn add<R>(self, other: Array<T, R, P>) -> Result<Array<T, AccessOp<P::Output, P>, P>, Error>
    where
        P: ElementwiseDual<L, R, T>,
    {
        if self.shape == other.shape {
            Ok(Array {
                shape: self.shape,
                access: self.platform.add(self.access, other.access)?,
                platform: self.platform,
                dtype: PhantomData,
            })
        } else {
            todo!("broadcast")
        }
    }

    pub fn sub<R>(self, other: Array<T, R, P>) -> Result<Array<T, AccessOp<P::Output, P>, P>, Error>
    where
        P: ElementwiseDual<L, R, T>,
    {
        if self.shape == other.shape {
            Ok(Array {
                shape: self.shape,
                access: self.platform.sub(self.access, other.access)?,
                platform: self.platform,
                dtype: PhantomData,
            })
        } else {
            todo!("broadcast")
        }
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
