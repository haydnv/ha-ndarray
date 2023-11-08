use std::borrow::Borrow;
use std::marker::PhantomData;

use crate::access::*;
use crate::buffer::{BufferConverter, BufferInstance};
use crate::ops::*;
use crate::{CType, Convert, Error, PlatformInstance, ReadBuf, Shape};

pub struct Array<T, A, P> {
    shape: Shape,
    access: A,
    platform: P,
    dtype: PhantomData<T>,
}

impl<T, A, P> Array<T, A, P> {
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

impl<T, L, P> Array<T, L, P> {
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
    A: ReadBuf<'a, T>,
    P: Reduce<A, T>,
{
    pub fn all(self) -> Result<bool, Error> {
        self.platform.all(self.access)
    }

    pub fn any(self) -> Result<bool, Error> {
        self.platform.any(self.access)
    }
}
