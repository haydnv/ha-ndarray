use crate::access::AccessOp;
use crate::{Error, PlatformInstance};

pub trait ElementwiseCompare<L, R, T>: PlatformInstance {
    type Output;

    fn eq(self, left: L, right: R) -> Result<AccessOp<Self::Output, Self>, Error>;
}

pub trait ElementwiseDual<L, R, T>: PlatformInstance {
    type Output;

    fn add(self, left: L, right: R) -> Result<AccessOp<Self::Output, Self>, Error>;
}

pub trait Reduce<A, T>: PlatformInstance {
    fn all(self, access: A) -> Result<bool, Error>;
}
