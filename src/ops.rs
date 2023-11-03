use crate::access::AccessOp;
use crate::{Error, PlatformInstance};

pub trait ElementwiseDual<L, R, T>: PlatformInstance {
    type Output;

    fn add(self, left: L, right: R) -> Result<AccessOp<Self::Output, Self>, Error>;
}
