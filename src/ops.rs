use crate::access::AccessOp;
use crate::PlatformInstance;

pub trait ElementwiseDual<L, R, T>: PlatformInstance {
    type Output;

    fn add(self, left: L, right: R) -> AccessOp<Self::Output, Self>;
}
