use std::fmt;

mod elementwise;
mod gather;
mod linalg;
mod random;
mod reduce;
mod slice;
mod view;

pub use elementwise::*;
pub use gather::*;
pub use linalg::*;
pub use random::*;
pub use reduce::*;
pub use slice::*;
pub use view::*;

pub(crate) const TILE_SIZE: usize = 8;
pub(crate) const WG_SIZE: usize = 64;

struct ArrayFormat<'a, T> {
    arr: &'a [T],
}

impl<'a, T> From<&'a [T]> for ArrayFormat<'a, T> {
    fn from(arr: &'a [T]) -> Self {
        Self { arr }
    }
}

impl<'a, T: fmt::Display> fmt::Display for ArrayFormat<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("{ ")?;

        for item in self.arr {
            write!(f, "{item}, ")?;
        }

        f.write_str(" }")
    }
}
