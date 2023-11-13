use std::fmt;

use super::div_ceil;

pub mod elementwise;
pub mod reduce;
pub mod view;

const TILE_SIZE: usize = 8;
const WG_SIZE: usize = 64;

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
