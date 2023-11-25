use std::fmt;

use crate::Error;

use super::{OpenCL, TILE_SIZE, WG_SIZE};

pub mod constructors;
pub mod elementwise;
pub mod linalg;
pub mod reduce;
pub mod slice;
pub mod view;

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

#[inline]
fn build(src: &str) -> Result<ocl::Program, Error> {
    ocl::Program::builder()
        .source(src)
        .build(OpenCL::context())
        .map_err(Error::from)
}
