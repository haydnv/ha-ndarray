use std::fmt;

use crate::Error;

use super::{OpenCL, TILE_SIZE, WG_SIZE};

pub mod constructors;
pub mod elementwise;
pub mod gather;
pub mod linalg;
pub mod reduce;
pub mod slice;
pub mod view;

pub trait Builder {
    fn build(self) -> Result<String, Error>;
}

#[derive(Clone, Hash, Eq, PartialEq, fmt::Debug)]
pub struct ElementDual {
    pub c_type: &'static str,
    pub name: &'static str,
    pub op: &'static str,
}

impl Builder for ElementDual {
    fn build(self) -> Result<String, Error> {
        Ok(format!(
            r#"
            inline {c_type} {name}(const {c_type} lhs, const {c_type} rhs) {{
                {op}
            }}
            "#,
            c_type = self.c_type,
            name = self.name,
            op = self.op
        ))
    }
}

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
