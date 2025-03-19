use std::fmt;

use crate::Error;

use super::{CLElement, OpenCL, TILE_SIZE, WG_SIZE};

pub mod constructors;
pub mod elementwise;
pub mod gather;
pub mod linalg;
pub mod reduce;
pub mod slice;
pub mod view;

#[derive(Clone, Eq, PartialEq, Hash, fmt::Debug)]
pub(super) enum CLExpr {
    Static(&'static str),
    String(String),
}

impl From<&'static str> for CLExpr {
    fn from(s: &'static str) -> Self {
        CLExpr::Static(s)
    }
}

impl From<String> for CLExpr {
    fn from(s: String) -> Self {
        CLExpr::String(s)
    }
}

impl fmt::Display for CLExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Static(s) => f.write_str(s),
            Self::String(s) => f.write_str(s),
        }
    }
}

pub trait Builder {
    fn build(self) -> String;
}

#[derive(Clone, Eq, PartialEq, Hash, fmt::Debug)]
pub struct ElementDual {
    pub(super) i_type: &'static str,
    pub(super) o_type: &'static str,
    pub(super) name: &'static str,
    pub(super) op: CLExpr,
}

impl ElementDual {
    pub(super) fn new<I, O, Op>(name: &'static str, op: Op) -> Self
    where
        I: CLElement,
        O: CLElement,
        Op: Into<CLExpr>,
    {
        Self {
            i_type: I::TYPE,
            o_type: O::TYPE,
            name,
            op: op.into(),
        }
    }
}

impl Builder for ElementDual {
    fn build(self) -> String {
        format!(
            r#"
            inline {o_type} {name}(const {i_type} lhs, const {i_type} rhs) {{
                {op}
            }}
            "#,
            i_type = self.i_type,
            o_type = self.o_type,
            name = self.name,
            op = self.op
        )
    }
}

pub struct ElementUnary {
    pub(super) i_type: &'static str,
    pub(super) o_type: &'static str,
    pub(super) name: &'static str,
    pub(super) op: CLExpr,
}

impl ElementUnary {
    pub(super) fn new<I, O, Op>(name: &'static str, op: Op) -> Self
    where
        I: CLElement,
        O: CLElement,
        Op: Into<CLExpr>,
    {
        Self {
            i_type: I::TYPE,
            o_type: O::TYPE,
            name,
            op: op.into(),
        }
    }
}

impl Builder for ElementUnary {
    fn build(self) -> String {
        format!(
            "
            inline {o_type} {name}(const {i_type} n) {{
                {op}
            }}
            ",
            i_type = self.i_type,
            o_type = self.o_type,
            name = self.name,
            op = self.op,
        )
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
