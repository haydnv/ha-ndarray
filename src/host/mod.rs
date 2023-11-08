use crate::access::AccessBuffer;

pub use buffer::*;
pub use platform::*;

mod buffer;
pub mod ops;
mod platform;

pub type Array<T> = crate::array::Array<T, AccessBuffer<Buffer<T>>, Host>;
