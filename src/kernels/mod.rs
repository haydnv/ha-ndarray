mod elementwise;
mod linalg;
mod reduce;
mod slice;
mod view;

pub use elementwise::*;
pub use linalg::*;
pub use reduce::*;
pub use slice::*;
pub use view::*;

// TODO: move to a custom Platform struct
const MIN_SIZE: usize = 1024;

// TODO: is there a good way to determine this at runtime?
const WG_SIZE: usize = 64;

#[inline]
fn div_ceil(num: usize, denom: usize) -> usize {
    if num % denom == 0 {
        num / denom
    } else {
        (num / denom) + 1
    }
}
