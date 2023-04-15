mod elementwise;
mod linalg;
mod reduce;

pub use elementwise::*;
pub use linalg::*;
pub use reduce::*;

// TODO: move to a custom Platform struct
const MIN_SIZE: usize = 1024;

// TODO: is there a good way to determine this at runtime?
const WG_SIZE: usize = 64;
