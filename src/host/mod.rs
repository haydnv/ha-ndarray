use smallvec::SmallVec;

pub use platform::*;
mod ops;
mod platform;

pub type StackVec<T> = SmallVec<[T; VEC_MIN_SIZE]>;
