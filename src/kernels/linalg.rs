use ocl::{Buffer, Error, Event, Queue};

use crate::CDatatype;

pub fn matmul<T: CDatatype>(
    queue: Queue,
    left: &Buffer<T>,
    right: &Buffer<T>,
    num_matrices: usize,
    dims: (usize, usize, usize),
    ewait: &Event,
) -> Result<Buffer<T>, Error> {
    let (a, b, c) = dims;
    debug_assert_eq!(num_matrices * a * b, left.len());
    debug_assert_eq!(num_matrices * b * c, right.len());

    todo!()
}
