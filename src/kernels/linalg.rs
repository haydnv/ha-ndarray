use ocl::core::Ulong4;
use ocl::{Buffer, Error, Event, Kernel, Program, Queue};

use crate::CDatatype;

pub fn matmul<T: CDatatype>(
    queue: Queue,
    left: Buffer<T>,
    right: Buffer<T>,
    batch_size: usize,
    dims: (usize, usize, usize),
    ewait: Event,
) -> Result<Buffer<T>, Error> {
    let (a, b, c) = dims;

    debug_assert!(batch_size > 0);
    debug_assert_eq!(batch_size * a * b, left.len());
    debug_assert_eq!(batch_size * b * c, right.len());

    let src = format!(
        r#"
        __kernel void matmul(
                ulong4 const dims,
                __global const {dtype}* restrict left,
                __global const {dtype}* restrict right,
                __global {dtype}* output)
        {{
            // x := output axis 0
            // y := reduce axis
            // z := output axis 1
            // w := matrix number

            const ulong x = get_global_id(1);
            const ulong z = get_global_id(2);
            const ulong w = get_global_id(0);

            {dtype} sum = 0;

            #pragma unroll
            for (ulong y = 0; y < dims.y; y++) {{
                {dtype} l = left[(w * dims.x * dims.y) + (x * dims.y) + y];
                {dtype} r = right[(w * dims.y * dims.z) + (y * dims.z) + z];
                sum += (l * r);
            }}

            output[(w * dims.x * dims.z) + (x * dims.z) + z] = sum;
        }}
        "#,
        dtype = T::TYPE_STR
    );

    let program = Program::builder().source(src).build(&queue.context())?;

    let dims = [a as u64, b as u64, c as u64, batch_size as u64];

    let output = Buffer::builder()
        .queue(queue.clone())
        .len(a * c * batch_size)
        .build()?;

    let kernel = Kernel::builder()
        .name("matmul")
        .program(&program)
        .queue(queue)
        .global_work_size((batch_size, a, c))
        .arg(Ulong4::from(dims))
        .arg(left)
        .arg(right)
        .arg(&output)
        .build()?;

    unsafe { kernel.cmd().ewait(&ewait).enq()? }

    Ok(output)
}
