use ocl::core::Ulong4;
use ocl::{Buffer, Error, Event, Kernel, Program, Queue};

use crate::CDatatype;

use super::WG_SIZE;

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

    let mut ewait = Some(ewait);

    // TODO: remove these conditions
    assert!(b <= WG_SIZE);

    let src = format!(
        r#"
        __kernel void matmul(
                const ulong4 dims,
                const ulong4 strides,
                __global const {dtype}* left,
                __global const {dtype}* right,
                __global {dtype}* output,
                __local {dtype}* partials)
        {{
            const ulong offset = get_global_id(0);
            const ulong4 coord = (offset / strides) % dims;

            // x := matrix number
            // y := output axis 0
            // z := output axis 1
            // w := axis to reduce

            // copy axis w from the left matrix
            partials[coord.w] = left[(coord.x * dims.y * dims.w) + (coord.y * dims.w) + coord.w];
            barrier(CLK_LOCAL_MEM_FENCE);

            // multiply by axis w from the right matrix
            partials[coord.w] *= right[(coord.x * dims.w * dims.z) + (coord.w * dims.z) + coord.z];
            barrier(CLK_LOCAL_MEM_FENCE);

            // sum over axis w

            if (coord.w == 0) {{
                // TODO: parallelize
                {dtype} sum = 0;
                for (uint b = 0; b < dims.w; b++) {{
                    sum += partials[b];
                }}

                ulong xyz = (coord.x * dims.y * dims.z) + (coord.y * dims.z) + coord.z;
                output[xyz] = sum;
            }}
        }}
        "#,
        dtype = T::TYPE_STR
    );

    let program = Program::builder().source(src).build(&queue.context())?;

    let wg_size = if b < WG_SIZE {
        b
    } else {
        assert_eq!(b % WG_SIZE, 0);
        WG_SIZE
    };

    let dims = [num_matrices as u64, a as u64, c as u64, b as u64];

    let strides = [(a * c * b) as u64, (c * b) as u64, b as u64, 1];

    let output = Buffer::builder()
        .queue(queue.clone())
        .len(a * c * num_matrices)
        .build()?;

    let kernel = Kernel::builder()
        .name("matmul")
        .program(&program)
        .queue(queue)
        .local_work_size(wg_size)
        .global_work_size(a * b * c * num_matrices)
        .arg(Ulong4::from(dims))
        .arg(Ulong4::from(strides))
        .arg(left)
        .arg(right)
        .arg(&output)
        .arg_local::<T>(wg_size)
        .build()?;

    if let Some(ewait) = ewait.take() {
        unsafe { kernel.cmd().ewait(ewait).enq()? }
    } else {
        unsafe { kernel.enq()? }
    }

    Ok(output)
}
