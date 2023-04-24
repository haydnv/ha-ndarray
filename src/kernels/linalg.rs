use ocl::core::Ulong4;
use ocl::{Buffer, Error, Event, Kernel, Program, Queue};

use crate::CDatatype;

use super::{div_ceil, WG_SIZE};

const TILE_SIZE: usize = 8;

pub fn matmul<T: CDatatype>(
    queue: Queue,
    left: Buffer<T>,
    right: Buffer<T>,
    batch_size: usize,
    dims: (usize, usize, usize),
    ewait: Event,
) -> Result<Buffer<T>, Error> {
    let (a, b, c) = dims;

    assert!(batch_size > 0);
    assert_eq!(batch_size * a * b, left.len());
    assert_eq!(batch_size * b * c, right.len());

    debug_assert_eq!(TILE_SIZE * TILE_SIZE, WG_SIZE);

    let src = format!(
        r#"
        __kernel void matmul(
                ulong4 const dims,
                ulong const reduce_tiles,
                __global const {dtype}* restrict left,
                __global const {dtype}* restrict right,
                __global {dtype}* output)
        {{
            // x := output axis 0
            // y := reduce axis
            // z := output axis 1
            // w := matrix number

            const ulong x_tile = get_global_id(1);
            const ulong z_tile = get_global_id(2);
            const ulong w = get_global_id(0);

            {dtype} tile[{TILE_SIZE}][{TILE_SIZE}];

            // initialize the local cache for the left and right tiles to zero
            {dtype} left_tile[{TILE_SIZE}][{TILE_SIZE}];
            {dtype} right_tile[{TILE_SIZE}][{TILE_SIZE}];

            // for each tile on the y axis
            for (ulong y_tile = 0; y_tile < reduce_tiles; y_tile++) {{
                // read the left and right tiles into the local cache
                #pragma unroll
                for (uint i = 0; i < {TILE_SIZE}; i++) {{
                    #pragma unroll
                    for (uint j = 0; j < {TILE_SIZE}; j++) {{
                        if ((x_tile * {TILE_SIZE}) + i < dims.x && (y_tile * {TILE_SIZE}) + j < dims.y) {{
                            ulong offset = (w * dims.x * dims.y) + (((x_tile * {TILE_SIZE}) + i) * dims.y) + ((y_tile * {TILE_SIZE}) + j);
                            left_tile[i][j] = left[offset];
                        }} else {{
                            left_tile[i][j] = 0;
                        }}

                        if ((y_tile * {TILE_SIZE}) + i < dims.y && (z_tile * {TILE_SIZE}) + j < dims.z) {{
                            ulong offset = (w * dims.y * dims.z) + (((y_tile * {TILE_SIZE}) + i) * dims.z) + ((z_tile * {TILE_SIZE}) + j);
                            right_tile[i][j] = right[offset];
                        }} else {{
                            right_tile[i][j] = 0;
                        }}
                    }}
                }}

                // tile += left tile @ right tile
                #pragma unroll
                for (uint i = 0; i < {TILE_SIZE}; i++) {{
                    #pragma unroll
                    for (uint j = 0; j < {TILE_SIZE}; j++) {{
                        #pragma unroll
                        for (uint k = 0; k < {TILE_SIZE}; k++) {{
                            tile[i][k] += left_tile[i][j] * right_tile[j][k];
                        }}
                    }}
                }}
            }}

            // write tile to output
            ulong offset = (w * dims.x * dims.z) + (x_tile * {TILE_SIZE} * dims.z) + (z_tile * {TILE_SIZE});

            #pragma unroll
            for (uint i = 0; i < {TILE_SIZE}; i++) {{
                #pragma unroll
                for (uint j = 0; j < {TILE_SIZE}; j++) {{
                    if (((x_tile * {TILE_SIZE}) + i) < dims.x && ((z_tile * {TILE_SIZE}) + j) < dims.z) {{
                        output[offset + (i * dims.z) + j] = tile[i][j];
                    }}
                }}
            }}
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
        .global_work_size((batch_size, div_ceil(a, TILE_SIZE), div_ceil(c, TILE_SIZE)))
        .arg(Ulong4::from(dims))
        .arg(div_ceil(b, TILE_SIZE))
        .arg(left)
        .arg(right)
        .arg(&output)
        .build()?;

    unsafe { kernel.cmd().ewait(&ewait).enq()? }

    Ok(output)
}
