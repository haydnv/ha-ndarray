use memoize::memoize;
use ocl::Program;

use crate::Error;

use super::{build, TILE_SIZE, WG_SIZE};

#[memoize]
pub fn pad_matrices(c_type: &'static str) -> Result<Program, Error> {
    let src = format!(
        r#"
        __kernel void pad_matrices(
                ulong const batch_size,
                ulong const stride_in,
                ulong const stride_out,
                __global const {c_type}* restrict input,
                __global {c_type}* restrict output)
        {{
            const ulong offset_in = get_global_id(0);

            const ulong w = offset_in / batch_size;     // matrix number
            const ulong x = offset_in % stride_in;      // row index
            const ulong y = offset_in;                  // column index

            const ulong offset_out = (w * stride_out) + x + y;

            output[offset_out] = input[offset_in];
        }}
    "#
    );

    build(&src)
}

#[memoize]
pub fn matmul(c_type: &'static str) -> Result<Program, Error> {
    debug_assert_eq!(TILE_SIZE * TILE_SIZE, WG_SIZE);

    let src = format!(
        r#"
        __kernel void matmul(
                ulong4 const dims,
                ulong const reduce_tiles,
                __global const {c_type}* restrict left,
                __global const {c_type}* restrict right,
                __global {c_type}* restrict output)
        {{
            // x := output axis 0
            // y := reduce axis
            // z := output axis 1
            // w := matrix number

            const ulong x_tile = get_global_id(1);
            const ulong z_tile = get_global_id(2);
            const ulong w = get_global_id(0);

            const ulong x_offset = x_tile * {TILE_SIZE};
            const ulong z_offset = z_tile * {TILE_SIZE};
            const ulong left_offset = w * dims.x * dims.y;
            const ulong right_offset = w * dims.y * dims.z;

            {c_type} tile[{TILE_SIZE}][{TILE_SIZE}];

            // initialize the local cache for the left and right tiles to zero
            {c_type} left_tile[{TILE_SIZE}][{TILE_SIZE}];
            {c_type} right_tile[{TILE_SIZE}][{TILE_SIZE}];

            // for each tile on the y axis
            for (ulong y_tile = 0; y_tile < reduce_tiles; y_tile++) {{
                const ulong y_offset = y_tile * {TILE_SIZE};

                // read the left and right tiles into the local cache
                #pragma unroll
                for (uint i = 0; i < {TILE_SIZE}; i++) {{
                    #pragma unroll
                    for (uint j = 0; j < {TILE_SIZE}; j++) {{
                        if (x_offset + i < dims.x && y_offset + j < dims.y) {{
                            ulong offset = left_offset + ((x_offset + i) * dims.y) + (y_offset + j);
                            left_tile[i][j] = left[offset];
                        }} else {{
                            left_tile[i][j] = 0;
                        }}

                        if (y_offset + i < dims.y && z_offset + j < dims.z) {{
                            ulong offset = right_offset + ((y_offset + i) * dims.z) + (z_offset + j);
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
            ulong offset = (w * dims.x * dims.z) + (x_offset * dims.z) + z_offset;

            #pragma unroll
            for (uint i = 0; i < {TILE_SIZE}; i++) {{
                #pragma unroll
                for (uint j = 0; j < {TILE_SIZE}; j++) {{
                    if ((x_offset + i) < dims.x && (z_offset + j) < dims.z) {{
                        output[offset + (i * dims.z) + j] = tile[i][j];
                    }}
                }}
            }}
        }}
        "#,
    );

    build(&src)
}
