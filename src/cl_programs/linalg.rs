use ocl::{Error, Program};

use crate::{CDatatype, Context};

use super::{TILE_SIZE, WG_SIZE};

pub fn matmul<T: CDatatype>(context: &Context) -> Result<Program, Error> {
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

    Program::builder().source(src).build(context.cl_context())
}
