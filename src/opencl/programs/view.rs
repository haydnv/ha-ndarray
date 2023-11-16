use memoize::memoize;
use ocl::Program;

use crate::{Error, Shape, Strides};

use super::{build, ArrayFormat};

// TODO: support SharedCache
#[memoize(Capacity: 1024)]
pub fn view(
    c_type: &'static str,
    shape: Shape,
    strides: Strides,
    source_strides: Strides,
) -> Result<Program, Error> {
    let ndim = shape.len();
    assert_eq!(strides.len(), ndim);

    let source_ndim = source_strides.len();

    let dims = ArrayFormat::from(shape.as_slice());
    let strides = ArrayFormat::from(strides.as_slice());
    let source_strides = ArrayFormat::from(source_strides.as_slice());

    let src = format!(
        r#"
        const uint ndim = {ndim};
        const ulong dims[{ndim}] = {dims};
        const ulong strides[{ndim}] = {strides};

        const ulong source_strides[{source_ndim}] = {source_strides};

        __kernel void view(
                __global const {c_type}* restrict input,
                __global {c_type}* restrict output)
        {{
            ulong offset = get_global_id(0);

            ulong coord[{ndim}];
            #pragma unroll
            for (uint x = 0; x < {ndim}; x++) {{
                if (strides[x] == 0) {{
                    coord[x] = 0;
                }} else {{
                    coord[x] = (offset / strides[x]) % dims[x];
                }}
            }}

            ulong source_offset = 0;
            #pragma unroll
            for (uint x = 0; x < {ndim}; x++) {{
                source_offset += coord[x] * source_strides[x];
            }}

            output[offset] = input[source_offset];
        }}
        "#,
    );

    build(&src)
}
