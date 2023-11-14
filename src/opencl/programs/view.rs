use ocl::{Context, Error, Program};

use crate::CType;

use super::ArrayFormat;

pub fn view<T: CType>(
    context: &Context,
    shape: &[usize],
    strides: &[usize],
    source_strides: &[usize],
) -> Result<Program, Error> {
    let ndim = shape.len();
    assert_eq!(strides.len(), ndim);

    let source_ndim = source_strides.len();

    let dims = ArrayFormat::from(shape);
    let strides = ArrayFormat::from(strides);
    let source_strides = ArrayFormat::from(source_strides);

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
        c_type = T::TYPE,
    );

    Program::builder().source(src).build(context)
}
