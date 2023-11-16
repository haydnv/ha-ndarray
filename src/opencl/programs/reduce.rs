use memoize::memoize;
use ocl::Program;

use crate::Error;

use super::build;

#[memoize]
pub fn all(c_type: &'static str) -> Result<Program, Error> {
    let src = format!(
        r#"
        __kernel void all(
                __global uchar* flag,
                __global const {c_type}* input)
        {{
            if (input[get_global_id(0)] == 0) {{
                flag[0] = 0;
            }}
        }}
        "#
    );

    build(&src)
}

#[memoize]
pub fn any(c_type: &'static str) -> Result<Program, Error> {
    let src = format!(
        r#"
        __kernel void any(
                __global uchar* flag,
                __global const {c_type}* input)
        {{
            if (input[get_global_id(0)] != 0) {{
                flag[0] = 1;
            }}
        }}
        "#
    );

    build(&src)
}

#[memoize]
pub fn reduce(c_type: &'static str, reduce: &'static str) -> Result<Program, Error> {
    let src = format!(
        r#"
        inline void add({c_type}* left, const {c_type} right) {{
            *left += right;
        }}

        inline void mul({c_type}* left, const {c_type} right) {{
            *left *= right;
        }}

        __kernel void reduce(
                const ulong size,
                __global const {c_type}* input,
                __global {c_type}* output,
                __local {c_type}* partials)
        {{
            const ulong offset = get_global_id(0);
            const uint group_size = get_local_size(0);
            const ulong a = offset / group_size;
            const uint b = offset % group_size;

            // copy from global to local memory
            partials[b] = input[offset];

            // reduce over local memory in parallel
            for (uint stride = group_size >> 1; stride > 0; stride = stride >> 1) {{
                barrier(CLK_LOCAL_MEM_FENCE);

                if (offset + stride < size) {{
                    uint next = b + stride;
                    if (next < group_size) {{
                        {reduce}(&partials[b], partials[b + stride]);
                    }}
                }}
            }}

            if (b == 0) {{
                output[a] = partials[b];
            }}
        }}
        "#,
    );

    build(&src)
}
