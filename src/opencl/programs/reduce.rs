use memoize::memoize;
use ocl::Program;

use crate::Error;

use super::build;

#[memoize]
pub fn fold_axis(c_type: &'static str, reduce: &'static str) -> Result<Program, Error> {
    let src = format!(
        r#"
        inline void add({c_type}* left, const {c_type} right) {{
            *left += right;
        }}

        inline void and({c_type}* left, const {c_type} right) {{
            if (left == 0) {{
                // no-op
            }} else if (right == 0) {{
                *left = 0;
            }}
        }}

        inline void mul({c_type}* left, const {c_type} right) {{
            *left *= right;
        }}

        inline void or({c_type}* left, const {c_type} right) {{
            if (left != 0) {{
                // no-op
            }} else if (right != 0) {{
                *left = right;
            }}
        }}

        __kernel void fold_axis(
            const ulong reduce_dim,
            const ulong target_dim,
            {c_type} init,
            __global const {c_type}* input,
            __global {c_type}* output)
        {{
            // the global offset in the output basis
            const ulong o_offset = get_global_id(0);

            // the local coordinate in the outer dimension
            const ulong a = o_offset / target_dim;

            // the local coordinate in the dimension to reduce
            const ulong b = o_offset % target_dim;

            // the global offset in the input basis
            const ulong i_offset = (a * reduce_dim) + b;

            {c_type} reduced = init;

            for (uint stride = i_offset; stride < (a + 1) * reduce_dim; stride += target_dim) {{
                {reduce}(&reduced, input[stride]);
            }}

            output[o_offset] = reduced;
        }}
        "#,
    );

    build(&src)
}

pub fn reduce_axis(c_type: &'static str, reduce: &'static str) -> Result<Program, Error> {
    let src = format!(
        r#"
        inline void add({c_type}* left, const {c_type} right) {{
            *left += right;
        }}

        inline void and({c_type}* left, const {c_type} right) {{
            if (left == 0) {{
                // no-op
            }} else if (right == 0) {{
                *left = 0;
            }}
        }}

        inline void mul({c_type}* left, const {c_type} right) {{
            *left *= right;
        }}

        inline void or({c_type}* left, const {c_type} right) {{
            if (left != 0) {{
                // no-op
            }} else if (right != 0) {{
                *left = right;
            }}
        }}

        __kernel void reduce(
                {c_type} init,
                __global const {c_type}* input,
                __global {c_type}* output,
                __local {c_type}* partials)
        {{
            const ulong offset = get_global_id(0);
            const uint reduce_dim = get_local_size(0);

            const ulong a = offset / reduce_dim;
            const uint b = offset % reduce_dim;

            // copy from global to local memory
            partials[b] = input[offset];

            // reduce over local memory in parallel
            for (uint stride = reduce_dim >> 1; stride > 0; stride = stride >> 1) {{
                barrier(CLK_LOCAL_MEM_FENCE);

                uint next = b + stride;
                if (next < reduce_dim) {{
                    {reduce}(&partials[b], partials[next]);
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

#[memoize]
pub fn reduce(c_type: &'static str, reduce: &'static str) -> Result<Program, Error> {
    let src = format!(
        r#"
        inline void add({c_type}* left, const {c_type} right) {{
            *left += right;
        }}

        inline void and({c_type}* left, const {c_type} right) {{
            if (left == 0) {{
                // no-op
            }} else if (right == 0) {{
                *left = 0;
            }}
        }}

        inline void mul({c_type}* left, const {c_type} right) {{
            *left *= right;
        }}

        inline void or({c_type}* left, const {c_type} right) {{
            if (left != 0) {{
                // no-op
            }} else if (right != 0) {{
                *left = right;
            }}
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
