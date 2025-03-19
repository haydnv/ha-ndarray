use memoize::memoize;
use ocl::Program;

use crate::Error;

use super::{build, Builder, ElementDual};

#[memoize]
pub fn fold_axis(op: ElementDual) -> Result<Program, Error> {
    let i_type = op.i_type;
    let o_type = op.o_type;
    let name = op.name;
    let op = op.build();

    let src = format!(
        r#"
        {op}

        __kernel void fold_axis(
            const ulong reduce_dim,
            const ulong target_dim,
            {i_type} init,
            __global const {i_type}* input,
            __global {o_type}* output)
        {{
            // the global offset in the output basis
            const ulong o_offset = get_global_id(0);

            // the local coordinate in the outer dimension
            const ulong a = o_offset / target_dim;

            // the local coordinate in the dimension to reduce
            const ulong b = o_offset % target_dim;

            // the global offset in the input basis
            const ulong i_offset = (a * reduce_dim) + b;

            {o_type} reduced = init;

            for (uint stride = i_offset; stride < (a + 1) * reduce_dim; stride += target_dim) {{
                reduced = {name}(reduced, input[stride]);
            }}

            output[o_offset] = reduced;
        }}
        "#,
    );

    build(&src)
}

pub fn reduce_axis(op: ElementDual) -> Result<Program, Error> {
    let i_type = op.i_type;
    let o_type = op.o_type;
    let name = op.name;
    let op = op.build();

    let src = format!(
        r#"
        {op}

        __kernel void reduce(
                {i_type} init,
                __global const {i_type}* input,
                __global {o_type}* output,
                __local {o_type}* partials)
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
                    partials[b] = {name}(partials[b], partials[next]);
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
pub fn reduce(op: ElementDual) -> Result<Program, Error> {
    let i_type = op.i_type;
    let o_type = op.o_type;
    let name = op.name;
    let op = op.build();

    let src = format!(
        r#"
        {op}

        __kernel void reduce(
                const ulong size,
                __global const {i_type}* input,
                __global {o_type}* output,
                __local {o_type}* partials)
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
                        partials[b] = {name}(partials[b], partials[b + stride]);
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
