use ocl::{Buffer, Context, Error, Kernel, Program, Queue};
use rayon::prelude::*;

use crate::CType;

use super::{div_ceil, WG_SIZE};

pub fn all<T: CType>(context: &Context) -> Result<Program, Error> {
    let src = format!(
        r#"
        __kernel void all(
                __global uchar* flag,
                __global const {c_type}* input)
        {{
            if (input[get_global_id(0)] == {zero}) {{
                flag[0] = 0;
            }}
        }}
        "#,
        c_type = T::TYPE,
        zero = T::ZERO,
    );

    Program::builder().source(src).build(context)
}

pub fn any<T: CType>(context: &Context) -> Result<Program, Error> {
    let src = format!(
        r#"
        __kernel void any(
                __global uchar* flag,
                __global const {c_type}* input)
        {{
            if (input[get_global_id(0)] != {zero}) {{
                flag[0] = 1;
            }}
        }}
        "#,
        c_type = T::TYPE,
        zero = T::ZERO,
    );

    Program::builder().source(src).build(context)
}

pub fn reduce<T: CType>(
    init: T,
    reduce: &'static str,
    queue: Queue,
    input: &Buffer<T>,
    collector: impl Fn(T, T) -> T + Send + Sync,
) -> Result<T, Error> {
    const MIN_SIZE: usize = 8192;

    let min_size = MIN_SIZE * num_cpus::get();

    if input.len() < min_size {
        let mut result = vec![init; input.len()];
        input.read(&mut result).enq()?;
        return Ok(result.into_par_iter().reduce(|| init, collector));
    }

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
        c_type = T::TYPE
    );

    let program = Program::builder().source(src).build(&queue.context())?;

    let mut buffer = {
        let output = Buffer::builder()
            .queue(queue.clone())
            .len(div_ceil(input.len(), WG_SIZE))
            .fill_val(init)
            .build()?;

        let kernel = Kernel::builder()
            .name("reduce")
            .program(&program)
            .queue(queue.clone())
            .local_work_size(WG_SIZE)
            .global_work_size(WG_SIZE * output.len())
            .arg(input.len() as u64)
            .arg(input)
            .arg(&output)
            .arg_local::<T>(WG_SIZE)
            .build()?;

        unsafe { kernel.enq()? };

        output
    };

    while buffer.len() >= min_size {
        let input = buffer;

        let output = Buffer::builder()
            .queue(queue.clone())
            .len(div_ceil(input.len(), WG_SIZE))
            .fill_val(init)
            .build()?;

        let kernel = Kernel::builder()
            .name("reduce")
            .program(&program)
            .queue(queue.clone())
            .local_work_size(WG_SIZE)
            .global_work_size(WG_SIZE * output.len())
            .arg(input.len() as u64)
            .arg(&input)
            .arg(&output)
            .arg_local::<T>(WG_SIZE)
            .build()?;

        unsafe { kernel.enq()? }

        buffer = output;
    }

    let mut result = vec![init; buffer.len()];
    buffer.read(&mut result).enq()?;

    queue.finish()?;

    Ok(result.into_par_iter().reduce(|| init, collector))
}
